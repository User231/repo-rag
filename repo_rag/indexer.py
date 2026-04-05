"""Indexing orchestrator: sources -> chunks -> vector store."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from repo_rag.chunker import Chunk, chunk_file, chunk_text, is_code_file
from repo_rag.config import RepoRagConfig
from repo_rag.sources import GitHubFetcher, LocalScanner, RawDocument, WebFetcher
from repo_rag.store import VectorStore

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class IndexResult:
    total_documents: int = 0
    total_chunks: int = 0
    by_source_type: dict[str, int] = field(default_factory=dict)
    by_language: dict[str, int] = field(default_factory=dict)
    duration_seconds: float = 0.0


# ── State management ─────────────────────────────────────────────────────


def _get_git_head(project_dir: Path) -> str | None:
    """Get current HEAD commit hash, or None if not a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _get_git_changed_files(project_dir: Path, since_commit: str) -> list[str] | None:
    """Get list of files changed since a commit. Returns None on error."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", since_commit, "HEAD"],
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
            return files
    except Exception:
        pass
    return None


def _load_state(config: RepoRagConfig) -> dict:
    """Load indexing state from cache."""
    if config.state_file.exists():
        try:
            return json.loads(config.state_file.read_text())
        except Exception:
            pass
    return {}


def _save_state(config: RepoRagConfig, state: dict) -> None:
    """Save indexing state to cache."""
    config.state_file.parent.mkdir(parents=True, exist_ok=True)
    config.state_file.write_text(json.dumps(state, indent=2))


# ── Chunk ID generation ─────────────────────────────────────────────────


def make_chunk_id(source: str, chunk_index: int) -> str:
    """Deterministic chunk ID from source + index."""
    raw = f"{source}:{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


# ── Main indexer ─────────────────────────────────────────────────────────


class Indexer:
    """Orchestrates: scan sources -> chunk -> embed -> upsert to Qdrant."""

    def __init__(self, config: RepoRagConfig, store: VectorStore):
        self.config = config
        self.store = store

    def index_all(self, *, force: bool = False, project_dir: Path | None = None) -> IndexResult:
        """Full indexing pipeline.

        If ``force``, recreate the collection from scratch.
        Otherwise, try incremental indexing based on git diff.
        """
        start = time.time()
        proj_dir = project_dir or Path.cwd()

        self.store.ensure_collection(recreate=force)

        if force:
            result = self._full_index(proj_dir, start)
            # Save state so next non-force run can do incremental
            current_commit = _get_git_head(proj_dir)
            if current_commit:
                _save_state(self.config, {"last_commit": current_commit})
            return result

        # Try incremental
        state = _load_state(self.config)
        last_commit = state.get("last_commit")
        current_commit = _get_git_head(proj_dir)

        if last_commit and current_commit:
            if last_commit == current_commit:
                # No changes since last index
                logger.info("Index is up-to-date (commit %s)", current_commit[:8])
                console.print(f"[green]Index is up-to-date[/green] (commit {current_commit[:8]})")
                return IndexResult(duration_seconds=time.time() - start)

            changed = _get_git_changed_files(proj_dir, last_commit)
            if changed is not None:
                result = self._incremental_index(proj_dir, changed, start)
                state["last_commit"] = current_commit
                _save_state(self.config, state)
                return result

        # Full index if no state or not a git repo
        result = self._full_index(proj_dir, start)

        # Save state
        if current_commit:
            state["last_commit"] = current_commit
            _save_state(self.config, state)

        return result

    def incremental_index_if_needed(self, project_dir: Path | None = None) -> IndexResult | None:
        """Check if index is stale and run incremental update if so.

        Called automatically from MCP server before search.
        Returns None if index is up-to-date, IndexResult if re-indexed.
        """
        proj_dir = project_dir or Path.cwd()

        if not self.store.collection_exists():
            return None  # Can't auto-index from scratch

        state = _load_state(self.config)
        last_commit = state.get("last_commit")
        current_commit = _get_git_head(proj_dir)

        if not last_commit or not current_commit:
            return None
        if last_commit == current_commit:
            return None

        changed = _get_git_changed_files(proj_dir, last_commit)
        if changed is None or not changed:
            # No changes or error — just update commit
            state["last_commit"] = current_commit
            _save_state(self.config, state)
            return None

        logger.info("Index stale (%d changed files). Running incremental update...", len(changed))
        start = time.time()
        result = self._incremental_index(proj_dir, changed, start)

        state["last_commit"] = current_commit
        _save_state(self.config, state)
        return result

    # ── Full index ───────────────────────────────────────────────────

    def _full_index(self, project_dir: Path, start: float) -> IndexResult:
        """Scan all sources, chunk everything, upsert."""
        result = IndexResult()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            # 1. Collect all documents
            task = progress.add_task("Scanning sources...", total=None)
            all_docs = self._collect_all_documents(project_dir)
            progress.update(task, completed=True, total=1, description=f"Scanned {len(all_docs)} files")
            result.total_documents = len(all_docs)

            if not all_docs:
                console.print("[yellow]No documents found to index.[/yellow]")
                result.duration_seconds = time.time() - start
                return result

            # 2. Chunk all documents
            task = progress.add_task("Chunking...", total=len(all_docs))
            all_chunks = []
            for doc in all_docs:
                chunks = self._chunk_document(doc)
                all_chunks.extend(chunks)
                self._count_result(result, doc, len(chunks))
                progress.advance(task)

            result.total_chunks = len(all_chunks)
            progress.update(task, description=f"Chunked into {len(all_chunks)} chunks")

            # 3. Upsert to Qdrant
            task = progress.add_task("Upserting to Qdrant...", total=len(all_chunks))
            batch_size = 64
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i + batch_size]
                self.store.upsert_chunks(batch)
                progress.advance(task, advance=len(batch))

        result.duration_seconds = time.time() - start
        return result

    # ── Incremental index ────────────────────────────────────────────

    def _incremental_index(
        self, project_dir: Path, changed_files: list[str], start: float,
    ) -> IndexResult:
        """Re-index only changed files."""
        result = IndexResult()

        # Delete old chunks for changed files
        self.store.delete_by_file_paths(changed_files)
        logger.info("Deleted old chunks for %d changed files", len(changed_files))

        # Re-scan only changed files that still exist
        all_chunks = []
        for rel_path in changed_files:
            fpath = project_dir / rel_path
            if not fpath.is_file():
                continue  # File was deleted

            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            lang = None
            source_type = "local_code" if is_code_file(rel_path) else "local_doc"

            doc = RawDocument(
                content=content,
                source=rel_path,
                source_type=source_type,
                title=rel_path,
                file_path=rel_path,
                language=lang,
            )
            chunks = self._chunk_document(doc)
            all_chunks.extend(chunks)
            self._count_result(result, doc, len(chunks))
            result.total_documents += 1

        # Upsert new chunks
        if all_chunks:
            self.store.upsert_chunks(all_chunks)

        result.total_chunks = len(all_chunks)
        result.duration_seconds = time.time() - start

        logger.info(
            "Incremental index: %d files, %d chunks in %.1fs",
            result.total_documents, result.total_chunks, result.duration_seconds,
        )
        return result

    # ── Document collection ──────────────────────────────────────────

    def _collect_all_documents(self, project_dir: Path) -> list[RawDocument]:
        """Collect documents from all configured sources."""
        docs: list[RawDocument] = []

        # Local sources
        for local_src in self.config.sources.local:
            scanner = LocalScanner(local_src, project_dir)
            docs.extend(scanner.scan())

        # Web sources
        if self.config.sources.web:
            fetcher = WebFetcher(self.config.cache_dir)
            for web_src in self.config.sources.web:
                doc = fetcher.fetch(web_src)
                if doc:
                    docs.append(doc)

        # GitHub sources
        if self.config.sources.github:
            fetcher = GitHubFetcher(self.config.cache_dir)
            for gh_src in self.config.sources.github:
                # README
                readme_doc = fetcher.fetch_readme(gh_src)
                if readme_doc:
                    docs.append(readme_doc)
                # Code (if clone=true)
                if gh_src.clone:
                    code_docs = fetcher.clone_and_scan(gh_src)
                    docs.extend(code_docs)

        return docs

    # ── Chunking ─────────────────────────────────────────────────────

    def _chunk_document(self, doc: RawDocument) -> list[dict]:
        """Chunk a document and return dicts ready for store.upsert_chunks()."""
        if doc.source_type in ("local_code", "github_code"):
            chunks = chunk_file(
                doc.content, doc.file_path, language=doc.language,
            )
        else:
            chunks = chunk_text(doc.content, language=doc.language)

        result = []
        for i, chunk in enumerate(chunks):
            chunk_id = make_chunk_id(doc.source, i)
            result.append({
                "id": chunk_id,
                "content": chunk.content,
                "metadata": {
                    "source": doc.source,
                    "source_type": doc.source_type,
                    "title": doc.title,
                    "file_path": doc.file_path,
                    "language": chunk.language or doc.language or "",
                    "tags": ",".join(doc.tags) if doc.tags else "",
                    "chunk_type": chunk.chunk_type,
                    "symbol_name": chunk.symbol_name or "",
                    "chunk_index": i,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "repo": doc.repo or "",
                },
            })
        return result

    @staticmethod
    def _count_result(result: IndexResult, doc: RawDocument, n_chunks: int) -> None:
        st = doc.source_type
        result.by_source_type[st] = result.by_source_type.get(st, 0) + n_chunks
        if doc.language:
            result.by_language[doc.language] = result.by_language.get(doc.language, 0) + n_chunks
