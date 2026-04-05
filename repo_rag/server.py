"""MCP server exposing search tools to Claude Code."""

from __future__ import annotations

import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from repo_rag.config import RepoRagConfig, find_config
from repo_rag.indexer import Indexer
from repo_rag.store import VectorStore

logger = logging.getLogger(__name__)


def create_mcp_server(config: RepoRagConfig) -> FastMCP:
    """Create a configured MCP server with search tools."""
    mcp = FastMCP(f"repo-rag-{config.name}")

    # Lazy-initialized singletons
    _store: VectorStore | None = None
    _indexer: Indexer | None = None

    def get_store() -> VectorStore:
        nonlocal _store
        if _store is None:
            _store = VectorStore(config)
        return _store

    def get_indexer() -> Indexer:
        nonlocal _indexer
        if _indexer is None:
            _indexer = Indexer(config, get_store())
        return _indexer

    def _resolve_project_dir() -> Path:
        """Best-effort: find the project dir from the config file location."""
        try:
            return find_config().parent
        except FileNotFoundError:
            return Path.cwd()

    def _auto_reindex() -> str | None:
        """Check if index is stale and run incremental update.

        Returns a status note string if reindexed, None otherwise.
        """
        store = get_store()
        if not store.collection_exists():
            return None

        indexer = get_indexer()
        project_dir = _resolve_project_dir()
        result = indexer.incremental_index_if_needed(project_dir)

        if result and result.total_chunks > 0:
            return (
                f"[Auto-reindexed {result.total_documents} changed files "
                f"({result.total_chunks} chunks) in {result.duration_seconds:.1f}s]\n\n"
            )
        return None

    # ── Tools ────────────────────────────────────────────────────────

    @mcp.tool()
    def search(
        query: str,
        top_k: int = 10,
        source_type: str | None = None,
        language: str | None = None,
        tags: str | None = None,
    ) -> str:
        """Search the indexed codebase using hybrid semantic + keyword search.

        Args:
            query: Natural language question or code concept to search for.
            top_k: Number of results to return (default 10).
            source_type: Filter by type: local_code, local_doc, web, github_code, github_doc.
            language: Filter by programming language (python, typescript, java, etc.).
            tags: Filter by tag.
        """
        store = get_store()
        if not store.collection_exists():
            return (
                f"No index found for '{config.name}'. "
                f"Run `repo-rag index` to create the initial index."
            )

        # Auto-reindex if stale
        reindex_note = _auto_reindex()

        filters: dict = {}
        if source_type:
            filters["source_type"] = source_type
        if language:
            filters["language"] = language
        if tags:
            filters["tags"] = tags

        results = store.hybrid_search(query, top_k=top_k, filters=filters or None)
        output = _format_results(results)

        if reindex_note:
            output = reindex_note + output
        return output

    @mcp.tool()
    def search_code(
        query: str,
        top_k: int = 10,
        language: str | None = None,
        file_pattern: str | None = None,
    ) -> str:
        """Search specifically for code examples and implementations.

        Args:
            query: What code pattern or implementation to find.
            top_k: Number of results to return (default 10).
            language: Filter by programming language.
            file_pattern: Filter by file path substring (e.g. "models", "handlers").
        """
        store = get_store()
        if not store.collection_exists():
            return (
                f"No index found for '{config.name}'. "
                f"Run `repo-rag index` to create the initial index."
            )

        # Auto-reindex if stale
        reindex_note = _auto_reindex()

        filters: dict = {"source_type": ["local_code", "github_code"]}
        if language:
            filters["language"] = language

        results = store.hybrid_search(query, top_k=top_k, filters=filters)

        # Apply file_pattern filter post-query (substring match)
        if file_pattern:
            results = [r for r in results if file_pattern in r.get("file_path", "")]

        output = _format_code_results(results)

        if reindex_note:
            output = reindex_note + output
        return output

    @mcp.tool()
    def list_indexed_files(pattern: str | None = None) -> str:
        """List all files that have been indexed.

        Args:
            pattern: Optional glob pattern to filter (e.g. "*.py", "src/**/*.ts").
        """
        store = get_store()
        if not store.collection_exists():
            return f"No index found for '{config.name}'."

        files = store.list_indexed_files(pattern)
        if not files:
            return "No files found matching the pattern." if pattern else "No files indexed."

        lines = [f"Indexed files ({len(files)} total):"]
        for f in files[:100]:  # Limit output
            lines.append(f"  {f['file_path']} ({f['chunks']} chunks)")
        if len(files) > 100:
            lines.append(f"  ... and {len(files) - 100} more")
        return "\n".join(lines)

    @mcp.tool()
    def collection_info() -> str:
        """Show statistics about the current index."""
        store = get_store()
        if not store.collection_exists():
            return f"No index found for '{config.name}'. Run `repo-rag index` to create it."

        info = store.collection_info()
        lines = [
            f"Collection: {config.name}",
            f"Status: {info.get('status', 'unknown')}",
            f"Total chunks: {info.get('points_count', 0)}",
        ]

        st_counts = store.get_field_counts("source_type")
        if st_counts:
            lines.append("\nBy source type:")
            for st, count in st_counts.items():
                lines.append(f"  {st}: {count}")

        lang_counts = store.get_field_counts("language")
        if lang_counts:
            lines.append("\nBy language:")
            for lang, count in lang_counts.items():
                if lang:
                    lines.append(f"  {lang}: {count}")

        return "\n".join(lines)

    return mcp


# ── Formatting helpers ───────────────────────────────────────────────────


def _format_results(results: list[dict]) -> str:
    """Format search results for LLM consumption."""
    if not results:
        return "No results found."

    lines = []
    for i, r in enumerate(results, 1):
        source_type = r.get("source_type", "")
        file_path = r.get("file_path", "")
        language = r.get("language", "")
        symbol = r.get("symbol_name", "")
        score = r.get("score", 0)
        content = r.get("content", "")

        header = f"--- Result {i} "
        header += f"[{source_type}] {file_path}"
        if symbol:
            header += f" → {symbol}"
        if language:
            header += f" ({language})"
        header += f" (score: {score:.3f}) ---"

        lines.append(header)
        lines.append(content)
        lines.append("")

    return "\n".join(lines)


def _format_code_results(results: list[dict]) -> str:
    """Format code search results with fenced code blocks."""
    if not results:
        return "No code results found."

    lines = []
    for i, r in enumerate(results, 1):
        file_path = r.get("file_path", "")
        language = r.get("language", "")
        symbol = r.get("symbol_name", "")
        chunk_type = r.get("chunk_type", "")
        score = r.get("score", 0)
        repo = r.get("repo", "")
        source = r.get("source", "")
        content = r.get("content", "")

        header = f"--- Result {i}: {file_path}"
        if symbol:
            header += f" → {symbol} ({chunk_type})"
        if repo:
            header += f" [{repo}]"
        header += f" (score: {score:.3f}) ---"

        lines.append(header)
        lines.append(f"```{language}")
        lines.append(content)
        lines.append("```")
        if source:
            lines.append(f"Source: {source}")
        lines.append("")

    return "\n".join(lines)
