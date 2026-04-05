"""CLI entry point for repo-rag."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
@click.version_option(package_name="repo-rag")
def cli():
    """repo-rag: Local semantic search for any codebase."""
    pass


# ── init ─────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--name", prompt="Project name", help="Name for the Qdrant collection")
@click.option("--dir", "directory", default=".", help="Directory to create config in")
def init(name: str, directory: str):
    """Create a repo-rag.yaml config template in the current directory."""
    from repo_rag.config import generate_template

    project_dir = Path(directory).resolve()
    path = generate_template(project_dir, name)
    console.print(f"[green]Created config:[/green] {path}")
    console.print(f"\nNext steps:")
    console.print(f"  1. Edit {path} to customize sources")
    console.print(f"  2. Start Qdrant: docker compose -f ~/git/my/repo-rag/docker-compose.yaml up -d")
    console.print(f"  3. Index: repo-rag index")


# ── index ────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--force", is_flag=True, help="Recreate collection from scratch")
@click.option("--config", "config_path", type=click.Path(exists=True), help="Path to repo-rag.yaml")
def index(force: bool, config_path: str | None):
    """Index sources based on repo-rag.yaml config."""
    from repo_rag.config import load_config
    from repo_rag.indexer import Indexer
    from repo_rag.store import VectorStore

    cfg = load_config(Path(config_path) if config_path else None)
    console.print(f"[bold]Indexing project:[/bold] {cfg.name}")

    store = VectorStore(cfg)
    indexer = Indexer(cfg, store)

    # Determine project dir from config file location
    if config_path:
        project_dir = Path(config_path).resolve().parent
    else:
        from repo_rag.config import find_config
        project_dir = find_config().parent

    result = indexer.index_all(force=force, project_dir=project_dir)

    # Summary
    console.print(f"\n[bold green]Done![/bold green] Indexed in {result.duration_seconds:.1f}s")
    console.print(f"  Documents: {result.total_documents}")
    console.print(f"  Chunks:    {result.total_chunks}")

    if result.by_source_type:
        console.print(f"\n  By source type:")
        for st, count in sorted(result.by_source_type.items()):
            console.print(f"    {st}: {count}")

    if result.by_language:
        console.print(f"\n  By language:")
        for lang, count in sorted(result.by_language.items()):
            console.print(f"    {lang}: {count}")


# ── search ───────────────────────────────────────────────────────────────


@cli.command()
@click.argument("query")
@click.option("--top-k", "-k", default=10, help="Number of results")
@click.option("--language", "-l", help="Filter by language")
@click.option("--code-only", is_flag=True, help="Search only code files")
@click.option("--config", "config_path", type=click.Path(exists=True), help="Path to repo-rag.yaml")
def search(query: str, top_k: int, language: str | None, code_only: bool, config_path: str | None):
    """Search the index from the terminal."""
    from repo_rag.config import load_config
    from repo_rag.store import VectorStore

    cfg = load_config(Path(config_path) if config_path else None)
    store = VectorStore(cfg)

    if not store.collection_exists():
        console.print("[red]No index found.[/red] Run 'repo-rag index' first.")
        raise SystemExit(1)

    filters = {}
    if code_only:
        filters["source_type"] = ["local_code", "github_code"]
    if language:
        filters["language"] = language

    results = store.hybrid_search(query, top_k=top_k, filters=filters or None)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    for i, r in enumerate(results, 1):
        score = r.get("score", 0)
        source_type = r.get("source_type", "")
        file_path = r.get("file_path", "")
        language = r.get("language", "")
        symbol = r.get("symbol_name", "")
        chunk_type = r.get("chunk_type", "")

        header = f"[bold cyan]#{i}[/bold cyan] "
        header += f"[dim]{source_type}[/dim] "
        if file_path:
            header += f"[green]{file_path}[/green]"
        if symbol:
            header += f" → [yellow]{symbol}[/yellow]"
        if language:
            header += f" [dim]({language})[/dim]"
        header += f" [dim]score={score:.3f}[/dim]"

        console.print(header)

        # Show content preview (first 300 chars)
        content = r.get("content", "")
        preview = content[:300] + "..." if len(content) > 300 else content
        console.print(f"  {preview}\n")


# ── info ─────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--config", "config_path", type=click.Path(exists=True), help="Path to repo-rag.yaml")
def info(config_path: str | None):
    """Show collection statistics."""
    from repo_rag.config import load_config
    from repo_rag.store import VectorStore

    cfg = load_config(Path(config_path) if config_path else None)
    store = VectorStore(cfg, lazy_embed=True)

    if not store.collection_exists():
        console.print(f"[yellow]Collection '{cfg.name}' does not exist.[/yellow]")
        console.print("Run 'repo-rag index' to create it.")
        return

    info = store.collection_info()
    console.print(f"\n[bold]Collection:[/bold] {cfg.name}")
    console.print(f"  Status: {info.get('status', 'unknown')}")
    console.print(f"  Points: {info.get('points_count', 0)}")

    # Breakdown by source type
    console.print(f"\n[bold]By source type:[/bold]")
    st_counts = store.get_field_counts("source_type")
    for st, count in st_counts.items():
        console.print(f"  {st}: {count}")

    # Breakdown by language
    console.print(f"\n[bold]By language:[/bold]")
    lang_counts = store.get_field_counts("language")
    for lang, count in lang_counts.items():
        if lang:
            console.print(f"  {lang}: {count}")

    console.print(f"\n[dim]Qdrant dashboard: {cfg.qdrant.url}/dashboard[/dim]")


# ── serve ────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--config", "config_path", type=click.Path(exists=True), help="Path to repo-rag.yaml")
def serve(config_path: str | None):
    """Start the MCP server (stdio transport for Claude Code)."""
    from repo_rag.config import load_config
    from repo_rag.server import create_mcp_server

    cfg = load_config(Path(config_path) if config_path else None)
    server = create_mcp_server(cfg)
    server.run()


if __name__ == "__main__":
    cli()
