"""Configuration models and YAML loading for repo-rag."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, PrivateAttr

CONFIG_FILENAME = "repo-rag.yaml"

# ── Source models ────────────────────────────────────────────────────────

DEFAULT_INCLUDE = [
    "*.py", "*.ts", "*.tsx", "*.js", "*.jsx", "*.md", "*.mdx",
    "*.go", "*.rs", "*.java", "*.kt", "*.cs", "*.fs",
    "*.ex", "*.exs", "*.php", "*.rb", "*.yaml", "*.yml",
    "*.toml", "*.json", "*.sh", "*.sql",
]

DEFAULT_EXCLUDE = [
    "node_modules", ".git", "__pycache__", "dist", "build",
    ".venv", "venv", "vendor", "target", ".next", ".nuxt",
    "*.min.js", "*.min.css", "*.map", "*.lock",
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "repos_cloned", "chroma_db", "ingested_sources",
    ".repo-rag",
]


class LocalSource(BaseModel):
    """A local directory to index."""

    path: str = "."
    include: list[str] = Field(default_factory=lambda: list(DEFAULT_INCLUDE))
    exclude: list[str] = Field(default_factory=lambda: list(DEFAULT_EXCLUDE))


class WebSource(BaseModel):
    """A web article URL to fetch and index."""

    url: str
    tags: list[str] = Field(default_factory=list)


class GitHubSource(BaseModel):
    """A GitHub repository to fetch README and/or clone for code indexing."""

    url: str
    clone: bool = True
    code_paths: list[str] | None = None
    tags: list[str] = Field(default_factory=list)


# ── Top-level config ─────────────────────────────────────────────────────


class QdrantConfig(BaseModel):
    url: str = "http://localhost:6333"


class EmbeddingConfig(BaseModel):
    model: str = "nomic-ai/nomic-embed-text-v1.5"


class SourcesConfig(BaseModel):
    local: list[LocalSource] = Field(default_factory=lambda: [LocalSource()])
    web: list[WebSource] = Field(default_factory=list)
    github: list[GitHubSource] = Field(default_factory=list)


class RepoRagConfig(BaseModel):
    """Root configuration loaded from repo-rag.yaml."""

    name: str
    cache: str = ".repo-rag"  # Relative to project root, or absolute path
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    sources: SourcesConfig = Field(default_factory=SourcesConfig)

    # Set after loading — the directory containing repo-rag.yaml
    _project_dir: Path | None = PrivateAttr(default=None)

    @property
    def cache_dir(self) -> Path:
        """Cache directory for web/github/repos/state.

        Resolved relative to the project root (where repo-rag.yaml lives).
        Can be overridden to an absolute path in config.
        """
        p = Path(self.cache)
        if not p.is_absolute():
            base = self._project_dir or Path.cwd()
            p = base / p
        p = p.resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def state_file(self) -> Path:
        return self.cache_dir / ".state"


# ── Loading helpers ──────────────────────────────────────────────────────


def find_config(start: Path | None = None) -> Path:
    """Walk up from *start* (default: cwd) to find repo-rag.yaml.

    Raises FileNotFoundError if not found.
    """
    current = (start or Path.cwd()).resolve()
    while True:
        candidate = current / CONFIG_FILENAME
        if candidate.is_file():
            return candidate
        parent = current.parent
        if parent == current:
            break
        current = parent
    raise FileNotFoundError(
        f"No {CONFIG_FILENAME} found in {start or Path.cwd()} or any parent directory. "
        f"Run 'repo-rag init' to create one."
    )


def load_config(path: Path | None = None) -> RepoRagConfig:
    """Load and validate config from a YAML file.

    If *path* is None, searches upward from cwd.
    """
    config_path = (path if path else find_config()).resolve()
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    cfg = RepoRagConfig(**raw)
    cfg._project_dir = config_path.parent
    return cfg


def generate_template(project_dir: Path, name: str) -> Path:
    """Write a default repo-rag.yaml template to *project_dir*.

    Returns the path of the created file.
    """
    dest = project_dir / CONFIG_FILENAME
    content = f"""\
# repo-rag configuration
# Docs: https://github.com/user/repo-rag

name: {name}
cache: .repo-rag

qdrant:
  url: http://localhost:6333

embedding:
  model: nomic-ai/nomic-embed-text-v1.5

sources:
  local:
    - path: .
      include:
        - "*.py"
        - "*.ts"
        - "*.tsx"
        - "*.js"
        - "*.jsx"
        - "*.md"
        - "*.go"
        - "*.rs"
        - "*.java"
        - "*.kt"
        - "*.cs"
        - "*.yaml"
      exclude:
        - node_modules
        - .git
        - __pycache__
        - dist
        - build
        - .venv
        - venv
        - vendor
        - target
        - "*.min.js"
        - "*.min.css"
        - "*.lock"
        - .repo-rag

  # Optional: web articles to index
  # web:
  #   - url: https://example.com/article
  #     tags: [docs]

  # Optional: GitHub repos to fetch & index
  # github:
  #   - url: https://github.com/owner/repo
  #     clone: true
  #     code_paths: [src/]
  #     tags: [reference]
"""
    dest.write_text(content)
    return dest
