"""Qdrant vector store wrapper with hybrid search (dense + BM25 sparse)."""

from __future__ import annotations

import logging
from typing import Any

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from repo_rag.config import RepoRagConfig

logger = logging.getLogger(__name__)

# Prefix required by nomic-embed-text for optimal performance.
# Other models ignore unknown prefixes gracefully.
_QUERY_PREFIX = "search_query: "
_DOC_PREFIX = "search_document: "

# Nomic-embed-text models use Matryoshka, we can truncate to save space.
# 768 is the full dimension; override if using a smaller model.
_DEFAULT_DIM = 768


class VectorStore:
    """Manages a Qdrant collection with dense + BM25 sparse vectors."""

    def __init__(self, config: RepoRagConfig, *, lazy_embed: bool = False):
        self.config = config
        self.collection_name = config.name
        self.client = QdrantClient(url=config.qdrant.url)
        self._embedder: SentenceTransformer | None = None
        self._dim: int | None = None
        if not lazy_embed:
            self._load_embedder()

    # ── Embedder ─────────────────────────────────────────────────────

    def _load_embedder(self) -> None:
        if self._embedder is not None:
            return
        logger.info("Loading embedding model: %s", self.config.embedding.model)
        self._embedder = SentenceTransformer(
            self.config.embedding.model, trust_remote_code=True, device="cpu",
        )
        self._dim = self._embedder.get_sentence_embedding_dimension()

    @property
    def embedder(self) -> SentenceTransformer:
        self._load_embedder()
        assert self._embedder is not None
        return self._embedder

    @property
    def dim(self) -> int:
        self._load_embedder()
        assert self._dim is not None
        return self._dim

    # ── Collection lifecycle ─────────────────────────────────────────

    def ensure_collection(self, *, recreate: bool = False) -> None:
        """Create the collection if it doesn't exist.  If *recreate*, drop first."""
        exists = self.client.collection_exists(self.collection_name)

        if exists and recreate:
            logger.info("Deleting collection %s for re-creation", self.collection_name)
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            logger.info("Creating collection %s (dim=%d)", self.collection_name, self.dim)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=self.dim,
                        distance=models.Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                    ),
                },
            )
            self._create_payload_indexes()

    def _create_payload_indexes(self) -> None:
        """Create payload indexes for efficient filtering."""
        keyword_fields = [
            "source_type", "language", "file_path",
            "tags", "chunk_type", "symbol_name", "source",
        ]
        for field in keyword_fields:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def collection_exists(self) -> bool:
        return self.client.collection_exists(self.collection_name)

    def delete_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

    # ── Upsert ───────────────────────────────────────────────────────

    def upsert_chunks(self, chunks: list[dict], batch_size: int = 32) -> int:
        """Embed and upsert chunks.

        Each chunk dict must have keys: ``id``, ``content``, ``metadata``.
        Returns total upserted count.
        """
        total = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c["content"] for c in batch]

            # Dense embeddings (with document prefix for nomic)
            prefixed = [f"{_DOC_PREFIX}{t}" for t in texts]
            dense_vectors = self.embedder.encode(prefixed, show_progress_bar=False)

            # Compute average doc length for BM25 tuning
            avg_len = sum(len(t.split()) for t in texts) / max(len(texts), 1)

            points = []
            for j, chunk in enumerate(batch):
                payload = {
                    "content": chunk["content"],
                    **chunk.get("metadata", {}),
                }
                point = models.PointStruct(
                    id=chunk["id"],
                    payload=payload,
                    vector={
                        "dense": dense_vectors[j].tolist(),
                        "sparse": models.Document(
                            text=chunk["content"],
                            model="Qdrant/bm25",
                            options={"avg_len": avg_len},
                        ),
                    },
                )
                points.append(point)

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            total += len(points)
            logger.debug("Upserted batch %d-%d (%d points)", i, i + len(batch), len(points))

        return total

    # ── Delete ───────────────────────────────────────────────────────

    def delete_by_file_paths(self, file_paths: list[str]) -> None:
        """Delete all points whose ``file_path`` is in *file_paths*."""
        if not file_paths:
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    should=[
                        models.FieldCondition(
                            key="file_path",
                            match=models.MatchValue(value=fp),
                        )
                        for fp in file_paths
                    ]
                )
            ),
        )

    def delete_by_source(self, source: str) -> None:
        """Delete all points from a given source."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source",
                            match=models.MatchValue(value=source),
                        )
                    ]
                )
            ),
        )

    # ── Search ───────────────────────────────────────────────────────

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict]:
        """Hybrid search: dense semantic + BM25 sparse, fused with RRF."""
        qdrant_filter = self._build_filter(filters)
        prefetch_limit = min(top_k * 3, 100)

        # Dense embedding (with query prefix for nomic)
        dense_vector = self.embedder.encode(
            f"{_QUERY_PREFIX}{query}", show_progress_bar=False,
        ).tolist()

        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=prefetch_limit,
                    filter=qdrant_filter,
                ),
                models.Prefetch(
                    query=models.Document(text=query, model="Qdrant/bm25"),
                    using="sparse",
                    limit=prefetch_limit,
                    filter=qdrant_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        return [
            {
                "content": pt.payload.get("content", ""),
                "score": pt.score,
                **{k: v for k, v in pt.payload.items() if k != "content"},
            }
            for pt in results.points
        ]

    def dense_search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict]:
        """Pure semantic search (no BM25 leg)."""
        qdrant_filter = self._build_filter(filters)
        dense_vector = self.embedder.encode(
            f"{_QUERY_PREFIX}{query}", show_progress_bar=False,
        ).tolist()

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_vector,
            using="dense",
            limit=top_k,
            filter=qdrant_filter,
            with_payload=True,
        )

        return [
            {
                "content": pt.payload.get("content", ""),
                "score": pt.score,
                **{k: v for k, v in pt.payload.items() if k != "content"},
            }
            for pt in results.points
        ]

    # ── Info ──────────────────────────────────────────────────────────

    def collection_info(self) -> dict:
        """Return collection stats."""
        if not self.collection_exists():
            return {"exists": False, "points_count": 0}

        info = self.client.get_collection(self.collection_name)
        return {
            "exists": True,
            "points_count": info.points_count,
            "segments_count": info.segments_count,
            "status": str(info.status),
        }

    def get_field_counts(self, field: str) -> dict[str, int]:
        """Scroll through collection to get value counts for a payload field."""
        counts: dict[str, int] = {}
        offset = None
        while True:
            result = self.client.scroll(
                collection_name=self.collection_name,
                limit=500,
                offset=offset,
                with_payload=[field],
                with_vectors=False,
            )
            points, next_offset = result
            for pt in points:
                val = pt.payload.get(field, "unknown")
                if val:
                    counts[val] = counts.get(val, 0) + 1
            if next_offset is None:
                break
            offset = next_offset
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def list_indexed_files(self, pattern: str | None = None) -> list[dict]:
        """Get unique file_path values with chunk counts."""
        counts: dict[str, int] = {}
        offset = None
        while True:
            result = self.client.scroll(
                collection_name=self.collection_name,
                limit=500,
                offset=offset,
                with_payload=["file_path", "language", "source_type"],
                with_vectors=False,
            )
            points, next_offset = result
            for pt in points:
                fp = pt.payload.get("file_path", "")
                if fp:
                    counts[fp] = counts.get(fp, 0) + 1
            if next_offset is None:
                break
            offset = next_offset

        files = [{"file_path": fp, "chunks": c} for fp, c in sorted(counts.items())]

        if pattern:
            from fnmatch import fnmatch
            files = [f for f in files if fnmatch(f["file_path"], pattern)]

        return files

    # ── Filter building ──────────────────────────────────────────────

    @staticmethod
    def _build_filter(filters: dict[str, Any] | None) -> models.Filter | None:
        if not filters:
            return None

        must: list[models.Condition] = []

        for key, val in filters.items():
            if val is None:
                continue
            if isinstance(val, list):
                must.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=val),
                    )
                )
            else:
                must.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=val),
                    )
                )

        return models.Filter(must=must) if must else None
