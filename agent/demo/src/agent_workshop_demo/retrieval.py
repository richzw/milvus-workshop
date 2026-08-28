"""Hybrid retrieval fallback that mirrors the Milvus search contract."""

from __future__ import annotations

from collections import Counter
from typing import Any, Literal, Protocol

from agent_workshop_demo.embedding import (
    cosine_similarity,
    dense_vector,
    sparse_vector,
)
from agent_workshop_demo.image_retrieval import (
    image_cosine_score,
    image_only_filters,
    require_image_space,
    validate_image_fingerprint,
    validate_image_search_top_k,
    validate_image_query_vector,
)
from agent_workshop_demo.models import (
    ImageSearchResult,
    KBChunk,
    SearchResult,
)
from agent_workshop_demo.validation import normalize_filters

OrderMode = Literal["relevance", "scalar"]
AGGREGATION_FIELDS = frozenset(
    {
        "source_type",
        "doc_type",
        "department",
        "has_image_vector",
        "doc_version",
        "is_current",
    }
)


class HybridRetriever(Protocol):
    """Search and aggregation interface shared by retrieval adapters."""

    supports_parallel_search: bool

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
        order_by: list[str] | None = None,
    ) -> list[SearchResult]: ...

    def aggregations(
        self,
        results: list[SearchResult],
        fields: list[str],
    ) -> dict[str, dict[str, int]]: ...

    def fetch_document_chunks(
        self,
        *,
        doc_id: str,
        doc_version: str,
        filters: dict[str, Any] | None = None,
        limit: int,
    ) -> list[KBChunk]: ...

    def fetch_chunks_by_ids(
        self,
        *,
        chunk_ids: list[str],
        filters: dict[str, Any] | None = None,
    ) -> list[KBChunk]: ...


class InMemoryHybridRetriever:
    """Deterministic local replacement for the planned Milvus adapter."""

    supports_parallel_search = True

    def __init__(self, chunks: list[KBChunk]) -> None:
        self.chunks = chunks

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
        order_by: list[str] | None = None,
        order_mode: OrderMode = "relevance",
    ) -> list[SearchResult]:
        """Return validated, score-first hybrid results."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        validated_filters = normalize_filters(filters)
        filtered = self._apply_filters(self.chunks, validated_filters)
        query_vector = dense_vector(query)
        query_sparse = sparse_vector(query)
        max_updated_at = max(
            (chunk.updated_at for chunk in filtered),
            default=1,
        )
        max_priority = max(
            (chunk.priority for chunk in filtered),
            default=1,
        )
        results: list[SearchResult] = []

        for item in filtered:
            dense_score = cosine_similarity(query_vector, item.text_vector)
            keyword_score = self._keyword_score(
                query_sparse,
                item.sparse_vector,
            )
            recency_score = item.updated_at / max_updated_at
            priority_score = item.priority / max_priority
            hybrid_score = (
                0.55 * dense_score
                + 0.35 * keyword_score
                + 0.05 * recency_score
                + 0.05 * priority_score
            )
            results.append(
                SearchResult(
                    chunk=item,
                    rank=0,
                    dense_score=dense_score,
                    keyword_score=keyword_score,
                    recency_score=recency_score,
                    priority_score=priority_score,
                    hybrid_score=hybrid_score,
                )
            )

        ordered = order_search_results(
            results,
            order_by or [],
            order_mode=order_mode,
        )[:top_k]
        return [
            SearchResult(
                chunk=result.chunk,
                rank=index,
                dense_score=result.dense_score,
                keyword_score=result.keyword_score,
                recency_score=result.recency_score,
                priority_score=result.priority_score,
                hybrid_score=result.hybrid_score,
            )
            for index, result in enumerate(ordered, start=1)
        ]

    def search_image_vector(
        self,
        query_vector: list[float],
        *,
        image_fingerprint: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[ImageSearchResult]:
        """Return image-only COSINE results with vector-space validation."""

        validate_image_search_top_k(top_k)
        validated_query = validate_image_query_vector(query_vector)
        validated_fingerprint = validate_image_fingerprint(image_fingerprint)
        validated_filters = image_only_filters(filters)
        filtered = self._apply_filters(
            self.chunks,
            validated_filters,
        )
        scored: list[tuple[float, KBChunk]] = []
        for chunk in filtered:
            require_image_space(
                chunk,
                expected_fingerprint=validated_fingerprint,
            )
            image_vector = chunk.image_vector
            if image_vector is None:
                raise ValueError("Image search candidate is missing image_vector")
            scored.append(
                (
                    image_cosine_score(
                        validated_query,
                        image_vector,
                    ),
                    chunk,
                )
            )
        ordered = sorted(
            scored,
            key=lambda item: (-item[0], item[1].chunk_id),
        )[:top_k]
        return [
            ImageSearchResult(
                chunk=chunk,
                rank=rank,
                image_score=score,
            )
            for rank, (score, chunk) in enumerate(ordered, start=1)
        ]

    def search_sparse(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
        order_by: list[str] | None = None,
    ) -> list[SearchResult]:
        """Return the deterministic lexical lane without dense contribution."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        filtered = self._apply_filters(self.chunks, normalize_filters(filters))
        query_sparse = sparse_vector(query)
        scored = sorted(
            (
                (self._keyword_score(query_sparse, item.sparse_vector), item)
                for item in filtered
            ),
            key=lambda item: (-item[0], item[1].chunk_id),
        )[:top_k]
        return [
            SearchResult(
                chunk=chunk,
                rank=rank,
                dense_score=0.0,
                keyword_score=score,
                recency_score=0.0,
                priority_score=0.0,
                hybrid_score=score,
                retrieval_profile="flat_bm25",
                retrieval_paths=("flat_bm25",),
            )
            for rank, (score, chunk) in enumerate(scored, start=1)
        ]

    def aggregations(
        self,
        results: list[SearchResult],
        fields: list[str],
    ) -> dict[str, dict[str, int]]:
        """Count requested public scalar fields over recalled results."""

        requested = validate_aggregation_fields(fields)
        output: dict[str, dict[str, int]] = {}
        for field in requested:
            values = (str(getattr(item.chunk, field)) for item in results)
            output[field] = dict(sorted(Counter(values).items()))
        return output

    def fetch_document_chunks(
        self,
        *,
        doc_id: str,
        doc_version: str,
        filters: dict[str, Any] | None = None,
        limit: int,
    ) -> list[KBChunk]:
        """Return authorized sibling chunks in stable document order."""

        if not doc_id.strip() or not doc_version.strip():
            raise ValueError("doc_id and doc_version must be non-empty")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        validated_filters = normalize_filters(filters)
        filtered = self._apply_filters(self.chunks, validated_filters)
        siblings = [
            chunk
            for chunk in filtered
            if chunk.doc_id == doc_id and chunk.doc_version == doc_version
        ]
        return sorted(
            siblings,
            key=lambda chunk: (
                chunk.chunk_index,
                chunk.page_no or 0,
                chunk.chunk_id,
            ),
        )[:limit]

    def fetch_chunks_by_ids(
        self,
        *,
        chunk_ids: list[str],
        filters: dict[str, Any] | None = None,
    ) -> list[KBChunk]:
        """Return authorized chunks for bounded cache freshness checks."""

        expected = list(dict.fromkeys(chunk_ids))
        if not 1 <= len(expected) <= 16 or any(not item.strip() for item in expected):
            raise ValueError("chunk_ids must contain 1..16 non-empty ids")
        validated_filters = normalize_filters(filters)
        allowed = self._apply_filters(self.chunks, validated_filters)
        by_id = {chunk.chunk_id: chunk for chunk in allowed}
        return [by_id[item] for item in expected if item in by_id]

    @staticmethod
    def _apply_filters(
        chunks: list[KBChunk],
        filters: dict[str, Any],
    ) -> list[KBChunk]:
        def allowed(item: KBChunk) -> bool:
            for field, expected in filters.items():
                if expected in ([], ""):
                    continue
                actual = getattr(item, field)
                if isinstance(expected, list):
                    if actual not in expected:
                        return False
                elif actual != expected:
                    return False
            return True

        return [item for item in chunks if allowed(item)]

    @staticmethod
    def _keyword_score(
        query_sparse: dict[str, float],
        chunk_sparse: dict[str, float],
    ) -> float:
        if not query_sparse:
            return 0.0
        overlap = set(query_sparse).intersection(chunk_sparse)
        return min(
            1.0,
            sum(chunk_sparse[token] for token in overlap) * 4,
        )


def parse_order_by(order_by: list[str]) -> list[tuple[str, str]]:
    """Validate public ordering clauses before any adapter I/O."""

    parsed: list[tuple[str, str]] = []
    for clause in order_by:
        parts = clause.split()
        field = parts[0] if parts else ""
        direction = parts[1].lower() if len(parts) > 1 else "asc"
        if (
            field not in {"updated_at", "priority"}
            or direction not in {"asc", "desc"}
            or len(parts) > 2
        ):
            raise ValueError(f"Unsupported order_by clause: {clause!r}")
        parsed.append((field, direction))
    return parsed


def validate_aggregation_fields(fields: list[str]) -> list[str]:
    """Return de-duplicated public facet fields or fail before adapter I/O."""

    requested = list(dict.fromkeys(fields))
    if any(field not in AGGREGATION_FIELDS for field in requested):
        raise ValueError("Unsupported aggregation field")
    return requested


def order_search_results(
    results: list[SearchResult],
    order_by: list[str],
    *,
    order_mode: OrderMode,
) -> list[SearchResult]:
    """Apply the shared relevance-first or scalar-primary order contract."""

    if order_mode not in {"relevance", "scalar"}:
        raise ValueError(f"Unsupported order_mode: {order_mode!r}")
    parsed = parse_order_by(order_by)

    def scalar_key(result: SearchResult) -> tuple[float, ...]:
        return tuple(
            float(getattr(result.chunk, field)) * (1 if direction == "asc" else -1)
            for field, direction in parsed
        )

    if order_mode == "scalar" and parsed:
        return sorted(
            results,
            key=lambda item: (
                *scalar_key(item),
                -item.hybrid_score,
                item.chunk.chunk_id,
            ),
        )
    return sorted(
        results,
        key=lambda item: (-item.hybrid_score, *scalar_key(item), item.chunk.chunk_id),
    )
