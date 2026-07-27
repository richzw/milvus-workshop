"""Hybrid retrieval fallback that mirrors the Milvus search contract."""

from __future__ import annotations

from collections import Counter
from typing import Any, Protocol

from agent_workshop_demo.embedding import (
    cosine_similarity,
    dense_vector,
    sparse_vector,
)
from agent_workshop_demo.models import KBChunk, SearchResult
from agent_workshop_demo.validation import normalize_filters


class HybridRetriever(Protocol):
    """Search and aggregation interface shared by retrieval adapters."""

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


class InMemoryHybridRetriever:
    """Deterministic local replacement for the planned Milvus adapter."""

    def __init__(self, chunks: list[KBChunk]) -> None:
        self.chunks = chunks

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
        order_by: list[str] | None = None,
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

        ordered = self._apply_order(results, order_by or [])[:top_k]
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

    def aggregations(
        self,
        results: list[SearchResult],
        fields: list[str],
    ) -> dict[str, dict[str, int]]:
        """Count requested public scalar fields over recalled results."""

        output: dict[str, dict[str, int]] = {}
        for field in fields:
            values = (str(getattr(item.chunk, field)) for item in results)
            output[field] = dict(sorted(Counter(values).items()))
        return output

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

    @staticmethod
    def _apply_order(
        results: list[SearchResult],
        order_by: list[str],
    ) -> list[SearchResult]:
        supported_fields = {"updated_at", "priority"}
        parsed: list[tuple[str, str]] = []
        for clause in order_by:
            parts = clause.split()
            field = parts[0] if parts else ""
            direction = parts[1].lower() if len(parts) > 1 else "asc"
            if (
                field not in supported_fields
                or direction not in {"asc", "desc"}
                or len(parts) > 2
            ):
                raise ValueError(f"Unsupported order_by clause: {clause!r}")
            parsed.append((field, direction))

        def key(result: SearchResult) -> tuple[float, ...]:
            tie_breakers: list[float] = []
            for field, direction in parsed:
                value = float(getattr(result.chunk, field))
                tie_breakers.append(value if direction == "asc" else -value)
            return (-result.hybrid_score, *tie_breakers)

        return sorted(results, key=key)
