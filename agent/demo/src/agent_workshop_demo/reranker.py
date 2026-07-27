"""Reranker interface and deterministic fallback implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod

from agent_workshop_demo.embedding import sparse_vector
from agent_workshop_demo.models import RerankedResult, SearchResult


class Reranker(ABC):
    """Answer-specific precision-ranking contract."""

    name = "base-reranker"

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: list[SearchResult],
        top_k: int,
    ) -> list[RerankedResult]:
        """Return chunks ordered by answer relevance."""


class RuleBasedReranker(Reranker):
    """Deterministic fallback used when no model reranker is configured."""

    name = "rule-based-reranker"

    def rerank(
        self,
        query: str,
        chunks: list[SearchResult],
        top_k: int,
    ) -> list[RerankedResult]:
        """Combine retrieval, overlap, recency, and priority scores."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        query_terms = set(sparse_vector(query))
        scored: list[tuple[float, SearchResult]] = []
        for result in chunks:
            chunk_terms = set(result.chunk.sparse_vector)
            overlap = len(query_terms.intersection(chunk_terms)) / max(
                len(query_terms),
                1,
            )
            score = (
                0.6 * result.hybrid_score
                + 0.2 * overlap
                + 0.1 * result.recency_score
                + 0.1 * result.priority_score
            )
            scored.append((score, result))

        ordered = sorted(scored, key=lambda item: item[0], reverse=True)
        return [
            RerankedResult(
                search_result=result,
                rerank=index,
                old_rank=result.rank,
                rerank_score=score,
            )
            for index, (score, result) in enumerate(ordered[:top_k], start=1)
        ]
