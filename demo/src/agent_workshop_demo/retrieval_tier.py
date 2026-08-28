"""Retrieval tier ladder: lexical baselines and the hybrid default.

Spec 15 treats retrieval complexity as a measured ladder. This module owns the
three tiers the repository can actually execute — T0 ``lexical_only``,
T1 ``lexical_rewrite`` and T2 ``hybrid_dense`` — plus the lexical retriever the
two baseline arms use. T3/T4/T5 are recorded design options, not code.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Final, Protocol

from agent_workshop_demo.models import KBChunk, SearchResult
from agent_workshop_demo.retrieval import OrderMode, order_search_results


class RetrievalTier(Enum):
    """Executable rungs of the spec 15 retrieval complexity ladder."""

    LEXICAL_ONLY = "lexical_only"
    LEXICAL_REWRITE = "lexical_rewrite"
    HYBRID_DENSE = "hybrid_dense"


TIER_CODES: Final[dict[RetrievalTier, str]] = {
    RetrievalTier.LEXICAL_ONLY: "T0",
    RetrievalTier.LEXICAL_REWRITE: "T1",
    RetrievalTier.HYBRID_DENSE: "T2",
}
LEXICAL_TIERS: Final = frozenset(
    {RetrievalTier.LEXICAL_ONLY, RetrievalTier.LEXICAL_REWRITE}
)
DEFAULT_TIER: Final = RetrievalTier.HYBRID_DENSE
LEXICAL_ONLY_CATALOG_VERSION: Final = "lexical_only_tier_no_catalog"


def parse_tier(value: RetrievalTier | str) -> RetrievalTier:
    """Return the registered tier or fail with a bounded message."""

    if isinstance(value, RetrievalTier):
        return value
    try:
        return RetrievalTier(value.strip())
    except ValueError as exc:
        raise ValueError("RETRIEVAL_TIER has an unsupported value") from exc


@dataclass(frozen=True)
class RetrievalTierConfig:
    """Validated retrieval tier selection for one runtime."""

    tier: RetrievalTier

    @property
    def tier_code(self) -> str:
        """Return the spec 15 ladder position, for example ``T2``."""

        return TIER_CODES[self.tier]

    @property
    def uses_dense_lane(self) -> bool:
        """Report whether stored chunk vectors are read at query time."""

        return self.tier is RetrievalTier.HYBRID_DENSE

    @property
    def uses_query_transformation(self) -> bool:
        """Report whether bounded transformation and entities are active."""

        return self.tier is not RetrievalTier.LEXICAL_ONLY

    def to_dict(self) -> dict[str, Any]:
        """Serialize a trace- and report-safe tier description."""

        return {
            "tier": self.tier.value,
            "tier_code": self.tier_code,
            "dense_lane": self.uses_dense_lane,
            "query_transformation": self.uses_query_transformation,
        }


def runtime_config_from_mapping(
    values: Mapping[str, str],
) -> RetrievalTierConfig:
    """Validate retrieval tier environment without performing I/O."""

    tier = parse_tier(values.get("RETRIEVAL_TIER", DEFAULT_TIER.value))
    struct_array = values.get("STRUCT_ARRAY_RETRIEVAL", "disabled").strip()
    if tier in LEXICAL_TIERS and struct_array not in {"", "disabled"}:
        raise ValueError(
            "StructArray profiles refine the hybrid_dense tier and cannot run "
            "under a lexical retrieval tier"
        )
    return RetrievalTierConfig(tier)


class SparseSearchRetriever(Protocol):
    """Retriever surface a lexical tier needs from its flat adapter."""

    supports_parallel_search: bool

    def search_sparse(
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


class LexicalOnlyRetriever:
    """Expose only the BM25 lane of a flat adapter as the T0/T1 arm.

    The wrapper deliberately does not implement ``search_profile`` or
    ``search_image_vector``: StructArray profiles and the image lane are
    hybrid-tier capabilities, so a lexical tier must not reach them.
    """

    def __init__(self, source: SparseSearchRetriever) -> None:
        if not callable(getattr(source, "search_sparse", None)):
            raise ValueError("A lexical retrieval tier requires a search_sparse lane")
        self.source = source
        self.supports_parallel_search = bool(
            getattr(source, "supports_parallel_search", False)
        )

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
        order_by: list[str] | None = None,
        order_mode: OrderMode = "relevance",
    ) -> list[SearchResult]:
        """Return ranked BM25 results under the shared ordering contract."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        clauses = list(order_by or [])
        results = self.source.search_sparse(
            query,
            top_k=top_k,
            filters=filters,
            order_by=clauses,
        )
        ordered = order_search_results(
            list(results),
            clauses,
            order_mode=order_mode,
        )[:top_k]
        return [
            replace(item, rank=index) for index, item in enumerate(ordered, start=1)
        ]

    def aggregations(
        self,
        results: list[SearchResult],
        fields: list[str],
    ) -> dict[str, dict[str, int]]:
        """Delegate facet counting to the wrapped adapter."""

        return self.source.aggregations(results, fields)

    def fetch_document_chunks(
        self,
        *,
        doc_id: str,
        doc_version: str,
        filters: dict[str, Any] | None = None,
        limit: int,
    ) -> list[KBChunk]:
        """Delegate bounded sibling expansion to the wrapped adapter."""

        return self.source.fetch_document_chunks(
            doc_id=doc_id,
            doc_version=doc_version,
            filters=filters,
            limit=limit,
        )

    def fetch_chunks_by_ids(
        self,
        *,
        chunk_ids: list[str],
        filters: dict[str, Any] | None = None,
    ) -> list[KBChunk]:
        """Delegate exact chunk lookup to the wrapped adapter."""

        return self.source.fetch_chunks_by_ids(
            chunk_ids=chunk_ids,
            filters=filters,
        )
