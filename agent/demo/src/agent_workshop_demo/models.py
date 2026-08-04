"""Public data models used by retrieval, workflow, CLI, and UI."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from agent_workshop_demo.config import VECTOR_DIMS


class QueryRoute(str, Enum):
    """Closed workflow route after bounded query classification."""

    DIRECT = "direct"
    RETRIEVAL = "retrieval"


@dataclass(frozen=True)
class QueryRouteResult:
    """Typed outcome of the combined classification and routing stage."""

    route: QueryRoute
    reason: str

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("query route reason must be non-empty")


@dataclass(frozen=True)
class RetrievalPlanResult:
    """Typed summary of one authorized initial retrieval plan."""

    selected_tools: tuple[str, ...]
    plan_count: int

    def __post_init__(self) -> None:
        if not self.selected_tools:
            raise ValueError("retrieval plan must select at least one tool")
        if not 1 <= self.plan_count <= 3:
            raise ValueError("retrieval plan count must be between 1 and 3")


class EvidenceAction(str, Enum):
    """Closed next action returned by evidence evaluation."""

    ANSWER = "answer"
    RETRY = "retry"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class EvidenceEvaluation:
    """Typed result of grading and optional retry-plan preparation."""

    action: EvidenceAction
    reason: str

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("evidence evaluation reason must be non-empty")


@dataclass(frozen=True)
class KBChunk:
    """One citation-addressable knowledge record."""

    doc_id: str
    chunk_id: str
    parent_id: str | None
    record_type: str
    source_type: str
    source_uri: str
    bucket: str | None
    object_key: str | None
    doc_type: str
    title: str
    section: str | None
    page_no: int | None
    chunk_index: int
    text: str
    text_summary: str | None
    language: str
    department: str
    updated_at: int
    created_at: int | None
    priority: int
    doc_version: str
    is_current: bool
    checksum: str | None
    metadata: dict[str, Any] | None
    has_image_vector: bool
    text_vector: list[float]
    sparse_vector: dict[str, float]
    image_vector: list[float] | None = None

    def __post_init__(self) -> None:
        if (
            not self.doc_id.strip()
            or not self.chunk_id.strip()
            or not self.doc_version.strip()
        ):
            raise ValueError("doc_id, chunk_id, and doc_version must be non-empty")
        if self.has_image_vector != (self.image_vector is not None):
            raise ValueError("has_image_vector must match image_vector nullability")
        if self.image_vector is not None:
            image_fingerprint = (
                self.metadata.get("image_embedding_fingerprint")
                if self.metadata
                else None
            )
            if (
                not isinstance(image_fingerprint, str)
                or not image_fingerprint.strip()
            ):
                raise ValueError(
                    "image vectors require metadata.image_embedding_fingerprint"
                )
            invalid_image_values = (
                len(self.image_vector) != VECTOR_DIMS["IMAGE_DIM"]
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in self.image_vector
                )
            )
            image_norm = (
                0.0
                if invalid_image_values
                else math.sqrt(
                    sum(float(value) ** 2 for value in self.image_vector)
                )
            )
            if invalid_image_values or not math.isclose(
                image_norm,
                1.0,
                rel_tol=1e-5,
                abs_tol=1e-5,
            ):
                raise ValueError(
                    "image_vector must contain exactly "
                    f"{VECTOR_DIMS['IMAGE_DIM']} finite L2-normalized values"
                )
        if self.record_type == "pdf_page" and self.page_no is None:
            raise ValueError("pdf_page records require page_no")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete storage record."""

        return asdict(self)

    def to_public_dict(self) -> dict[str, Any]:
        """Serialize fields safe and useful for the teaching UI."""

        data = asdict(self)
        data.pop("text_vector", None)
        data.pop("sparse_vector", None)
        data.pop("image_vector", None)
        return data

    def citation(self, citation_id: str) -> dict[str, Any]:
        """Build a stable chunk/page/version citation."""

        return {
            "citation_id": citation_id,
            "title": self.title,
            "source_uri": self.source_uri,
            "page_no": self.page_no,
            "doc_id": self.doc_id,
            "doc_version": self.doc_version,
            "chunk_id": self.chunk_id,
            "section": self.section,
        }

    def snippet(self) -> str:
        """Return a bounded UI snippet."""

        if self.text_summary:
            return self.text_summary
        suffix = "..." if len(self.text) > 220 else ""
        return self.text[:220] + suffix


@dataclass(frozen=True)
class SearchResult:
    """Normalized high-recall search result."""

    chunk: KBChunk
    rank: int
    dense_score: float
    keyword_score: float
    recency_score: float
    priority_score: float
    hybrid_score: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize a UI-safe result."""

        data = self.chunk.to_public_dict()
        data.update(
            {
                "rank": self.rank,
                "dense_score": round(self.dense_score, 4),
                "keyword_score": round(self.keyword_score, 4),
                "recency_score": round(self.recency_score, 4),
                "priority_score": round(self.priority_score, 4),
                "hybrid_score": round(self.hybrid_score, 4),
                "score": round(self.hybrid_score, 4),
            }
        )
        return data


@dataclass(frozen=True)
class ImageSearchResult:
    """Normalized image-vector similarity result without public vectors."""

    chunk: KBChunk
    rank: int
    image_score: float

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("image search rank must be greater than zero")
        if not math.isfinite(self.image_score):
            raise ValueError("image search score must be finite")
        if not self.chunk.has_image_vector:
            raise ValueError("image search results require an image vector")

    def to_dict(self) -> dict[str, Any]:
        """Serialize scalar provenance and score while omitting vectors."""

        data = self.chunk.to_public_dict()
        data.update(
            {
                "rank": self.rank,
                "image_score": round(self.image_score, 4),
                "score": round(self.image_score, 4),
                "retrieval_mode": "image_vector",
            }
        )
        return data


@dataclass(frozen=True)
class RerankedResult:
    """A search result after answer-specific precision ranking."""

    search_result: SearchResult
    rerank: int
    old_rank: int
    rerank_score: float
    selected: bool = False

    @property
    def chunk(self) -> KBChunk:
        """Expose the underlying knowledge record."""

        return self.search_result.chunk

    def to_dict(self) -> dict[str, Any]:
        """Serialize a UI-safe reranked result."""

        data = self.search_result.to_dict()
        data.update(
            {
                "rerank": self.rerank,
                "old_rank": self.old_rank,
                "rerank_score": round(self.rerank_score, 4),
                "selected": self.selected,
            }
        )
        return data


@dataclass
class AgentState:
    """Mutable state owned by one workflow invocation."""

    user_query: str
    query_id: str
    session_id: str
    recalled_memories: list[dict[str, Any]] = field(default_factory=list)
    memory_context: str = ""
    memory_status: str = "empty"
    memory_recall_decision: str = "skipped"
    memory_recall_mode: str = "none"
    memory_recall_reason: str = "not_applicable"
    memory_requested_count: int | None = None
    memory_recall_types: list[str] = field(default_factory=list)
    memory_written_count: int = 0
    memory_ttl_seconds: int = 86_400
    remembered_statement: str | None = None
    selective_memory_status: str = "empty"
    selective_memory_pack: dict[str, Any] = field(default_factory=dict)
    selective_memory_private_values: list[str] = field(default_factory=list)
    selective_memory_written_count: int = 0
    selective_memory_retention_class: str | None = None
    selective_memory_selection_reasons: list[str] = field(default_factory=list)
    selective_memory_consolidation_status: str = "not_run"
    selective_memory_selector_name: str | None = None
    selective_memory_selector_model: str | None = None
    selective_memory_selector_fallback_reason: str | None = None
    response_cache_status: str = "miss"
    response_cache_candidates: list[Any] = field(default_factory=list)
    response_cache_candidate_count: int = 0
    response_cache_match_type: str | None = None
    response_cache_similarity: float | None = None
    response_cache_source_query_id: str | None = None
    response_cache_fallback_reason: str | None = None
    response_cache_expires_at: int | None = None
    response_cache_written_count: int = 0
    intent: str = "private_knowledge"
    query_type: str = "unknown"
    classifier_name: str = "not_invoked"
    classifier_model: str | None = None
    classification_confidence: float | None = None
    classification_fallback_reason: str | None = None
    entity_catalog_version: str = ""
    matched_entities: list[dict[str, Any]] = field(default_factory=list)
    ambiguous_entities: list[dict[str, Any]] = field(default_factory=list)
    need_retrieval: bool = True
    retrieval_decision: dict[str, Any] = field(default_factory=dict)
    permission_decision: dict[str, Any] = field(default_factory=dict)
    selected_tools: list[str] = field(default_factory=list)
    tool_selection_reasons: dict[str, str] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    query_plan: list[dict[str, Any]] = field(default_factory=list)
    retrieval_provenance: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    rewritten_queries: list[str] = field(default_factory=list)
    query_rewrite_rounds: list[dict[str, Any]] = field(default_factory=list)
    retrieval_goal: str = "focused"
    document_expansions: list[dict[str, Any]] = field(default_factory=list)
    version_scope: dict[str, Any] = field(
        default_factory=lambda: {"mode": "current", "doc_versions": []}
    )
    search_mode: str = "hybrid"
    retrieval_execution_mode: str = "sequential"
    search_filters: dict[str, Any] = field(default_factory=dict)
    search_order_by: list[str] = field(default_factory=list)
    milvus_top_k: int = 20
    retrieved_chunks: list[SearchResult] = field(default_factory=list)
    candidate_pool_fingerprint: str | None = None
    candidate_pool_unchanged: bool = False
    retrieval_stop_reason: str | None = None
    reranker_name: str = "not_run"
    reranker_model: str | None = None
    reranker_fallback_reason: str | None = None
    reranker_sticky_fallback_reason: str | None = None
    reranker_primary_attempt_count: int = 0
    reranker_fallback_only_count: int = 0
    reranker_top_k: int = 8
    reranked_chunks: list[RerankedResult] = field(default_factory=list)
    enough_evidence: bool = False
    evidence_grade: dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retry: int = 3
    retry_queries: list[str] = field(default_factory=list)
    answer: str = ""
    terminal_status: str = "running"
    citations: list[dict[str, Any]] = field(default_factory=list)
    answer_generator_name: str = "not_invoked"
    answer_model: str | None = None
    generation_fallback_reason: str | None = None
    generation_mode: str = "validated_buffered"
    generation_context_count: int = 0
    generation_resolved_entity_count: int = 0
    generation_context_truncated_count: int = 0
    answer_validation: dict[str, Any] = field(default_factory=dict)
    aggregations: dict[str, dict[str, int]] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    stage_latency_ms: dict[str, float] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)
