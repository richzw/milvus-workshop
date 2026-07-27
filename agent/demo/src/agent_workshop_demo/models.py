"""Public data models used by retrieval, workflow, CLI, and UI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


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
            raise ValueError(
                "doc_id, chunk_id, and doc_version must be non-empty"
            )
        if self.has_image_vector != (self.image_vector is not None):
            raise ValueError(
                "has_image_vector must match image_vector nullability"
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
    memory_written_count: int = 0
    memory_ttl_seconds: int = 86_400
    remembered_statement: str | None = None
    intent: str = "private_knowledge"
    query_type: str = "unknown"
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
    retrieval_provenance: dict[str, list[dict[str, Any]]] = field(
        default_factory=dict
    )
    rewritten_queries: list[str] = field(default_factory=list)
    query_rewrite_rounds: list[dict[str, Any]] = field(default_factory=list)
    version_scope: dict[str, Any] = field(
        default_factory=lambda: {"mode": "current", "doc_versions": []}
    )
    search_mode: str = "hybrid"
    search_filters: dict[str, Any] = field(default_factory=dict)
    search_order_by: list[str] = field(default_factory=list)
    milvus_top_k: int = 20
    retrieved_chunks: list[SearchResult] = field(default_factory=list)
    reranker_name: str = "rule-based-reranker"
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
