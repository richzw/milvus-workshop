"""Safe, presentation-facing events emitted by Agentic RAG workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any, Literal

EventKind = Literal[
    "stage_completed",
    "tool_completed",
    "retry_scheduled",
]
EventStatus = Literal["completed", "warning"]
SAFE_DETAIL_FIELDS = frozenset(
    {
        "allowed",
        "allowed_department_count",
        "ambiguity_count",
        "candidate_count",
        "citation_count",
        "context_count",
        "doc_version",
        "enough_evidence",
        "fallback_reason",
        "generator_name",
        "intent",
        "matched_entity_count",
        "memory_status",
        "memory_types",
        "missing_aspects",
        "mode",
        "model",
        "need_retrieval",
        "plan_count",
        "query_type",
        "relevant_count",
        "result_count",
        "recalled_count",
        "retry_count",
        "round",
        "selected_tools",
        "tool",
        "ttl_seconds",
        "valid",
        "version_mode",
        "written_count",
    }
)


def details_are_safe(details: object) -> bool:
    """Return whether details contain only bounded presentation metadata."""

    if not isinstance(details, dict):
        return False
    if not set(details).issubset(SAFE_DETAIL_FIELDS):
        return False
    for value in details.values():
        if isinstance(value, str) and len(value) > 120:
            return False
        if isinstance(value, float) and not isfinite(value):
            return False
        if value is None or isinstance(value, (str, int, float, bool)):
            continue
        if (
            isinstance(value, list)
            and len(value) <= 16
            and all(
                isinstance(item, str) and len(item) <= 120
                for item in value
            )
        ):
            continue
        return False
    return True


@dataclass(frozen=True)
class WorkflowEvent:
    """A bounded event safe to expose in the teaching UI."""

    query_id: str
    sequence: int
    kind: EventKind
    stage: str
    title: str
    summary: str
    status: EventStatus = "completed"
    elapsed_ms: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.query_id.strip():
            raise ValueError("Workflow event query_id must be non-empty")
        if self.sequence < 1:
            raise ValueError("Workflow event sequence must be positive")
        if self.kind not in {
            "stage_completed",
            "tool_completed",
            "retry_scheduled",
        }:
            raise ValueError("Workflow event kind is unsupported")
        if self.status not in {"completed", "warning"}:
            raise ValueError("Workflow event status is unsupported")
        if not self.stage.strip():
            raise ValueError("Workflow event stage must be non-empty")
        if not self.title.strip() or len(self.title) > 80:
            raise ValueError(
                "Workflow event title must contain 1 to 80 characters"
            )
        if not self.summary.strip() or len(self.summary) > 300:
            raise ValueError(
                "Workflow event summary must contain 1 to 300 characters"
            )
        if self.elapsed_ms is not None and (
            self.elapsed_ms < 0 or not isfinite(self.elapsed_ms)
        ):
            raise ValueError(
                "Workflow event elapsed_ms must be finite and non-negative"
            )
        if not details_are_safe(self.details):
            raise ValueError("Workflow event details contain unsafe fields")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation."""

        return asdict(self)


class WorkflowEventEmitter:
    """Assign one contiguous sequence to a query's workflow events."""

    def __init__(self, query_id: str) -> None:
        if not query_id.strip():
            raise ValueError("Workflow event query_id must be non-empty")
        self.query_id = query_id
        self._next_sequence = 1

    def emit(
        self,
        *,
        kind: EventKind,
        stage: str,
        title: str,
        summary: str,
        status: EventStatus = "completed",
        elapsed_ms: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the next validated event envelope."""

        event = WorkflowEvent(
            query_id=self.query_id,
            sequence=self._next_sequence,
            kind=kind,
            stage=stage,
            title=title,
            summary=summary,
            status=status,
            elapsed_ms=elapsed_ms,
            details={} if details is None else details,
        )
        self._next_sequence += 1
        return {"type": "trace_event", "event": event.to_dict()}
