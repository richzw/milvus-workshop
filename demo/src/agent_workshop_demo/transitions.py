"""Shared, fail-closed transition contract for local and LangGraph runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from agent_workshop_demo.models import AgentState, EvidenceAction


class WorkflowNode(str, Enum):
    """Closed logical node names used by conditional workflow edges."""

    CLASSIFY_AND_ROUTE = "classify_and_route"
    RESOLVE_TERMINOLOGY = "resolve_terminology"
    CHECK_PERMISSION = "check_permission"
    TRY_GROUNDED_CACHE = "try_grounded_cache"
    RECALL_AUTHORIZED_EXPERIENCE = "recall_authorized_experience"
    EXECUTE_TOOL_PLAN = "execute_tool_plan"
    RERANK_EVIDENCE = "rerank_evidence"
    EVALUATE_EVIDENCE = "evaluate_evidence"
    PREPARE_GENERATION_CONTEXT = "prepare_generation_context"
    GENERATE_CANDIDATE_ANSWER = "generate_candidate_answer"
    OUTPUT_GATE = "output_gate"


class TransitionReason(str, Enum):
    """Registered reasons safe to expose in tests and internal trace."""

    DIRECT_ROUTE = "direct_route"
    RETRIEVAL_ROUTE = "retrieval_route"
    CLARIFICATION_REQUIRED = "clarification_required"
    ENTITIES_RESOLVED = "entities_resolved"
    PERMISSION_DENIED = "permission_denied"
    PERMISSION_ALLOWED = "permission_allowed"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    NO_PROGRESS = "no_progress"
    EVIDENCE_PROGRESS = "evidence_progress"
    SUPPLEMENTARY_RETRY = "supplementary_retry"
    EVIDENCE_TERMINAL = "evidence_terminal"
    EVIDENCE_READY_FOR_CONTEXT = "evidence_ready_for_context"


@dataclass(frozen=True)
class WorkflowTransition:
    """One validated edge selected by the shared transition contract."""

    next_node: WorkflowNode
    reason: TransitionReason


def next_transition(
    completed_node: WorkflowNode,
    state: AgentState,
    *,
    evidence_action: EvidenceAction | None = None,
) -> WorkflowTransition:
    """Return the only valid next edge for the completed logical node."""

    if completed_node is WorkflowNode.CLASSIFY_AND_ROUTE:
        if state.need_retrieval:
            return WorkflowTransition(
                WorkflowNode.RESOLVE_TERMINOLOGY,
                TransitionReason.RETRIEVAL_ROUTE,
            )
        if state.terminal_status == "running":
            raise ValueError("direct route requires a terminal answer")
        return WorkflowTransition(
            WorkflowNode.OUTPUT_GATE,
            TransitionReason.DIRECT_ROUTE,
        )

    if completed_node is WorkflowNode.RESOLVE_TERMINOLOGY:
        if state.terminal_status == "clarification_required":
            return WorkflowTransition(
                WorkflowNode.OUTPUT_GATE,
                TransitionReason.CLARIFICATION_REQUIRED,
            )
        if state.terminal_status != "running":
            raise ValueError("resolved terminology requires a running state")
        return WorkflowTransition(
            WorkflowNode.CHECK_PERMISSION,
            TransitionReason.ENTITIES_RESOLVED,
        )

    if completed_node is WorkflowNode.CHECK_PERMISSION:
        if state.permission_decision.get("allowed") is True:
            if state.terminal_status != "running":
                raise ValueError("allowed permission requires a running state")
            return WorkflowTransition(
                WorkflowNode.TRY_GROUNDED_CACHE,
                TransitionReason.PERMISSION_ALLOWED,
            )
        if state.terminal_status != "permission_denied":
            raise ValueError("denied permission requires permission_denied state")
        return WorkflowTransition(
            WorkflowNode.OUTPUT_GATE,
            TransitionReason.PERMISSION_DENIED,
        )

    if completed_node is WorkflowNode.TRY_GROUNDED_CACHE:
        if state.terminal_status == "answered_from_cache":
            return WorkflowTransition(
                WorkflowNode.OUTPUT_GATE,
                TransitionReason.CACHE_HIT,
            )
        if state.terminal_status != "running":
            raise ValueError("cache miss requires a running state")
        return WorkflowTransition(
            WorkflowNode.RECALL_AUTHORIZED_EXPERIENCE,
            TransitionReason.CACHE_MISS,
        )

    if completed_node is WorkflowNode.EXECUTE_TOOL_PLAN:
        if state.candidate_pool_unchanged:
            if state.terminal_status != "abstained":
                raise ValueError("no progress requires an abstained state")
            return WorkflowTransition(
                WorkflowNode.GENERATE_CANDIDATE_ANSWER,
                TransitionReason.NO_PROGRESS,
            )
        if state.terminal_status != "running":
            raise ValueError("evidence progress requires a running state")
        return WorkflowTransition(
            WorkflowNode.RERANK_EVIDENCE,
            TransitionReason.EVIDENCE_PROGRESS,
        )

    if completed_node is WorkflowNode.EVALUATE_EVIDENCE:
        if evidence_action is None:
            raise ValueError("evidence evaluation requires an EvidenceAction")
        if evidence_action is EvidenceAction.RETRY:
            if state.terminal_status != "running":
                raise ValueError("retry requires a running state")
            return WorkflowTransition(
                WorkflowNode.EXECUTE_TOOL_PLAN,
                TransitionReason.SUPPLEMENTARY_RETRY,
            )
        if evidence_action is EvidenceAction.ANSWER:
            if not state.enough_evidence or state.terminal_status != "answered":
                raise ValueError("answer action requires sufficient evidence")
            return WorkflowTransition(
                WorkflowNode.PREPARE_GENERATION_CONTEXT,
                TransitionReason.EVIDENCE_READY_FOR_CONTEXT,
            )
        elif state.enough_evidence or state.terminal_status != "abstained":
            raise ValueError("abstain action requires insufficient evidence")
        return WorkflowTransition(
            WorkflowNode.GENERATE_CANDIDATE_ANSWER,
            TransitionReason.EVIDENCE_TERMINAL,
        )

    raise ValueError(f"node {completed_node.value!r} has no conditional transition")


def mark_no_progress_abstention(state: AgentState) -> None:
    """Record the execute-tool node's no-progress terminal state."""

    state.enough_evidence = False
    state.terminal_status = "abstained"
    grade = dict(state.evidence_grade)
    grade.update(
        {
            "enough_evidence": False,
            "reason": "Supplementary retrieval produced no new evidence.",
            "retry_count": state.retry_count,
            "max_retry": state.max_retry,
            "stop_reason": "no_progress",
            "candidate_pool_unchanged": True,
        }
    )
    state.evidence_grade = grade
