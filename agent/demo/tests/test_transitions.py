from __future__ import annotations

import unittest
from collections.abc import Callable
from unittest.mock import patch

from agent_workshop_demo.models import AgentState, EvidenceAction
from agent_workshop_demo.transitions import (
    TransitionReason,
    WorkflowTransition,
    WorkflowNode,
    mark_no_progress_abstention,
    next_transition,
)
from agent_workshop_demo.workflow import AgenticRAGWorkflow


def state() -> AgentState:
    return AgentState(
        user_query="test query",
        query_id="query_transition",
        session_id="session_transition",
    )


class WorkflowTransitionTests(unittest.TestCase):
    def test_every_registered_branch_returns_the_expected_edge(self) -> None:
        cases: list[
            tuple[
                WorkflowNode,
                Callable[[AgentState], None],
                EvidenceAction | None,
                WorkflowNode,
                TransitionReason,
            ]
        ] = [
            (
                WorkflowNode.CLASSIFY_AND_ROUTE,
                lambda item: (
                    setattr(item, "need_retrieval", False),
                    setattr(item, "terminal_status", "answered_without_retrieval"),
                ),
                None,
                WorkflowNode.OUTPUT_GATE,
                TransitionReason.DIRECT_ROUTE,
            ),
            (
                WorkflowNode.CLASSIFY_AND_ROUTE,
                lambda item: None,
                None,
                WorkflowNode.RESOLVE_TERMINOLOGY,
                TransitionReason.RETRIEVAL_ROUTE,
            ),
            (
                WorkflowNode.RESOLVE_TERMINOLOGY,
                lambda item: setattr(
                    item,
                    "terminal_status",
                    "clarification_required",
                ),
                None,
                WorkflowNode.OUTPUT_GATE,
                TransitionReason.CLARIFICATION_REQUIRED,
            ),
            (
                WorkflowNode.RESOLVE_TERMINOLOGY,
                lambda item: None,
                None,
                WorkflowNode.CHECK_PERMISSION,
                TransitionReason.ENTITIES_RESOLVED,
            ),
            (
                WorkflowNode.CHECK_PERMISSION,
                lambda item: setattr(
                    item,
                    "permission_decision",
                    {"allowed": True},
                ),
                None,
                WorkflowNode.TRY_GROUNDED_CACHE,
                TransitionReason.PERMISSION_ALLOWED,
            ),
            (
                WorkflowNode.CHECK_PERMISSION,
                lambda item: (
                    setattr(item, "permission_decision", {"allowed": False}),
                    setattr(item, "terminal_status", "permission_denied"),
                ),
                None,
                WorkflowNode.OUTPUT_GATE,
                TransitionReason.PERMISSION_DENIED,
            ),
            (
                WorkflowNode.TRY_GROUNDED_CACHE,
                lambda item: setattr(
                    item,
                    "terminal_status",
                    "answered_from_cache",
                ),
                None,
                WorkflowNode.OUTPUT_GATE,
                TransitionReason.CACHE_HIT,
            ),
            (
                WorkflowNode.TRY_GROUNDED_CACHE,
                lambda item: None,
                None,
                WorkflowNode.RECALL_AUTHORIZED_EXPERIENCE,
                TransitionReason.CACHE_MISS,
            ),
            (
                WorkflowNode.EXECUTE_TOOL_PLAN,
                lambda item: setattr(
                    item,
                    "candidate_pool_unchanged",
                    True,
                ),
                None,
                WorkflowNode.GENERATE_CANDIDATE_ANSWER,
                TransitionReason.NO_PROGRESS,
            ),
            (
                WorkflowNode.EXECUTE_TOOL_PLAN,
                lambda item: None,
                None,
                WorkflowNode.RERANK_EVIDENCE,
                TransitionReason.EVIDENCE_PROGRESS,
            ),
            (
                WorkflowNode.EVALUATE_EVIDENCE,
                lambda item: None,
                EvidenceAction.RETRY,
                WorkflowNode.EXECUTE_TOOL_PLAN,
                TransitionReason.SUPPLEMENTARY_RETRY,
            ),
            (
                WorkflowNode.EVALUATE_EVIDENCE,
                lambda item: (
                    setattr(item, "enough_evidence", True),
                    setattr(item, "terminal_status", "answered"),
                ),
                EvidenceAction.ANSWER,
                WorkflowNode.PREPARE_GENERATION_CONTEXT,
                TransitionReason.EVIDENCE_READY_FOR_CONTEXT,
            ),
            (
                WorkflowNode.EVALUATE_EVIDENCE,
                lambda item: setattr(item, "terminal_status", "abstained"),
                EvidenceAction.ABSTAIN,
                WorkflowNode.GENERATE_CANDIDATE_ANSWER,
                TransitionReason.EVIDENCE_TERMINAL,
            ),
        ]

        for completed, arrange, action, expected_node, expected_reason in cases:
            with self.subTest(completed=completed, reason=expected_reason):
                item = state()
                arrange(item)
                if (
                    completed is WorkflowNode.EXECUTE_TOOL_PLAN
                    and item.candidate_pool_unchanged
                ):
                    mark_no_progress_abstention(item)
                result = next_transition(
                    completed,
                    item,
                    evidence_action=action,
                )
                self.assertIs(result.next_node, expected_node)
                self.assertIs(result.reason, expected_reason)

    def test_impossible_states_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "terminal answer"):
            item = state()
            item.need_retrieval = False
            next_transition(WorkflowNode.CLASSIFY_AND_ROUTE, item)

        with self.assertRaisesRegex(ValueError, "EvidenceAction"):
            next_transition(WorkflowNode.EVALUATE_EVIDENCE, state())

        with self.assertRaisesRegex(ValueError, "permission_denied"):
            item = state()
            item.permission_decision = {"allowed": False}
            next_transition(WorkflowNode.CHECK_PERMISSION, item)

    def test_execute_tool_node_owns_no_progress_terminal_state(self) -> None:
        item = state()
        item.retry_count = 2
        item.candidate_pool_unchanged = True

        mark_no_progress_abstention(item)
        next_transition(WorkflowNode.EXECUTE_TOOL_PLAN, item)

        self.assertEqual(item.terminal_status, "abstained")
        self.assertEqual(item.evidence_grade["stop_reason"], "no_progress")
        self.assertEqual(item.evidence_grade["retry_count"], 2)

    def test_local_dispatcher_follows_returned_next_node(self) -> None:
        workflow = AgenticRAGWorkflow()

        with patch(
            "agent_workshop_demo.workflow.next_transition",
            return_value=WorkflowTransition(
                WorkflowNode.OUTPUT_GATE,
                TransitionReason.DIRECT_ROUTE,
            ),
        ):
            item, _started = workflow._prepare_answer_state(
                "Milvus 架构",
                None,
                "session_dispatch",
                "query_dispatch",
            )

        self.assertIn("classify_and_route", item.stage_latency_ms)
        self.assertNotIn("resolve_terminology", item.stage_latency_ms)
        self.assertNotIn("check_permission", item.stage_latency_ms)


if __name__ == "__main__":
    unittest.main()
