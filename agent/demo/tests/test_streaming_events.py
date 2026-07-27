from __future__ import annotations

import unittest
from typing import Any

from agent_workshop_demo.events import WorkflowEvent, WorkflowEventEmitter
from agent_workshop_demo.knowledge_tools import PermissionDecision
from agent_workshop_demo.langgraph_workflow import (
    LangGraphAgenticRAGWorkflow,
)
from agent_workshop_demo.streamlit_app import (
    StreamConsumer,
    append_trace_event,
    render_trace_timeline,
    safe_query_error,
)
from agent_workshop_demo.workflow import (
    AgenticRAGWorkflow,
    WorkflowStageError,
)


class StreamingEventTests(unittest.TestCase):
    def test_grounded_stream_orders_trace_validation_answer_and_final(
        self,
    ) -> None:
        events = list(
            AgenticRAGWorkflow().stream(
                "我们 S3 文档同步流程是怎么设计的？",
                query_id="query_stream_order",
            )
        )

        trace_events = [
            item["event"]
            for item in events
            if item["type"] == "trace_event"
        ]
        self.assertEqual(
            [item["sequence"] for item in trace_events],
            list(range(1, len(trace_events) + 1)),
        )
        self.assertEqual(
            {item["query_id"] for item in trace_events},
            {"query_stream_order"},
        )
        stages = [item["stage"] for item in trace_events]
        self.assertIn("verify_answer", stages)
        self.assertIn(
            "tool_completed",
            [item["kind"] for item in trace_events],
        )
        verify_index = next(
            index
            for index, item in enumerate(events)
            if item.get("event", {}).get("stage") == "verify_answer"
        )
        answer_index = next(
            index
            for index, item in enumerate(events)
            if item["type"] == "answer_delta"
        )
        self.assertLess(verify_index, answer_index)
        self.assertEqual(events[-1]["type"], "final")

    def test_retry_is_visible_without_exposing_rewritten_query(self) -> None:
        events = list(
            AgenticRAGWorkflow().stream(
                "不存在的采购宇宙飞船编号是什么？",
                query_id="query_stream_retry",
            )
        )
        retry_events = [
            item["event"]
            for item in events
            if item["type"] == "trace_event"
            and item["event"]["kind"] == "retry_scheduled"
        ]

        self.assertTrue(retry_events)
        for event in retry_events:
            self.assertNotIn("query", event["details"])
            self.assertNotIn("filters", event["details"])
            self.assertTrue(event["details"]["missing_aspects"])

    def test_exact_version_tool_event_identifies_edition(self) -> None:
        events = list(
            AgenticRAGWorkflow().stream(
                "GO按钮 v1 表示什么？",
                query_id="query_exact_version",
            )
        )
        tool_events = [
            item["event"]
            for item in events
            if item["type"] == "trace_event"
            and item["event"]["kind"] == "tool_completed"
        ]

        self.assertTrue(tool_events)
        self.assertEqual(tool_events[0]["details"]["version_mode"], "exact")
        self.assertEqual(tool_events[0]["details"]["doc_version"], "v1")

    def test_event_contract_rejects_invalid_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "sequence"):
            WorkflowEvent(
                query_id="query_1",
                sequence=0,
                kind="stage_completed",
                stage="classify_query",
                title="Done",
                summary="Done.",
            )
        emitter = WorkflowEventEmitter("query_1")
        first = emitter.emit(
            kind="stage_completed",
            stage="classify_query",
            title="Done",
            summary="Done.",
        )
        second = emitter.emit(
            kind="tool_completed",
            stage="retrieve",
            title="Tool",
            summary="One result.",
        )
        self.assertEqual(first["event"]["sequence"], 1)
        self.assertEqual(second["event"]["sequence"], 2)
        with self.assertRaisesRegex(ValueError, "unsafe"):
            emitter.emit(
                kind="stage_completed",
                stage="rewrite_query",
                title="Unsafe",
                summary="Unsafe.",
                details={"query": "private rewrite"},
            )

    def test_ui_accepts_only_active_in_order_events_and_escapes_html(
        self,
    ) -> None:
        event = WorkflowEvent(
            query_id="query_ui",
            sequence=1,
            kind="stage_completed",
            stage="classify_query",
            title="<script>alert(1)</script>",
            summary="<b>summary</b>",
        ).to_dict()
        events: list[dict[str, Any]] = []

        self.assertTrue(
            append_trace_event(events, event, query_id="query_ui")
        )
        self.assertFalse(
            append_trace_event(events, event, query_id="query_ui")
        )
        wrong_query = dict(event, sequence=2, query_id="query_other")
        self.assertFalse(
            append_trace_event(events, wrong_query, query_id="query_ui")
        )
        rendered = render_trace_timeline(events)
        self.assertNotIn("<script>", rendered)
        self.assertNotIn("<b>", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        markdown = dict(
            event,
            sequence=2,
            title="![track](https://example.test/pixel)",
        )
        self.assertTrue(
            append_trace_event(events, markdown, query_id="query_ui")
        )
        rendered = render_trace_timeline(events)
        self.assertNotIn("<img", rendered)
        self.assertNotIn("<a ", rendered)
        self.assertNotIn("![", rendered)
        self.assertNotIn("](", rendered)
        self.assertIn("&#33;&#91;track&#93;&#40;", rendered)
        unsafe = dict(event, sequence=3, details={"filters": {"acl": "*"}})
        self.assertFalse(
            append_trace_event(events, unsafe, query_id="query_ui")
        )

    def test_ui_error_does_not_render_raw_dependency_message(self) -> None:
        error = WorkflowStageError(
            "milvus_hybrid_retrieve",
            "query_safe_error",
            RuntimeError("token=secret-value"),
        )

        rendered = safe_query_error(error)

        self.assertIn("milvus_hybrid_retrieve", rendered)
        self.assertIn("query_safe_error", rendered)
        self.assertNotIn("secret-value", rendered)

    def test_stream_consumer_rejects_invalid_terminal_protocol(self) -> None:
        consumer = StreamConsumer("query_consumer")
        consumer.consume(
            {
                "type": "answer_delta",
                "text": "validated answer",
            }
        )
        with self.assertRaisesRegex(RuntimeError, "invalid final"):
            consumer.consume(
                {
                    "type": "final",
                    "response": {
                        "query_id": "query_other",
                        "answer": "validated answer",
                    },
                }
            )

        valid = StreamConsumer("query_consumer")
        valid.consume({"type": "answer_delta", "text": "answer"})
        valid.consume(
            {
                "type": "final",
                "response": {
                    "query_id": "query_consumer",
                    "answer": "answer",
                },
            }
        )
        with self.assertRaisesRegex(RuntimeError, "after final"):
            valid.consume({"type": "answer_delta", "text": "extra"})
        with self.assertRaisesRegex(RuntimeError, "after final"):
            valid.consume(
                {
                    "type": "final",
                    "response": {
                        "query_id": "query_consumer",
                        "answer": "answer",
                    },
                }
            )
        with self.assertRaisesRegex(RuntimeError, "without a terminal"):
            StreamConsumer("query_incomplete").finish()
        trace_after_answer = StreamConsumer("query_trace_after_answer")
        trace_after_answer.consume({"type": "answer_delta", "text": "answer"})
        with self.assertRaisesRegex(RuntimeError, "trace after answer"):
            trace_after_answer.consume(
                {
                    "type": "trace_event",
                    "event": WorkflowEvent(
                        query_id="query_trace_after_answer",
                        sequence=1,
                        kind="stage_completed",
                        stage="verify_answer",
                        title="Verified",
                        summary="Verified.",
                    ).to_dict(),
                }
            )

    def test_terminal_branches_have_safe_complete_streams(self) -> None:
        class DenyPermissionChecker:
            def check(
                self,
                *,
                session_id: str,
                intent: str,
                query_type: str,
            ) -> PermissionDecision:
                del session_id, intent, query_type
                return PermissionDecision(
                    allowed=False,
                    allowed_departments=(),
                    reason="denied",
                    checker_name="test",
                )

        cases = [
            (AgenticRAGWorkflow(), "你好", "answered_without_retrieval"),
            (
                AgenticRAGWorkflow(),
                "帮我删除产品路线图",
                "refused_unsupported_operation",
            ),
            (
                AgenticRAGWorkflow(),
                "段位是什么意思？",
                "clarification_required",
            ),
            (
                AgenticRAGWorkflow(
                    permission_checker=DenyPermissionChecker()
                ),
                "请查看内部产品路线图",
                "permission_denied",
            ),
        ]
        for index, (workflow, question, expected) in enumerate(cases):
            query_id = f"query_terminal_{index}"
            events = list(workflow.stream(question, query_id=query_id))
            consumer = StreamConsumer(query_id)
            for event in events:
                consumer.consume(event)
            self.assertEqual(
                consumer.finish()["terminal_status"],
                expected,
            )

    def test_langgraph_adapter_rejects_missing_final_update(self) -> None:
        workflow = AgenticRAGWorkflow()

        class MissingFinalGraph:
            def stream(
                self,
                payload: dict[str, Any],
                *,
                stream_mode: str,
            ) -> Any:
                self_stream_mode = stream_mode
                if self_stream_mode != "updates":
                    raise AssertionError("unexpected stream mode")
                state = payload["state"]
                workflow._measure_stage(
                    state,
                    "classify_query",
                    lambda: workflow.classify_query(state),
                )
                yield {"classify_query": payload}

        adapter = LangGraphAgenticRAGWorkflow.__new__(
            LangGraphAgenticRAGWorkflow
        )
        adapter.workflow = workflow
        adapter.app = MissingFinalGraph()  # type: ignore[assignment]

        with self.assertRaisesRegex(RuntimeError, "without a terminal"):
            list(
                adapter.stream(
                    "你好",
                    query_id="query_missing_graph_final",
                )
            )

    def test_langgraph_adapter_releases_answer_after_verify_update(
        self,
    ) -> None:
        workflow = AgenticRAGWorkflow()

        class VerifiedGraph:
            def stream(
                self,
                payload: dict[str, Any],
                *,
                stream_mode: str,
            ) -> Any:
                if stream_mode != "updates":
                    raise AssertionError("unexpected stream mode")
                state = payload["state"]
                state.answer = "validated answer"
                state.answer_validation = {
                    "valid": True,
                    "mode": "abstention",
                }
                state.stage_latency_ms["verify_answer"] = 1.0
                payload["answer_chunks"] = ["validated ", "answer"]
                yield {"verify_answer": payload}
                payload["response"] = {
                    "query_id": state.query_id,
                    "answer": state.answer,
                }
                yield {"finalize": payload}

        adapter = LangGraphAgenticRAGWorkflow.__new__(
            LangGraphAgenticRAGWorkflow
        )
        adapter.workflow = workflow
        adapter.app = VerifiedGraph()  # type: ignore[assignment]

        events = list(
            adapter.stream(
                "你好",
                query_id="query_verified_graph",
            )
        )

        self.assertEqual(events[0]["type"], "trace_event")
        self.assertEqual(events[0]["event"]["stage"], "verify_answer")
        self.assertEqual(events[1]["type"], "answer_delta")
        self.assertEqual(events[-1]["type"], "final")


if __name__ == "__main__":
    unittest.main()
