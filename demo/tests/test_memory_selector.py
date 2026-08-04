from __future__ import annotations

import json
import unittest
from typing import Any
from unittest.mock import Mock

from agent_workshop_demo.schema.collections import MEMORY_EVENTS_COLLECTION
from agent_workshop_demo.selective_memory import (
    LLMMemorySelector,
    LocalSelectiveMemoryStore,
    MemorySelection,
    RuleBasedMemorySelector,
    SelectiveMemoryService,
    build_memory_selector,
    event_from_storage,
)
from agent_workshop_demo.workflow import AgenticRAGWorkflow


class RecordingResponses:
    def __init__(
        self,
        *,
        payload: object | None = None,
        error: Exception | None = None,
    ) -> None:
        self.payload = payload
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        if self.error is not None:
            raise self.error
        return type(
            "Response",
            (),
            {"output_text": json.dumps(self.payload)},
        )()


class RecordingClient:
    def __init__(
        self,
        *,
        payload: object | None = None,
        error: Exception | None = None,
    ) -> None:
        self.responses = RecordingResponses(payload=payload, error=error)


class FixedAmbiguousSelector:
    def select(
        self,
        *,
        query: str,
        terminal_status: str,
        remembered_statement: str | None,
    ) -> MemorySelection:
        del query, terminal_status, remembered_statement
        return MemorySelection(
            event_type="user_statement",
            salience_score=0.5,
            selection_reason=("future_utility_ambiguous",),
            retention_class="ephemeral",
            decay_profile="episode_fast",
        )


class MemorySelectorTests(unittest.TestCase):
    def test_out_of_band_and_protected_rules_make_zero_model_calls(self) -> None:
        client = RecordingClient(payload={"decision": "promote_candidate"})
        selector = LLMMemorySelector(
            client=client,
            model="memory-selector-test",
        )

        ordinary = selector.select(
            query="你好",
            terminal_status="answered_without_retrieval",
            remembered_statement=None,
        )
        failure = selector.select(
            query="没有找到资料",
            terminal_status="abstained",
            remembered_statement=None,
        )
        protected = selector.select(
            query="请记住以后用中文",
            terminal_status="memory_saved",
            remembered_statement="以后用中文",
        )

        self.assertEqual(ordinary.retention_class, "ephemeral")
        self.assertEqual(failure.retention_class, "candidate")
        self.assertEqual(protected.retention_class, "protected")
        self.assertEqual(client.responses.calls, [])

    def test_in_band_model_can_only_promote_candidate_with_strict_schema(
        self,
    ) -> None:
        client = RecordingClient(payload={"decision": "promote_candidate"})
        selector = LLMMemorySelector(
            client=client,
            model="memory-selector-test",
            fallback=FixedAmbiguousSelector(),
            timeout_seconds=3.5,
        )

        result = selector.select(
            query="这个步骤下次可能还要用",
            terminal_status="answered",
            remembered_statement=None,
        )

        self.assertEqual(result.salience_score, 0.5)
        self.assertEqual(result.retention_class, "candidate")
        self.assertEqual(result.decay_profile, "experience_balanced")
        self.assertEqual(result.selector_name, "openai")
        self.assertEqual(result.selector_model, "memory-selector-test")
        self.assertIsNone(result.selector_fallback_reason)
        self.assertIn("llm_promote_candidate", result.selection_reason)
        self.assertEqual(len(client.responses.calls), 1)
        call = client.responses.calls[0]
        self.assertEqual(call["timeout"], 3.5)
        self.assertEqual(call["max_output_tokens"], 50)
        output_format = call["text"]["format"]
        self.assertEqual(output_format["type"], "json_schema")
        self.assertTrue(output_format["strict"])
        self.assertFalse(
            output_format["schema"]["additionalProperties"]
        )
        prompt = json.loads(call["input"])
        self.assertEqual(prompt["user_query"], "这个步骤下次可能还要用")
        self.assertNotIn("assistant_answer", prompt)

        ephemeral = LLMMemorySelector(
            client=RecordingClient(payload={"decision": "ephemeral"}),
            model="memory-selector-test",
            fallback=FixedAmbiguousSelector(),
        ).select(
            query="一次性信息",
            terminal_status="answered",
            remembered_statement=None,
        )
        self.assertEqual(ephemeral.retention_class, "ephemeral")
        self.assertIn("llm_ephemeral", ephemeral.selection_reason)

    def test_configured_builder_reaches_model_from_real_rule_signal(
        self,
    ) -> None:
        client = RecordingClient(
            payload={"decision": "promote_candidate"}
        )
        selector = build_memory_selector(
            {
                "MEMORY_SELECTOR": "openai",
                "OPENAI_API_KEY": "test-key",
                "OPENAI_MEMORY_SELECTOR_MODEL": "memory-selector-test",
            },
            client_factory=lambda _api_key: client,
        )

        result = selector.select(
            query="这个排障步骤下次可能还会复用",
            terminal_status="answered",
            remembered_statement=None,
        )

        self.assertEqual(result.retention_class, "candidate")
        self.assertEqual(result.salience_score, 0.4)
        self.assertEqual(
            result.selection_reason,
            (
                "future_utility_ambiguous",
                "llm_promote_candidate",
            ),
        )
        self.assertEqual(len(client.responses.calls), 1)

    def test_future_phrases_without_reuse_utility_stay_ordinary(self) -> None:
        client = RecordingClient(
            payload={"decision": "promote_candidate"}
        )
        selector = LLMMemorySelector(
            client=client,
            model="memory-selector-test",
        )

        for query in (
            "下次可能会下雨吗？",
            "以后可能发生什么？",
            "将来或许没有发布计划",
            "we may discuss weather later",
        ):
            with self.subTest(query=query):
                result = selector.select(
                    query=query,
                    terminal_status="answered",
                    remembered_statement=None,
                )
                self.assertEqual(result.salience_score, 0.2)
                self.assertEqual(result.retention_class, "ephemeral")
        self.assertEqual(client.responses.calls, [])

    def test_invalid_output_and_timeout_return_exact_rule_decision(self) -> None:
        invalid_payloads = [
            {"decision": "protected"},
            {"decision": "ephemeral", "reason": "private rationale"},
            {"decision": True},
            ["promote_candidate"],
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                client = RecordingClient(payload=payload)
                selector = LLMMemorySelector(
                    client=client,
                    model="memory-selector-test",
                    fallback=FixedAmbiguousSelector(),
                )

                result = selector.select(
                    query="ambiguous",
                    terminal_status="answered",
                    remembered_statement=None,
                )

                self.assertEqual(result.retention_class, "ephemeral")
                self.assertEqual(
                    result.selection_reason,
                    ("future_utility_ambiguous",),
                )
                self.assertEqual(result.decay_profile, "episode_fast")
                self.assertEqual(
                    result.selector_fallback_reason,
                    "invalid_model_output",
                )

        timeout = LLMMemorySelector(
            client=RecordingClient(
                error=TimeoutError("secret provider response")
            ),
            model="memory-selector-test",
            fallback=FixedAmbiguousSelector(),
        ).select(
            query="ambiguous",
            terminal_status="answered",
            remembered_statement=None,
        )
        self.assertEqual(timeout.retention_class, "ephemeral")
        self.assertEqual(timeout.selector_fallback_reason, "timeout")
        self.assertNotIn("secret", repr(timeout))

    def test_persisted_event_has_selector_metadata_and_prompt_omits_answer(
        self,
    ) -> None:
        client = RecordingClient(payload={"decision": "promote_candidate"})
        store = LocalSelectiveMemoryStore()
        service = SelectiveMemoryService(
            store,
            selector=LLMMemorySelector(
                client=client,
                model="memory-selector-test",
            ),
        )

        result = service.persist_turn(
            session_id="session_selector",
            query_id="query_selector",
            query="这个排障步骤下次可能还会复用",
            answer="SECRET_ASSISTANT_ANSWER",
            terminal_status="answered",
            remembered_statement=None,
            now_ms=1_000,
        )

        self.assertEqual(result.retention_class, "candidate")
        self.assertEqual(result.selector_name, "openai")
        saved = store.list_events(
            "session_selector",
            now_ms=1_000,
        )
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0].selector_name, "openai")
        self.assertEqual(saved[0].selector_model, "memory-selector-test")
        self.assertIsNone(saved[0].selector_fallback_reason)
        self.assertNotIn(
            "SECRET_ASSISTANT_ANSWER",
            client.responses.calls[0]["input"],
        )
        self.assertEqual(event_from_storage(saved[0].to_dict()), saved[0])

    def test_builder_is_optional_and_validates_configuration(self) -> None:
        client_factory = Mock(return_value=RecordingClient())
        rule = build_memory_selector(
            {
                "MEMORY_SELECTOR": "rule_based",
                "OPENAI_API_KEY": "unused",
                "OPENAI_MEMORY_SELECTOR_MODEL": "unused",
            },
            client_factory=client_factory,
        )
        self.assertIsInstance(rule, RuleBasedMemorySelector)
        client_factory.assert_not_called()

        unavailable = build_memory_selector(
            {"MEMORY_SELECTOR": "openai"},
            client_factory=client_factory,
        )
        result = unavailable.select(
            query="这个步骤下次可能还会复用",
            terminal_status="answered",
            remembered_statement=None,
        )
        self.assertEqual(result.selector_name, "rule_based_fallback")
        self.assertEqual(result.selector_fallback_reason, "not_configured")
        client_factory.assert_not_called()

        for values, message in (
            ({"MEMORY_SELECTOR": "maybe"}, "MEMORY_SELECTOR"),
            (
                {"MEMORY_SELECTOR_AMBIGUITY_MIN": "0.39"},
                "ambiguity band",
            ),
            (
                {
                    "MEMORY_SELECTOR_AMBIGUITY_MIN": "0.55",
                    "MEMORY_SELECTOR_AMBIGUITY_MAX": "0.50",
                },
                "ambiguity band",
            ),
            (
                {"OPENAI_MEMORY_SELECTOR_TIMEOUT_SECONDS": "0"},
                "TIMEOUT",
            ),
        ):
            configured = {"MEMORY_SELECTOR": "openai", **values}
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, message):
                    build_memory_selector(configured)

    def test_memory_event_schema_persists_selector_metadata(self) -> None:
        fields = {
            field["name"]: field
            for field in MEMORY_EVENTS_COLLECTION["fields"]
        }
        self.assertFalse(fields["selector_name"]["nullable"])
        self.assertTrue(fields["selector_model"]["nullable"])
        self.assertTrue(fields["selector_fallback_reason"]["nullable"])

    def test_workflow_trace_exposes_only_safe_selector_metadata(self) -> None:
        client = RecordingClient(
            payload={"decision": "promote_candidate"}
        )
        workflow = AgenticRAGWorkflow(
            selective_memory=SelectiveMemoryService(
                selector=LLMMemorySelector(
                    client=client,
                    model="memory-selector-test",
                )
            ),
            wall_clock_ms=lambda: 1_000,
        )

        response = workflow.run(
            "你好，这个步骤下次可能还会复用",
            session_id="session_selector",
            query_id="query_selector",
        )

        selective = response["trace"]["memory"]["selective"]
        self.assertEqual(selective["selector_name"], "openai")
        self.assertEqual(selective["model"], "memory-selector-test")
        self.assertIsNone(selective["fallback_reason"])
        serialized = json.dumps(selective, ensure_ascii=False)
        self.assertNotIn("user_query", serialized)
        self.assertNotIn("rule_decision", serialized)
        self.assertNotIn("instructions", serialized)

    def test_constructor_rejects_band_outside_rule_contract(self) -> None:
        for minimum, maximum in (
            (0.39, 0.60),
            (0.40, 0.61),
            (0.55, 0.50),
        ):
            with self.subTest(minimum=minimum, maximum=maximum):
                with self.assertRaisesRegex(ValueError, "ambiguity band"):
                    LLMMemorySelector(
                        client=RecordingClient(),
                        model="memory-selector-test",
                        fallback=FixedAmbiguousSelector(),
                        ambiguity_min=minimum,
                        ambiguity_max=maximum,
                    )


if __name__ == "__main__":
    unittest.main()
