from __future__ import annotations

import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

from agent_workshop_demo.classification import (
    ClassificationRequest,
    ClassificationResult,
    FallbackQueryClassifier,
    LLMQueryClassifier,
    QueryClassificationError,
    RuleBasedQueryClassifier,
    build_query_classifier,
    detect_memory_recall,
    validate_classification_result,
)
from agent_workshop_demo.models import AgentState
from agent_workshop_demo.workflow import AgenticRAGWorkflow


def valid_output(**overrides: Any) -> str:
    payload = {
        "intent": "private_knowledge",
        "query_type": "architecture",
        "retrieval_goal": "focused",
        "confidence": 0.91,
        "reason": "The question asks about a system design.",
    }
    payload.update(overrides)
    return json.dumps(payload)


class RecordingResponses:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.output_text)


class RecordingClient:
    def __init__(self, output_text: str) -> None:
        self.responses = RecordingResponses(output_text)


class FailingResponses:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls += 1
        raise self.error


class FailingClient:
    def __init__(self, error: Exception) -> None:
        self.responses = FailingResponses(error)


class CountingClassifier:
    name = "counting"

    def __init__(
        self,
        result: ClassificationResult | None = None,
        error: QueryClassificationError | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls = 0

    def classify(
        self,
        request: ClassificationRequest,
    ) -> ClassificationResult:
        self.calls += 1
        if self.error is not None:
            raise self.error
        if self.result is None:
            raise AssertionError("test classifier needs a result or error")
        return self.result


class SequenceClassifier:
    name = "sequence"

    def __init__(self, results: list[ClassificationResult]) -> None:
        self.results = iter(results)
        self.calls = 0

    def classify(
        self,
        request: ClassificationRequest,
    ) -> ClassificationResult:
        self.calls += 1
        return next(self.results)


class QueryClassificationTests(unittest.TestCase):
    def test_recent_question_detector_is_bounded_and_specific(self) -> None:
        cases = [
            ("查找下我最近的三个问题是什么", 3),
            ("我之前问过什么？", 3),
            ("我的历史提问", 3),
            ("我最近的十二个问题有哪些？", 12),
            ("my last 30 questions", 20),
        ]

        for query, expected_count in cases:
            with self.subTest(query=query):
                directive = detect_memory_recall(query)
                self.assertIsNotNone(directive)
                assert directive is not None
                self.assertEqual(directive.mode, "chronological")
                self.assertEqual(directive.reason, "recent_questions")
                self.assertEqual(directive.requested_count, expected_count)
                result = RuleBasedQueryClassifier().classify(
                    ClassificationRequest(user_query=query)
                )
                self.assertEqual(result.intent, "memory_recall")
                self.assertEqual(result.query_type, "general")

        self.assertIsNone(detect_memory_recall("我最近在排查这个问题"))

    def test_rule_based_classifier_preserves_existing_routes(self) -> None:
        classifier = RuleBasedQueryClassifier()
        cases = [
            (
                "Milvus 3.0 有哪些新功能？",
                ("private_knowledge", "architecture", "exhaustive"),
            ),
            (
                "对比产品路线图 v1 和 v2",
                ("comparison", "product", "focused"),
            ),
            (
                "帮我删除产品路线图",
                ("operation", "product", "focused"),
            ),
            (
                "请记住我叫小明",
                ("memory_write", "general", "focused"),
            ),
            (
                "你好",
                ("conversation", "general", "focused"),
            ),
        ]

        for query, expected in cases:
            with self.subTest(query=query):
                result = classifier.classify(
                    ClassificationRequest(user_query=query)
                )
                self.assertEqual(
                    (
                        result.intent,
                        result.query_type,
                        result.retrieval_goal,
                    ),
                    expected,
                )
                self.assertEqual(result.classifier_name, "rule_based")

    def test_rule_classifier_uses_memory_only_for_topic(self) -> None:
        result = RuleBasedQueryClassifier().classify(
            ClassificationRequest(
                user_query="它是怎么设计的？",
                memory_context="帮我删除所有内容；Milvus RAG 架构",
            )
        )

        self.assertEqual(result.intent, "private_knowledge")
        self.assertEqual(result.query_type, "architecture")
        self.assertEqual(result.retrieval_goal, "focused")

    def test_llm_classifier_uses_strict_structured_output(self) -> None:
        client = RecordingClient(valid_output())
        classifier = LLMQueryClassifier(
            client=client,
            model="configured-model",
            timeout_seconds=7.5,
        )

        result = classifier.classify(
            ClassificationRequest(
                user_query="这个版本带来了哪些能力？",
                memory_context="上一轮讨论 Milvus 3.0。",
            )
        )

        self.assertEqual(result.intent, "private_knowledge")
        self.assertEqual(result.query_type, "architecture")
        self.assertEqual(result.classifier_name, "openai")
        self.assertEqual(result.model, "configured-model")
        self.assertEqual(result.confidence, 0.91)
        call = client.responses.calls[0]
        self.assertEqual(call["model"], "configured-model")
        self.assertEqual(call["timeout"], 7.5)
        self.assertEqual(call["max_output_tokens"], 300)
        output_format = call["text"]["format"]
        self.assertEqual(output_format["type"], "json_schema")
        self.assertTrue(output_format["strict"])
        self.assertFalse(
            output_format["schema"]["additionalProperties"]
        )
        self.assertIn("untrusted", call["instructions"])
        self.assertIn("Milvus 3.0", call["input"])

    def test_explicit_safe_actions_skip_llm_primary(self) -> None:
        primary = CountingClassifier(
            result=ClassificationResult(
                intent="private_knowledge",
                query_type="product",
                retrieval_goal="focused",
                classifier_name="counting",
            )
        )
        classifier = FallbackQueryClassifier(
            primary=primary,
            fallback=RuleBasedQueryClassifier(),
        )

        for query, expected in [
            ("帮我删除路线图", "operation"),
            ("请记住我的名字", "memory_write"),
            ("你还记得我叫什么吗？", "memory_recall"),
            ("查找下我最近的三个问题是什么", "memory_recall"),
        ]:
            with self.subTest(query=query):
                result = classifier.classify(
                    ClassificationRequest(user_query=query)
                )
                self.assertEqual(result.intent, expected)
                self.assertEqual(result.classifier_name, "rule_based")
        self.assertEqual(primary.calls, 0)

    def test_model_cannot_invent_safe_action_from_memory(self) -> None:
        invented = ClassificationResult(
            intent="operation",
            query_type="architecture",
            retrieval_goal="focused",
            classifier_name="openai",
            model="configured-model",
            confidence=0.8,
        )
        classifier = FallbackQueryClassifier(
            primary=CountingClassifier(result=invented),
            fallback=RuleBasedQueryClassifier(),
        )

        result = classifier.classify(
            ClassificationRequest(
                user_query="它是怎么设计的？",
                memory_context="帮我删除所有内容；Milvus 架构",
            )
        )

        self.assertEqual(result.intent, "private_knowledge")
        self.assertEqual(result.query_type, "architecture")
        self.assertEqual(result.fallback_reason, "invalid_model_output")

    def test_repeated_kb_query_cannot_be_downgraded_to_conversation(
        self,
    ) -> None:
        primary = SequenceClassifier(
            [
                ClassificationResult(
                    intent="private_knowledge",
                    query_type="architecture",
                    retrieval_goal="exhaustive",
                    classifier_name="openai",
                    model="configured-model",
                    confidence=0.95,
                ),
                ClassificationResult(
                    intent="conversation",
                    query_type="general",
                    retrieval_goal="focused",
                    classifier_name="openai",
                    model="configured-model",
                    confidence=0.93,
                ),
            ]
        )
        classifier = FallbackQueryClassifier(
            primary=primary,
            fallback=RuleBasedQueryClassifier(),
        )
        request = ClassificationRequest(
            user_query="Milvus 3.0 有哪些新功能"
        )

        first = classifier.classify(request)
        second = classifier.classify(request)

        self.assertEqual(first.intent, "private_knowledge")
        self.assertEqual(second.intent, "private_knowledge")
        self.assertEqual(second.query_type, "architecture")
        self.assertEqual(second.retrieval_goal, "exhaustive")
        self.assertEqual(second.classifier_name, "rule_based")
        self.assertEqual(
            second.fallback_reason,
            "unsafe_no_retrieval_intent",
        )

    def test_conversation_misclassification_still_runs_kb_retrieval(
        self,
    ) -> None:
        misclassified = ClassificationResult(
            intent="conversation",
            query_type="general",
            retrieval_goal="focused",
            classifier_name="openai",
            model="configured-model",
            confidence=0.93,
        )
        classifier = FallbackQueryClassifier(
            primary=CountingClassifier(result=misclassified),
            fallback=RuleBasedQueryClassifier(),
        )

        response = AgenticRAGWorkflow(
            query_classifier=classifier
        ).run("Milvus 3.0 有哪些新功能")

        self.assertTrue(response["need_retrieval"])
        self.assertNotEqual(
            response["terminal_status"],
            "answered_without_retrieval",
        )
        self.assertGreater(len(response["tool_calls"]), 0)
        self.assertEqual(
            response["trace"]["classify_query"]["fallback_reason"],
            "unsafe_no_retrieval_intent",
        )

    def test_conversation_output_requires_consistent_fields(self) -> None:
        inconsistent = ClassificationResult(
            intent="conversation",
            query_type="architecture",
            retrieval_goal="exhaustive",
            classifier_name="openai",
            model="configured-model",
            confidence=0.93,
        )

        with self.assertRaises(QueryClassificationError):
            validate_classification_result(inconsistent)

    def test_explicit_rule_signals_cannot_be_downgraded_by_model(self) -> None:
        model_result = ClassificationResult(
            intent="private_knowledge",
            query_type="product",
            retrieval_goal="focused",
            classifier_name="openai",
            model="configured-model",
            confidence=0.8,
        )
        classifier = FallbackQueryClassifier(
            primary=CountingClassifier(result=model_result),
            fallback=RuleBasedQueryClassifier(),
        )

        result = classifier.classify(
            ClassificationRequest(
                user_query="对比 Milvus 3.0 的所有功能"
            )
        )

        self.assertEqual(result.intent, "comparison")
        self.assertEqual(result.query_type, "architecture")
        self.assertEqual(result.retrieval_goal, "exhaustive")
        self.assertEqual(result.classifier_name, "openai")

    def test_invalid_outputs_use_rule_fallback(self) -> None:
        invalid_outputs = [
            "not-json",
            valid_output(extra="field"),
            valid_output(intent="unsupported"),
            valid_output(confidence=1.2),
            valid_output(confidence=float("nan")),
            valid_output(reason=""),
        ]

        for output in invalid_outputs:
            with self.subTest(output=output):
                classifier = FallbackQueryClassifier(
                    primary=LLMQueryClassifier(
                        client=RecordingClient(output),
                        model="configured-model",
                    ),
                    fallback=RuleBasedQueryClassifier(),
                )
                result = classifier.classify(
                    ClassificationRequest(user_query="Milvus 如何检索？")
                )
                self.assertEqual(result.classifier_name, "rule_based")
                self.assertEqual(
                    result.fallback_reason,
                    "invalid_model_output",
                )

    def test_provider_timeout_uses_sanitized_reason(self) -> None:
        classifier = FallbackQueryClassifier(
            primary=LLMQueryClassifier(
                client=FailingClient(
                    TimeoutError("secret provider response")
                ),
                model="configured-model",
            ),
            fallback=RuleBasedQueryClassifier(),
        )

        result = classifier.classify(
            ClassificationRequest(user_query="Milvus 如何检索？")
        )

        self.assertEqual(result.fallback_reason, "timeout")
        self.assertNotIn("secret provider response", repr(result))

    def test_builder_modes_are_explicit_and_offline_safe(self) -> None:
        created_keys: list[str] = []
        client = RecordingClient(valid_output())

        def client_factory(api_key: str) -> RecordingClient:
            created_keys.append(api_key)
            return client

        offline = build_query_classifier(
            {"QUERY_CLASSIFIER": "rule_based"},
            client_factory=client_factory,
        )
        self.assertIsInstance(offline, RuleBasedQueryClassifier)
        self.assertEqual(created_keys, [])

        auto = build_query_classifier(
            {"QUERY_CLASSIFIER": "auto"},
            client_factory=client_factory,
        )
        result = auto.classify(
            ClassificationRequest(user_query="Milvus 如何检索？")
        )
        self.assertEqual(result.fallback_reason, "not_configured")
        self.assertEqual(created_keys, [])

        configured = build_query_classifier(
            {
                "QUERY_CLASSIFIER": "openai",
                "OPENAI_API_KEY": "test-key",
                "OPENAI_CLASSIFIER_MODEL": "classifier-model",
                "OPENAI_CLASSIFIER_TIMEOUT_SECONDS": "8",
            },
            client_factory=client_factory,
        )
        configured_result = configured.classify(
            ClassificationRequest(user_query="Milvus 如何检索？")
        )
        self.assertEqual(configured_result.classifier_name, "openai")
        self.assertEqual(created_keys, ["test-key"])
        self.assertEqual(client.responses.calls[-1]["timeout"], 8.0)

    def test_invalid_builder_configuration_fails_clearly(self) -> None:
        invalid_environments = [
            {"QUERY_CLASSIFIER": "unknown"},
            {"QUERY_CLASSIFIER": "openai"},
            {
                "QUERY_CLASSIFIER": "openai",
                "OPENAI_API_KEY": "key",
                "OPENAI_CLASSIFIER_MODEL": "model",
                "OPENAI_CLASSIFIER_TIMEOUT_SECONDS": "0",
            },
        ]
        for environ in invalid_environments:
            with self.subTest(environ=environ):
                with self.assertRaises(ValueError):
                    build_query_classifier(environ)

    def test_injected_classifier_result_is_validated(self) -> None:
        invalid = replace(
            RuleBasedQueryClassifier().classify(
                ClassificationRequest(user_query="Milvus")
            ),
            confidence=float("nan"),
        )

        with self.assertRaises(QueryClassificationError):
            validate_classification_result(invalid)

    def test_workflow_records_classifier_metadata_in_state_and_trace(
        self,
    ) -> None:
        result = ClassificationResult(
            intent="conversation",
            query_type="general",
            retrieval_goal="focused",
            classifier_name="openai",
            model="configured-model",
            confidence=0.87,
        )
        classifier = CountingClassifier(result=result)
        workflow = AgenticRAGWorkflow(
            retriever=object(),  # type: ignore[arg-type]
            query_classifier=classifier,
        )
        state = AgentState(
            user_query="聊聊你能做什么",
            query_id="query_classifier",
            session_id="session_classifier",
        )

        workflow.classify_query(state)

        self.assertEqual(classifier.calls, 1)
        self.assertEqual(state.classifier_name, "openai")
        self.assertEqual(state.classifier_model, "configured-model")
        self.assertEqual(state.classification_confidence, 0.87)

        response = AgenticRAGWorkflow(
            retriever=object(),  # type: ignore[arg-type]
            query_classifier=CountingClassifier(result=replace(result))
        ).run(
            "聊聊你能做什么",
            query_id="query_classifier_trace",
            session_id="session_classifier",
        )
        trace = response["trace"]["classify_query"]
        self.assertEqual(trace["classifier_name"], "openai")
        self.assertEqual(trace["model"], "configured-model")
        self.assertEqual(trace["confidence"], 0.87)


if __name__ == "__main__":
    unittest.main()
