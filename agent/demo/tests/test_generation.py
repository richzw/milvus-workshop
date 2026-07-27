from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

from agent_workshop_demo.generation import (
    DeterministicAnswerGenerator,
    FallbackAnswerGenerator,
    GenerationContext,
    GenerationRequest,
    OpenAIAnswerGenerator,
    build_answer_generator,
)


class RecordingResponses:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.output_text)


class RecordingOpenAIClient:
    def __init__(self, output_text: str) -> None:
        self.responses = RecordingResponses(output_text)


class FailingResponses:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def create(self, **kwargs: Any) -> SimpleNamespace:
        raise self.error


class FailingOpenAIClient:
    def __init__(self, error: Exception) -> None:
        self.responses = FailingResponses(error)


class GenerationTests(unittest.TestCase):
    @staticmethod
    def generation_request() -> GenerationRequest:
        return GenerationRequest(
            query_id="query_generation",
            user_query="同步流程如何工作？",
            resolved_entities=[
                {
                    "entity_id": "ui.go_button",
                    "entity": "GO按钮",
                    "comment": "表示跳转或领取按钮",
                }
            ],
            version_scope={"mode": "current"},
            contexts=[
                GenerationContext(
                    citation_id="C1",
                    chunk_id="chunk_1",
                    doc_id="doc_sync",
                    doc_version="v2",
                    title="同步设计",
                    page_no=None,
                    section="Pipeline",
                    text="扫描对象存储并提取元数据。",
                ),
                GenerationContext(
                    citation_id="C2",
                    chunk_id="chunk_2",
                    doc_id="doc_sync",
                    doc_version="v2",
                    title="索引设计",
                    page_no=3,
                    section=None,
                    text="切分并生成向量后写入 Milvus。",
                ),
            ],
            memory_context=["上一轮讨论的是 S3 文档同步。"],
        )

    def test_deterministic_generator_returns_grounded_citations(self) -> None:
        request = self.generation_request()

        result = DeterministicAnswerGenerator().generate(request)

        self.assertEqual(result.generator_name, "deterministic")
        self.assertIsNone(result.model)
        self.assertIsNone(result.fallback_reason)
        self.assertEqual(result.referenced_citation_ids, ["C1", "C2"])
        self.assertIn("扫描对象存储", result.text)
        self.assertIn("写入 Milvus", result.text)
        self.assertIn("[C1]", result.text)
        self.assertIn("[C2]", result.text)

    def test_openai_generator_synthesizes_valid_cited_answer(self) -> None:
        client = RecordingOpenAIClient(
            "系统先扫描对象，随后写入 Milvus。[C1][C2]"
        )
        generator = OpenAIAnswerGenerator(
            client=client,
            model="configured-model",
            timeout_seconds=12.5,
        )

        result = generator.generate(self.generation_request())

        self.assertEqual(result.generator_name, "openai")
        self.assertEqual(result.model, "configured-model")
        self.assertEqual(result.referenced_citation_ids, ["C1", "C2"])
        call = client.responses.calls[0]
        self.assertEqual(call["model"], "configured-model")
        self.assertEqual(call["timeout"], 12.5)
        self.assertIn("[C1]", call["input"])
        self.assertIn("扫描对象存储", call["input"])
        self.assertIn("<entity_info>", call["input"])
        self.assertIn("<memory_context>", call["input"])
        self.assertIn("上一轮讨论的是 S3", call["input"])
        self.assertIn("GO按钮", call["input"])
        self.assertIn("doc_version=v2", call["input"])
        self.assertIn("untrusted", call["instructions"])

    def test_comparison_fallback_labels_each_document_version(self) -> None:
        request = self.generation_request()
        request = GenerationRequest(
            query_id=request.query_id,
            user_query=request.user_query,
            resolved_entities=request.resolved_entities,
            version_scope={"mode": "comparison"},
            contexts=[
                request.contexts[0],
                GenerationContext(
                    citation_id="C2",
                    chunk_id="chunk_2",
                    doc_id="doc_sync",
                    doc_version="v1",
                    title="索引设计",
                    page_no=3,
                    section=None,
                    text="旧版先生成向量后写入 Milvus。",
                ),
            ],
        )

        result = DeterministicAnswerGenerator().generate(request)

        self.assertIn("[版本 v2]", result.text)
        self.assertIn("[版本 v1]", result.text)

    def test_invalid_model_citation_uses_deterministic_fallback(self) -> None:
        primary = OpenAIAnswerGenerator(
            client=RecordingOpenAIClient("没有依据的回答。[C9]"),
            model="configured-model",
        )
        generator = FallbackAnswerGenerator(
            primary=primary,
            fallback=DeterministicAnswerGenerator(),
        )

        result = generator.generate(self.generation_request())

        self.assertEqual(result.generator_name, "deterministic")
        self.assertEqual(result.fallback_reason, "invalid_model_output")
        self.assertEqual(result.referenced_citation_ids, ["C1", "C2"])
        self.assertNotIn("[C9]", result.text)

    def test_provider_timeout_uses_sanitized_fallback_reason(self) -> None:
        primary = OpenAIAnswerGenerator(
            client=FailingOpenAIClient(TimeoutError("secret provider body")),
            model="configured-model",
        )
        generator = FallbackAnswerGenerator(
            primary=primary,
            fallback=DeterministicAnswerGenerator(),
        )

        result = generator.generate(self.generation_request())

        self.assertEqual(result.generator_name, "deterministic")
        self.assertEqual(result.fallback_reason, "timeout")
        self.assertNotIn("secret provider body", repr(result))

    def test_auto_mode_without_credentials_uses_offline_fallback(self) -> None:
        generator = build_answer_generator({"ANSWER_GENERATOR": "auto"})

        result = generator.generate(self.generation_request())

        self.assertEqual(result.generator_name, "deterministic")
        self.assertEqual(result.fallback_reason, "not_configured")

    def test_openai_mode_uses_explicit_environment_configuration(self) -> None:
        client = RecordingOpenAIClient("综合后的答案。[C1]")
        received_keys: list[str] = []

        def client_factory(api_key: str) -> RecordingOpenAIClient:
            received_keys.append(api_key)
            return client

        generator = build_answer_generator(
            {
                "ANSWER_GENERATOR": "openai",
                "OPENAI_API_KEY": "test-key",
                "OPENAI_MODEL": "configured-model",
                "OPENAI_TIMEOUT_SECONDS": "8.5",
            },
            client_factory=client_factory,
        )

        result = generator.generate(self.generation_request())

        self.assertEqual(received_keys, ["test-key"])
        self.assertEqual(result.generator_name, "openai")
        self.assertEqual(result.model, "configured-model")
        self.assertEqual(client.responses.calls[0]["timeout"], 8.5)

    def test_invalid_generation_configuration_fails_clearly(self) -> None:
        invalid_environments = [
            {"ANSWER_GENERATOR": "unknown"},
            {"ANSWER_GENERATOR": "openai"},
            {
                "ANSWER_GENERATOR": "openai",
                "OPENAI_API_KEY": "key",
                "OPENAI_MODEL": "model",
                "OPENAI_TIMEOUT_SECONDS": "0",
            },
        ]

        for environ in invalid_environments:
            with self.subTest(environ=environ):
                with self.assertRaises(ValueError):
                    build_answer_generator(environ)


if __name__ == "__main__":
    unittest.main()
