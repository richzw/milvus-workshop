from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from typing import Any

from agent_workshop_demo.query_transform import (
    OpenAIQueryTransformer,
    QueryTransformRequest,
    RuleBasedQueryTransformer,
    build_query_transformer,
)


class RecordingResponses:
    def __init__(self, output: object) -> None:
        self.output = output
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=json.dumps(self.output, ensure_ascii=False))


class QueryTransformTests(unittest.TestCase):
    def test_rule_policy_selects_one_bounded_strategy(self) -> None:
        transformer = RuleBasedQueryTransformer()
        cases = [
            ("Force Merge 是什么？", "identity", ["primary"]),
            ("它怎么弄？", "rewrite", ["primary"]),
            ("Milvus Force Merge 为什么这样工作？", "step_back", ["background", "primary"]),
            ("比较 Milvus v2.6 和 v3.0", "decompose", ["aspect", "aspect"]),
        ]
        for query, strategy, roles in cases:
            with self.subTest(query=query):
                result = transformer.transform(
                    QueryTransformRequest(
                        query,
                        resolved_entities=("Milvus",) if "它" in query else (),
                    )
                )
                self.assertEqual(result.strategy, strategy)
                self.assertEqual([item.query_role for item in result.items], roles)
                self.assertLessEqual(len(result.items), 3)

    def test_step_back_and_decompose_preserve_versions_and_named_terms(self) -> None:
        transformer = RuleBasedQueryTransformer()
        for query in (
            "Milvus v3.0 Force Merge 为什么不能在线执行？",
            "比较 Milvus v2.6 和 v3.0 的 Force Merge",
        ):
            result = transformer.transform(QueryTransformRequest(query))
            for item in result.items:
                self.assertIn("Milvus", item.query)
                self.assertIn("Force Merge", item.query)
                if "v3.0" in query:
                    self.assertIn("v3.0", item.query)
                if "v2.6" in query:
                    self.assertIn("v2.6", item.query)

    def test_openai_transformer_uses_one_strict_request(self) -> None:
        responses = RecordingResponses(
            {
                "strategy": "step_back",
                "items": [
                    {
                        "query": "Milvus v3.0 Force Merge 如何工作？；背景架构",
                        "query_role": "background",
                        "depends_on": [],
                    },
                    {
                        "query": "Milvus v3.0 Force Merge 如何工作？",
                        "query_role": "primary",
                        "depends_on": [],
                    },
                ],
            }
        )
        transformer = OpenAIQueryTransformer(
            client=SimpleNamespace(responses=responses),
            model="test-model",
        )

        result = transformer.transform(
            QueryTransformRequest("Milvus v3.0 Force Merge 如何工作？")
        )

        self.assertEqual(result.strategy, "step_back")
        self.assertEqual(len(responses.calls), 1)
        self.assertEqual(
            responses.calls[0]["text"]["format"]["schema"]["additionalProperties"],
            False,
        )

    def test_invalid_provider_output_falls_back_without_losing_terms(self) -> None:
        responses = RecordingResponses(
            {
                "strategy": "rewrite",
                "items": [
                    {
                        "query": "合并怎么工作？",
                        "query_role": "primary",
                        "depends_on": [],
                    }
                ],
            }
        )
        transformer = build_query_transformer(
            {
                "QUERY_TRANSFORMER": "openai",
                "OPENAI_API_KEY": "test-key",
                "OPENAI_MODEL": "test-model",
            },
            client_factory=lambda _key: SimpleNamespace(responses=responses),
        )

        result = transformer.transform(
            QueryTransformRequest("Milvus v3.0 Force Merge 如何工作？")
        )

        self.assertEqual(result.fallback_reason, "invalid_model_output")
        self.assertTrue(all("Milvus" in item.query for item in result.items))
        self.assertEqual(len(responses.calls), 1)

    def test_provider_cannot_drop_unregistered_product_or_feature_terms(self) -> None:
        responses = RecordingResponses(
            {
                "strategy": "rewrite",
                "items": [
                    {
                        "query": "v3.0 如何工作？",
                        "query_role": "primary",
                        "depends_on": [],
                    }
                ],
            }
        )
        transformer = build_query_transformer(
            {
                "QUERY_TRANSFORMER": "openai",
                "OPENAI_API_KEY": "test-key",
                "OPENAI_MODEL": "test-model",
            },
            client_factory=lambda _key: SimpleNamespace(responses=responses),
        )

        result = transformer.transform(
            QueryTransformRequest("Milvus v3.0 Force Merge 如何工作？")
        )

        self.assertEqual(result.fallback_reason, "invalid_model_output")
        self.assertTrue(
            all(
                "Milvus v3.0 Force Merge 如何工作？" in item.query
                for item in result.items
            )
        )

    def test_default_builder_is_offline_and_explicit_mode_is_strict(self) -> None:
        self.assertEqual(build_query_transformer({}).name, "rule_based")
        with self.assertRaises(ValueError):
            build_query_transformer({"QUERY_TRANSFORMER": "openai"})


if __name__ == "__main__":
    unittest.main()
