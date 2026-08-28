from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any, cast

from agent_workshop_demo.eval_runner import (
    EvalScenarioPermissionChecker,
    EvaluationWorkflow,
)
from agent_workshop_demo.knowledge_tools import PermissionDecision
from agent_workshop_demo.langgraph_workflow import build_default_workflow
from agent_workshop_demo.retrieval import InMemoryHybridRetriever
from agent_workshop_demo.retrieval_tier import (
    LEXICAL_ONLY_CATALOG_VERSION,
    LexicalOnlyRetriever,
    RetrievalTier,
    RetrievalTierConfig,
    SparseSearchRetriever,
    parse_tier,
    runtime_config_from_mapping,
)
from agent_workshop_demo.retrieval_tier_eval import (
    QUALITY_METRICS,
    TIER_ARMS,
    default_tier_justification,
    evaluate_retrieval_tiers,
)
from agent_workshop_demo.sample_data import load_kb_chunks
from agent_workshop_demo.workflow import AgenticRAGWorkflow

QUESTIONS = Path(__file__).resolve().parents[1] / "eval/questions.json"
GOLDEN = Path(__file__).resolve().parents[1] / "eval/golden_answers.yaml"
REGISTRY = Path(__file__).resolve().parents[1] / "eval/metric_registry.json"


def _arm_factory(arm: str) -> Any:
    def factory(scenario: dict[str, str]) -> EvaluationWorkflow:
        return cast(
            EvaluationWorkflow,
            AgenticRAGWorkflow(
                permission_checker=EvalScenarioPermissionChecker(
                    allowed=scenario["permission"] == "allow"
                ),
                retrieval_tier=RetrievalTier(arm),
            ),
        )

    return factory


class RetrievalTierConfigTests(unittest.TestCase):
    def test_default_tier_is_hybrid_dense(self) -> None:
        config = runtime_config_from_mapping({})

        self.assertIs(config.tier, RetrievalTier.HYBRID_DENSE)
        self.assertEqual(config.tier_code, "T2")
        self.assertTrue(config.uses_dense_lane)
        self.assertTrue(config.uses_query_transformation)

    def test_lexical_only_disables_dense_lane_and_transformation(self) -> None:
        config = runtime_config_from_mapping({"RETRIEVAL_TIER": "lexical_only"})

        self.assertEqual(config.tier_code, "T0")
        self.assertFalse(config.uses_dense_lane)
        self.assertFalse(config.uses_query_transformation)
        self.assertEqual(
            config.to_dict(),
            {
                "tier": "lexical_only",
                "tier_code": "T0",
                "dense_lane": False,
                "query_transformation": False,
            },
        )

    def test_lexical_rewrite_keeps_transformation_without_dense(self) -> None:
        config = runtime_config_from_mapping({"RETRIEVAL_TIER": "lexical_rewrite"})

        self.assertEqual(config.tier_code, "T1")
        self.assertFalse(config.uses_dense_lane)
        self.assertTrue(config.uses_query_transformation)

    def test_unsupported_tier_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            runtime_config_from_mapping({"RETRIEVAL_TIER": "on_the_fly"})
        with self.assertRaises(ValueError):
            parse_tier("hot_cold")

    def test_struct_array_profile_cannot_run_under_a_lexical_tier(self) -> None:
        with self.assertRaises(ValueError):
            runtime_config_from_mapping(
                {
                    "RETRIEVAL_TIER": "lexical_only",
                    "STRUCT_ARRAY_RETRIEVAL": "struct_element",
                }
            )

    def test_disabled_struct_array_is_allowed_under_a_lexical_tier(self) -> None:
        config = runtime_config_from_mapping(
            {
                "RETRIEVAL_TIER": "lexical_rewrite",
                "STRUCT_ARRAY_RETRIEVAL": "disabled",
            }
        )

        self.assertIs(config.tier, RetrievalTier.LEXICAL_REWRITE)


class LexicalOnlyRetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.flat = InMemoryHybridRetriever(load_kb_chunks())
        self.retriever = LexicalOnlyRetriever(
            cast(SparseSearchRetriever, self.flat)
        )

    def test_search_returns_bm25_only_results(self) -> None:
        results = self.retriever.search(
            "S3 文档同步流程 chunking Milvus insertion",
            top_k=5,
            filters={"department": "engineering"},
            order_by=["updated_at desc", "priority desc"],
        )

        self.assertTrue(results)
        self.assertEqual([item.rank for item in results], list(range(1, len(results) + 1)))
        for item in results:
            self.assertEqual(item.dense_score, 0.0)
            self.assertEqual(item.retrieval_profile, "flat_bm25")
            self.assertEqual(item.retrieval_paths, ("flat_bm25",))
        scores = [item.hybrid_score for item in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_search_rejects_non_positive_top_k(self) -> None:
        with self.assertRaises(ValueError):
            self.retriever.search("Milvus", top_k=0)

    def test_dense_and_struct_lanes_are_not_exposed(self) -> None:
        self.assertFalse(hasattr(self.retriever, "search_profile"))
        self.assertFalse(hasattr(self.retriever, "search_image_vector"))

    def test_source_without_sparse_lane_is_rejected(self) -> None:
        class _NoSparse:
            supports_parallel_search = False

        with self.assertRaises(ValueError):
            LexicalOnlyRetriever(cast(SparseSearchRetriever, _NoSparse()))

    def test_scalar_and_exact_lookups_are_delegated(self) -> None:
        results = self.retriever.search("Milvus", top_k=5)
        aggregations = self.retriever.aggregations(results, ["department"])
        chunks = self.retriever.fetch_chunks_by_ids(
            chunk_ids=["doc_s3_sync_design_c003"],
        )
        siblings = self.retriever.fetch_document_chunks(
            doc_id="doc_s3_sync_design",
            doc_version=chunks[0].doc_version,
            limit=5,
        )

        self.assertTrue(aggregations["department"])
        self.assertEqual(chunks[0].chunk_id, "doc_s3_sync_design_c003")
        self.assertTrue(siblings)


class TieredWorkflowTests(unittest.TestCase):
    def test_hybrid_tier_remains_the_default(self) -> None:
        response = AgenticRAGWorkflow().run("RAG 架构里 Milvus 负责哪一层？")

        self.assertEqual(response["retrieval_tier"], "hybrid_dense")
        self.assertEqual(response["search_mode"], "hybrid")
        self.assertEqual(
            {item["retrieval_profile"] for item in response["milvus_recalled"]},
            {"flat_hybrid"},
        )

    def test_lexical_tiers_use_only_the_bm25_lane(self) -> None:
        for tier in ("lexical_only", "lexical_rewrite"):
            with self.subTest(tier=tier):
                response = AgenticRAGWorkflow(retrieval_tier=tier).run(
                    "我们 S3 文档同步流程是怎么设计的？"
                )

                self.assertEqual(response["retrieval_tier"], tier)
                self.assertEqual(response["search_mode"], "lexical")
                self.assertEqual(response["trace"]["milvus_search"]["tier"], tier)
                self.assertEqual(
                    {
                        item["retrieval_profile"]
                        for item in response["milvus_recalled"]
                    },
                    {"flat_bm25"},
                )
                for item in response["milvus_recalled"]:
                    self.assertEqual(item["dense_score"], 0.0)

    def test_lexical_only_disables_transformation_and_the_entity_catalog(
        self,
    ) -> None:
        workflow = AgenticRAGWorkflow(retrieval_tier="lexical_only")
        response = workflow.run("领取按钮现在表示什么？")

        self.assertEqual(
            workflow.entity_catalog.catalog_version,
            LEXICAL_ONLY_CATALOG_VERSION,
        )
        self.assertEqual(workflow.entity_catalog.entities, ())
        self.assertEqual(response["matched_entities"], [])
        self.assertEqual(response["query_transformation"]["strategy"], "identity")

    def test_lexical_rewrite_keeps_transformation_and_the_entity_catalog(
        self,
    ) -> None:
        workflow = AgenticRAGWorkflow(retrieval_tier="lexical_rewrite")

        self.assertNotEqual(
            workflow.entity_catalog.catalog_version,
            LEXICAL_ONLY_CATALOG_VERSION,
        )
        self.assertTrue(workflow.entity_catalog.entities)

    def test_unsupported_workflow_tier_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AgenticRAGWorkflow(retrieval_tier="pre_embedded_at_scale")

    def test_configured_environment_selects_the_tier(self) -> None:
        workflow = build_default_workflow(
            environ={"RETRIEVAL_TIER": "lexical_only"},
        )
        response = workflow.run("RAG 架构里 Milvus 负责哪一层？")

        self.assertEqual(response["retrieval_tier"], "lexical_only")
        self.assertEqual(response["search_mode"], "lexical")

    def test_permission_denial_is_unchanged_by_the_tier(self) -> None:
        class _DenyAll:
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
                )

        response = AgenticRAGWorkflow(
            retrieval_tier="lexical_only",
            permission_checker=_DenyAll(),
        ).run("请查看产品路线图里的下一季度计划。")

        self.assertEqual(response["terminal_status"], "permission_denied")
        self.assertEqual(response["milvus_recalled"], [])


class RetrievalTierEvalTests(unittest.TestCase):
    def test_three_arms_are_reported_independently(self) -> None:
        report = evaluate_retrieval_tiers(
            questions_path=QUESTIONS,
            golden_answers_path=GOLDEN,
            arm_workflow_factories={arm: _arm_factory(arm) for arm in TIER_ARMS},
            metric_registry_path=REGISTRY,
        )

        self.assertEqual(report["report_version"], "retrieval-tier-eval-v1")
        self.assertEqual(report["evaluation"]["scoring"], "per_arm_only")
        self.assertEqual([arm["arm"] for arm in report["arms"]], list(TIER_ARMS))
        for arm in report["arms"]:
            self.assertEqual(arm["status"], "complete")
            for metric in QUALITY_METRICS:
                self.assertIsInstance(arm["quality"][metric], float)
        lexical_only = report["arms"][0]
        self.assertEqual(lexical_only["operational"]["cost_per_request"], 0.0)

    def test_missing_arm_stays_in_the_denominator(self) -> None:
        report = evaluate_retrieval_tiers(
            questions_path=QUESTIONS,
            golden_answers_path=GOLDEN,
            arm_workflow_factories={
                "hybrid_dense": _arm_factory("hybrid_dense"),
            },
            metric_registry_path=REGISTRY,
        )

        statuses = {arm["arm"]: arm["status"] for arm in report["arms"]}
        self.assertEqual(statuses["lexical_only"], "evaluation_incomplete")
        self.assertEqual(statuses["lexical_rewrite"], "evaluation_incomplete")
        self.assertEqual(statuses["hybrid_dense"], "complete")
        self.assertEqual(
            report["default_tier_justification"]["status"],
            "evaluation_incomplete",
        )
        self.assertEqual(
            report["default_tier_justification"]["reasons"],
            ["arm_not_evaluated"],
        )

    def test_unknown_arm_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_retrieval_tiers(
                questions_path=QUESTIONS,
                golden_answers_path=GOLDEN,
                arm_workflow_factories={"on_the_fly": _arm_factory("hybrid_dense")},
                metric_registry_path=REGISTRY,
            )

    def test_default_tier_without_quality_gain_is_teaching_goal_only(self) -> None:
        report = evaluate_retrieval_tiers(
            questions_path=QUESTIONS,
            golden_answers_path=GOLDEN,
            arm_workflow_factories={arm: _arm_factory(arm) for arm in TIER_ARMS},
            metric_registry_path=REGISTRY,
            latency_budget_ms=60_000.0,
        )
        justification = report["default_tier_justification"]

        self.assertEqual(justification["status"], "teaching_goal_only")
        self.assertFalse(justification["quality_gain"])
        self.assertTrue(justification["within_latency_budget"])
        self.assertIn(
            "no_quality_gain_over_lexical_rewrite",
            justification["reasons"],
        )

    def test_unapproved_latency_budget_blocks_quality_evidence(self) -> None:
        arms: dict[str, dict[str, Any]] = {
            "lexical_only": {"status": "complete", "arm": "lexical_only"},
            "lexical_rewrite": {
                "arm": "lexical_rewrite",
                "status": "complete",
                "quality": {metric: 0.5 for metric in QUALITY_METRICS},
                "latency": {
                    "end_to_end_ms": {"complete": True, "p95": 10.0},
                },
            },
            "hybrid_dense": {
                "arm": "hybrid_dense",
                "status": "complete",
                "quality": {metric: 0.9 for metric in QUALITY_METRICS},
                "latency": {
                    "end_to_end_ms": {"complete": True, "p95": 25.0},
                },
            },
        }
        unapproved = default_tier_justification(arms, latency_budget_ms=None)
        approved = default_tier_justification(arms, latency_budget_ms=30.0)
        exceeded = default_tier_justification(arms, latency_budget_ms=20.0)

        self.assertEqual(unapproved["status"], "teaching_goal_only")
        self.assertIn("latency_budget_not_approved", unapproved["reasons"])
        self.assertEqual(approved["status"], "quality_evidence")
        self.assertTrue(approved["quality_gain"])
        self.assertEqual(exceeded["status"], "teaching_goal_only")
        self.assertIn("p95_latency_over_budget", exceeded["reasons"])


class RetrievalTierConfigDataclassTests(unittest.TestCase):
    def test_config_is_immutable(self) -> None:
        config = RetrievalTierConfig(RetrievalTier.HYBRID_DENSE)

        with self.assertRaises(Exception):
            config.tier = RetrievalTier.LEXICAL_ONLY  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
