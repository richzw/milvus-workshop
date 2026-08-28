from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any

from agent_workshop_demo.knowledge_tools import PermissionDecision
from agent_workshop_demo.langgraph_workflow import build_default_workflow
from agent_workshop_demo.models import SearchResult
from agent_workshop_demo.retrieval import InMemoryHybridRetriever
from agent_workshop_demo.sample_data import load_kb_chunks
from agent_workshop_demo.workflow import AgenticRAGWorkflow


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
            reason="The demo principal is not allowed.",
            checker_name="deny-test",
        )


class SearchMustNotRunRetriever(InMemoryHybridRetriever):
    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
        order_by: list[str] | None = None,
        order_mode: Any = "relevance",
    ) -> list[SearchResult]:
        raise AssertionError("search must not run before permission")


class VersionCrowdingRetriever(InMemoryHybridRetriever):
    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
        order_by: list[str] | None = None,
        order_mode: Any = "relevance",
    ) -> list[SearchResult]:
        del query, top_k, order_by, order_mode
        version = str((filters or {}).get("doc_version"))
        template = next(
            chunk
            for chunk in load_kb_chunks()
            if chunk.doc_id == "doc_go_button_guide"
            and chunk.doc_version == version
        )
        count = 20 if version == "v1" else 2
        base_score = 0.95 if version == "v1" else 0.5
        return [
            SearchResult(
                chunk=replace(
                    template,
                    chunk_id=f"crowding_{version}_{index:03d}",
                ),
                rank=index,
                dense_score=base_score,
                keyword_score=base_score,
                recency_score=1.0,
                priority_score=1.0,
                hybrid_score=base_score - index / 1000,
            )
            for index in range(1, count + 1)
        ]


class AgenticToolTests(unittest.TestCase):
    def test_single_domain_question_selects_code_tool(self) -> None:
        response = AgenticRAGWorkflow().run(
            "我们 S3 文档同步流程是怎么设计的？"
        )

        self.assertEqual(response["intent"], "private_knowledge")
        self.assertEqual(response["selected_tools"], ["search_code_docs"])
        self.assertTrue(response["permission_decision"]["allowed"])
        self.assertEqual(len(response["tool_calls"]), 1)
        self.assertEqual(
            response["tool_calls"][0]["filters"]["department"],
            ["engineering"],
        )
        self.assertTrue(response["answer_validation"]["valid"])

    def test_comparison_runs_parallel_meeting_and_product_tools(self) -> None:
        response = AgenticRAGWorkflow().run(
            "本季度客户最关心的问题有没有被产品路线图覆盖？"
        )

        self.assertEqual(response["intent"], "comparison")
        self.assertEqual(
            response["selected_tools"],
            ["search_meeting_notes", "search_product_docs"],
        )
        self.assertEqual(
            [call["tool"] for call in response["tool_calls"]],
            ["search_meeting_notes", "search_product_docs"],
        )
        self.assertEqual(response["retry_count"], 0)
        self.assertTrue(
            all(not item["depends_on"] for item in response["query_plan"])
        )
        self.assertEqual(response["terminal_status"], "answered")
        self.assertEqual(response["evidence_grade"]["missing_aspects"], [])
        self.assertTrue(response["answer_validation"]["valid"])
        cited = {item["chunk_id"] for item in response["citations"]}
        self.assertIn("doc_customer_meeting_notes_c001", cited)
        self.assertIn("doc_product_roadmap_c001", cited)

    def test_default_langgraph_path_runs_the_same_multi_hop_plan(self) -> None:
        response = build_default_workflow().run(
            "本季度客户最关心的问题有没有被产品路线图覆盖？"
        )

        self.assertEqual(
            [call["tool"] for call in response["tool_calls"]],
            ["search_meeting_notes", "search_product_docs"],
        )
        self.assertEqual(response["terminal_status"], "answered")
        self.assertTrue(response["trace"]["answer_validation"]["valid"])

    def test_permission_denial_stops_before_retrieval(self) -> None:
        workflow = AgenticRAGWorkflow(
            retriever=SearchMustNotRunRetriever(load_kb_chunks()),
            permission_checker=DenyPermissionChecker(),
        )

        response = workflow.run("请查看内部产品路线图")

        self.assertEqual(response["terminal_status"], "permission_denied")
        self.assertEqual(response["tool_calls"], [])
        self.assertEqual(response["milvus_recalled"], [])
        self.assertTrue(response["answer_validation"]["valid"])

    def test_operation_request_is_refused_without_tools(self) -> None:
        response = AgenticRAGWorkflow().run("帮我删除产品路线图")

        self.assertEqual(response["intent"], "operation")
        self.assertEqual(
            response["terminal_status"],
            "refused_unsupported_operation",
        )
        self.assertEqual(response["selected_tools"], [])
        self.assertEqual(response["tool_calls"], [])

    def test_entity_alias_rewrite_uses_current_document_version(self) -> None:
        response = AgenticRAGWorkflow().run(
            "领取按钮现在表示什么？"
        )

        self.assertEqual(
            response["matched_entities"][0]["entity_id"],
            "ui.go_button",
        )
        self.assertEqual(response["version_scope"]["mode"], "current")
        self.assertEqual(
            response["tool_calls"][0]["filters"]["is_current"],
            True,
        )
        self.assertNotIn("doc_version", response["tool_calls"][0]["filters"])
        self.assertEqual(
            {item["doc_version"] for item in response["citations"]},
            {"v2"},
        )
        self.assertTrue(
            all(
                "GO按钮" in query
                for query in response["rewritten_queries"]
            )
        )

    def test_exact_version_does_not_fall_back_to_current(self) -> None:
        response = AgenticRAGWorkflow().run(
            "跳转按钮在 v1 表示什么？"
        )

        self.assertEqual(response["version_scope"]["mode"], "exact")
        self.assertEqual(
            response["tool_calls"][0]["filters"]["doc_version"],
            "v1",
        )
        self.assertNotIn("is_current", response["tool_calls"][0]["filters"])
        self.assertEqual(
            {item["doc_version"] for item in response["citations"]},
            {"v1"},
        )

        missing = AgenticRAGWorkflow().run(
            "跳转按钮在 v9 表示什么？"
        )
        self.assertEqual(missing["terminal_status"], "abstained")
        self.assertEqual(missing["citations"], [])
        self.assertTrue(
            all(
                call["filters"]["doc_version"] == "v9"
                for call in missing["tool_calls"]
            )
        )

    def test_product_associated_bare_version_is_exact(self) -> None:
        item = AgenticRAGWorkflow().create_state(
            "介绍下 Milvus 3.0 Force Merge"
        )
        item.intent = "private_knowledge"
        item.query_type = "architecture"

        AgenticRAGWorkflow().resolve_terminology(item)

        self.assertEqual(item.version_scope["mode"], "exact")
        self.assertEqual(item.version_scope["doc_versions"], ["v3.0"])

        unqualified = AgenticRAGWorkflow().create_state(
            "指标达到 3.0 是什么意思？"
        )
        unqualified.intent = "private_knowledge"
        unqualified.query_type = "unknown"
        AgenticRAGWorkflow().resolve_terminology(unqualified)
        self.assertEqual(unqualified.version_scope["mode"], "current")

    def test_explicit_version_comparison_keeps_evidence_partitioned(self) -> None:
        response = AgenticRAGWorkflow().run(
            "比较 GO按钮 v1 和 v2 的变化"
        )

        self.assertEqual(response["version_scope"]["mode"], "comparison")
        self.assertEqual(
            {
                call["filters"]["doc_version"]
                for call in response["tool_calls"]
            },
            {"v1", "v2"},
        )
        self.assertEqual(
            {item["doc_version"] for item in response["citations"]},
            {"v1", "v2"},
        )
        self.assertIn("版本 v1", response["answer"])
        self.assertIn("版本 v2", response["answer"])
        self.assertTrue(response["answer_validation"]["valid"])

    def test_version_comparison_requires_evidence_for_both_sides(self) -> None:
        response = AgenticRAGWorkflow().run(
            "比较 GO按钮 v1 和 v9 的变化"
        )

        self.assertEqual(response["terminal_status"], "abstained")
        self.assertEqual(response["citations"], [])
        self.assertIn(
            {"mode": "exact", "doc_version": "v9"},
            response["evidence_grade"]["missing_version_scopes"],
        )

    def test_version_comparison_reserves_recall_and_rerank_per_side(
        self,
    ) -> None:
        response = AgenticRAGWorkflow(
            retriever=VersionCrowdingRetriever(load_kb_chunks())
        ).run("比较 GO按钮 v1 和 v2 的变化")

        self.assertEqual(response["terminal_status"], "answered")
        self.assertEqual(
            {item["doc_version"] for item in response["milvus_recalled"]},
            {"v1", "v2"},
        )
        self.assertEqual(
            {item["doc_version"] for item in response["citations"]},
            {"v1", "v2"},
        )

    def test_two_versions_without_comparison_intent_require_clarification(
        self,
    ) -> None:
        question = "GO按钮在 v1 v2 表示什么？"
        responses = [
            AgenticRAGWorkflow().run(question),
            build_default_workflow().run(question),
        ]

        for response in responses:
            self.assertEqual(
                response["terminal_status"],
                "clarification_required",
            )
            self.assertEqual(response["tool_calls"], [])
            self.assertEqual(
                response["ambiguous_entities"][0]["status"],
                "ambiguous_version_scope",
            )

    def test_ambiguous_entity_stops_before_retrieval_in_both_runtimes(
        self,
    ) -> None:
        local = AgenticRAGWorkflow().run("段位是什么意思？")
        graph = build_default_workflow().run("段位是什么意思？")

        for response in [local, graph]:
            self.assertEqual(
                response["terminal_status"],
                "clarification_required",
            )
            self.assertEqual(response["tool_calls"], [])
            self.assertEqual(response["milvus_recalled"], [])
            self.assertEqual(
                response["ambiguous_entities"][0]["status"],
                "ambiguous",
            )
            self.assertTrue(response["answer_validation"]["valid"])

    def test_streamlit_has_no_manual_metadata_controls(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "agent_workshop_demo"
            / "streamlit_app.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("st.multiselect", source)
        self.assertNotIn("Search Controls", source)
        self.assertNotIn("filters=filters", source)


if __name__ == "__main__":
    unittest.main()
