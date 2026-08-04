from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any

from agent_workshop_demo.response_cache import (
    CachedEvidence,
    GroundedResponseCacheRecord,
    GroundedResponseCacheStore,
    MilvusGroundedResponseCacheStore,
    ResponseCacheCandidate,
    ResponseCacheError,
    build_cache_record,
    query_constraints,
)
from agent_workshop_demo.langgraph_workflow import build_default_workflow
from agent_workshop_demo.knowledge_tools import PermissionDecision
from agent_workshop_demo.models import KBChunk
from agent_workshop_demo.retrieval import InMemoryHybridRetriever
from agent_workshop_demo.sample_data import load_kb_chunks
from agent_workshop_demo.workflow import AgenticRAGWorkflow


class CountingResponseCache(GroundedResponseCacheStore):
    def __init__(self) -> None:
        super().__init__()
        self.search_count = 0

    def search(
        self,
        query: str,
        *,
        session_id: str,
        top_k: int,
        now_ms: int | None = None,
    ) -> list[ResponseCacheCandidate]:
        self.search_count += 1
        return super().search(
            query,
            session_id=session_id,
            top_k=top_k,
            now_ms=now_ms,
        )


def cache_record(
    *,
    query: str = "Milvus 3.0 有哪些新功能",
    session_id: str = "session_cache",
    created_at: int = 1_000,
    expires_at: int = 2_000,
) -> GroundedResponseCacheRecord:
    chunk = load_kb_chunks()[0]
    return build_cache_record(
        session_id=session_id,
        source_query_id="query_source",
        user_query=query,
        intent="private_knowledge",
        query_type="architecture",
        retrieval_goal="exhaustive",
        version_scope={
            "mode": "current",
            "doc_versions": [],
            "sides": [{"mode": "current"}],
        },
        entity_ids=[],
        permission_scope_hash_value="permission-hash",
        kb_revision="demo-v1",
        answer="缓存回答。[C1]",
        citations=[
            {
                "citation_id": "C1",
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "doc_version": chunk.doc_version,
            }
        ],
        evidence=[
            CachedEvidence(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                doc_version=chunk.doc_version,
                checksum=chunk.checksum or "checksum",
                is_current=chunk.is_current,
            )
        ],
        created_at=created_at,
        expires_at=expires_at,
    )


class ResponseCacheStoreTests(unittest.TestCase):
    def test_query_constraints_preserve_version_and_polarity(self) -> None:
        self.assertEqual(
            query_constraints("Milvus 3.0 有哪些新功能"),
            ["polarity:positive", "version:3.0"],
        )
        self.assertEqual(
            query_constraints("Milvus 2.6 不支持哪些功能"),
            ["polarity:negative", "version:2.6"],
        )

    def test_exact_semantic_expiry_and_session_scope(self) -> None:
        store = GroundedResponseCacheStore(now_ms=1_500)
        record = cache_record()
        self.assertEqual(store.upsert(record), 1)

        exact = store.search(
            "Milvus 3.0 有哪些新功能",
            session_id="session_cache",
            top_k=3,
            now_ms=1_500,
        )
        semantic = store.search(
            "Milvus 3.0 有哪些新功能？",
            session_id="session_cache",
            top_k=3,
            now_ms=1_500,
        )

        self.assertEqual(exact[0].match_type, "exact")
        self.assertEqual(exact[0].similarity, 1.0)
        self.assertEqual(semantic[0].match_type, "semantic")
        self.assertEqual(semantic[0].similarity, 1.0)
        self.assertEqual(
            store.search(
                "Milvus 3.0 有哪些新功能",
                session_id="session_other",
                top_k=3,
                now_ms=1_500,
            ),
            [],
        )
        self.assertEqual(
            store.search(
                "Milvus 3.0 有哪些新功能",
                session_id="session_cache",
                top_k=3,
                now_ms=2_000,
            ),
            [],
        )

    def test_upsert_and_clear_are_session_scoped(self) -> None:
        store = GroundedResponseCacheStore(now_ms=1_000)
        store.upsert(cache_record())
        store.upsert(
            cache_record(
                session_id="session_other",
                created_at=1_100,
                expires_at=2_100,
            )
        )

        self.assertEqual(store.delete_session("session_cache"), 1)
        self.assertEqual(len(store.records), 1)
        self.assertEqual(store.records[0].session_id, "session_other")

    def test_milvus_store_persists_and_validates_exact_scope(self) -> None:
        class Client:
            def __init__(self) -> None:
                self.row: dict[str, Any] | None = None
                self.loaded: list[str] = []
                self.deleted: list[str] = []

            def has_collection(self, *, collection_name: str) -> bool:
                del collection_name
                return True

            def load_collection(self, *, collection_name: str) -> None:
                self.loaded.append(collection_name)

            def query(self, **kwargs: Any) -> list[dict[str, Any]]:
                del kwargs
                return [] if self.row is None else [self.row]

            def search(self, **kwargs: Any) -> list[list[dict[str, Any]]]:
                del kwargs
                return [[]]

            def delete(self, **kwargs: Any) -> dict[str, int]:
                self.deleted.append(kwargs["filter"])
                return {"delete_count": 1}

            def insert(self, **kwargs: Any) -> dict[str, int]:
                self.row = kwargs["data"][0]
                return {"insert_count": 1}

            def flush(self, **kwargs: Any) -> None:
                del kwargs
                return None

        client = Client()
        store = MilvusGroundedResponseCacheStore(client)
        record = cache_record()
        store.ensure_collection_ready()

        self.assertEqual(store.upsert(record), 1)
        result = store.search(
            "Milvus 3.0 有哪些新功能",
            session_id="session_cache",
            top_k=3,
            now_ms=1_500,
        )

        self.assertEqual(client.loaded, ["grounded_response_cache"])
        self.assertEqual(result[0].record.answer, record.answer)
        self.assertEqual(result[0].match_type, "exact")
        self.assertEqual(store.delete_session("session_cache"), 1)


class WorkflowResponseCacheTests(unittest.TestCase):
    def test_streamlit_cache_key_tracks_revision_and_cache_config(self) -> None:
        source = Path(
            "demo/src/agent_workshop_demo/streamlit_app.py"
        ).read_text(encoding="utf-8")

        self.assertIn("MILVUS_RESPONSE_CACHE_COLLECTION_NAME", source)
        self.assertIn("RESPONSE_CACHE_SIMILARITY_THRESHOLD", source)
        self.assertIn('os.getenv("KB_REVISION"', source)
        self.assertIn("cached answers", source)

    def test_repeated_question_returns_validated_cached_answer(self) -> None:
        workflow = AgenticRAGWorkflow()
        first = workflow.run(
            "Milvus 3.0 有哪些新功能",
            session_id="session_repeat",
            query_id="query_first",
        )
        second = workflow.run(
            "Milvus 3.0 有哪些新功能",
            session_id="session_repeat",
            query_id="query_second",
        )

        self.assertEqual(first["terminal_status"], "answered")
        self.assertEqual(second["terminal_status"], "answered_from_cache")
        self.assertEqual(second["answer"], first["answer"])
        self.assertEqual(second["citations"], first["citations"])
        self.assertEqual(second["tool_calls"], [])
        self.assertEqual(second["reranked"], [])
        self.assertEqual(second["response_cache_match_type"], "exact")
        self.assertEqual(
            second["answer_validation"]["mode"],
            "cached_grounded",
        )
        self.assertNotIn("response_cache_candidates", second)

    def test_only_allowed_grounded_routes_search_the_cache(self) -> None:
        class DenyPermission:
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
                    reason="Denied for cache routing test.",
                    checker_name="deny-cache-test",
                )

        for runtime in ("local", "langgraph"):
            with self.subTest(runtime=runtime):
                store = CountingResponseCache()
                build = (
                    AgenticRAGWorkflow
                    if runtime == "local"
                    else build_default_workflow
                )
                workflow = build(response_cache=store)
                session_id = f"session_cache_routes_{runtime}"
                for question in (
                    "你好",
                    "请记住我喜欢蓝色",
                    "你还记得我喜欢什么吗？",
                    "帮我删除产品路线图",
                    "段位是什么意思？",
                ):
                    workflow.run(question, session_id=session_id)
                if runtime == "local":
                    AgenticRAGWorkflow(
                        response_cache=store,
                        permission_checker=DenyPermission(),
                    ).run(
                        "请查看内部产品路线图",
                        session_id=session_id,
                    )

                self.assertEqual(store.search_count, 0)

    def test_cache_lookup_is_single_stage_before_authorized_experience(self) -> None:
        store = CountingResponseCache()
        workflow = AgenticRAGWorkflow(response_cache=store)
        first = workflow.run(
            "Milvus 3.0 有哪些新功能",
            session_id="session_cache_order",
            query_id="query_cache_order_source",
        )
        second = workflow.run(
            "Milvus 3.0 有哪些新功能",
            session_id="session_cache_order",
            query_id="query_cache_order_hit",
        )

        self.assertEqual(store.search_count, 2)
        self.assertIn(
            "recall_authorized_experience",
            first["metrics"]["stage_latency_ms"],
        )
        self.assertIn(
            "try_grounded_cache",
            second["metrics"]["stage_latency_ms"],
        )
        self.assertNotIn(
            "recall_authorized_experience",
            second["metrics"]["stage_latency_ms"],
        )
        self.assertNotIn(
            "validate_response_cache",
            second["metrics"]["stage_latency_ms"],
        )

    def test_semantically_equivalent_question_uses_cache(self) -> None:
        workflow = AgenticRAGWorkflow()
        workflow.run(
            "Milvus 3.0 有哪些新功能",
            session_id="session_semantic",
            query_id="query_semantic_source",
        )
        response = workflow.run(
            "Milvus 3.0 有哪些新功能？",
            session_id="session_semantic",
            query_id="query_semantic_hit",
        )

        self.assertEqual(
            response["terminal_status"],
            "answered_from_cache",
        )
        self.assertEqual(response["response_cache_match_type"], "semantic")
        self.assertEqual(response["response_cache_similarity"], 1.0)

    def test_langgraph_runtime_has_the_same_cache_short_circuit(self) -> None:
        workflow = build_default_workflow()
        workflow.run(
            "Milvus 3.0 有哪些新功能",
            session_id="session_graph_cache",
            query_id="query_graph_source",
        )
        response = workflow.run(
            "Milvus 3.0 有哪些新功能",
            session_id="session_graph_cache",
            query_id="query_graph_hit",
        )

        self.assertEqual(
            response["terminal_status"],
            "answered_from_cache",
        )
        self.assertEqual(response["tool_calls"], [])

    def test_changed_evidence_checksum_falls_back_to_rag(self) -> None:
        retriever = InMemoryHybridRetriever(load_kb_chunks())
        workflow = AgenticRAGWorkflow(retriever=retriever)
        first = workflow.run(
            "Milvus 3.0 有哪些新功能",
            session_id="session_stale",
            query_id="query_stale_source",
        )
        cited_id = str(first["citations"][0]["chunk_id"])
        retriever.chunks = [
            (
                replace(chunk, checksum=f"{chunk.checksum}-changed")
                if chunk.chunk_id == cited_id
                else chunk
            )
            for chunk in retriever.chunks
        ]

        response = workflow.run(
            "Milvus 3.0 有哪些新功能",
            session_id="session_stale",
            query_id="query_stale_retry",
        )

        self.assertEqual(response["terminal_status"], "answered")
        self.assertTrue(response["tool_calls"])
        self.assertEqual(
            response["response_cache_fallback_reason"],
            "cache_stale",
        )

    def test_kb_revision_change_invalidates_cached_answer(self) -> None:
        store = GroundedResponseCacheStore()
        first = AgenticRAGWorkflow(
            response_cache=store,
            kb_revision="revision-1",
        )
        first.run(
            "Milvus 3.0 有哪些新功能",
            session_id="session_revision",
            query_id="query_revision_source",
        )
        second = AgenticRAGWorkflow(
            response_cache=store,
            kb_revision="revision-2",
        ).run(
            "Milvus 3.0 有哪些新功能",
            session_id="session_revision",
            query_id="query_revision_retry",
        )

        self.assertEqual(second["terminal_status"], "answered")
        self.assertTrue(second["tool_calls"])
        self.assertEqual(
            second["response_cache_fallback_reason"],
            "cache_stale",
        )

    def test_permission_scope_change_invalidates_cached_answer(self) -> None:
        class EngineeringOnly:
            def check(
                self,
                *,
                session_id: str,
                intent: str,
                query_type: str,
            ) -> PermissionDecision:
                del session_id, intent, query_type
                return PermissionDecision(
                    allowed=True,
                    allowed_departments=("engineering",),
                    reason="Restricted test scope.",
                )

        store = GroundedResponseCacheStore()
        AgenticRAGWorkflow(response_cache=store).run(
            "Milvus 3.0 有哪些新功能",
            session_id="session_permission",
            query_id="query_permission_source",
        )
        response = AgenticRAGWorkflow(
            response_cache=store,
            permission_checker=EngineeringOnly(),
        ).run(
            "Milvus 3.0 有哪些新功能",
            session_id="session_permission",
            query_id="query_permission_retry",
        )

        self.assertEqual(response["terminal_status"], "answered")
        self.assertTrue(response["tool_calls"])
        self.assertEqual(
            response["response_cache_fallback_reason"],
            "cache_stale",
        )

    def test_expired_answer_and_cache_failure_fall_back_to_rag(self) -> None:
        now = [1_000]
        workflow = AgenticRAGWorkflow(
            response_cache_ttl_seconds=1,
            wall_clock_ms=lambda: now[0],
        )
        workflow.run(
            "Milvus 3.0 有哪些新功能",
            session_id="session_expiry",
            query_id="query_expiry_source",
        )
        now[0] = 2_001
        expired = workflow.run(
            "Milvus 3.0 有哪些新功能",
            session_id="session_expiry",
            query_id="query_expiry_retry",
        )
        self.assertEqual(expired["terminal_status"], "answered")
        self.assertTrue(expired["tool_calls"])

        class FailingCache(GroundedResponseCacheStore):
            def search(
                self,
                query: str,
                *,
                session_id: str,
                top_k: int,
                now_ms: int | None = None,
            ) -> list[ResponseCacheCandidate]:
                del query, session_id, top_k, now_ms
                raise ResponseCacheError("secret dependency detail")

        failed = AgenticRAGWorkflow(response_cache=FailingCache()).run(
            "Milvus 3.0 有哪些新功能",
            session_id="session_failure",
            query_id="query_failure_retry",
        )
        self.assertEqual(failed["terminal_status"], "answered")
        self.assertTrue(failed["tool_calls"])
        self.assertEqual(
            failed["response_cache_fallback_reason"],
            "cache_unavailable",
        )
        self.assertNotIn("secret dependency detail", str(failed))

        class FailingEvidenceRetriever(InMemoryHybridRetriever):
            fail_validation = False

            def fetch_chunks_by_ids(
                self,
                *,
                chunk_ids: list[str],
                filters: dict[str, Any] | None = None,
            ) -> list[KBChunk]:
                if self.fail_validation:
                    raise OSError("secret Milvus evidence error")
                return super().fetch_chunks_by_ids(
                    chunk_ids=chunk_ids,
                    filters=filters,
                )

        retriever = FailingEvidenceRetriever(load_kb_chunks())
        evidence_workflow = AgenticRAGWorkflow(retriever=retriever)
        evidence_workflow.run(
            "Milvus 3.0 有哪些新功能",
            session_id="session_evidence_failure",
            query_id="query_evidence_source",
        )
        retriever.fail_validation = True
        evidence_failed = evidence_workflow.run(
            "Milvus 3.0 有哪些新功能",
            session_id="session_evidence_failure",
            query_id="query_evidence_retry",
        )
        self.assertEqual(evidence_failed["terminal_status"], "answered")
        self.assertTrue(evidence_failed["tool_calls"])
        self.assertNotIn("secret Milvus evidence error", str(evidence_failed))

    def test_clear_memory_also_clears_grounded_response_cache(self) -> None:
        workflow = AgenticRAGWorkflow()
        workflow.run(
            "Milvus 3.0 有哪些新功能",
            session_id="session_clear_cache",
            query_id="query_clear_cache",
        )

        deleted = workflow.clear_memory("session_clear_cache")

        self.assertGreaterEqual(deleted, 2)
        response = workflow.run(
            "Milvus 3.0 有哪些新功能",
            session_id="session_clear_cache",
            query_id="query_after_clear",
        )
        self.assertEqual(response["terminal_status"], "answered")


if __name__ == "__main__":
    unittest.main()
