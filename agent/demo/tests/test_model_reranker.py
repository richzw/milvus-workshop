from __future__ import annotations

import json
import math
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from agent_workshop_demo.models import SearchResult
from agent_workshop_demo.reranker import (
    FallbackReranker,
    OpenAIModelReranker,
    RERANK_SCHEMA,
    Reranker,
    RerankerError,
    RerankRun,
    RuleBasedReranker,
    build_reranker,
    validate_rerank_run,
)
from agent_workshop_demo.sample_data import load_kb_chunks
from agent_workshop_demo.workflow import AgenticRAGWorkflow


@dataclass
class _FakeResponse:
    output_text: str


class _FakeResponses:
    def __init__(
        self,
        *,
        payload: object | None = None,
        error: Exception | None = None,
    ) -> None:
        self.payload = payload
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return _FakeResponse(
            output_text=json.dumps(self.payload, allow_nan=True)
        )


class _FakeClient:
    def __init__(
        self,
        *,
        payload: object | None = None,
        error: Exception | None = None,
    ) -> None:
        self.responses = _FakeResponses(payload=payload, error=error)


class _StaticReranker(Reranker):
    name = "invalid-test-double"

    def __init__(self, run: RerankRun) -> None:
        self.run = run

    def rerank(
        self,
        query: str,
        chunks: list[SearchResult],
        top_k: int,
    ) -> RerankRun:
        return self.run


class _FailingReranker(Reranker):
    name = "failing-test-double"

    def rerank(
        self,
        query: str,
        chunks: list[SearchResult],
        top_k: int,
    ) -> RerankRun:
        raise RerankerError("secret provider body")


def _search_results(count: int = 3) -> list[SearchResult]:
    chunks = [chunk for chunk in load_kb_chunks() if chunk.is_current][:count]
    return [
        SearchResult(
            chunk=chunk,
            rank=index,
            dense_score=0.9 - index / 100,
            keyword_score=0.8,
            recency_score=0.7,
            priority_score=0.6,
            hybrid_score=0.9 - index / 100,
        )
        for index, chunk in enumerate(chunks, start=1)
    ]


def _payload(
    chunks: list[SearchResult],
    scores: list[float] | None = None,
) -> dict[str, list[dict[str, str | float]]]:
    values = scores or [
        (index + 1) / max(len(chunks), 1)
        for index in range(len(chunks))
    ]
    return {
        "rankings": [
            {
                "chunk_id": result.chunk.chunk_id,
                "relevance_score": values[index],
            }
            for index, result in enumerate(chunks)
        ]
    }


class ModelRerankerTests(unittest.TestCase):
    def test_openai_reranker_uses_strict_schema_and_complete_pool(self) -> None:
        chunks = _search_results()
        client = _FakeClient(payload=_payload(chunks, [0.7, 0.9, 0.7]))
        reranker = OpenAIModelReranker(
            client=client,
            model="configured-model",
            timeout_seconds=8,
        )

        run = reranker.rerank("Milvus 如何检索？", chunks, top_k=3)

        self.assertEqual(run.reranker_name, "openai")
        self.assertEqual(run.model, "configured-model")
        self.assertIsNone(run.fallback_reason)
        self.assertEqual(
            [item.chunk.chunk_id for item in run.results],
            [
                chunks[1].chunk.chunk_id,
                chunks[0].chunk.chunk_id,
                chunks[2].chunk.chunk_id,
            ],
        )
        self.assertEqual([item.rerank for item in run.results], [1, 2, 3])
        self.assertEqual([item.old_rank for item in run.results], [2, 1, 3])

        self.assertEqual(len(client.responses.calls), 1)
        call = client.responses.calls[0]
        self.assertEqual(call["model"], "configured-model")
        self.assertEqual(call["timeout"], 8)
        self.assertEqual(
            call["text"]["format"],
            {
                "type": "json_schema",
                "name": "candidate_ranking",
                "schema": RERANK_SCHEMA,
                "strict": True,
            },
        )
        request = json.loads(call["input"])
        self.assertEqual(
            {item["chunk_id"] for item in request["candidates"]},
            {item.chunk.chunk_id for item in chunks},
        )

    def test_invalid_model_batches_fall_back_without_partial_scores(self) -> None:
        chunks = _search_results()
        valid = _payload(chunks)
        invalid_payloads: list[object] = [
            {"rankings": valid["rankings"][:-1]},
            {
                "rankings": [
                    valid["rankings"][0],
                    valid["rankings"][0],
                    valid["rankings"][2],
                ]
            },
            {
                "rankings": [
                    *valid["rankings"][:-1],
                    {"chunk_id": "invented", "relevance_score": 0.5},
                ]
            },
            _payload(chunks, [0.1, math.nan, 0.3]),
            _payload(chunks, [0.1, 1.1, 0.3]),
            {"unexpected": []},
        ]

        expected = RuleBasedReranker().rerank(
            "Milvus 如何检索？",
            chunks,
            top_k=3,
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                reranker = FallbackReranker(
                    primary=OpenAIModelReranker(
                        client=_FakeClient(payload=payload),
                        model="configured-model",
                    ),
                    fallback=RuleBasedReranker(),
                )
                run = reranker.rerank(
                    "Milvus 如何检索？",
                    chunks,
                    top_k=3,
                )

                self.assertEqual(
                    run.fallback_reason,
                    "invalid_model_output",
                )
                self.assertEqual(run.reranker_name, "rule-based-reranker")
                self.assertIsNone(run.model)
                self.assertEqual(run.results, expected.results)

    def test_provider_failures_use_sanitized_reason_codes(self) -> None:
        chunks = _search_results()
        cases = [
            (TimeoutError("secret timeout body"), "timeout"),
            (
                type("AuthenticationError", (RuntimeError,), {})(
                    "secret auth body"
                ),
                "authentication_error",
            ),
            (
                type("RateLimitError", (RuntimeError,), {})(
                    "secret rate body"
                ),
                "rate_limited",
            ),
            (RuntimeError("secret provider body"), "provider_error"),
        ]
        for error, reason in cases:
            with self.subTest(reason=reason):
                reranker = FallbackReranker(
                    primary=OpenAIModelReranker(
                        client=_FakeClient(error=error),
                        model="configured-model",
                    ),
                    fallback=RuleBasedReranker(),
                )
                run = reranker.rerank("Milvus", chunks, top_k=3)
                self.assertEqual(run.fallback_reason, reason)
                self.assertNotIn("secret", run.fallback_reason or "")

        injected = FallbackReranker(
            primary=_FailingReranker(),
            fallback=RuleBasedReranker(),
        ).rerank("Milvus", chunks, top_k=3)
        self.assertEqual(injected.fallback_reason, "provider_error")
        self.assertNotIn("secret", injected.fallback_reason or "")

    def test_fallback_only_skips_primary_and_preserves_registered_reason(self) -> None:
        chunks = _search_results()
        primary = _FailingReranker()
        reranker = FallbackReranker(
            primary=primary,
            fallback=RuleBasedReranker(),
        )

        run = reranker.rerank_fallback_only(
            "Milvus",
            chunks,
            top_k=3,
            reason_code="timeout",
        )

        self.assertEqual(run.reranker_name, "rule-based-reranker")
        self.assertEqual(run.fallback_reason, "timeout")
        with self.assertRaisesRegex(ValueError, "registered"):
            reranker.rerank_fallback_only(
                "Milvus",
                chunks,
                top_k=3,
                reason_code="secret",
            )

    def test_builder_modes_are_explicit_and_lazy(self) -> None:
        chunks = _search_results()
        calls: list[str] = []
        client = _FakeClient(payload=_payload(chunks))

        def client_factory(key: str) -> _FakeClient:
            calls.append(key)
            return client

        offline = build_reranker(
            {
                "RERANKER": "rule_based",
                "OPENAI_API_KEY": "ignored",
                "OPENAI_RERANKER_MODEL": "ignored",
            },
            client_factory=client_factory,
        )
        self.assertIsInstance(offline, RuleBasedReranker)
        self.assertEqual(calls, [])

        auto = build_reranker({"RERANKER": "auto"})
        auto_run = auto.rerank("Milvus", chunks, top_k=3)
        self.assertEqual(auto_run.fallback_reason, "not_configured")
        self.assertEqual(auto_run.reranker_name, "rule-based-reranker")

        configured = build_reranker(
            {
                "RERANKER": "openai",
                "OPENAI_API_KEY": "test-key",
                "OPENAI_RERANKER_MODEL": "reranker-model",
                "OPENAI_RERANKER_TIMEOUT_SECONDS": "7",
            },
            client_factory=client_factory,
        )
        configured_run = configured.rerank("Milvus", chunks, top_k=3)
        self.assertEqual(calls, ["test-key"])
        self.assertEqual(configured_run.reranker_name, "openai")
        self.assertEqual(configured_run.model, "reranker-model")
        self.assertEqual(client.responses.calls[0]["timeout"], 7)

        with self.assertRaisesRegex(ValueError, "are required"):
            build_reranker({"RERANKER": "openai"})
        with self.assertRaisesRegex(ValueError, "must be positive"):
            build_reranker(
                {
                    "RERANKER": "openai",
                    "OPENAI_API_KEY": "key",
                    "OPENAI_RERANKER_MODEL": "model",
                    "OPENAI_RERANKER_TIMEOUT_SECONDS": "0",
                }
            )

        def failed_client_factory(key: str) -> _FakeClient:
            raise RuntimeError(f"secret init failure for {key}")

        initialization_fallback = build_reranker(
            {
                "RERANKER": "openai",
                "OPENAI_API_KEY": "secret-key",
                "OPENAI_RERANKER_MODEL": "model",
            },
            client_factory=failed_client_factory,
        )
        init_run = initialization_fallback.rerank(
            "Milvus",
            chunks,
            top_k=3,
        )
        self.assertEqual(init_run.fallback_reason, "provider_error")
        self.assertEqual(init_run.reranker_name, "rule-based-reranker")

    def test_exhaustive_multi_side_pool_is_bounded_and_fully_ranked(self) -> None:
        seed = _search_results(1)[0]
        chunks = [
            replace(
                seed,
                chunk=replace(
                    seed.chunk,
                    chunk_id=f"expanded_chunk_{index:03d}",
                    text="候选证据" * 2_000,
                ),
                rank=index,
            )
            for index in range(1, 41)
        ]
        client = _FakeClient(payload=_payload(chunks))
        workflow = AgenticRAGWorkflow(
            reranker=OpenAIModelReranker(
                client=client,
                model="reranker-model",
            )
        )
        state = workflow.create_state("对比两个版本的全部功能")
        state.retrieved_chunks = chunks
        state.document_expansions = [
            {
                "result_chunk_ids": [
                    item.chunk.chunk_id for item in chunks[:20]
                ]
            },
            {
                "result_chunk_ids": [
                    item.chunk.chunk_id for item in chunks[20:]
                ]
            },
        ]

        workflow.rerank_evidence(state)

        self.assertEqual(len(state.reranked_chunks), 40)
        self.assertEqual(
            {item.chunk.chunk_id for item in state.reranked_chunks},
            {item.chunk.chunk_id for item in chunks},
        )
        self.assertLessEqual(
            len(client.responses.calls[0]["input"]),
            96_000,
        )

    def test_streamlit_cache_key_tracks_reranker_configuration(self) -> None:
        source = Path(
            "demo/src/agent_workshop_demo/streamlit_app.py"
        ).read_text(encoding="utf-8")

        self.assertIn('os.getenv("RERANKER"', source)
        self.assertIn('os.getenv("OPENAI_RERANKER_MODEL"', source)
        self.assertIn(
            'os.getenv("OPENAI_RERANKER_TIMEOUT_SECONDS"',
            source,
        )

    def test_workflow_trace_reports_actual_model_or_fallback(self) -> None:
        chunks = _search_results()
        client = _FakeClient(payload=_payload(chunks, [0.9, 0.8, 0.7]))
        model_reranker = FallbackReranker(
            primary=OpenAIModelReranker(
                client=client,
                model="reranker-model",
            ),
            fallback=RuleBasedReranker(),
        )
        workflow = AgenticRAGWorkflow(reranker=model_reranker)
        state = workflow.create_state(
            "Milvus 如何检索？",
            query_id="query_reranker",
        )
        state.retrieved_chunks = chunks

        workflow.rerank_evidence(state)
        workflow._finalize(state, workflow._clock())

        self.assertEqual(state.reranker_name, "openai")
        self.assertEqual(state.reranker_model, "reranker-model")
        self.assertIsNone(state.reranker_fallback_reason)
        self.assertEqual(
            state.trace["reranker"],
            {
                "name": "openai",
                "model": "reranker-model",
                "fallback_active": False,
                "fallback_reason": None,
                "sticky_fallback_reason": None,
                "primary_attempt_count": 1,
                "fallback_only_count": 0,
                "input_candidates": 3,
                "processed_candidates": 3,
                "output_top_k": 8,
            },
        )

        failed = AgenticRAGWorkflow(
            reranker=FallbackReranker(
                primary=OpenAIModelReranker(
                    client=_FakeClient(error=TimeoutError("secret")),
                    model="reranker-model",
                ),
                fallback=RuleBasedReranker(),
            )
        )
        failed_state = failed.create_state(
            "Milvus 如何检索？",
            query_id="query_reranker_fallback",
        )
        failed_state.retrieved_chunks = chunks
        failed.rerank_evidence(failed_state)
        failed._finalize(failed_state, failed._clock())
        self.assertEqual(
            failed_state.trace["reranker"]["name"],
            "rule-based-reranker",
        )
        self.assertTrue(
            failed_state.trace["reranker"]["fallback_active"]
        )
        self.assertEqual(
            failed_state.trace["reranker"]["fallback_reason"],
            "timeout",
        )
        self.assertEqual(
            failed_state.trace["reranker"]["sticky_fallback_reason"],
            "timeout",
        )

    def test_workflow_pins_fallback_only_within_one_query(self) -> None:
        first = _search_results(3)
        second = _search_results(4)
        client = _FakeClient(error=TimeoutError("secret"))
        workflow = AgenticRAGWorkflow(
            reranker=FallbackReranker(
                primary=OpenAIModelReranker(
                    client=client,
                    model="reranker-model",
                ),
                fallback=RuleBasedReranker(),
            )
        )
        state = workflow.create_state("Milvus 如何检索？")
        state.retrieved_chunks = first

        workflow.rerank_evidence(state)
        state.retrieved_chunks = second
        workflow.rerank_evidence(state)

        self.assertEqual(len(client.responses.calls), 1)
        self.assertEqual(state.reranker_primary_attempt_count, 1)
        self.assertEqual(state.reranker_fallback_only_count, 1)
        self.assertEqual(state.reranker_sticky_fallback_reason, "timeout")

        next_query = workflow.create_state("Milvus 如何索引？")
        next_query.retrieved_chunks = first
        workflow.rerank_evidence(next_query)
        self.assertEqual(len(client.responses.calls), 2)

    def test_injected_reranker_run_is_validated(self) -> None:
        chunks = _search_results()
        invalid = RerankRun(
            results=(),
            reranker_name="injected",
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "invalid_model_output",
        ):
            validate_rerank_run(invalid, chunks=chunks, top_k=3)

        valid = RuleBasedReranker().rerank("Milvus", chunks, top_k=3)
        invalid_runs = [
            RerankRun(
                results=valid.results,
                reranker_name=cast(Any, 123),
            ),
            replace(
                valid,
                results=(
                    replace(
                        valid.results[0],
                        rerank_score=cast(Any, "bad"),
                    ),
                    *valid.results[1:],
                ),
            ),
        ]
        for invalid_run in invalid_runs:
            with self.subTest(run=invalid_run):
                fallback = FallbackReranker(
                    primary=_StaticReranker(invalid_run),
                    fallback=RuleBasedReranker(),
                ).rerank("Milvus", chunks, top_k=3)
                self.assertEqual(
                    fallback.fallback_reason,
                    "invalid_model_output",
                )
                self.assertEqual(
                    fallback.reranker_name,
                    "rule-based-reranker",
                )

    def test_no_retrieval_query_reports_reranker_not_run(self) -> None:
        response = AgenticRAGWorkflow().run("你好")

        self.assertEqual(response["trace"]["reranker"]["name"], "not_run")
        self.assertFalse(
            response["trace"]["reranker"]["fallback_active"]
        )
        self.assertEqual(
            response["trace"]["reranker"]["processed_candidates"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
