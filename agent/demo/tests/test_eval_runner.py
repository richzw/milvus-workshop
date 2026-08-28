"""Layered RAG eval report, reliability, and baseline contracts."""

from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from unittest.mock import patch

from agent_workshop_demo.eval_governance import DEFAULT_METRIC_REGISTRY_PATH
from agent_workshop_demo.eval_runner import evaluate_questions
from agent_workshop_demo.knowledge_tools import SEARCH_TOOLS, KnowledgeSearchTool


class _StreamingWorkflow:
    def __init__(self, *, include_fact: bool = True, sequence: int = 1) -> None:
        self.include_fact = include_fact
        self.sequence = sequence

    def run(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        *,
        session_id: str | None = None,
        query_id: str | None = None,
    ) -> dict[str, Any]:
        del question, filters, session_id
        return self._response(query_id or "eval_query")

    def stream(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        *,
        session_id: str | None = None,
        query_id: str | None = None,
    ) -> Iterable[dict[str, Any]]:
        del question, filters, session_id
        resolved_query_id = query_id or "eval_query"
        response = self._response(resolved_query_id)
        stages = (
            "recall_memory",
            "classify_and_route",
            "resolve_terminology",
            "check_permission",
            "try_grounded_cache",
            "recall_authorized_experience",
            "plan_retrieval",
            "execute_tool_plan",
            "rerank_evidence",
            "evaluate_evidence",
            "prepare_generation_context",
            "generate_answer_streaming",
            "verify_answer",
        )
        for offset, stage in enumerate(stages):
            yield {
                "type": "trace_event",
                "event": {
                    "sequence": self.sequence + offset,
                    "query_id": resolved_query_id,
                    "stage": stage,
                    "kind": "stage_completed",
                },
            }
        yield {"type": "answer_delta", "text": response["answer"]}
        yield {"type": "final", "response": response}

    def _response(self, query_id: str) -> dict[str, Any]:
        answer = "required fact" if self.include_fact else "different answer"
        selected = {
            "chunk_id": "chunk_1",
            "doc_id": "doc_1",
            "doc_version": "v1",
            "is_current": True,
            "selected": True,
        }
        return {
            "query_id": query_id,
            "terminal_status": "answered",
            "retry_count": 0,
            "trace": {
                "query_id": query_id,
                "terminal_status": "answered",
            },
            "answer": answer,
            "answer_validation": {"valid": True},
            "enough_evidence": True,
            "milvus_recalled": [selected],
            "reranked": [selected],
            "citations": [selected],
            "selected_tools": ["search_code_docs"],
            "matched_entities": [],
            "version_scope": {"mode": "current", "doc_versions": []},
            "query_plan": [
                {
                    "subquery_id": "sq1",
                    "tool": "search_code_docs",
                    "query": "test question engineering docs",
                    "query_role": "primary",
                    "round": 0,
                }
            ],
            "query_transformation": {
                "strategy": "identity",
                "item_roles": ["primary"],
                "item_count": 1,
                "transformer_name": "rule_based",
            },
            "context_compression": {
                "configured_mode": "disabled",
                "effective_mode": "disabled",
                "compressor_name": "identity",
                "before_chars": 20,
                "after_chars": 20,
                "retained_source_count": 1,
                "fallback_reason": None,
            },
            "tool_calls": [
                {
                    "tool": "search_code_docs",
                    "filters": {
                        "department": ["engineering"],
                        "is_current": True,
                    },
                    "version_scope": {"mode": "current"},
                }
            ],
            "permission_decision": {
                "allowed": True,
                "allowed_departments": ["engineering"],
            },
            "metrics": {
                "latency_ms": 4.0,
                "retrieval_latency_ms": 1.0,
                "rerank_latency_ms": 1.0,
                "generation_latency_ms": 1.0,
            },
        }


class _NoToolWorkflow(_StreamingWorkflow):
    def _response(self, query_id: str) -> dict[str, Any]:
        response = super()._response(query_id)
        response["tool_calls"] = []
        return response


class _UnauthorizedToolWorkflow(_StreamingWorkflow):
    def _response(self, query_id: str) -> dict[str, Any]:
        response = super()._response(query_id)
        response["permission_decision"]["allowed_departments"] = ["product"]
        return response


class _PermissionBypassWorkflow(_StreamingWorkflow):
    def _response(self, query_id: str) -> dict[str, Any]:
        response = super()._response(query_id)
        response["terminal_status"] = "permission_denied"
        response["trace"]["terminal_status"] = "permission_denied"
        response["permission_decision"]["allowed"] = False
        return response


class _PermissionDeniedWorkflow(_PermissionBypassWorkflow):
    def _response(self, query_id: str) -> dict[str, Any]:
        response = super()._response(query_id)
        response.update(
            {
                "answer": "",
                "enough_evidence": False,
                "milvus_recalled": [],
                "reranked": [],
                "citations": [],
                "selected_tools": [],
                "query_plan": [],
                "tool_calls": [],
            }
        )
        return response

    def stream(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        *,
        session_id: str | None = None,
        query_id: str | None = None,
    ) -> Iterable[dict[str, Any]]:
        del question, filters, session_id
        resolved_query_id = query_id or "eval_query"
        response = self._response(resolved_query_id)
        for sequence, stage in enumerate(
            (
                "recall_memory",
                "classify_and_route",
                "resolve_terminology",
                "check_permission",
            ),
            start=1,
        ):
            yield {
                "type": "trace_event",
                "event": {
                    "sequence": sequence,
                    "query_id": resolved_query_id,
                    "stage": stage,
                    "kind": "stage_completed",
                },
            }
        yield {"type": "final", "response": response}


class _CostWithoutProfileWorkflow(_StreamingWorkflow):
    def _response(self, query_id: str) -> dict[str, Any]:
        response = super()._response(query_id)
        response["metrics"].update(
            {
                "provider_call_count": 1,
                "input_tokens": 20,
                "output_tokens": 10,
                "estimated_cost": 0.01,
            }
        )
        return response


class _NonFiniteUsageWorkflow(_StreamingWorkflow):
    def _response(self, query_id: str) -> dict[str, Any]:
        response = super()._response(query_id)
        response["metrics"].update(
            {
                "retrieval_latency_ms": -1,
                "provider_call_count": 1,
                "input_tokens": 20,
                "output_tokens": 10,
                "estimated_cost": float("nan"),
                "cost_profile": "test-price-v1",
            }
        )
        return response


class _UnboundedSupplementaryWorkflow(_StreamingWorkflow):
    def _response(self, query_id: str) -> dict[str, Any]:
        response = super()._response(query_id)
        response["query_plan"].extend(
            [
                {
                    "subquery_id": "sq2",
                    "tool": "search_code_docs",
                    "round": 1,
                },
                {
                    "subquery_id": "sq3",
                    "tool": "search_code_docs",
                    "round": 1,
                },
            ]
        )
        return response


class _MissingTerminalPathWorkflow(_StreamingWorkflow):
    def stream(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        *,
        session_id: str | None = None,
        query_id: str | None = None,
    ) -> Iterable[dict[str, Any]]:
        del question, filters, session_id
        resolved_query_id = query_id or "eval_query"
        response = self._response(resolved_query_id)
        for sequence, stage in enumerate(
            ("recall_memory", "classify_and_route", "verify_answer"),
            start=1,
        ):
            yield {
                "type": "trace_event",
                "event": {
                    "sequence": sequence,
                    "query_id": resolved_query_id,
                    "stage": stage,
                    "kind": "stage_completed",
                },
            }
        yield {"type": "answer_delta", "text": response["answer"]}
        yield {"type": "final", "response": response}


class _CacheHitWorkflow(_StreamingWorkflow):
    def _response(self, query_id: str) -> dict[str, Any]:
        response = super()._response(query_id)
        response["terminal_status"] = "answered_from_cache"
        response["trace"]["terminal_status"] = "answered_from_cache"
        response["selected_tools"] = []
        response["query_plan"] = []
        response["tool_calls"] = []
        # A real cache hit skips recall, rerank and selection; only the cached
        # citations survive. Keeping a populated `reranked` list here would
        # hide the very shape this double exists to represent.
        response["milvus_recalled"] = []
        response["reranked"] = []
        response["metrics"]["response_cache_hit"] = True
        return response

    def stream(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        *,
        session_id: str | None = None,
        query_id: str | None = None,
    ) -> Iterable[dict[str, Any]]:
        del question, filters, session_id
        resolved_query_id = query_id or "eval_query"
        response = self._response(resolved_query_id)
        stages = (
            "recall_memory",
            "classify_and_route",
            "resolve_terminology",
            "check_permission",
            "try_grounded_cache",
        )
        for sequence, stage in enumerate(stages, start=1):
            yield {
                "type": "trace_event",
                "event": {
                    "sequence": sequence,
                    "query_id": resolved_query_id,
                    "stage": stage,
                    "kind": "stage_completed",
                },
            }
        yield {"type": "answer_delta", "text": response["answer"]}
        yield {"type": "final", "response": response}


class _OverSelectionWorkflow(_StreamingWorkflow):
    def _response(self, query_id: str) -> dict[str, Any]:
        response = super()._response(query_id)
        selected = [
            {
                "chunk_id": f"chunk_{index}",
                "doc_id": "doc_1",
                "doc_version": "v1",
                "is_current": True,
                "selected": True,
            }
            for index in range(1, 7)
        ]
        response["milvus_recalled"] = selected
        response["reranked"] = selected
        response["citations"] = [selected[-1]]
        return response


class _StepBackWorkflow(_StreamingWorkflow):
    """A step-back plan: one background query plus the original-retaining one."""

    def _response(self, query_id: str) -> dict[str, Any]:
        response = super()._response(query_id)
        response["query_transformation"] = {
            "strategy": "step_back",
            "item_roles": ["background", "primary"],
            "item_count": 2,
            "transformer_name": "rule_based",
        }
        return response


class _SelectiveCompressionWorkflow(_StreamingWorkflow):
    """A run whose selective compression succeeded without falling back."""

    def _response(self, query_id: str) -> dict[str, Any]:
        response = super()._response(query_id)
        response["context_compression"] = {
            "configured_mode": "selective",
            "effective_mode": "selective",
            "compressor_name": "openai",
            "before_chars": 40,
            "after_chars": 20,
            "retained_source_count": 1,
            "fallback_reason": None,
        }
        return response


class _AbstainWithGenerationContextWorkflow(_StreamingWorkflow):
    """An abstention that still ran `prepare_generation_context`."""

    def _response(self, query_id: str) -> dict[str, Any]:
        response = super()._response(query_id)
        response["terminal_status"] = "abstained"
        response["trace"]["terminal_status"] = "abstained"
        response["enough_evidence"] = False
        response["citations"] = []
        return response


class _InvalidCompressionWorkflow(_StreamingWorkflow):
    """A run whose compression projection lost every retained source."""

    def _response(self, query_id: str) -> dict[str, Any]:
        response = super()._response(query_id)
        response["context_compression"] = {
            "configured_mode": "selective",
            "effective_mode": "selective",
            "compressor_name": "openai",
            "before_chars": 20,
            "after_chars": 40,
            "retained_source_count": 0,
            "fallback_reason": None,
        }
        return response


class _ExhaustiveWorkflow(_StreamingWorkflow):
    """An exhaustive document query releasing `context_count` sibling chunks."""

    def __init__(self, *, context_count: int = 16, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.context_count = context_count

    def _response(self, query_id: str) -> dict[str, Any]:
        response = super()._response(query_id)
        siblings = [
            {
                "chunk_id": f"chunk_{index}",
                "doc_id": "doc_1",
                "doc_version": "v1",
                "is_current": True,
                "selected": True,
            }
            for index in range(1, self.context_count + 1)
        ]
        response["retrieval_goal"] = "exhaustive"
        response["milvus_recalled"] = siblings
        response["reranked"] = siblings
        # Cite the last sibling: it is a legal context only under the
        # exhaustive branch of the two-branch generation-context bound.
        response["citations"] = [siblings[0], siblings[-1]]
        return response


class _RecordingWorkflow(_StreamingWorkflow):
    def __init__(self, query_ids: list[str]) -> None:
        super().__init__()
        self.query_ids = query_ids

    def stream(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        *,
        session_id: str | None = None,
        query_id: str | None = None,
    ) -> Iterable[dict[str, Any]]:
        self.query_ids.append(str(query_id))
        yield from super().stream(
            question,
            filters,
            session_id=session_id,
            query_id=query_id,
        )


class _MalformedEventWorkflow(_StreamingWorkflow):
    def stream(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        *,
        session_id: str | None = None,
        query_id: str | None = None,
    ) -> Iterable[dict[str, Any]]:
        del question, filters, session_id
        resolved_query_id = query_id or "eval_query"
        yield {
            "type": "trace_event",
            "event": {
                "sequence": "not-an-int",
                "query_id": resolved_query_id,
                "stage": "verify_answer",
                "kind": "stage_completed",
            },
        }
        yield {"type": "answer_delta", "text": "required fact"}
        yield {"type": "final", "response": self._response(resolved_query_id)}


class EvalRunnerTests(unittest.TestCase):
    def _fixtures(self, root: Path) -> tuple[Path, Path]:
        questions_path = root / "questions.json"
        golden_path = root / "golden.json"
        questions_path.write_text(
            json.dumps(
                [
                    {
                        "question_id": "case_1",
                        "question": "test question",
                        "expected_sources": ["chunk_1"],
                        "expected_tools": ["search_code_docs"],
                    }
                ]
            ),
            encoding="utf-8",
        )
        golden_path.write_text(
            json.dumps(
                {
                    "case_1": {
                        "required_facts": ["required fact"],
                        "required_citations": ["chunk_1"],
                    }
                }
            ),
            encoding="utf-8",
        )
        return questions_path, golden_path

    def test_live_trials_report_pass_at_k_and_gate_on_pass_power_k(self) -> None:
        outcomes = iter((False, True, True))

        with tempfile.TemporaryDirectory() as tmpdir:
            questions_path, golden_path = self._fixtures(Path(tmpdir))
            report = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=lambda: _StreamingWorkflow(
                    include_fact=next(outcomes)
                ),
                live_providers=True,
                trials=3,
            )

        self.assertEqual(report["reliability"]["pass_at_k"], 1.0)
        self.assertEqual(report["reliability"]["pass_power_k"], 0.0)
        self.assertEqual(report["reliability"]["gate_metric"], "pass_power_k")
        fact_metric = report["metric_portfolio"]["goal"]["goal.required_fact_coverage"]
        self.assertEqual(fact_metric["decision_status"], "fail")
        self.assertEqual(fact_metric["gate_basis"], "all_applicable_trials")
        self.assertEqual(fact_metric["applicable_trial_count"], 3)
        self.assertEqual(fact_metric["passing_trial_count"], 2)
        self.assertEqual(report["num_trials"], 3)
        self.assertEqual(
            report["dimensions"]["outcome"]["case_pass_rate"]["value"],
            0.6667,
        )
        self.assertEqual(report["cases"][0]["first_failure_layer"], "outcome")
        self.assertEqual(len(report["cases"][0]["trials"]), 3)

    def test_live_gate_fails_when_mean_passes_but_one_trial_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            questions_path, golden_path = self._fixtures(root)
            question = json.loads(questions_path.read_text(encoding="utf-8"))[0]
            golden = json.loads(golden_path.read_text(encoding="utf-8"))["case_1"]
            questions = []
            goldens = {}
            for index in range(4):
                case_id = f"case_{index + 1}"
                questions.append({**question, "question_id": case_id})
                goldens[case_id] = golden
            questions_path.write_text(json.dumps(questions), encoding="utf-8")
            golden_path.write_text(json.dumps(goldens), encoding="utf-8")
            outcomes = iter([False, *([True] * 11)])
            report = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=lambda: _StreamingWorkflow(
                    include_fact=next(outcomes)
                ),
                live_providers=True,
                trials=3,
            )

        metric = report["metric_portfolio"]["goal"]["goal.required_fact_coverage"]
        self.assertGreaterEqual(metric["value"], 0.9)
        self.assertEqual(metric["decision_status"], "fail")
        self.assertEqual(metric["applicable_trial_count"], 12)
        self.assertEqual(metric["passing_trial_count"], 11)

    def test_transformation_and_compression_dimensions_are_baseline_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            questions_path, golden_path = self._fixtures(Path(tmpdir))
            report = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=_StreamingWorkflow,
            )

        self.assertEqual(
            report["dimensions"]["tool"]["query_transformation_contract_rate"]["value"],
            1.0,
        )
        self.assertEqual(
            report["dimensions"]["tool"]["original_query_retention_rate"]["value"],
            1.0,
        )
        self.assertEqual(
            report["dimensions"]["outcome"]["context_compression_provenance_rate"][
                "value"
            ],
            1.0,
        )

    def test_live_case_summary_is_independent_of_trial_order(self) -> None:
        outcomes = iter((True, False, True))

        with tempfile.TemporaryDirectory() as tmpdir:
            questions_path, golden_path = self._fixtures(Path(tmpdir))
            report = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=lambda: _StreamingWorkflow(
                    include_fact=next(outcomes)
                ),
                live_providers=True,
                trials=3,
            )

        self.assertFalse(report["cases"][0]["case_passed"])
        self.assertEqual(report["cases"][0]["first_failure_layer"], "outcome")

    def test_live_trials_reject_single_trial(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            questions_path, golden_path = self._fixtures(Path(tmpdir))
            with self.assertRaisesRegex(ValueError, "at least 3 trials"):
                evaluate_questions(
                    questions_path=questions_path,
                    golden_answers_path=golden_path,
                    workflow_factory=_StreamingWorkflow,
                    live_providers=True,
                    trials=1,
                )

    def test_factory_rejects_reused_singleton(self) -> None:
        singleton = _StreamingWorkflow()
        with tempfile.TemporaryDirectory() as tmpdir:
            questions_path, golden_path = self._fixtures(Path(tmpdir))
            with self.assertRaisesRegex(ValueError, "fresh instance"):
                evaluate_questions(
                    questions_path=questions_path,
                    golden_answers_path=golden_path,
                    workflow_factory=lambda: singleton,
                    live_providers=True,
                    trials=3,
                )

    def test_first_failure_layer_prefers_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            questions_path, golden_path = self._fixtures(Path(tmpdir))
            report = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=lambda: _StreamingWorkflow(sequence=2),
            )

        case = report["cases"][0]
        self.assertEqual(case["first_failure_layer"], "trajectory")
        self.assertEqual(
            case["failure_reasons"]["trajectory"],
            ["non_contiguous_events"],
        )

    def test_malformed_event_becomes_registered_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            questions_path, golden_path = self._fixtures(Path(tmpdir))
            report = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=_MalformedEventWorkflow,
            )

        self.assertEqual(
            report["cases"][0]["failure_reasons"]["trajectory"],
            ["malformed_trace_event"],
        )

    def test_terminal_path_cannot_skip_required_stages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            questions_path, golden_path = self._fixtures(Path(tmpdir))
            report = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=_MissingTerminalPathWorkflow,
            )

        self.assertEqual(
            report["cases"][0]["failure_reasons"]["trajectory"],
            ["terminal_path_mismatch"],
        )

    def test_cache_hit_uses_cache_validation_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            questions_path, golden_path = self._fixtures(root)
            questions = json.loads(questions_path.read_text(encoding="utf-8"))
            questions[0].update(
                {
                    "expected_tools": [],
                    "expect_tool_calls": False,
                    "expected_terminal_status": "answered_from_cache",
                    "expected_response_cache_hit": True,
                }
            )
            questions_path.write_text(json.dumps(questions), encoding="utf-8")
            report = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=_CacheHitWorkflow,
            )

        trial = report["cases"][0]["trials"][0]
        self.assertTrue(trial["layer_results"]["trajectory"])
        self.assertIsNotNone(trial["time_to_first_token_ms"])
        self.assertTrue(trial["case_passed"])

    def test_compatible_baseline_adds_only_active_metric_delta(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            questions_path, golden_path = self._fixtures(root)
            baseline = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=_StreamingWorkflow,
            )
            baseline_path = root / "baseline.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            report = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=_StreamingWorkflow,
                baseline_path=baseline_path,
                provider_profile={"kind": "injected_workflow"},
                dataset_profile={"kind": "caller_managed"},
            )

        metric = report["metric_portfolio"]["goal"]["goal.required_fact_coverage"]
        self.assertEqual(metric["baseline"], 1.0)
        self.assertEqual(metric["delta"], 0.0)
        self.assertNotIn(
            "baseline",
            report["dimensions"]["outcome"]["case_pass_rate"],
        )

    def test_foreign_runtime_baseline_keeps_quality_deltas_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            questions_path, golden_path = self._fixtures(root)
            baseline = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=_StreamingWorkflow,
            )
            baseline["runtime"] = {
                **baseline["runtime"],
                "processor": "another-machine-cpu",
            }
            baseline_path = root / "baseline.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            report = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=_StreamingWorkflow,
                baseline_path=baseline_path,
                provider_profile={"kind": "injected_workflow"},
                dataset_profile={"kind": "caller_managed"},
            )

        metric = report["metric_portfolio"]["goal"]["goal.required_fact_coverage"]
        self.assertEqual(metric["baseline"], 1.0)
        self.assertEqual(metric["delta"], 0.0)
        self.assertFalse(report["baseline"]["runtime_compatible"])
        self.assertEqual(
            report["baseline"]["operational_delta_skipped_reason"],
            "runtime_profile_mismatch",
        )
        latency = report["metric_portfolio"]["operational"][
            "operational.end_to_end_latency_p95"
        ]
        self.assertIsNotNone(latency["value"])
        self.assertIsNone(latency["baseline"])
        self.assertIsNone(latency["delta"])

    def test_v3_portfolio_separates_roles_and_decision_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            questions_path, golden_path = self._fixtures(Path(tmpdir))
            report = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=_StreamingWorkflow,
            )

        self.assertEqual(report["report_version"], "rag-eval-v3")
        portfolio = report["metric_portfolio"]
        self.assertEqual(
            {"goal", "guardrail", "operational"},
            {role for role in ("goal", "guardrail", "operational") if portfolio[role]},
        )
        self.assertEqual(
            portfolio["goal"]["goal.required_fact_coverage"]["decision_status"],
            "pass",
        )
        self.assertEqual(
            portfolio["operational"]["operational.cost_per_request"]["decision_status"],
            "observational",
        )
        permission_metric = portfolio["guardrail"]["guardrail.permission_bypass_count"]
        self.assertIsNone(permission_metric["value"])
        self.assertEqual(permission_metric["decision_status"], "evaluation_incomplete")
        self.assertIn(
            "guardrail.permission_bypass_count",
            portfolio["incomplete_metrics"],
        )
        self.assertEqual(report["operational"]["cost_per_request"], 0.0)
        self.assertGreater(report["operational"]["completed_requests_per_hour"], 0)

    def test_permission_bypass_is_a_blocking_guardrail_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            questions_path, golden_path = self._fixtures(root)
            questions = json.loads(questions_path.read_text(encoding="utf-8"))
            questions[0]["expected_terminal_status"] = "permission_denied"
            questions_path.write_text(json.dumps(questions), encoding="utf-8")
            report = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=_PermissionBypassWorkflow,
            )

        metric = report["metric_portfolio"]["guardrail"][
            "guardrail.permission_bypass_count"
        ]
        self.assertEqual(metric["value"], 1)
        self.assertEqual(metric["decision_status"], "fail")
        self.assertIn(
            "guardrail.permission_bypass_count",
            report["metric_portfolio"]["failed_metrics"],
        )

    def test_permission_denial_without_retrieval_passes_guardrail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            questions_path, golden_path = self._fixtures(root)
            questions = json.loads(questions_path.read_text(encoding="utf-8"))
            questions[0]["expected_terminal_status"] = "permission_denied"
            questions_path.write_text(json.dumps(questions), encoding="utf-8")
            report = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=_PermissionDeniedWorkflow,
            )

        self.assertEqual(report["permission_denial_case_count"], 1)
        metric = report["metric_portfolio"]["guardrail"][
            "guardrail.permission_bypass_count"
        ]
        self.assertEqual(metric["value"], 0)
        self.assertEqual(metric["decision_status"], "pass")

    def test_live_usage_is_incomplete_when_provider_omits_cost_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            questions_path, golden_path = self._fixtures(Path(tmpdir))
            report = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=_StreamingWorkflow,
                live_providers=True,
                trials=3,
            )

        metric = report["metric_portfolio"]["operational"][
            "operational.cost_per_request"
        ]
        self.assertIsNone(metric["value"])
        self.assertEqual(metric["decision_status"], "evaluation_incomplete")

    def test_live_cost_requires_price_profile_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            questions_path, golden_path = self._fixtures(Path(tmpdir))
            report = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=_CostWithoutProfileWorkflow,
                live_providers=True,
                trials=3,
            )

        self.assertIsNone(report["operational"]["cost_per_request"])
        metric = report["metric_portfolio"]["operational"][
            "operational.cost_per_request"
        ]
        self.assertEqual(metric["decision_status"], "evaluation_incomplete")

    def test_non_finite_usage_and_negative_stage_latency_are_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            questions_path, golden_path = self._fixtures(Path(tmpdir))
            report = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=_NonFiniteUsageWorkflow,
                live_providers=True,
                trials=3,
            )

        self.assertIsNone(report["operational"]["cost_per_request"])
        self.assertEqual(report["latency"]["retrieval_latency_ms"]["sample_count"], 0)
        self.assertFalse(report["latency"]["retrieval_latency_ms"]["complete"])
        self.assertIsNone(report["latency"]["retrieval_latency_ms"]["p95"])
        json.dumps(report, allow_nan=False)

    def test_string_department_tool_config_is_one_scope_not_characters(self) -> None:
        tool = SEARCH_TOOLS["search_code_docs"]
        replacement = KnowledgeSearchTool(
            name=tool.name,
            description=tool.description,
            filters={**dict(tool.filters), "department": "engineering"},
            query_hint=tool.query_hint,
        )
        with (
            patch.dict(SEARCH_TOOLS, {"search_code_docs": replacement}),
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            questions_path, golden_path = self._fixtures(Path(tmpdir))
            report = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=_StreamingWorkflow,
            )

        case = report["cases"][0]
        self.assertNotIn(
            "unauthorized_tool_scope",
            case["failure_reasons"]["tool"],
        )
        self.assertTrue(case["layer_results"]["tool"])

    def test_live_run_rejects_deterministic_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            questions_path, golden_path = self._fixtures(root)
            baseline = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=_StreamingWorkflow,
            )
            baseline_path = root / "baseline.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "run profile"):
                evaluate_questions(
                    questions_path=questions_path,
                    golden_answers_path=golden_path,
                    workflow_factory=_StreamingWorkflow,
                    live_providers=True,
                    trials=3,
                    baseline_path=baseline_path,
                    provider_profile={"kind": "injected_workflow"},
                    dataset_profile={"kind": "caller_managed"},
                )

    def test_baseline_rejects_different_metric_registry_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            questions_path, golden_path = self._fixtures(root)
            baseline = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=_StreamingWorkflow,
            )
            baseline_path = root / "baseline.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            registry = json.loads(
                DEFAULT_METRIC_REGISTRY_PATH.read_text(encoding="utf-8")
            )
            registry["metrics"][0]["owner"] = "new-owner"
            registry_path = root / "registry.json"
            registry_path.write_text(json.dumps(registry), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "metric registry"):
                evaluate_questions(
                    questions_path=questions_path,
                    golden_answers_path=golden_path,
                    workflow_factory=_StreamingWorkflow,
                    baseline_path=baseline_path,
                    metric_registry_path=registry_path,
                    provider_profile={"kind": "injected_workflow"},
                    dataset_profile={"kind": "caller_managed"},
                )

    def test_factory_and_query_identity_are_fresh_across_runs(self) -> None:
        instances: list[_RecordingWorkflow] = []
        query_ids: list[str] = []

        def factory() -> _RecordingWorkflow:
            workflow = _RecordingWorkflow(query_ids)
            instances.append(workflow)
            return workflow

        with tempfile.TemporaryDirectory() as tmpdir:
            questions_path, golden_path = self._fixtures(Path(tmpdir))
            for _ in range(2):
                evaluate_questions(
                    questions_path=questions_path,
                    golden_answers_path=golden_path,
                    workflow_factory=factory,
                )

        self.assertEqual(len(instances), 2)
        self.assertIsNot(instances[0], instances[1])
        self.assertEqual(len(set(query_ids)), 2)

    def test_missing_denominators_are_null(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            questions_path = root / "questions.json"
            golden_path = root / "golden.json"
            questions_path.write_text(
                json.dumps(
                    [
                        {
                            "question_id": "case_1",
                            "question": "test question",
                            "expected_sources": [],
                            "expected_tools": ["search_code_docs"],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            golden_path.write_text(
                json.dumps(
                    {
                        "case_1": {
                            "required_facts": [],
                            "required_citations": [],
                        }
                    }
                ),
                encoding="utf-8",
            )
            report = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=_StreamingWorkflow,
            )

        self.assertIsNone(report["recall_at_k"])
        self.assertIsNone(report["entity_resolution_accuracy"])
        self.assertIsNone(
            report["dimensions"]["outcome"]["retrieval_recall_at_k"]["value"]
        )

    def test_missing_tool_invocation_fails_tool_layer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            questions_path, golden_path = self._fixtures(Path(tmpdir))
            report = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=_NoToolWorkflow,
            )

        case = report["cases"][0]
        self.assertEqual(case["first_failure_layer"], "tool")
        self.assertIn("missing_tool_invocation", case["failure_reasons"]["tool"])
        self.assertEqual(
            report["dimensions"]["tool"]["invocation_accuracy"]["value"],
            0.0,
        )

    def test_tool_scope_must_match_permission_and_registered_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            questions_path, golden_path = self._fixtures(Path(tmpdir))
            report = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=_UnauthorizedToolWorkflow,
            )

        case = report["cases"][0]
        self.assertEqual(case["first_failure_layer"], "tool")
        self.assertIn("unauthorized_tool_scope", case["failure_reasons"]["tool"])

    def test_supplementary_plan_is_bounded_per_round(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            questions_path, golden_path = self._fixtures(Path(tmpdir))
            report = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=_UnboundedSupplementaryWorkflow,
            )

        self.assertIn(
            "supplementary_plan_bound_exceeded",
            report["cases"][0]["failure_reasons"]["tool"],
        )

    def test_selected_context_recall_uses_only_first_five(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            questions_path, golden_path = self._fixtures(root)
            questions = json.loads(questions_path.read_text(encoding="utf-8"))
            questions[0]["expected_sources"] = ["chunk_6"]
            questions_path.write_text(json.dumps(questions), encoding="utf-8")
            golden = json.loads(golden_path.read_text(encoding="utf-8"))
            golden["case_1"]["required_citations"] = ["chunk_6"]
            golden_path.write_text(json.dumps(golden), encoding="utf-8")
            report = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=_OverSelectionWorkflow,
            )

        self.assertEqual(report["selected_context_recall_at_5"], 0.0)
        self.assertEqual(report["citation_coverage"], 1.0)
        self.assertFalse(report["cases"][0]["case_passed"])
        self.assertEqual(
            report["cases"][0]["failure_reasons"]["outcome"],
            ["citation_not_selected", "selected_context_bound_exceeded"],
        )

    def test_exhaustive_query_may_release_sixteen_sibling_contexts(self) -> None:
        """Spec 12 § 5.9 caps focused answers at 5 and exhaustive ones at 16."""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            questions_path, golden_path = self._fixtures(root)
            questions = json.loads(questions_path.read_text(encoding="utf-8"))
            questions[0]["expected_sources"] = ["chunk_1", "chunk_16"]
            questions_path.write_text(json.dumps(questions), encoding="utf-8")
            golden = json.loads(golden_path.read_text(encoding="utf-8"))
            golden["case_1"]["required_citations"] = ["chunk_1", "chunk_16"]
            golden_path.write_text(json.dumps(golden), encoding="utf-8")
            report = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=_ExhaustiveWorkflow,
            )

        case = report["cases"][0]
        self.assertEqual(case["failure_reasons"]["outcome"], [])
        self.assertTrue(case["case_passed"])
        # The gated goal metric keeps its top-5 window despite the wider bound:
        # only chunk_1 of the two expected sources is inside the first five.
        self.assertEqual(report["selected_context_recall_at_5"], 0.5)

    def test_exhaustive_query_beyond_sixteen_contexts_still_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            questions_path, golden_path = self._fixtures(root)
            questions = json.loads(questions_path.read_text(encoding="utf-8"))
            questions[0]["expected_sources"] = ["chunk_1", "chunk_17"]
            questions_path.write_text(json.dumps(questions), encoding="utf-8")
            golden = json.loads(golden_path.read_text(encoding="utf-8"))
            golden["case_1"]["required_citations"] = ["chunk_1", "chunk_17"]
            golden_path.write_text(json.dumps(golden), encoding="utf-8")
            report = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=lambda: _ExhaustiveWorkflow(context_count=17),
            )

        case = report["cases"][0]
        self.assertFalse(case["case_passed"])
        self.assertEqual(
            case["failure_reasons"]["outcome"],
            ["citation_not_selected", "selected_context_bound_exceeded"],
        )

    def test_abstention_may_not_project_a_generation_context(self) -> None:
        """Spec 70 § 4.6 grades the forbidden half of a terminal stage path."""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            questions_path, golden_path = self._fixtures(root)
            questions = json.loads(questions_path.read_text(encoding="utf-8"))
            questions[0]["should_abstain"] = True
            questions[0]["expected_terminal_status"] = "abstained"
            questions_path.write_text(json.dumps(questions), encoding="utf-8")
            report = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=_AbstainWithGenerationContextWorkflow,
            )

        case = report["cases"][0]
        self.assertFalse(case["case_passed"])
        self.assertIn(
            "forbidden_stage_present",
            case["failure_reasons"]["trajectory"],
        )
        self.assertEqual(case["first_failure_layer"], "trajectory")

    def test_invalid_compression_provenance_fails_the_case(self) -> None:
        """Spec 70 § 4.6 makes compression provenance an outcome judgement."""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            questions_path, golden_path = self._fixtures(root)
            report = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=_InvalidCompressionWorkflow,
            )

        case = report["cases"][0]
        self.assertFalse(case["case_passed"])
        self.assertEqual(
            case["failure_reasons"]["outcome"],
            ["compression_provenance_invalid"],
        )

    def test_absent_compression_is_not_an_outcome_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            questions_path, golden_path = self._fixtures(root)
            report = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=_StreamingWorkflow,
            )

        case = report["cases"][0]
        self.assertEqual(case["failure_reasons"]["outcome"], [])
        self.assertTrue(case["case_passed"])

    def test_fixture_pins_the_transformation_strategy_and_roles(self) -> None:
        """Spec 70 § 4.6 grades strategy and item roles against the fixture."""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            questions_path, golden_path = self._fixtures(root)
            questions = json.loads(questions_path.read_text(encoding="utf-8"))
            questions[0]["expected_transformation_strategy"] = "step_back"
            questions[0]["expected_query_roles"] = ["background", "primary"]
            questions[0]["expected_plan_item_count"] = 2
            questions_path.write_text(json.dumps(questions), encoding="utf-8")

            matching = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=_StepBackWorkflow,
            )
            # The default double runs `identity` with a single primary item.
            diverging = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=_StreamingWorkflow,
            )

        self.assertTrue(matching["cases"][0]["case_passed"])
        self.assertEqual(matching["cases"][0]["failure_reasons"]["trajectory"], [])
        self.assertFalse(diverging["cases"][0]["case_passed"])
        self.assertEqual(
            diverging["cases"][0]["failure_reasons"]["trajectory"],
            # `layer_reasons` sorts each layer's registered codes.
            [
                "plan_item_count_mismatch",
                "query_roles_mismatch",
                "transformation_strategy_mismatch",
            ],
        )
        self.assertEqual(diverging["cases"][0]["first_failure_layer"], "trajectory")

    def test_fixture_pins_the_compression_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            questions_path, golden_path = self._fixtures(root)
            questions = json.loads(questions_path.read_text(encoding="utf-8"))
            questions[0]["expected_compression_mode"] = "selective"
            questions[0]["expected_compression_fallback"] = False
            questions_path.write_text(json.dumps(questions), encoding="utf-8")

            matching = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=_SelectiveCompressionWorkflow,
            )
            # The default double leaves compression disabled.
            diverging = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=_StreamingWorkflow,
            )

        self.assertTrue(matching["cases"][0]["case_passed"])
        self.assertFalse(diverging["cases"][0]["case_passed"])
        self.assertEqual(
            diverging["cases"][0]["failure_reasons"]["outcome"],
            ["compression_path_mismatch"],
        )

    def test_unpinned_fixture_ignores_transformation_and_compression(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            questions_path, golden_path = self._fixtures(root)
            report = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=_StepBackWorkflow,
            )

        self.assertTrue(report["cases"][0]["case_passed"])

    def test_invalid_transformation_expectations_fail_closed(self) -> None:
        cases = (
            {"expected_transformation_strategy": "summarise"},
            {"expected_query_roles": ["primary", "unknown_role"]},
            {"expected_plan_item_count": 4},
            {"expected_compression_mode": "auto"},
            # roles and plan count must agree with each other
            {"expected_query_roles": ["primary"], "expected_plan_item_count": 2},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                questions_path, golden_path = self._fixtures(root)
                questions = json.loads(questions_path.read_text(encoding="utf-8"))
                questions[0].update(overrides)
                questions_path.write_text(json.dumps(questions), encoding="utf-8")
                with self.assertRaises(ValueError):
                    evaluate_questions(
                        questions_path=questions_path,
                        golden_answers_path=golden_path,
                        workflow_factory=_StreamingWorkflow,
                    )

    def test_cache_hit_keeps_citations_without_a_selected_context(self) -> None:
        """Spec 10c § 6: a hit skips selection but must preserve its citations."""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            questions_path, golden_path = self._fixtures(root)
            questions = json.loads(questions_path.read_text(encoding="utf-8"))
            questions[0]["expected_terminal_status"] = "answered_from_cache"
            # A hit invokes no tools (spec 10c § 6).
            questions[0]["expected_tools"] = []
            questions[0]["expect_tool_calls"] = False
            questions_path.write_text(json.dumps(questions), encoding="utf-8")
            report = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=_CacheHitWorkflow,
            )

        case = report["cases"][0]
        self.assertEqual(case["failure_reasons"]["outcome"], [])
        self.assertTrue(case["case_passed"])
        # No retrieval ran, so the retrieval recalls have no denominator and
        # must not be scored as zero.
        self.assertIsNone(report["recall_at_k"])
        self.assertIsNone(report["selected_context_recall_at_5"])
        self.assertEqual(report["citation_coverage"], 1.0)

    def test_cache_hit_without_a_citation_fails(self) -> None:
        class _UncitedCacheHitWorkflow(_CacheHitWorkflow):
            def _response(self, query_id: str) -> dict[str, Any]:
                response = super()._response(query_id)
                response["citations"] = []
                return response

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            questions_path, golden_path = self._fixtures(root)
            questions = json.loads(questions_path.read_text(encoding="utf-8"))
            questions[0]["expected_terminal_status"] = "answered_from_cache"
            questions[0]["expected_tools"] = []
            questions[0]["expect_tool_calls"] = False
            questions_path.write_text(json.dumps(questions), encoding="utf-8")
            report = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=_UncitedCacheHitWorkflow,
            )

        self.assertIn(
            "citation_not_selected",
            report["cases"][0]["failure_reasons"]["outcome"],
        )

    def test_scenario_prelude_replays_turns_in_the_graded_session(self) -> None:
        """Spec 70 § 3 controlled pre-state, built from ordinary turns."""

        seen: list[tuple[str, str | None]] = []

        class _RecordingPreludeWorkflow(_StreamingWorkflow):
            def stream(
                self,
                question: str,
                filters: dict[str, Any] | None = None,
                *,
                session_id: str | None = None,
                query_id: str | None = None,
            ) -> Iterable[dict[str, Any]]:
                seen.append((question, session_id))
                return super().stream(
                    question,
                    filters,
                    session_id=session_id,
                    query_id=query_id,
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            questions_path, golden_path = self._fixtures(root)
            questions = json.loads(questions_path.read_text(encoding="utf-8"))
            questions[0]["scenario"] = {"prelude": ["warm up", "second warm up"]}
            questions_path.write_text(json.dumps(questions), encoding="utf-8")
            report = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=_RecordingPreludeWorkflow,
            )

        self.assertEqual(
            [question for question, _ in seen],
            ["warm up", "second warm up", "test question"],
        )
        # One session across the prelude and the graded turn.
        self.assertEqual(len({session for _, session in seen}), 1)
        # Only the graded turn is reported.
        self.assertEqual(len(report["cases"]), 1)
        self.assertTrue(report["cases"][0]["case_passed"])

    def test_prelude_only_scenario_needs_no_injecting_factory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            questions_path, golden_path = self._fixtures(root)
            questions = json.loads(questions_path.read_text(encoding="utf-8"))
            questions[0]["scenario"] = {"prelude": ["warm up"]}
            questions_path.write_text(json.dumps(questions), encoding="utf-8")
            report = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=_StreamingWorkflow,
            )

        self.assertTrue(report["cases"][0]["case_passed"])

    def test_permission_scenario_still_requires_the_injecting_factory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            questions_path, golden_path = self._fixtures(root)
            questions = json.loads(questions_path.read_text(encoding="utf-8"))
            questions[0]["scenario"] = {"permission": "deny"}
            questions_path.write_text(json.dumps(questions), encoding="utf-8")
            with self.assertRaises(ValueError):
                evaluate_questions(
                    questions_path=questions_path,
                    golden_answers_path=golden_path,
                    workflow_factory=_StreamingWorkflow,
                )

    def test_invalid_scenario_prelude_fails_closed(self) -> None:
        scenarios: tuple[dict[str, Any], ...] = (
            {"prelude": []},
            {"prelude": ["a", "b", "c", "d"]},
            {"prelude": [""]},
            {"prelude": "warm up"},
            {"prelude": ["warm up"], "unknown": 1},
        )
        for scenario in scenarios:
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                questions_path, golden_path = self._fixtures(root)
                questions = json.loads(questions_path.read_text(encoding="utf-8"))
                questions[0]["scenario"] = scenario
                questions_path.write_text(json.dumps(questions), encoding="utf-8")
                with self.assertRaises(ValueError):
                    evaluate_questions(
                        questions_path=questions_path,
                        golden_answers_path=golden_path,
                        workflow_factory=_StreamingWorkflow,
                    )

    def test_fixture_pins_whether_the_reranker_fell_back(self) -> None:
        """Spec 12 § 5.6 exposes which implementation actually ranked."""

        class _FallbackRerankWorkflow(_StreamingWorkflow):
            def _response(self, query_id: str) -> dict[str, Any]:
                response = super()._response(query_id)
                response["reranker_name"] = "rule-based-reranker"
                response["reranker_fallback_reason"] = "not_configured"
                return response

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            questions_path, golden_path = self._fixtures(root)
            questions = json.loads(questions_path.read_text(encoding="utf-8"))
            questions[0]["expected_reranker_fallback"] = True
            questions_path.write_text(json.dumps(questions), encoding="utf-8")

            degraded = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=_FallbackRerankWorkflow,
            )
            # The default double reports no fallback.
            healthy = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=_StreamingWorkflow,
            )

        self.assertTrue(degraded["cases"][0]["case_passed"])
        self.assertFalse(healthy["cases"][0]["case_passed"])
        self.assertEqual(
            healthy["cases"][0]["failure_reasons"]["trajectory"],
            ["reranker_fallback_mismatch"],
        )

    def test_unregistered_reranker_fallback_reason_fails(self) -> None:
        class _LeakyRerankWorkflow(_StreamingWorkflow):
            def _response(self, query_id: str) -> dict[str, Any]:
                response = super()._response(query_id)
                response["reranker_fallback_reason"] = "Connection refused by host"
                return response

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            questions_path, golden_path = self._fixtures(root)
            questions = json.loads(questions_path.read_text(encoding="utf-8"))
            questions[0]["expected_reranker_fallback"] = True
            questions_path.write_text(json.dumps(questions), encoding="utf-8")
            report = evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                workflow_factory=_LeakyRerankWorkflow,
            )

        self.assertEqual(
            report["cases"][0]["failure_reasons"]["trajectory"],
            ["unregistered_reranker_fallback_reason"],
        )

    def test_scenario_reranker_reaches_the_injecting_factory(self) -> None:
        seen: list[dict[str, str]] = []

        def factory(scenario: dict[str, str]) -> _StreamingWorkflow:
            seen.append(dict(scenario))
            return _StreamingWorkflow()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            questions_path, golden_path = self._fixtures(root)
            questions = json.loads(questions_path.read_text(encoding="utf-8"))
            questions[0]["scenario"] = {"reranker": "fallback"}
            questions_path.write_text(json.dumps(questions), encoding="utf-8")
            evaluate_questions(
                questions_path=questions_path,
                golden_answers_path=golden_path,
                scenario_workflow_factory=factory,
            )

        self.assertEqual(seen, [{"permission": "allow", "reranker": "fallback"}])

    def test_scenario_reranker_requires_the_injecting_factory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            questions_path, golden_path = self._fixtures(root)
            questions = json.loads(questions_path.read_text(encoding="utf-8"))
            questions[0]["scenario"] = {"reranker": "fallback"}
            questions_path.write_text(json.dumps(questions), encoding="utf-8")
            with self.assertRaises(ValueError):
                evaluate_questions(
                    questions_path=questions_path,
                    golden_answers_path=golden_path,
                    workflow_factory=_StreamingWorkflow,
                )

    def test_invalid_scenario_reranker_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            questions_path, golden_path = self._fixtures(root)
            questions = json.loads(questions_path.read_text(encoding="utf-8"))
            questions[0]["scenario"] = {"reranker": "openai"}
            questions_path.write_text(json.dumps(questions), encoding="utf-8")
            with self.assertRaises(ValueError):
                evaluate_questions(
                    questions_path=questions_path,
                    golden_answers_path=golden_path,
                    scenario_workflow_factory=lambda scenario: _StreamingWorkflow(),
                )

    def test_stale_transcript_review_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            questions_path, golden_path = self._fixtures(root)
            review_path = root / "review.json"
            review_path.write_text(
                json.dumps(
                    {
                        "schema_version": "rag-eval-review-v1",
                        "reviews": [
                            {
                                "question_id": "case_1",
                                "failure_layer": "outcome",
                                "reason_code": "required_fact_missing",
                                "attribution": "task_quality",
                                "owner": "workshop-author",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "absent from this run"):
                evaluate_questions(
                    questions_path=questions_path,
                    golden_answers_path=golden_path,
                    workflow_factory=_StreamingWorkflow,
                    review_path=review_path,
                )

    def test_question_fixture_rejects_unknown_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            questions_path, golden_path = self._fixtures(root)
            questions = json.loads(questions_path.read_text(encoding="utf-8"))
            questions[0]["shortcut_score"] = 1.0
            questions_path.write_text(json.dumps(questions), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "fields are invalid"):
                evaluate_questions(
                    questions_path=questions_path,
                    golden_answers_path=golden_path,
                    workflow_factory=_StreamingWorkflow,
                )

    def test_question_fixture_rejects_unknown_scenario(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            questions_path, golden_path = self._fixtures(root)
            questions = json.loads(questions_path.read_text(encoding="utf-8"))
            questions[0]["scenario"] = {"permission": "bypass"}
            questions_path.write_text(json.dumps(questions), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "scenario is invalid"):
                evaluate_questions(
                    questions_path=questions_path,
                    golden_answers_path=golden_path,
                    workflow_factory=_StreamingWorkflow,
                )

    def test_scenario_case_requires_scenario_aware_factory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            questions_path, golden_path = self._fixtures(root)
            questions = json.loads(questions_path.read_text(encoding="utf-8"))
            questions[0]["scenario"] = {"permission": "deny"}
            questions_path.write_text(json.dumps(questions), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "scenario_workflow_factory"):
                evaluate_questions(
                    questions_path=questions_path,
                    golden_answers_path=golden_path,
                    workflow_factory=_StreamingWorkflow,
                )


if __name__ == "__main__":
    unittest.main()
