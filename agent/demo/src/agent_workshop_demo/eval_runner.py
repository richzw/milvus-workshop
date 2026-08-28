"""Golden-question evaluation for retrieval and citation behavior."""

from __future__ import annotations

import ast
import json
import math
import platform
import re
import sys
import unicodedata
import uuid
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, Protocol, cast

from agent_workshop_demo.config import (
    DEFAULT_SEARCH_PARAMS,
    MAX_EXHAUSTIVE_CONTEXTS,
)
from agent_workshop_demo.eval_governance import (
    MetricDefinition,
    MetricRegistry,
    load_metric_registry,
)
from agent_workshop_demo.reranker import RERANK_FALLBACK_REASONS
from agent_workshop_demo.knowledge_tools import (
    ALL_DEPARTMENTS,
    REGISTERED_TOOL_NAMES,
    SEARCH_TOOLS,
    KnowledgeSearchTool,
    PermissionDecision,
)
from agent_workshop_demo.validation import normalize_filters

REPORT_VERSION = "rag-eval-v3"
GRADER = "L1_programmatic"
FAILURE_LAYERS = ("trajectory", "tool", "outcome")
TOOL_INVOCATION_REASONS = frozenset(
    {
        "forbidden_tool_invocation",
        "missing_tool_invocation",
        "tool_call_selection_mismatch",
    }
)
TOOL_NON_STRUCTURAL_REASONS = frozenset(
    {
        "entity_resolution_mismatch",
        "unexpected_tools",
        "version_scope_mismatch",
        *TOOL_INVOCATION_REASONS,
    }
)
TRANSFORMATION_STRATEGIES = frozenset(
    {"decompose", "identity", "rewrite", "step_back"}
)
QUERY_ROLES = frozenset({"aspect", "background", "hop", "primary"})
COMPRESSION_MODES = frozenset(
    {"disabled", "extraction", "selective", "summary"}
)
SCENARIO_RERANKERS = frozenset({"fallback", "rule_based"})
TERMINAL_STATUSES = frozenset(
    {
        "abstained",
        "answered",
        "answered_from_cache",
        "answered_from_memory",
        "answered_without_retrieval",
        "clarification_required",
        "memory_not_found",
        "memory_saved",
        "memory_write_failed",
        "permission_denied",
        "refused_unsupported_operation",
    }
)
STAGE_ORDER = {
    name: index
    for index, name in enumerate(
        (
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
    )
}


class EvaluationWorkflow(Protocol):
    """Minimal workflow surface consumed by the evaluator."""

    def run(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        *,
        session_id: str | None = None,
        query_id: str | None = None,
    ) -> dict[str, Any]: ...


class EvalScenarioPermissionChecker:
    """Apply only the strict, fixture-declared eval permission scenario."""

    def __init__(self, *, allowed: bool) -> None:
        self.allowed = allowed

    def check(
        self,
        *,
        session_id: str,
        intent: str,
        query_type: str,
    ) -> PermissionDecision:
        """Return the fixture decision without consulting the question."""

        del session_id, intent, query_type
        return PermissionDecision(
            allowed=self.allowed,
            allowed_departments=ALL_DEPARTMENTS if self.allowed else (),
            reason=(
                "Eval fixture grants synthetic corpus access."
                if self.allowed
                else "Eval fixture denies synthetic corpus access."
            ),
            checker_name="eval-scenario-permission",
        )


def evaluate_questions(
    *,
    questions_path: Path,
    golden_answers_path: Path | None = None,
    workflow_factory: Callable[[], EvaluationWorkflow] | None = None,
    scenario_workflow_factory: (
        Callable[[dict[str, str]], EvaluationWorkflow] | None
    ) = None,
    top_k: int = 20,
    trials: int = 1,
    live_providers: bool = False,
    baseline_path: Path | None = None,
    review_path: Path | None = None,
    metric_registry_path: Path | None = None,
    provider_profile: dict[str, str] | None = None,
    dataset_profile: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Evaluate layered RAG behavior with isolated, repeatable trials."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if (workflow_factory is None) == (scenario_workflow_factory is None):
        raise ValueError(
            "exactly one workflow_factory or scenario_workflow_factory is required"
        )
    if trials <= 0:
        raise ValueError("trials must be greater than zero")
    if live_providers and trials < 3:
        raise ValueError("live provider evaluation requires at least 3 trials")
    if not live_providers and trials != 1:
        raise ValueError("deterministic evaluation runs exactly one trial")
    if baseline_path is not None and (
        provider_profile is None or dataset_profile is None
    ):
        raise ValueError(
            "baseline comparison requires explicit provider and dataset profiles"
        )
    metric_registry = load_metric_registry(metric_registry_path)
    questions = _validate_questions(
        json.loads(questions_path.read_text(encoding="utf-8"))
    )
    if scenario_workflow_factory is None and any(
        _question_scenario(question) != {"permission": "allow", "reranker": "rule_based"}
        for question in questions
    ):
        raise ValueError("scenario cases require scenario_workflow_factory")
    question_ids = {
        str(question["question_id"])
        for question in questions
        if isinstance(question, dict) and "question_id" in question
    }
    if len(question_ids) != len(questions):
        raise ValueError("questions must have unique question_id values")

    resolved_golden_path = golden_answers_path
    if resolved_golden_path is None:
        candidate = questions_path.with_name("golden_answers.yaml")
        resolved_golden_path = candidate if candidate.exists() else None
    golden_answers = (
        _load_golden_answers(resolved_golden_path)
        if resolved_golden_path is not None
        else {}
    )
    if golden_answers and set(golden_answers) != question_ids:
        raise ValueError(
            "questions and golden answers must contain identical question IDs"
        )

    factory = workflow_factory
    scenario_factory = scenario_workflow_factory
    cases: list[dict[str, Any]] = []
    workflow_instances: list[EvaluationWorkflow] = []
    run_nonce = uuid.uuid4().hex[:12]
    evaluation_started = perf_counter()

    for question_index, question in enumerate(questions, start=1):
        question_id = str(question["question_id"])
        golden = golden_answers.get(question_id, {})
        filters = question.get("metadata_filters") or {}
        question_sources = set(question.get("expected_sources", []))
        expected = set(golden.get("required_citations", question_sources))
        if golden and expected != question_sources:
            raise ValueError(
                f"Question {question_id!r} expected_sources do not match "
                "golden required_citations"
            )
        trial_results: list[dict[str, Any]] = []
        for trial_number in range(1, trials + 1):
            runner = (
                scenario_factory(_question_scenario(question))
                if scenario_factory is not None
                else cast(Callable[[], EvaluationWorkflow], factory)()
            )
            if any(runner is existing for existing in workflow_instances):
                raise ValueError("workflow_factory must return a fresh instance")
            workflow_instances.append(runner)
            query_id = f"eval_{run_nonce}_{question_index}_{trial_number}"
            session_id = f"eval_session_{run_nonce}_{question_index}_{trial_number}"
            for prelude_index, prelude_question in enumerate(
                _question_prelude(question), start=1
            ):
                _execute_trial(
                    runner,
                    prelude_question,
                    filters=_expand_filters(filters),
                    session_id=session_id,
                    query_id=f"{query_id}_prelude_{prelude_index}",
                )
            response, stream_events, ttft_ms, observed_latency_ms = _execute_trial(
                runner,
                str(question["question"]),
                filters=_expand_filters(filters),
                session_id=session_id,
                query_id=query_id,
            )
            trial_results.append(
                _evaluate_trial(
                    question=question,
                    golden=golden,
                    expected=expected,
                    response=response,
                    stream_events=stream_events,
                    query_id=query_id,
                    top_k=top_k,
                    trial_number=trial_number,
                    time_to_first_token_ms=ttft_ms,
                    observed_latency_ms=observed_latency_ms,
                )
            )
        representative = _summarize_case_trials(trial_results)
        trial_passes = [bool(item["case_passed"]) for item in trial_results]
        representative["trials"] = trial_results
        representative["trial_count"] = trials
        representative["pass_at_k"] = any(trial_passes) if live_providers else None
        representative["pass_power_k"] = all(trial_passes) if live_providers else None
        cases.append(representative)

    all_trials = [trial for case in cases for trial in case["trials"]]
    evaluation_elapsed_seconds = max(perf_counter() - evaluation_started, 1e-9)
    total_trials = len(all_trials) or 1
    aggregates = {
        "num_questions": len(cases),
        "num_trials": len(all_trials),
        "recall_at_k": _mean(item["recall_at_k"] for item in all_trials),
        "reranked_recall_at_8": _mean(
            item["reranked_recall_at_8"] for item in all_trials
        ),
        "selected_context_recall_at_5": _mean(
            item["selected_context_recall_at_5"] for item in all_trials
        ),
        "citation_coverage": _mean(item["citation_coverage"] for item in all_trials),
        "citation_precision": _mean(item["citation_precision"] for item in all_trials),
        "citation_resolve_rate": _mean(
            1.0 if item["citation_subset_valid"] else 0.0 for item in all_trials
        ),
        "required_fact_coverage": _mean(
            item["required_fact_coverage"] for item in all_trials
        ),
        "abstention_accuracy": round(
            sum(1 for item in all_trials if item["abstention_correct"]) / total_trials,
            4,
        ),
        "tool_selection_accuracy": round(
            sum(1 for item in all_trials if item["tool_selection_correct"])
            / total_trials,
            4,
        ),
        "tool_invocation_accuracy": round(
            sum(1 for item in all_trials if item["tool_invocation_correct"])
            / total_trials,
            4,
        ),
        "entity_resolution_accuracy": _mean(
            (
                1.0
                if item["entity_resolution_correct"]
                else (0.0 if item["entity_resolution_correct"] is not None else None)
            )
            for item in all_trials
        ),
        "version_scope_accuracy": _mean(
            (
                1.0
                if item["version_scope_correct"]
                else (0.0 if item["version_scope_correct"] is not None else None)
            )
            for item in all_trials
        ),
        "cross_version_contamination_count": sum(
            int(item["cross_version_contamination_count"]) for item in all_trials
        ),
        "permission_denial_case_count": sum(
            int(item["permission_guardrail_applicable"]) for item in all_trials
        ),
        "permission_bypass_count": (
            sum(
                int(item["permission_bypass_count"])
                for item in all_trials
                if item["permission_guardrail_applicable"]
            )
            if any(item["permission_guardrail_applicable"] for item in all_trials)
            else None
        ),
        "query_transformation_contract_rate": _mean(
            item["query_transformation_contract_valid"] for item in all_trials
        ),
        "original_query_retention_rate": _mean(
            item["original_query_retained"] for item in all_trials
        ),
        "step_back_primary_coverage": _mean(
            item["step_back_primary_present"] for item in all_trials
        ),
        "context_compression_provenance_rate": _mean(
            item["context_compression_provenance_valid"] for item in all_trials
        ),
        "context_compression_reduction_ratio": _mean(
            item["context_compression_reduction_ratio"] for item in all_trials
        ),
        "context_compression_fallback_rate": _mean(
            item["context_compression_fallback"] for item in all_trials
        ),
        "enough_evidence_rate": round(
            sum(1 for item in all_trials if item["enough_evidence"]) / total_trials,
            4,
        ),
        "cases": cases,
    }
    report = {
        "report_version": REPORT_VERSION,
        "evaluation": {
            "mode": "live_providers" if live_providers else "deterministic",
            "trials_per_question": trials,
            "top_k": top_k,
            "question_ids": [str(item["question_id"]) for item in cases],
            "grader_layer": GRADER,
            "provider_profile": _validated_profile(
                provider_profile,
                default_kind="injected_workflow",
            ),
            "dataset_profile": _validated_profile(
                dataset_profile,
                default_kind="caller_managed",
            ),
            "observed_provider_profile": _observed_provider_profile(all_trials),
        },
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.system(),
            "machine": platform.machine(),
            "processor": _processor_identity(),
            "memory_mb": _memory_megabytes(),
            "byteorder": sys.byteorder,
        },
        **aggregates,
    }
    report["dimensions"] = _dimension_report(report)
    report["reliability"] = _reliability_report(cases, live_providers, trials)
    report["latency"] = _latency_report(cases)
    report["operational"] = _operational_report(
        all_trials,
        live_providers=live_providers,
        elapsed_seconds=evaluation_elapsed_seconds,
    )
    report["transcript_review"] = _transcript_review(cases, review_path)
    report["metric_portfolio"] = _metric_portfolio(
        report,
        metric_registry,
        live_providers=live_providers,
    )
    _attach_baseline(report, baseline_path)
    return report


def _summarize_case_trials(
    trial_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build an order-independent case summary over every trial."""

    first = trial_results[0]
    summary = {
        name: first[name]
        for name in (
            "question_id",
            "category",
            "expected_sources",
            "required_facts",
            "should_abstain",
            "expected_tools",
            "expected_entities",
            "expected_version_scope",
            "expected_doc_versions",
        )
    }
    for name in (
        "recall_at_k",
        "reranked_recall_at_8",
        "selected_context_recall_at_5",
        "citation_coverage",
        "citation_precision",
        "required_fact_coverage",
        "latency_ms",
        "retrieval_latency_ms",
        "rerank_latency_ms",
        "generation_latency_ms",
        "time_to_first_token_ms",
    ):
        summary[name] = _mean(item[name] for item in trial_results)
    summary["layer_results"] = {
        layer: all(item["layer_results"][layer] for item in trial_results)
        for layer in FAILURE_LAYERS
    }
    summary["failure_reasons"] = {
        layer: sorted(
            {
                reason
                for item in trial_results
                for reason in item["failure_reasons"][layer]
            }
        )
        for layer in FAILURE_LAYERS
    }
    summary["first_failure_layer"] = next(
        (layer for layer in FAILURE_LAYERS if not summary["layer_results"][layer]),
        None,
    )
    summary["case_passed"] = all(item["case_passed"] for item in trial_results)
    summary["terminal_statuses"] = sorted(
        {str(item["terminal_status"]) for item in trial_results}
    )
    return summary


def _observed_provider_profile(
    trials: list[dict[str, Any]],
) -> dict[str, str]:
    keys = (
        "classifier",
        "reranker",
        "generator",
        "memory_selector",
        "query_transformer",
        "context_compressor",
    )
    return {
        key: ",".join(
            sorted(
                {
                    str(trial["provider_observations"].get(key, "not_reported"))
                    for trial in trials
                }
            )
        )
        for key in keys
    }


def _validated_profile(
    profile: dict[str, str] | None,
    *,
    default_kind: str,
) -> dict[str, str]:
    output = {"kind": default_kind} if profile is None else dict(profile)
    if not output or len(output) > 16:
        raise ValueError("eval profile must contain 1 to 16 fields")
    if any(
        not isinstance(key, str)
        or not key
        or len(key) > 64
        or not isinstance(value, str)
        or not value
        or len(value) > 128
        for key, value in output.items()
    ):
        raise ValueError("eval profile fields must be bounded non-empty strings")
    return dict(sorted(output.items()))


def _processor_identity() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.casefold().startswith("model name") and ":" in line:
                value = line.split(":", 1)[1].strip()
                if value:
                    return value[:256]
    except OSError:
        pass
    return (platform.processor() or platform.machine() or "unknown")[:256]


def _memory_megabytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                kibibytes = int(line.split()[1])
                return round(kibibytes / 1024)
    except (OSError, ValueError, IndexError):
        pass
    return None


def _execute_trial(
    workflow: EvaluationWorkflow,
    question: str,
    *,
    filters: dict[str, Any],
    session_id: str,
    query_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], float | None, float]:
    """Run one isolated trial and retain only bounded stream envelopes."""

    stream_method = getattr(workflow, "stream", None)
    if not callable(stream_method):
        started = perf_counter()
        response = workflow.run(
            question,
            filters=filters,
            session_id=session_id,
            query_id=query_id,
        )
        elapsed_ms = round((perf_counter() - started) * 1000, 4)
        return response, [], None, elapsed_ms
    started = perf_counter()
    first_token_ms: float | None = None
    verification_seen = False
    events: list[dict[str, Any]] = []
    for event in stream_method(
        question,
        filters=filters,
        session_id=session_id,
        query_id=query_id,
    ):
        if not isinstance(event, dict):
            raise ValueError("evaluation stream envelopes must be mappings")
        events.append(event)
        trace_event = event.get("event")
        if (
            event.get("type") == "trace_event"
            and isinstance(trace_event, dict)
            and trace_event.get("kind") == "stage_completed"
            and trace_event.get("stage") in {"try_grounded_cache", "verify_answer"}
        ):
            verification_seen = True
        if (
            event.get("type") == "answer_delta"
            and first_token_ms is None
            and verification_seen
            and isinstance(event.get("text"), str)
            and bool(str(event["text"]))
        ):
            first_token_ms = round((perf_counter() - started) * 1000, 4)
    finals = [item for item in events if item.get("type") == "final"]
    if len(finals) != 1 or not isinstance(finals[0].get("response"), dict):
        raise ValueError("evaluation stream must contain exactly one final response")
    elapsed_ms = round((perf_counter() - started) * 1000, 4)
    return dict(finals[0]["response"]), events, first_token_ms, elapsed_ms


def _evaluate_trial(
    *,
    question: dict[str, Any],
    golden: dict[str, Any],
    expected: set[str],
    response: dict[str, Any],
    stream_events: list[dict[str, Any]],
    query_id: str,
    top_k: int,
    trial_number: int,
    time_to_first_token_ms: float | None,
    observed_latency_ms: float,
) -> dict[str, Any]:
    """Apply independent L1 graders and identify the first failing layer."""

    terminal_status = str(response.get("terminal_status", ""))
    recalled = {
        str(item["chunk_id"]) for item in response.get("milvus_recalled", [])[:top_k]
    }
    reranked = {str(item["chunk_id"]) for item in response.get("reranked", [])[:8]}
    selected_items = [
        item for item in response.get("reranked", []) if item.get("selected")
    ]
    answer_context_limit = _answer_context_limit(response)
    # `selected` stays the top-5 window because it feeds the gated
    # `selected_context_recall_at_5` goal metric. Citation and bound grading use
    # every released context, which is up to 16 for an exhaustive query.
    selected = {str(item["chunk_id"]) for item in selected_items[:5]}
    released_contexts = {
        str(item["chunk_id"]) for item in selected_items[:answer_context_limit]
    }
    cited = {str(item["chunk_id"]) for item in response.get("citations", [])}
    expected_recalled = expected.intersection(recalled)
    expected_reranked = expected.intersection(reranked)
    expected_selected = expected.intersection(selected)
    expected_cited = expected.intersection(cited)
    # A cache hit runs no retrieval, rerank or selection stage, so it has no
    # applicable denominator for the three retrieval recalls. Spec 70 § 4.0a
    # forbids scoring a missing sample as 0, which would penalise the very
    # short-circuit 10c § 6 requires. Citation coverage still applies.
    retrieval_ran = expected and terminal_status != "answered_from_cache"
    recall = len(expected_recalled) / len(expected) if retrieval_ran else None
    reranked_recall = (
        len(expected_reranked) / len(expected) if retrieval_ran else None
    )
    selected_recall = len(expected_selected) / len(expected) if retrieval_ran else None
    coverage = len(expected_cited) / len(expected) if expected else None
    precision = (
        len(expected_cited) / len(cited) if cited else (1.0 if not expected else 0.0)
    )
    required_facts = [str(value) for value in golden.get("required_facts", [])]
    normalized_answer = _normalize_fact_text(str(response.get("answer", "")))
    matched_facts = [
        fact
        for fact in required_facts
        if _normalize_fact_text(fact) in normalized_answer
    ]
    fact_coverage = len(matched_facts) / len(required_facts) if required_facts else None
    should_abstain = bool(question.get("should_abstain", False))
    did_abstain = terminal_status == "abstained"
    permission_guardrail_applicable, permission_bypass_count = (
        _permission_guardrail_result(
            question=question,
            response=response,
            stream_events=stream_events,
        )
    )
    expected_tools = [str(value) for value in question.get("expected_tools", [])]
    selected_tools = [str(value) for value in response.get("selected_tools", [])]
    expected_entities = [str(value) for value in question.get("expected_entities", [])]
    matched_entities = [
        str(item["entity_id"]) for item in response.get("matched_entities", [])
    ]
    expected_version_scope = question.get("expected_version_scope")
    expected_doc_versions = sorted(
        str(value) for value in question.get("expected_doc_versions", [])
    )
    response_version_scope = response.get("version_scope", {})
    actual_version_scope = response_version_scope.get("mode")
    actual_doc_versions = sorted(
        str(value) for value in response_version_scope.get("doc_versions", [])
    )
    version_scope_correct = (
        None
        if expected_version_scope is None
        else (
            actual_version_scope == expected_version_scope
            and (
                not expected_doc_versions
                or actual_doc_versions == expected_doc_versions
            )
        )
    )
    entity_resolution_correct = (
        None if not expected_entities else matched_entities == expected_entities
    )
    tool_selection_correct = selected_tools == expected_tools
    contamination_count = _cross_version_contamination_count(response)
    # A cache hit skips tools, rerank and selection entirely (spec 10c § 6), so
    # there is no this-turn selected context to be a subset of. Its citations
    # were revalidated against live evidence inside `try_grounded_cache`; what
    # must still hold is that at least one survived.
    answered_from_cache = terminal_status == "answered_from_cache"
    citation_subset_valid = (
        bool(cited) if answered_from_cache else cited.issubset(released_contexts)
    )
    selected_context_bound_valid = len(selected_items) <= answer_context_limit
    answer_validation = response.get("answer_validation", {})
    grounded_answer = terminal_status in {"answered", "answered_from_cache"}
    self_check_valid = not grounded_answer or (
        isinstance(answer_validation, dict) and answer_validation.get("valid") is True
    )

    compression_metrics = _compression_metrics(response)
    compression_provenance_valid = cast(
        "bool | None",
        compression_metrics["context_compression_provenance_valid"],
    )
    compression_fallback = cast(
        "bool | None",
        compression_metrics["context_compression_fallback"],
    )
    raw_compression = response.get("context_compression")
    compression_mode = (
        raw_compression.get("effective_mode")
        if isinstance(raw_compression, dict)
        else None
    )

    layer_reasons: dict[str, list[str]] = {
        "trajectory": _trajectory_failures(
            response,
            stream_events,
            query_id=query_id,
            question=question,
        ),
        "tool": _tool_failures(
            response,
            question=question,
            expected_tools=expected_tools,
            tool_selection_correct=tool_selection_correct,
            entity_resolution_correct=entity_resolution_correct,
            version_scope_correct=version_scope_correct,
        ),
        "outcome": _outcome_failures(
            question=question,
            expected=expected,
            coverage=coverage,
            fact_coverage=fact_coverage,
            abstention_correct=should_abstain == did_abstain,
            citation_subset_valid=citation_subset_valid,
            selected_context_bound_valid=selected_context_bound_valid,
            compression_provenance_valid=compression_provenance_valid,
            compression_mode=compression_mode,
            compression_fallback=compression_fallback,
            self_check_valid=self_check_valid,
            contamination_count=contamination_count,
            terminal_status=terminal_status,
            expected_terminal_status=str(
                question.get(
                    "expected_terminal_status",
                    "abstained" if should_abstain else "answered",
                )
            ),
            expected_stop_reason=(
                str(question["expected_stop_reason"])
                if "expected_stop_reason" in question
                else None
            ),
            response=response,
        ),
    }
    layer_reasons = {
        layer: sorted(set(reasons)) for layer, reasons in layer_reasons.items()
    }
    layer_results = {layer: not reasons for layer, reasons in layer_reasons.items()}
    tool_reason_set = set(layer_reasons["tool"])
    first_failure_layer = next(
        (layer for layer in FAILURE_LAYERS if not layer_results[layer]),
        None,
    )
    metrics = response.get("metrics", {})
    provider_usage = _provider_usage(metrics)
    transformation_metrics = _transformation_metrics(
        response, str(question["question"])
    )
    return {
        "question_id": str(question["question_id"]),
        "trial": trial_number,
        "category": question.get("category"),
        "expected_sources": sorted(expected),
        "recalled_sources": sorted(recalled),
        "reranked_sources": sorted(reranked),
        "selected_sources": sorted(selected),
        "cited_sources": sorted(cited),
        "expected_recalled_sources": sorted(expected_recalled),
        "expected_reranked_sources": sorted(expected_reranked),
        "expected_selected_sources": sorted(expected_selected),
        "expected_cited_sources": sorted(expected_cited),
        "recall_at_k": _optional_round(recall),
        "reranked_recall_at_8": _optional_round(reranked_recall),
        "selected_context_recall_at_5": _optional_round(selected_recall),
        "citation_coverage": _optional_round(coverage),
        "citation_precision": round(precision, 4),
        "citation_subset_valid": citation_subset_valid,
        "selected_context_bound_valid": selected_context_bound_valid,
        "required_facts": required_facts,
        "matched_facts": matched_facts,
        "required_fact_coverage": _optional_round(fact_coverage),
        "should_abstain": should_abstain,
        "did_abstain": did_abstain,
        "abstention_correct": should_abstain == did_abstain,
        "expected_tools": expected_tools,
        "selected_tools": selected_tools,
        "tool_selection_correct": tool_selection_correct,
        "tool_invocation_correct": not bool(
            tool_reason_set.intersection(TOOL_INVOCATION_REASONS)
        ),
        "tool_structural_correct": not bool(
            tool_reason_set.difference(TOOL_NON_STRUCTURAL_REASONS)
        ),
        "expected_entities": expected_entities,
        "matched_entities": matched_entities,
        "entity_resolution_correct": entity_resolution_correct,
        "expected_version_scope": expected_version_scope,
        "actual_version_scope": actual_version_scope,
        "expected_doc_versions": expected_doc_versions,
        "actual_doc_versions": actual_doc_versions,
        "version_scope_correct": version_scope_correct,
        "cross_version_contamination_count": contamination_count,
        "permission_guardrail_applicable": permission_guardrail_applicable,
        "permission_bypass_count": permission_bypass_count,
        "recall_hit": bool(expected_recalled),
        "citation_hit": bool(expected_cited),
        "enough_evidence": bool(response.get("enough_evidence", False)),
        "terminal_status": terminal_status,
        "layer_results": layer_results,
        "failure_reasons": layer_reasons,
        "first_failure_layer": first_failure_layer,
        "case_passed": all(layer_results.values()),
        "latency_ms": observed_latency_ms,
        "retrieval_latency_ms": _non_negative_numeric_metric(
            metrics,
            "retrieval_latency_ms",
        ),
        "rerank_latency_ms": _non_negative_numeric_metric(metrics, "rerank_latency_ms"),
        "generation_latency_ms": _non_negative_numeric_metric(
            metrics,
            "generation_latency_ms",
        ),
        "time_to_first_token_ms": (
            time_to_first_token_ms if self_check_valid else None
        ),
        "provider_observations": _provider_observations(response),
        **provider_usage,
        **transformation_metrics,
        **compression_metrics,
    }


def _transformation_metrics(
    response: dict[str, Any],
    original_query: str,
) -> dict[str, bool | None]:
    transformation = response.get("query_transformation")
    plan = response.get("query_plan")
    if not isinstance(transformation, dict) or not transformation:
        return {
            "query_transformation_contract_valid": None,
            "original_query_retained": None,
            "step_back_primary_present": None,
        }
    strategy = transformation.get("strategy")
    roles = transformation.get("item_roles")
    item_count = transformation.get("item_count")
    plan_items = plan if isinstance(plan, list) else []
    contract_valid = (
        strategy in {"identity", "rewrite", "step_back", "decompose"}
        and isinstance(roles, list)
        and isinstance(item_count, int)
        and not isinstance(item_count, bool)
        and 1 <= item_count <= 3
        and len(roles) == item_count
        and len(plan_items) <= 3
    )
    normalized_original = _normalize_query_text(original_query)
    initial_plan_items = [
        item for item in plan_items if isinstance(item, dict) and item.get("round") == 0
    ]
    retained = bool(initial_plan_items) and all(
        isinstance(item, dict)
        and isinstance(item.get("query"), str)
        and normalized_original in _normalize_query_text(str(item["query"]))
        for item in initial_plan_items
    )
    primary_present: bool | None = None
    if strategy == "step_back":
        primary_plan_valid = any(
            isinstance(item, dict)
            and item.get("query_role") == "primary"
            and normalized_original in _normalize_query_text(str(item.get("query", "")))
            for item in plan_items
        )
        tool_calls = response.get("tool_calls")
        primary_result_ids = {
            str(chunk_id)
            for call in (tool_calls if isinstance(tool_calls, list) else [])
            if isinstance(call, dict) and call.get("query_role") == "primary"
            for chunk_id in call.get("result_chunk_ids", [])
            if isinstance(chunk_id, str)
        }
        selected_ids = {
            str(item["chunk_id"])
            for item in response.get("reranked", [])
            if isinstance(item, dict)
            and item.get("selected") is True
            and isinstance(item.get("chunk_id"), str)
        }
        primary_present = primary_plan_valid and bool(
            primary_result_ids.intersection(selected_ids)
        )
    return {
        "query_transformation_contract_valid": contract_valid,
        "original_query_retained": retained,
        "step_back_primary_present": primary_present,
    }


def _compression_metrics(response: dict[str, Any]) -> dict[str, float | bool | None]:
    compression = response.get("context_compression")
    if not isinstance(compression, dict) or not compression:
        return {
            "context_compression_provenance_valid": None,
            "context_compression_reduction_ratio": None,
            "context_compression_fallback": None,
        }
    before = compression.get("before_chars")
    after = compression.get("after_chars")
    retained = compression.get("retained_source_count")
    effective_mode = compression.get("effective_mode")
    valid_numbers = all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in (before, after, retained)
    )
    before_count = cast(int, before) if valid_numbers else 0
    after_count = cast(int, after) if valid_numbers else 0
    retained_count = cast(int, retained) if valid_numbers else 0
    provenance_valid = (
        valid_numbers
        and effective_mode in {"disabled", "selective", "summary", "extraction"}
        and retained_count > 0
        and after_count <= before_count
    )
    reduction = (
        round(1.0 - (after_count / before_count), 4)
        if valid_numbers and before_count > 0
        else None
    )
    return {
        "context_compression_provenance_valid": provenance_valid,
        "context_compression_reduction_ratio": reduction,
        "context_compression_fallback": compression.get("fallback_reason")
        not in {None, "below_trigger", "not_configured"},
    }


def _normalize_query_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _provider_observations(response: dict[str, Any]) -> dict[str, str]:
    def identity(name_field: str, model_field: str) -> str:
        name = str(response.get(name_field) or "not_reported")
        model = str(response.get(model_field) or "none")
        return f"{name}:{model}"

    transformation = response.get("query_transformation")
    compression = response.get("context_compression")
    transformer = transformation if isinstance(transformation, dict) else {}
    compressor = compression if isinstance(compression, dict) else {}
    return {
        "classifier": identity("classifier_name", "classifier_model"),
        "reranker": identity("reranker_name", "reranker_model"),
        "generator": identity("answer_generator_name", "answer_model"),
        "memory_selector": identity(
            "selective_memory_selector_name",
            "selective_memory_selector_model",
        ),
        "query_transformer": (
            f"{transformer.get('transformer_name') or 'not_reported'}:"
            f"{transformer.get('model') or 'none'}:"
            f"{transformer.get('strategy') or 'not_run'}"
        ),
        "context_compressor": (
            f"{compressor.get('compressor_name') or 'not_reported'}:"
            f"{compressor.get('model') or 'none'}:"
            f"{compressor.get('effective_mode') or 'not_run'}"
        ),
    }


def _answer_context_limit(response: dict[str, Any]) -> int:
    """Return the generation-context bound for this question's retrieval goal.

    Spec 12 § 5.9 and 13 § 8 make the cap two-branch: at most five projections
    for a focused question, or at most sixteen bounded siblings for an
    exhaustive document query. Grading only the focused branch both misses an
    over-cap exhaustive answer and fails a legal one.
    """

    if response.get("retrieval_goal") == "exhaustive":
        return MAX_EXHAUSTIVE_CONTEXTS
    return int(DEFAULT_SEARCH_PARAMS["answer_context_top_k"])


def _transformation_fixture_failures(
    response: dict[str, Any],
    question: dict[str, Any],
) -> list[str]:
    """Compare the executed transformation with the fixture's expectation.

    Spec 70 § 4.6 grades strategy and item roles "与 fixture 一致"; without an
    expected-value channel the runner could only prove the transformation was
    self-consistent, never that it was the one the case is about.
    """

    expectations = {
        "expected_transformation_strategy",
        "expected_query_roles",
        "expected_plan_item_count",
    }.intersection(question)
    if not expectations:
        return []
    transformation = response.get("query_transformation")
    if not isinstance(transformation, dict) or not transformation:
        return ["transformation_not_reported"]
    reasons: list[str] = []
    expected_strategy = question.get("expected_transformation_strategy")
    if (
        expected_strategy is not None
        and transformation.get("strategy") != expected_strategy
    ):
        reasons.append("transformation_strategy_mismatch")
    expected_roles = question.get("expected_query_roles")
    if expected_roles is not None:
        actual_roles = transformation.get("item_roles")
        if not isinstance(actual_roles, list) or actual_roles != list(expected_roles):
            reasons.append("query_roles_mismatch")
    expected_count = question.get("expected_plan_item_count")
    if expected_count is not None and transformation.get("item_count") != expected_count:
        reasons.append("plan_item_count_mismatch")
    return reasons


def _reranker_fixture_failures(
    response: dict[str, Any],
    question: dict[str, Any],
) -> list[str]:
    """Compare which reranker implementation actually produced the ranking.

    Spec 12 § 5.6 requires trace to name the implementation that ranked and
    whether a registered fallback was active; 70 § 3 wants a golden case for
    the degraded path so G5's recoverable demo is measured, not assumed.
    """

    expected = question.get("expected_reranker_fallback")
    if expected is None:
        return []
    reason = response.get("reranker_fallback_reason")
    actual = reason is not None
    if actual and reason not in RERANK_FALLBACK_REASONS:
        return ["unregistered_reranker_fallback_reason"]
    return [] if actual is expected else ["reranker_fallback_mismatch"]


def _trajectory_failures(
    response: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    query_id: str,
    question: dict[str, Any],
) -> list[str]:
    reasons: list[str] = _transformation_fixture_failures(response, question)
    reasons.extend(_reranker_fixture_failures(response, question))
    terminal_status = response.get("terminal_status")
    trace = response.get("trace", {})
    if terminal_status not in TERMINAL_STATUSES:
        reasons.append("invalid_terminal_status")
    if response.get("query_id") != query_id:
        reasons.append("cross_query_response")
    if not isinstance(trace, dict) or trace.get("terminal_status") != terminal_status:
        reasons.append("terminal_trace_mismatch")
    elif trace.get("query_id") != query_id:
        reasons.append("cross_query_trace")
    retry_count = response.get("retry_count")
    if (
        isinstance(retry_count, bool)
        or not isinstance(retry_count, int)
        or not 0 <= retry_count <= 3
    ):
        reasons.append("retry_limit_exceeded")
    if not events:
        reasons.append("stream_unavailable")
        return reasons
    if events[-1].get("type") != "final":
        reasons.append("final_not_last")
    if sum(1 for item in events if item.get("type") == "final") != 1:
        reasons.append("invalid_final_count")
    trace_events = [
        item.get("event") for item in events if item.get("type") == "trace_event"
    ]
    if any(
        not isinstance(item, dict)
        or not isinstance(item.get("sequence"), int)
        or isinstance(item.get("sequence"), bool)
        or int(item["sequence"]) < 1
        or not isinstance(item.get("query_id"), str)
        or not isinstance(item.get("stage"), str)
        or not str(item["stage"])
        or item.get("kind")
        not in {"stage_completed", "tool_completed", "retry_scheduled"}
        for item in trace_events
    ):
        reasons.append("malformed_trace_event")
        return reasons
    validated_trace_events = [cast(dict[str, Any], item) for item in trace_events]
    sequences = [int(item["sequence"]) for item in validated_trace_events]
    if sequences != list(range(1, len(sequences) + 1)):
        reasons.append("non_contiguous_events")
    if any(item.get("query_id") != query_id for item in validated_trace_events):
        reasons.append("cross_query_event")
    stage_events = [
        item
        for item in validated_trace_events
        if item.get("kind") in {"stage_completed", "retry_scheduled"}
    ]
    stages = [str(item["stage"]) for item in stage_events]
    if any(stage not in STAGE_ORDER for stage in stages):
        reasons.append("unknown_trajectory_stage")
    elif not _legal_stage_order(stages):
        reasons.append("illegal_stage_order")
    else:
        if not _terminal_path_valid(str(terminal_status), stages):
            reasons.append("terminal_path_mismatch")
        forbidden = _forbidden_stages(str(terminal_status)).intersection(stages)
        if forbidden:
            reasons.append("forbidden_stage_present")
    if terminal_status in {"answered", "answered_from_cache"}:
        validation_stage = (
            "try_grounded_cache"
            if terminal_status == "answered_from_cache"
            else "verify_answer"
        )
        validation_positions = [
            index
            for index, item in enumerate(events)
            if isinstance(item.get("event"), dict)
            and item["event"].get("kind") == "stage_completed"
            and item["event"].get("stage") == validation_stage
        ]
        answer_positions = [
            index
            for index, item in enumerate(events)
            if item.get("type") == "answer_delta"
        ]
        if (
            not validation_positions
            or not answer_positions
            or min(validation_positions) > min(answer_positions)
        ):
            reasons.append("answer_before_verification")
    return reasons


def _legal_stage_order(stages: list[str]) -> bool:
    previous: str | None = None
    for stage in stages:
        if previous is not None and STAGE_ORDER[stage] < STAGE_ORDER[previous]:
            if not (previous == "evaluate_evidence" and stage == "execute_tool_plan"):
                return False
        previous = stage
    return (
        bool(stages)
        and stages[0] == "recall_memory"
        and ("classify_and_route" in stages)
    )


# Terminal statuses whose complete stage path is fixed. These are graded by
# exact equality, so their forbidden stages are already covered.
_EARLY_TERMINAL_PATHS: dict[str, tuple[str, ...]] = {
    "answered_without_retrieval": ("recall_memory", "classify_and_route"),
    "answered_from_memory": ("recall_memory", "classify_and_route"),
    "memory_not_found": ("recall_memory", "classify_and_route"),
    "memory_saved": ("recall_memory", "classify_and_route"),
    "memory_write_failed": ("recall_memory", "classify_and_route"),
    "refused_unsupported_operation": (
        "recall_memory",
        "classify_and_route",
    ),
    "clarification_required": (
        "recall_memory",
        "classify_and_route",
        "resolve_terminology",
    ),
    "permission_denied": (
        "recall_memory",
        "classify_and_route",
        "resolve_terminology",
        "check_permission",
    ),
    "answered_from_cache": (
        "recall_memory",
        "classify_and_route",
        "resolve_terminology",
        "check_permission",
        "try_grounded_cache",
    ),
}


def _required_stages(terminal_status: str) -> list[str]:
    """Return the stages a terminal status must reach, in execution order."""

    required = [
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
    ]
    if terminal_status == "answered":
        required.append("prepare_generation_context")
    required.extend(["generate_answer_streaming", "verify_answer"])
    return required


def _forbidden_stages(terminal_status: str) -> frozenset[str]:
    """Return the stages a terminal status must never reach.

    Spec 70 § 4.6 grades the required *and* forbidden halves of a terminal
    path, and calls out that `prepare_generation_context` may appear only on an
    evidence-sufficient path. Grading presence alone lets an abstention that
    projected a generation context pass unnoticed.
    """

    if terminal_status in _EARLY_TERMINAL_PATHS or terminal_status not in {
        "answered",
        "abstained",
    }:
        return frozenset()
    return frozenset(STAGE_ORDER).difference(_required_stages(terminal_status))


def _terminal_path_valid(terminal_status: str, stages: list[str]) -> bool:
    if terminal_status in _EARLY_TERMINAL_PATHS:
        return stages == list(_EARLY_TERMINAL_PATHS[terminal_status])
    if terminal_status not in {"answered", "abstained"}:
        return False
    required = _required_stages(terminal_status)
    return all(stage in stages for stage in required) and stages[-2:] == [
        "generate_answer_streaming",
        "verify_answer",
    ]


def _tool_failures(
    response: dict[str, Any],
    *,
    question: dict[str, Any],
    expected_tools: list[str],
    tool_selection_correct: bool,
    entity_resolution_correct: bool | None,
    version_scope_correct: bool | None,
) -> list[str]:
    reasons: list[str] = []
    if not tool_selection_correct:
        reasons.append("unexpected_tools")
    selected_tools = [str(item) for item in response.get("selected_tools", [])]
    if any(item not in REGISTERED_TOOL_NAMES for item in selected_tools):
        reasons.append("unregistered_tool")
    query_plan = response.get("query_plan", [])
    if not isinstance(query_plan, list):
        return [*reasons, "invalid_query_plan"]
    initial_items = [
        item for item in query_plan if isinstance(item, dict) and item.get("round") == 0
    ]
    if len(initial_items) > 3:
        reasons.append("initial_plan_bound_exceeded")
    if any(
        not isinstance(item, dict)
        or str(item.get("tool", "")) not in REGISTERED_TOOL_NAMES
        for item in query_plan
    ):
        reasons.append("invalid_query_plan_tool")
    initial_plan_tools = {
        str(item.get("tool")) for item in initial_items if isinstance(item, dict)
    }
    if initial_plan_tools != set(selected_tools):
        reasons.append("plan_selection_mismatch")
    tool_calls = response.get("tool_calls", [])
    if not isinstance(tool_calls, list):
        return [*reasons, "invalid_tool_calls"]
    if any(
        not isinstance(item, dict)
        or str(item.get("tool", "")) not in REGISTERED_TOOL_NAMES
        for item in tool_calls
    ):
        reasons.append("invalid_tool_call")
    call_tools = {
        str(item.get("tool")) for item in tool_calls if isinstance(item, dict)
    }
    if not call_tools.issubset(set(selected_tools)):
        reasons.append("tool_call_selection_mismatch")
    expect_tool_calls = bool(question.get("expect_tool_calls", bool(expected_tools)))
    if expect_tool_calls and not set(expected_tools).issubset(call_tools):
        reasons.append("missing_tool_invocation")
    if not expect_tool_calls and tool_calls:
        reasons.append("forbidden_tool_invocation")
    if any(
        isinstance(item, dict)
        and (
            isinstance(item.get("round"), bool)
            or not isinstance(item.get("round"), int)
            or item.get("round") not in {0, 1, 2, 3}
        )
        for item in query_plan
    ):
        reasons.append("invalid_plan_round")
    supplementary_counts = Counter(
        int(item["round"])
        for item in query_plan
        if isinstance(item, dict)
        and isinstance(item.get("round"), int)
        and not isinstance(item.get("round"), bool)
        and int(item["round"]) in {1, 2, 3}
    )
    if sum(supplementary_counts.values()) > 3 or any(
        count > 1 for count in supplementary_counts.values()
    ):
        reasons.append("supplementary_plan_bound_exceeded")
    if len(tool_calls) > 6 or len(tool_calls) > len(query_plan):
        reasons.append("tool_call_bound_exceeded")
    permission = response.get("permission_decision")
    allowed_departments: set[str] | None = None
    if tool_calls:
        if (
            not isinstance(permission, dict)
            or permission.get("allowed") is not True
            or not isinstance(permission.get("allowed_departments"), list)
            or not all(
                isinstance(item, str) and item
                for item in permission.get("allowed_departments", [])
            )
        ):
            reasons.append("missing_permission_scope")
        else:
            allowed_departments = set(permission["allowed_departments"])
    for item in tool_calls:
        if not isinstance(item, dict):
            continue
        filters = item.get("filters")
        if not isinstance(filters, dict):
            reasons.append("invalid_tool_filters")
            continue
        try:
            normalized_filters = normalize_filters(filters)
        except (TypeError, ValueError):
            reasons.append("invalid_tool_filters")
        else:
            if (
                "department" in normalized_filters
                and not normalized_filters["department"]
            ):
                reasons.append("empty_tool_department_scope")
            tool_name = str(item.get("tool", ""))
            tool = SEARCH_TOOLS.get(tool_name)
            raw_departments = normalized_filters.get("department", [])
            departments = set(
                [raw_departments]
                if isinstance(raw_departments, str)
                else raw_departments
            )
            configured_departments = _configured_tool_departments(tool)
            if (
                configured_departments is None
                or allowed_departments is None
                or departments
                != configured_departments.intersection(allowed_departments)
            ):
                reasons.append("unauthorized_tool_scope")
        scope = item.get("version_scope")
        if not _valid_tool_version_scope(scope):
            reasons.append("invalid_tool_version_scope")
    if entity_resolution_correct is False:
        reasons.append("entity_resolution_mismatch")
    if version_scope_correct is False:
        reasons.append("version_scope_mismatch")
    return reasons


def _configured_tool_departments(
    tool: KnowledgeSearchTool | None,
) -> set[str] | None:
    if tool is None:
        return None
    configured = tool.filters.get("department", ())
    if isinstance(configured, str):
        return {configured}
    if isinstance(configured, tuple):
        return set(configured)
    return None


def _valid_tool_version_scope(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    mode = value.get("mode")
    if mode == "current":
        return value.get("doc_version") is None
    return (
        mode == "exact"
        and isinstance(value.get("doc_version"), str)
        and bool(str(value["doc_version"]).strip())
    )


def _compression_path_mismatch(
    question: dict[str, Any],
    *,
    mode: object,
    fallback: bool | None,
) -> bool:
    """Report whether the executed compression path contradicts the fixture.

    Spec 70 § 3 requires both a case where selective compression succeeds and
    one where malformed output falls back atomically, so a fixture must be able
    to say which of the two it is testing.
    """

    expected_mode = question.get("expected_compression_mode")
    if expected_mode is not None and mode != expected_mode:
        return True
    expected_fallback = question.get("expected_compression_fallback")
    return expected_fallback is not None and fallback is not expected_fallback


def _outcome_failures(
    *,
    question: dict[str, Any],
    expected: set[str],
    coverage: float | None,
    fact_coverage: float | None,
    abstention_correct: bool,
    citation_subset_valid: bool,
    selected_context_bound_valid: bool,
    compression_provenance_valid: bool | None,
    compression_mode: object,
    compression_fallback: bool | None,
    self_check_valid: bool,
    contamination_count: int,
    terminal_status: str,
    expected_terminal_status: str,
    expected_stop_reason: str | None,
    response: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if terminal_status != expected_terminal_status:
        reasons.append("terminal_status_mismatch")
    if not abstention_correct:
        reasons.append("abstention_mismatch")
    if not citation_subset_valid:
        reasons.append("citation_not_selected")
    if not selected_context_bound_valid:
        reasons.append("selected_context_bound_exceeded")
    if compression_provenance_valid is False:
        reasons.append("compression_provenance_invalid")
    if _compression_path_mismatch(
        question,
        mode=compression_mode,
        fallback=compression_fallback,
    ):
        reasons.append("compression_path_mismatch")
    if expected and coverage != 1.0:
        reasons.append("required_citation_missing")
    if fact_coverage is not None and fact_coverage != 1.0:
        reasons.append("required_fact_missing")
    if not self_check_valid:
        reasons.append("grounded_self_check_invalid")
    if contamination_count:
        reasons.append("cross_version_contamination")
    if (
        expected_stop_reason is not None
        and response.get("retrieval_stop_reason") != expected_stop_reason
    ):
        reasons.append("stop_reason_mismatch")
    metrics = response.get("metrics", {})
    for fixture_field, metric_field, reason in (
        (
            "expected_memory_written_count",
            "num_written_memories",
            "memory_state_mismatch",
        ),
        (
            "expected_cache_written_count",
            "response_cache_written_count",
            "cache_state_mismatch",
        ),
        (
            "expected_response_cache_hit",
            "response_cache_hit",
            "cache_hit_mismatch",
        ),
    ):
        if fixture_field in question and (
            not isinstance(metrics, dict)
            or metrics.get(metric_field) != question[fixture_field]
        ):
            reasons.append(reason)
    return reasons


def _permission_guardrail_result(
    *,
    question: dict[str, Any],
    response: dict[str, Any],
    stream_events: list[dict[str, Any]],
) -> tuple[bool, int]:
    """Return applicability and bypass count for a permission-denial trial."""

    decision = response.get("permission_decision")
    explicitly_denied = isinstance(decision, dict) and decision.get("allowed") is False
    denial_expected = question.get("expected_terminal_status") == "permission_denied"
    terminal_denied = response.get("terminal_status") == "permission_denied"
    if not (explicitly_denied or denial_expected or terminal_denied):
        return False, 0
    retrieval_stages = {
        "try_grounded_cache",
        "recall_authorized_experience",
        "plan_retrieval",
        "execute_tool_plan",
        "rerank_evidence",
        "evaluate_evidence",
        "prepare_generation_context",
        "generate_answer_streaming",
    }
    observed_stages = {
        str(event["event"].get("stage"))
        for event in stream_events
        if event.get("type") == "trace_event" and isinstance(event.get("event"), dict)
    }
    retrieval_activity = bool(
        response.get("tool_calls")
        or response.get("milvus_recalled")
        or response.get("reranked")
        or observed_stages.intersection(retrieval_stages)
    )
    return True, int(retrieval_activity)


def _provider_usage(metrics: object) -> dict[str, float | str | None]:
    """Extract optional non-negative provider usage from response metrics."""

    return {
        "provider_call_count": _non_negative_numeric_metric(
            metrics, "provider_call_count"
        ),
        "input_tokens": _non_negative_numeric_metric(metrics, "input_tokens"),
        "output_tokens": _non_negative_numeric_metric(metrics, "output_tokens"),
        "estimated_cost": _non_negative_numeric_metric(metrics, "estimated_cost"),
        "cost_profile": _bounded_optional_metric_string(metrics, "cost_profile"),
    }


def _non_negative_numeric_metric(metrics: object, name: str) -> float | None:
    value = _numeric_metric(metrics, name)
    return value if value is not None and value >= 0 else None


def _bounded_optional_metric_string(metrics: object, name: str) -> str | None:
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(name)
    if not isinstance(value, str) or not value or len(value) > 128:
        return None
    return value


def _numeric_metric(metrics: object, name: str) -> float | None:
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return round(numeric, 4)


def _dimension_report(report: dict[str, Any]) -> dict[str, Any]:
    cases = report["cases"]
    trials = [trial for case in cases for trial in case["trials"]]
    trajectory_rate = _mean(
        1.0 if item["layer_results"]["trajectory"] else 0.0 for item in trials
    )
    structure_rate = _mean(
        1.0 if item["tool_structural_correct"] else 0.0 for item in trials
    )
    case_pass_rate = _mean(1.0 if item["case_passed"] else 0.0 for item in trials)
    return {
        "trajectory": {
            "contract_pass_rate": _metric(trajectory_rate),
        },
        "tool": {
            "invocation_accuracy": _metric(report["tool_invocation_accuracy"]),
            "selection_accuracy": _metric(report["tool_selection_accuracy"]),
            "structural_accuracy": _metric(structure_rate),
            "entity_resolution_accuracy": _metric(report["entity_resolution_accuracy"]),
            "version_scope_accuracy": _metric(report["version_scope_accuracy"]),
            "query_transformation_contract_rate": _metric(
                report["query_transformation_contract_rate"]
            ),
            "original_query_retention_rate": _metric(
                report["original_query_retention_rate"]
            ),
            "step_back_primary_coverage": _metric(report["step_back_primary_coverage"]),
        },
        "outcome": {
            "case_pass_rate": _metric(case_pass_rate),
            "retrieval_recall_at_k": _metric(report["recall_at_k"]),
            "reranked_recall_at_8": _metric(report["reranked_recall_at_8"]),
            "selected_context_recall_at_5": _metric(
                report["selected_context_recall_at_5"]
            ),
            "citation_coverage": _metric(report["citation_coverage"]),
            "citation_precision": _metric(report["citation_precision"]),
            "required_fact_coverage": _metric(report["required_fact_coverage"]),
            "abstention_accuracy": _metric(report["abstention_accuracy"]),
            "cross_version_contamination_count": _metric(
                report["cross_version_contamination_count"]
            ),
            "context_compression_provenance_rate": _metric(
                report["context_compression_provenance_rate"]
            ),
            "context_compression_reduction_ratio": _metric(
                report["context_compression_reduction_ratio"]
            ),
            "context_compression_fallback_rate": _metric(
                report["context_compression_fallback_rate"]
            ),
        },
    }


def _metric(value: float | int | None) -> dict[str, Any]:
    return {
        "grader": GRADER,
        "value": value,
    }


def _reliability_report(
    cases: list[dict[str, Any]],
    live_providers: bool,
    trials: int,
) -> dict[str, Any]:
    if not live_providers:
        return {
            "trials_per_question": trials,
            "pass_at_k": None,
            "pass_power_k": None,
            "gate_metric": "deterministic_case_pass",
        }
    return {
        "trials_per_question": trials,
        "pass_at_k": _mean(1.0 if item["pass_at_k"] else 0.0 for item in cases),
        "pass_power_k": _mean(1.0 if item["pass_power_k"] else 0.0 for item in cases),
        "gate_metric": "pass_power_k",
    }


def _latency_report(cases: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    expected_sample_count = sum(len(case["trials"]) for case in cases)
    for name in (
        "latency_ms",
        "retrieval_latency_ms",
        "rerank_latency_ms",
        "generation_latency_ms",
        "time_to_first_token_ms",
    ):
        values = sorted(
            float(trial[name])
            for case in cases
            for trial in case["trials"]
            if trial[name] is not None
        )
        complete = len(values) == expected_sample_count and expected_sample_count > 0
        output[name] = {
            "sample_count": len(values),
            "expected_sample_count": expected_sample_count,
            "complete": complete,
            "median": round(median(values), 4) if complete else None,
            "p95": _percentile(values, 0.95) if complete else None,
        }
    return output


def _operational_report(
    trials: list[dict[str, Any]],
    *,
    live_providers: bool,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Build profile-bound scalar usage and throughput observations."""

    trial_count = len(trials)
    if live_providers:
        provider_calls = _complete_mean(
            trial.get("provider_call_count") for trial in trials
        )
        input_tokens = _complete_mean(trial.get("input_tokens") for trial in trials)
        output_tokens = _complete_mean(trial.get("output_tokens") for trial in trials)
        cost_profile_complete = bool(trials) and all(
            trial.get("cost_profile") is not None for trial in trials
        )
        cost_per_request = (
            _complete_mean(trial.get("estimated_cost") for trial in trials)
            if cost_profile_complete
            else None
        )
        cost_profiles = sorted(
            {
                str(trial["cost_profile"])
                for trial in trials
                if trial.get("cost_profile") is not None
            }
        )
    else:
        provider_calls = 0.0
        input_tokens = 0.0
        output_tokens = 0.0
        cost_per_request = 0.0
        cost_profiles = ["deterministic_offline_non_billable"]
    completed_requests_per_hour = (
        None if trial_count == 0 else round(trial_count * 3600.0 / elapsed_seconds, 4)
    )
    return {
        "profile": {
            "concurrency": 1,
            "warmup_trials": 0,
            "measurement": "sequential_wall_clock",
        },
        "run_elapsed_ms": round(elapsed_seconds * 1000.0, 4),
        "completed_requests_per_hour": completed_requests_per_hour,
        "provider_calls_per_request": provider_calls,
        "input_tokens_per_request": input_tokens,
        "output_tokens_per_request": output_tokens,
        "cost_per_request": cost_per_request,
        "cost_profiles": cost_profiles,
    }


def _complete_mean(values: Iterable[object]) -> float | None:
    numbers: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        numbers.append(float(value))
    if not numbers:
        return None
    return round(sum(numbers) / len(numbers), 4)


def _metric_portfolio(
    report: dict[str, Any],
    registry: MetricRegistry,
    *,
    live_providers: bool,
) -> dict[str, Any]:
    """Project only active, decision-bound registry metrics into the report."""

    role_groups: dict[str, dict[str, Any]] = {
        "goal": {},
        "guardrail": {},
        "operational": {},
    }
    failed_metrics: list[str] = []
    incomplete_metrics: list[str] = []
    trials = [trial for case in report["cases"] for trial in case["trials"]]
    for definition in registry.active_metrics:
        value = _measurement_value(report, definition.measurement)
        if live_providers and definition.threshold_or_budget.mode != "baseline_only":
            decision_status, applicable_count, passed_count = _live_gate_status(
                definition,
                trials,
            )
            gate_basis = "all_applicable_trials"
        else:
            decision_status = _decision_status(definition, value)
            applicable_count = None
            passed_count = None
            gate_basis = "aggregate"
        record = {
            "question": definition.question,
            "source": {
                "kind": definition.source_kind,
                "ref": definition.source_ref,
            },
            "owner": definition.owner,
            "grader_id": definition.grader_id,
            "grader_version": definition.grader_version,
            "grader_layer": definition.grader_layer,
            "dataset_segment": definition.dataset_segment,
            "measurement": definition.measurement,
            "value": value,
            "baseline": None,
            "delta": None,
            "threshold_or_budget": definition.threshold_or_budget.to_dict(),
            "decision_status": decision_status,
            "gate_basis": gate_basis,
            "applicable_trial_count": applicable_count,
            "passing_trial_count": passed_count,
            "decision_action": list(definition.decision_actions),
            "cost_class": definition.cost_class,
            "run_cadence": definition.run_cadence,
            "retirement_condition": definition.retirement_condition,
        }
        role_groups[definition.role][definition.metric_id] = record
        if decision_status == "fail":
            failed_metrics.append(definition.metric_id)
        elif decision_status == "evaluation_incomplete":
            incomplete_metrics.append(definition.metric_id)
    return {
        "registry_version": registry.version,
        "registry_checksum": registry.checksum,
        **role_groups,
        "failed_metrics": failed_metrics,
        "incomplete_metrics": incomplete_metrics,
    }


def _live_gate_status(
    definition: MetricDefinition,
    trials: list[dict[str, Any]],
) -> tuple[str, int, int]:
    """Gate a live metric on every applicable trial, never on its mean alone."""

    values = [
        value
        for trial in trials
        if (value := _trial_measurement_value(trial, definition.measurement))
        is not None
    ]
    if not values:
        return "evaluation_incomplete", 0, 0
    statuses = [_decision_status(definition, value) for value in values]
    passed_count = sum(status == "pass" for status in statuses)
    return (
        "pass" if passed_count == len(values) else "fail",
        len(values),
        passed_count,
    )


def _trial_measurement_value(
    trial: dict[str, Any], measurement: str
) -> float | int | None:
    """Resolve the per-trial scalar behind a live quality gate."""

    direct = {
        "aggregate.recall_at_k": "recall_at_k",
        "aggregate.selected_context_recall_at_5": "selected_context_recall_at_5",
        "aggregate.required_fact_coverage": "required_fact_coverage",
        "aggregate.permission_bypass_count": "permission_bypass_count",
        "aggregate.cross_version_contamination_count": (
            "cross_version_contamination_count"
        ),
    }
    if measurement == "aggregate.citation_resolve_rate":
        return 1 if trial.get("citation_subset_valid") is True else 0
    if measurement == "aggregate.abstention_accuracy":
        return 1 if trial.get("abstention_correct") is True else 0
    if measurement == "aggregate.permission_bypass_count" and not trial.get(
        "permission_guardrail_applicable"
    ):
        return None
    field = direct.get(measurement)
    if field is None:
        raise ValueError(f"measurement {measurement!r} has no live trial evaluator")
    return _report_number(trial.get(field))


def _measurement_value(report: dict[str, Any], measurement: str) -> float | int | None:
    """Resolve one allow-listed scalar measurement without arbitrary path access."""

    if measurement == "aggregate.recall_at_k":
        return _report_number(report.get("recall_at_k"))
    if measurement == "aggregate.selected_context_recall_at_5":
        return _report_number(report.get("selected_context_recall_at_5"))
    if measurement == "aggregate.required_fact_coverage":
        return _report_number(report.get("required_fact_coverage"))
    if measurement == "aggregate.citation_resolve_rate":
        return _report_number(report.get("citation_resolve_rate"))
    if measurement == "aggregate.abstention_accuracy":
        return _report_number(report.get("abstention_accuracy"))
    if measurement == "aggregate.permission_bypass_count":
        return _report_number(report.get("permission_bypass_count"))
    if measurement == "aggregate.cross_version_contamination_count":
        return _report_number(report.get("cross_version_contamination_count"))
    if measurement == "latency.latency_ms.p95":
        latency = report.get("latency")
        latency_ms = latency.get("latency_ms") if isinstance(latency, dict) else None
        return _report_number(
            latency_ms.get("p95") if isinstance(latency_ms, dict) else None
        )
    if measurement == "operational.cost_per_request":
        operational = report.get("operational")
        return _report_number(
            operational.get("cost_per_request")
            if isinstance(operational, dict)
            else None
        )
    if measurement == "operational.completed_requests_per_hour":
        operational = report.get("operational")
        return _report_number(
            operational.get("completed_requests_per_hour")
            if isinstance(operational, dict)
            else None
        )
    raise ValueError(f"registered measurement {measurement!r} has no evaluator")


def _report_number(value: object) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)):
        return None
    return value


def _decision_status(
    definition: MetricDefinition,
    value: float | int | None,
) -> str:
    if value is None:
        return "evaluation_incomplete"
    threshold = definition.threshold_or_budget
    if threshold.mode == "baseline_only":
        return "observational"
    if threshold.operator is None or threshold.value is None:
        raise ValueError(f"metric {definition.metric_id!r} has no gate comparator")
    actual = float(value)
    expected = threshold.value
    if threshold.operator == "eq":
        passed = math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9)
    elif threshold.operator == "gte":
        passed = actual >= expected
    elif threshold.operator == "lte":
        passed = actual <= expected
    else:
        raise ValueError(f"metric {definition.metric_id!r} has unknown gate comparator")
    return "pass" if passed else "fail"


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    index = max(0, min(len(values) - 1, round((len(values) - 1) * fraction)))
    return round(values[index], 4)


def _transcript_review(
    cases: list[dict[str, Any]],
    review_path: Path | None,
) -> dict[str, Any]:
    first_layers: Counter[str] = Counter()
    reasons: dict[str, Counter[str]] = {layer: Counter() for layer in FAILURE_LAYERS}
    failed_case_ids: set[str] = set()
    for case in cases:
        for trial in case["trials"]:
            first = trial["first_failure_layer"]
            if first is not None:
                first_layers[str(first)] += 1
                failed_case_ids.add(str(case["question_id"]))
            for layer in FAILURE_LAYERS:
                reasons[layer].update(trial["failure_reasons"][layer])
    reviews = _load_transcript_reviews(review_path)
    observed = {
        (str(case["question_id"]), layer, reason)
        for case in cases
        for trial in case["trials"]
        for layer in FAILURE_LAYERS
        for reason in trial["failure_reasons"][layer]
    }
    reviewed = {
        (
            str(item["question_id"]),
            str(item["failure_layer"]),
            str(item["reason_code"]),
        )
        for item in reviews
    }
    stale = reviewed.difference(observed)
    if stale:
        raise ValueError("transcript review contains failures absent from this run")
    unreviewed = sorted(observed.difference(reviewed))
    return {
        "failed_case_ids": sorted(failed_case_ids),
        "first_failure_layer_counts": {
            layer: first_layers[layer] for layer in FAILURE_LAYERS
        },
        "reason_counts": {
            layer: dict(sorted(reasons[layer].items())) for layer in FAILURE_LAYERS
        },
        "reviews": reviews,
        "unreviewed_failures": [
            {
                "question_id": question_id,
                "failure_layer": layer,
                "reason_code": reason,
            }
            for question_id, layer, reason in unreviewed
        ],
        "human_attribution_required": bool(unreviewed),
        "allowed_attributions": [
            "capability_gap",
            "task_quality",
            "scaffold_entanglement",
        ],
    }


def _load_transcript_reviews(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to load transcript review {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        "rag-eval-review-v1"
    ):
        raise ValueError("transcript review schema_version is incompatible")
    raw_reviews = payload.get("reviews")
    if not isinstance(raw_reviews, list):
        raise ValueError("transcript review reviews must be a list")
    allowed_fields = {
        "question_id",
        "failure_layer",
        "reason_code",
        "attribution",
        "owner",
    }
    allowed_attributions = {
        "capability_gap",
        "task_quality",
        "scaffold_entanglement",
    }
    output: list[dict[str, str]] = []
    identities: set[tuple[str, str, str]] = set()
    for raw in raw_reviews:
        if not isinstance(raw, dict) or set(raw) != allowed_fields:
            raise ValueError("transcript review entry fields are invalid")
        if not all(isinstance(raw[field], str) and raw[field] for field in raw):
            raise ValueError("transcript review values must be non-empty strings")
        if raw["failure_layer"] not in FAILURE_LAYERS:
            raise ValueError("transcript review failure_layer is invalid")
        if raw["attribution"] not in allowed_attributions:
            raise ValueError("transcript review attribution is invalid")
        identity = (
            raw["question_id"],
            raw["failure_layer"],
            raw["reason_code"],
        )
        if identity in identities:
            raise ValueError("transcript review entries must be unique")
        identities.add(identity)
        output.append({field: str(raw[field]) for field in sorted(allowed_fields)})
    return output


def _attach_baseline(report: dict[str, Any], path: Path | None) -> None:
    if path is None:
        return
    try:
        baseline = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to load eval baseline {path}: {exc}") from exc
    if (
        not isinstance(baseline, dict)
        or baseline.get("report_version") != REPORT_VERSION
    ):
        raise ValueError("eval baseline report_version is incompatible")
    current_evaluation = report["evaluation"]
    baseline_evaluation = baseline.get("evaluation", {})
    if (
        not isinstance(baseline_evaluation, dict)
        or baseline_evaluation.get("question_ids") != current_evaluation["question_ids"]
        or baseline_evaluation.get("top_k") != current_evaluation["top_k"]
        or baseline_evaluation.get("mode") != current_evaluation["mode"]
        or baseline_evaluation.get("trials_per_question")
        != current_evaluation["trials_per_question"]
        or baseline_evaluation.get("provider_profile")
        != current_evaluation["provider_profile"]
        or baseline_evaluation.get("dataset_profile")
        != current_evaluation["dataset_profile"]
        or baseline_evaluation.get("observed_provider_profile")
        != current_evaluation["observed_provider_profile"]
        or baseline_evaluation.get("grader_layer") != current_evaluation["grader_layer"]
    ):
        raise ValueError("eval baseline run profile is incompatible")
    runtime_compatible = baseline.get("runtime") == report["runtime"]
    current_portfolio = report.get("metric_portfolio")
    baseline_portfolio = baseline.get("metric_portfolio")
    if not isinstance(current_portfolio, dict) or not isinstance(
        baseline_portfolio, dict
    ):
        raise ValueError("eval baseline metric_portfolio is missing")
    if baseline_portfolio.get("registry_version") != current_portfolio.get(
        "registry_version"
    ) or baseline_portfolio.get("registry_checksum") != current_portfolio.get(
        "registry_checksum"
    ):
        raise ValueError("eval baseline metric registry is incompatible")
    current_operational = report.get("operational")
    baseline_operational = baseline.get("operational")
    operational_profile_compatible = (
        runtime_compatible
        and isinstance(current_operational, dict)
        and isinstance(baseline_operational, dict)
        and current_operational.get("profile") == baseline_operational.get("profile")
        and current_operational.get("cost_profiles")
        == baseline_operational.get("cost_profiles")
    )
    for role in ("goal", "guardrail", "operational"):
        current_group = current_portfolio.get(role)
        baseline_group = baseline_portfolio.get(role)
        if not isinstance(current_group, dict) or not isinstance(baseline_group, dict):
            raise ValueError(f"eval baseline metric role {role!r} is missing")
        if set(current_group) != set(baseline_group):
            raise ValueError(f"eval baseline metric role {role!r} is incompatible")
        for metric_id, current_metric_value in current_group.items():
            baseline_metric_value = baseline_group[metric_id]
            if not isinstance(current_metric_value, dict) or not isinstance(
                baseline_metric_value, dict
            ):
                raise ValueError(f"eval baseline metric {metric_id!r} is invalid")
            for field in (
                "measurement",
                "grader_id",
                "grader_version",
                "grader_layer",
                "dataset_segment",
                "threshold_or_budget",
            ):
                if baseline_metric_value.get(field) != current_metric_value.get(field):
                    raise ValueError(
                        f"eval baseline metric {metric_id!r} contract is incompatible"
                    )
            baseline_value = _report_number(baseline_metric_value.get("value"))
            if (
                baseline_metric_value.get("value") is not None
                and baseline_value is None
            ):
                raise ValueError(f"eval baseline metric {metric_id!r} value is invalid")
            if role == "operational" and not operational_profile_compatible:
                current_metric_value["baseline"] = None
                current_metric_value["delta"] = None
                continue
            current_value = _report_number(current_metric_value.get("value"))
            current_metric_value["baseline"] = baseline_value
            current_metric_value["delta"] = (
                None
                if baseline_value is None or current_value is None
                else round(float(current_value) - float(baseline_value), 4)
            )
    report["baseline"] = {
        "path": path.as_posix(),
        "report_version": REPORT_VERSION,
        "registry_checksum": current_portfolio["registry_checksum"],
        "runtime_compatible": runtime_compatible,
        "operational_profile_compatible": operational_profile_compatible,
        "operational_delta_skipped_reason": (
            None
            if operational_profile_compatible
            else (
                "runtime_profile_mismatch"
                if not runtime_compatible
                else "operational_profile_mismatch"
            )
        ),
    }


def _expand_filters(filters: dict[str, Any]) -> dict[str, Any]:
    output = {
        "source_type": ["local", "s3"],
        "doc_type": ["markdown", "pdf", "text", "image"],
    }
    output.update(filters)
    return output


def _cross_version_contamination_count(
    response: dict[str, Any],
) -> int:
    mode = response.get("version_scope", {}).get("mode")
    recalled = response.get("milvus_recalled", [])
    selected = [item for item in response.get("reranked", []) if item.get("selected")]
    citations = response.get("citations", [])
    records = [*recalled, *selected, *citations]
    violations = 0
    by_doc: dict[str, set[str]] = {}
    version_scope = response.get("version_scope", {})
    requested_versions = version_scope.get(
        "doc_versions",
        [],
    )
    scopes = version_scope.get("sides", [])
    for item in records:
        doc_id = str(item.get("doc_id", ""))
        doc_version = str(item.get("doc_version", ""))
        if doc_id and doc_version:
            by_doc.setdefault(doc_id, set()).add(doc_version)
        if mode == "current" and item.get("is_current") is False:
            violations += 1
        if (
            mode == "exact"
            and requested_versions
            and doc_version != requested_versions[0]
        ):
            violations += 1
        if (
            mode == "comparison"
            and scopes
            and not any(_record_matches_scope(item, scope) for scope in scopes)
        ):
            violations += 1
    if mode == "comparison":
        violations += sum(
            1
            for scope in scopes
            if not any(_record_matches_scope(item, scope) for item in selected)
        )
        return violations
    violations += sum(
        len(versions) - 1 for versions in by_doc.values() if len(versions) > 1
    )
    return violations


def _record_matches_scope(
    item: dict[str, Any],
    scope: dict[str, Any],
) -> bool:
    if scope.get("mode") == "current":
        return item.get("is_current") is True
    if scope.get("mode") == "exact":
        return item.get("doc_version") == scope.get("doc_version")
    return False


def _optional_round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _mean(values: Iterable[float | None]) -> float | None:
    included = [value for value in values if value is not None]
    if not included:
        return None
    return round(sum(included) / len(included), 4)


def _normalize_fact_text(value: str) -> str:
    return re.sub(r"[\W_]+", " ", value.casefold()).strip()


def _load_golden_answers(path: Path) -> dict[str, dict[str, Any]]:
    """Load the repository's deliberately small YAML fixture without extras."""

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Unable to read golden answers {path}: {exc}") from exc

    try:
        parsed_json = json.loads(raw)
    except json.JSONDecodeError:
        parsed_json = None
    if parsed_json is not None:
        if not isinstance(parsed_json, dict):
            raise ValueError("golden answers must be a mapping")
        return {
            str(key): _validate_golden_case(str(key), value)
            for key, value in parsed_json.items()
        }

    lines = raw.splitlines()
    output: dict[str, dict[str, Any]] = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        if line.startswith(" ") or not line.endswith(":"):
            raise ValueError(f"Unsupported golden answers YAML at line {index + 1}")
        question_id = line[:-1].strip()
        if not question_id or question_id in output:
            raise ValueError("golden answer IDs must be non-empty and unique")
        case: dict[str, Any] = {}
        output[question_id] = case
        index += 1
        while index < len(lines) and (
            not lines[index].strip() or lines[index].startswith("  ")
        ):
            if not lines[index].strip():
                index += 1
                continue
            field_line = lines[index][2:]
            if ":" not in field_line:
                raise ValueError(f"Unsupported golden answers YAML at line {index + 1}")
            field, raw_value = field_line.split(":", 1)
            field = field.strip()
            raw_value = raw_value.strip()
            index += 1
            if raw_value == ">":
                parts: list[str] = []
                while index < len(lines) and lines[index].startswith("    "):
                    parts.append(lines[index].strip())
                    index += 1
                case[field] = " ".join(parts)
            elif not raw_value:
                values: list[Any] = []
                while index < len(lines) and lines[index].startswith("    - "):
                    values.append(_parse_yaml_scalar(lines[index][6:].strip()))
                    index += 1
                case[field] = values
            else:
                case[field] = _parse_yaml_scalar(raw_value)

    return {key: _validate_golden_case(key, value) for key, value in output.items()}


def _parse_yaml_scalar(value: str) -> Any:
    lowered = value.casefold()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def _validate_golden_case(
    question_id: str,
    value: Any,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Golden answer {question_id!r} must be a mapping")
    if not set(value).issubset(
        {"question", "golden_answer", "required_facts", "required_citations"}
    ):
        raise ValueError(f"Golden answer {question_id!r} has unknown fields")
    required_facts = value.get("required_facts")
    required_citations = value.get("required_citations")
    if not _bounded_unique_strings(required_facts, maximum=32):
        raise ValueError(f"Golden answer {question_id!r} requires a string facts list")
    if not _bounded_unique_strings(required_citations, maximum=32):
        raise ValueError(f"Golden answer {question_id!r} requires a citation ID list")
    for field in ("question", "golden_answer"):
        if field in value and not _bounded_string(value[field], maximum=8_000):
            raise ValueError(f"Golden answer {question_id!r} has invalid {field}")
    return dict(value)


def _validate_questions(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not 1 <= len(value) <= 100:
        raise ValueError("questions file must contain 1 to 100 cases")
    allowed = {
        "category",
        "expect_tool_calls",
        "expected_cache_written_count",
        "expected_compression_fallback",
        "expected_compression_mode",
        "expected_doc_versions",
        "expected_entities",
        "expected_memory_written_count",
        "expected_plan_item_count",
        "expected_query_roles",
        "expected_reranker_fallback",
        "expected_response_cache_hit",
        "expected_sources",
        "expected_stop_reason",
        "expected_terminal_status",
        "expected_tools",
        "expected_transformation_strategy",
        "expected_version_scope",
        "metadata_filters",
        "question",
        "question_id",
        "scenario",
        "should_abstain",
    }
    output: list[dict[str, Any]] = []
    for raw in value:
        if (
            not isinstance(raw, dict)
            or not set(raw).issubset(allowed)
            or not {"question_id", "question", "expected_sources"}.issubset(raw)
        ):
            raise ValueError("question case fields are invalid")
        if not _bounded_string(raw["question_id"], maximum=128) or not _bounded_string(
            raw["question"], maximum=8_000
        ):
            raise ValueError("question identity and text must be bounded strings")
        for field in (
            "expected_sources",
            "expected_tools",
            "expected_entities",
            "expected_doc_versions",
        ):
            if field in raw and not _bounded_unique_strings(raw[field], maximum=32):
                raise ValueError(f"question {field} must be a bounded unique list")
        if any(
            item not in REGISTERED_TOOL_NAMES for item in raw.get("expected_tools", [])
        ):
            raise ValueError("question expected_tools contains an unknown tool")
        for field in ("category", "expected_stop_reason"):
            if field in raw and not _bounded_string(raw[field], maximum=128):
                raise ValueError(f"question {field} must be a bounded string")
        if raw.get("expected_stop_reason") not in {
            None,
            "duplicate_retry_query",
            "no_progress",
            "retry_exhausted",
        }:
            raise ValueError("question expected_stop_reason is invalid")
        if raw.get("expected_terminal_status") not in {None, *TERMINAL_STATUSES}:
            raise ValueError("question expected_terminal_status is invalid")
        if raw.get("expected_transformation_strategy") not in {
            None,
            *TRANSFORMATION_STRATEGIES,
        }:
            raise ValueError("question expected_transformation_strategy is invalid")
        roles = raw.get("expected_query_roles")
        if roles is not None and (
            not isinstance(roles, list)
            or not 1 <= len(roles) <= 3
            or any(item not in QUERY_ROLES for item in roles)
        ):
            raise ValueError("question expected_query_roles is invalid")
        if raw.get("expected_compression_mode") not in {None, *COMPRESSION_MODES}:
            raise ValueError("question expected_compression_mode is invalid")
        if "expected_plan_item_count" in raw and (
            isinstance(raw["expected_plan_item_count"], bool)
            or not isinstance(raw["expected_plan_item_count"], int)
            or not 1 <= raw["expected_plan_item_count"] <= 3
        ):
            raise ValueError("question expected_plan_item_count is invalid")
        if roles is not None and raw.get("expected_plan_item_count", len(roles)) != len(
            roles
        ):
            raise ValueError("question expected_query_roles conflicts with plan count")
        if raw.get("expected_version_scope") not in {
            None,
            "comparison",
            "current",
            "exact",
        }:
            raise ValueError("question expected_version_scope is invalid")
        for field in (
            "expect_tool_calls",
            "expected_compression_fallback",
            "expected_reranker_fallback",
            "expected_response_cache_hit",
            "should_abstain",
        ):
            if field in raw and not isinstance(raw[field], bool):
                raise ValueError(f"question {field} must be boolean")
        for field in (
            "expected_cache_written_count",
            "expected_memory_written_count",
        ):
            if field in raw and (
                isinstance(raw[field], bool)
                or not isinstance(raw[field], int)
                or not 0 <= raw[field] <= 100
            ):
                raise ValueError(f"question {field} must be a bounded integer")
        filters = raw.get("metadata_filters", {})
        if not isinstance(filters, dict):
            raise ValueError("question metadata_filters must be a mapping")
        normalize_filters(filters)
        scenario = raw.get("scenario", {})
        if (
            not isinstance(scenario, dict)
            or not set(scenario).issubset({"permission", "prelude", "reranker"})
            or scenario.get("permission", "allow") not in {"allow", "deny"}
        ):
            raise ValueError("question scenario is invalid")
        prelude = scenario.get("prelude")
        if prelude is not None and (
            not isinstance(prelude, list)
            or not 1 <= len(prelude) <= 3
            or any(not _bounded_string(item, maximum=8_000) for item in prelude)
        ):
            raise ValueError("question scenario prelude is invalid")
        if scenario.get("reranker", "rule_based") not in SCENARIO_RERANKERS:
            raise ValueError("question scenario reranker is invalid")
        if (
            raw.get("should_abstain") is True
            and raw.get("expected_terminal_status", "abstained") != "abstained"
        ):
            raise ValueError("abstention expectation conflicts with terminal status")
        output.append(dict(raw))
    return output


def _question_scenario(question: dict[str, Any]) -> dict[str, str]:
    """Return the dependency-injection half of a scenario.

    `prelude` is deliberately excluded: it is same-session conversation the
    runner replays through the normal `stream()` path, not a dependency the
    factory has to build.
    """

    raw = cast(dict[str, Any], question.get("scenario", {}))
    return {
        "permission": str(raw.get("permission", "allow")),
        "reranker": str(raw.get("reranker", "rule_based")),
    }


def _question_prelude(question: dict[str, Any]) -> list[str]:
    """Return the same-session turns to replay before the graded question."""

    raw = question.get("scenario")
    if not isinstance(raw, dict):
        return []
    prelude = raw.get("prelude")
    return [str(item) for item in prelude] if isinstance(prelude, list) else []


def _bounded_string(value: object, *, maximum: int) -> bool:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= maximum


def _bounded_unique_strings(value: object, *, maximum: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) <= maximum
        and all(_bounded_string(item, maximum=512) for item in value)
        and len(set(value)) == len(value)
    )
