"""Comparative retrieval-tier evaluation for the spec 15 ladder.

Spec 70 § 4.2d fixes three arms — `lexical_only` (T0), `lexical_rewrite` (T1)
and `hybrid_dense` (T2) — over one corpus, question set, permission/version
scope, reranker and generator. Each arm is reported independently; the module
never emits a blended cross-arm score, and a missing arm stays in the
denominator as `evaluation_incomplete`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final

from agent_workshop_demo.eval_runner import EvaluationWorkflow, evaluate_questions
from agent_workshop_demo.retrieval_tier import TIER_CODES, RetrievalTier

RETRIEVAL_TIER_EVAL_VERSION: Final = "retrieval-tier-eval-v1"
TIER_ARMS: Final[tuple[str, ...]] = (
    RetrievalTier.LEXICAL_ONLY.value,
    RetrievalTier.LEXICAL_REWRITE.value,
    RetrievalTier.HYBRID_DENSE.value,
)
BASELINE_ARM: Final = RetrievalTier.LEXICAL_ONLY.value
DEFAULT_ARM: Final = RetrievalTier.HYBRID_DENSE.value
QUALITY_METRICS: Final[tuple[str, ...]] = (
    "recall_at_k",
    "selected_context_recall_at_5",
    "required_fact_coverage",
    "citation_resolve_rate",
    "abstention_accuracy",
)
COMPARISONS: Final[tuple[tuple[str, str], ...]] = (
    (RetrievalTier.LEXICAL_ONLY.value, RetrievalTier.LEXICAL_REWRITE.value),
    (RetrievalTier.LEXICAL_REWRITE.value, RetrievalTier.HYBRID_DENSE.value),
)
_TOLERANCE: Final = 1e-9

ArmFactory = Callable[[dict[str, str]], EvaluationWorkflow]


def evaluate_retrieval_tiers(
    *,
    questions_path: Path,
    golden_answers_path: Path | None = None,
    arm_workflow_factories: Mapping[str, ArmFactory],
    top_k: int = 20,
    metric_registry_path: Path | None = None,
    provider_profile: dict[str, str] | None = None,
    dataset_profile: dict[str, str] | None = None,
    latency_budget_ms: float | None = None,
) -> dict[str, Any]:
    """Run the fixed tier arms and report each one independently."""

    unknown = sorted(set(arm_workflow_factories) - set(TIER_ARMS))
    if unknown:
        raise ValueError(
            "Unsupported retrieval tier arms: " + ", ".join(unknown)
        )
    if latency_budget_ms is not None and latency_budget_ms <= 0:
        raise ValueError("latency_budget_ms must be positive when provided")
    arms: dict[str, dict[str, Any]] = {}
    runtime: dict[str, Any] | None = None
    question_ids: list[str] = []
    for arm in TIER_ARMS:
        factory = arm_workflow_factories.get(arm)
        if factory is None:
            arms[arm] = _incomplete_arm(arm, "arm_not_configured")
            continue
        report = evaluate_questions(
            questions_path=questions_path,
            golden_answers_path=golden_answers_path,
            scenario_workflow_factory=factory,
            top_k=top_k,
            metric_registry_path=metric_registry_path,
            provider_profile=provider_profile,
            dataset_profile=dataset_profile,
        )
        arms[arm] = _arm_report(arm, report)
        if runtime is None:
            runtime = dict(report["runtime"])
            question_ids = list(report["evaluation"]["question_ids"])
    comparisons = [
        _comparison(arms[baseline], arms[candidate])
        for baseline, candidate in COMPARISONS
    ]
    return {
        "report_version": RETRIEVAL_TIER_EVAL_VERSION,
        "evaluation": {
            "top_k": top_k,
            "question_ids": question_ids,
            "arms": list(TIER_ARMS),
            "baseline_arm": BASELINE_ARM,
            "default_arm": DEFAULT_ARM,
            "scoring": "per_arm_only",
            "provider_profile": dict(provider_profile or {}),
            "dataset_profile": dict(dataset_profile or {}),
        },
        "runtime": runtime or {},
        "arms": [arms[arm] for arm in TIER_ARMS],
        "comparisons": comparisons,
        "default_tier_justification": default_tier_justification(
            arms,
            latency_budget_ms=latency_budget_ms,
        ),
    }


def _incomplete_arm(arm: str, reason: str) -> dict[str, Any]:
    """Keep an unevaluated arm in the denominator with a registered reason."""

    return {
        "arm": arm,
        "tier_code": TIER_CODES[RetrievalTier(arm)],
        "status": "evaluation_incomplete",
        "reason": reason,
        "quality": {metric: None for metric in QUALITY_METRICS},
        "latency": {},
        "operational": {},
        "case_pass_count": None,
        "num_questions": None,
    }


def _arm_report(arm: str, report: dict[str, Any]) -> dict[str, Any]:
    """Extract only the § 4.2d fields from one full rag-eval-v3 report."""

    latency = report.get("latency", {})
    operational = report.get("operational", {})
    cases = report.get("cases", [])
    return {
        "arm": arm,
        "tier_code": TIER_CODES[RetrievalTier(arm)],
        "status": "complete",
        "quality": {metric: report.get(metric) for metric in QUALITY_METRICS},
        "latency": {
            "end_to_end_ms": dict(latency.get("latency_ms", {})),
            "retrieval_ms": dict(latency.get("retrieval_latency_ms", {})),
        },
        "operational": {
            "provider_calls_per_request": operational.get(
                "provider_calls_per_request"
            ),
            "input_tokens_per_request": operational.get("input_tokens_per_request"),
            "output_tokens_per_request": operational.get("output_tokens_per_request"),
            "cost_per_request": operational.get("cost_per_request"),
            "cost_profiles": list(operational.get("cost_profiles", [])),
        },
        "case_pass_count": sum(1 for case in cases if case.get("case_passed")),
        "num_questions": report.get("num_questions"),
    }


def _comparison(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Report per-metric deltas without collapsing them into one number."""

    if baseline["status"] != "complete" or candidate["status"] != "complete":
        return {
            "baseline_arm": baseline["arm"],
            "candidate_arm": candidate["arm"],
            "status": "evaluation_incomplete",
            "quality_deltas": {metric: None for metric in QUALITY_METRICS},
            "regressions": [],
            "improvements": [],
            "p95_latency_delta_ms": None,
        }
    deltas: dict[str, float | None] = {}
    regressions: list[str] = []
    improvements: list[str] = []
    for metric in QUALITY_METRICS:
        before = baseline["quality"].get(metric)
        after = candidate["quality"].get(metric)
        if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
            deltas[metric] = None
            continue
        delta = round(float(after) - float(before), 6)
        deltas[metric] = delta
        if delta < -_TOLERANCE:
            regressions.append(metric)
        elif delta > _TOLERANCE:
            improvements.append(metric)
    return {
        "baseline_arm": baseline["arm"],
        "candidate_arm": candidate["arm"],
        "status": "complete",
        "quality_deltas": deltas,
        "regressions": regressions,
        "improvements": improvements,
        "p95_latency_delta_ms": _p95_delta(baseline, candidate),
    }


def _p95_delta(baseline: dict[str, Any], candidate: dict[str, Any]) -> float | None:
    """Return added end-to-end P95 latency, or None when a sample is partial."""

    before = _p95(baseline)
    after = _p95(candidate)
    if before is None or after is None:
        return None
    return round(after - before, 4)


def _p95(arm: dict[str, Any]) -> float | None:
    """Return a complete end-to-end P95 sample or None."""

    sample = arm.get("latency", {}).get("end_to_end_ms", {})
    if not sample.get("complete"):
        return None
    value = sample.get("p95")
    return float(value) if isinstance(value, (int, float)) else None


def default_tier_justification(
    arms: Mapping[str, dict[str, Any]],
    *,
    latency_budget_ms: float | None,
) -> dict[str, Any]:
    """State whether quality evidence, not the teaching goal, keeps T2 default."""

    rewrite = arms[RetrievalTier.LEXICAL_REWRITE.value]
    hybrid = arms[DEFAULT_ARM]
    reasons: list[str] = []
    if rewrite["status"] != "complete" or hybrid["status"] != "complete":
        return {
            "default_tier": DEFAULT_ARM,
            "status": "evaluation_incomplete",
            "quality_gain": None,
            "regressions": [],
            "latency_budget_ms": latency_budget_ms,
            "p95_latency_delta_ms": None,
            "within_latency_budget": None,
            "reasons": ["arm_not_evaluated"],
        }
    comparison = _comparison(rewrite, hybrid)
    quality_gain = bool(comparison["improvements"]) and not comparison["regressions"]
    if not quality_gain:
        reasons.append("no_quality_gain_over_lexical_rewrite")
    observed_p95 = _p95(hybrid)
    if latency_budget_ms is None:
        within_budget: bool | None = None
        reasons.append("latency_budget_not_approved")
    elif observed_p95 is None:
        within_budget = None
        reasons.append("latency_sample_incomplete")
    else:
        within_budget = observed_p95 <= latency_budget_ms
        if not within_budget:
            reasons.append("p95_latency_over_budget")
    status = (
        "quality_evidence"
        if quality_gain and within_budget is True
        else "teaching_goal_only"
    )
    return {
        "default_tier": DEFAULT_ARM,
        "status": status,
        "quality_gain": quality_gain,
        "regressions": comparison["regressions"],
        "latency_budget_ms": latency_budget_ms,
        "p95_latency_delta_ms": comparison["p95_latency_delta_ms"],
        "within_latency_budget": within_budget,
        "reasons": reasons,
    }
