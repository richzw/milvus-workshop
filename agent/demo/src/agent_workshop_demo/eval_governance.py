"""Strict governance artifacts for the maintainable evaluation portfolio."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import cast

REGISTRY_VERSION = "eval-metric-registry-v1"
ERROR_ANALYSIS_VERSION = "eval-error-analysis-v1"
DEFAULT_METRIC_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2] / "eval" / "metric_registry.json"
)

ROLES = frozenset({"goal", "guardrail", "operational"})
SOURCE_KINDS = frozenset(
    {"observed_failure", "product_goal", "hard_constraint", "incident"}
)
GRADER_LAYERS = frozenset({"L1_programmatic", "L2_judge", "L3_human"})
COST_CLASSES = frozenset({"near_zero", "metered", "manual"})
RUN_CADENCES = frozenset({"per_pr", "nightly", "release", "monthly", "incident"})
DATASET_SEGMENTS = frozenset({"rag_core"})
METRIC_STATUSES = frozenset({"candidate", "active", "retired"})
THRESHOLD_MODES = frozenset({"gate", "budget", "baseline_only"})
THRESHOLD_OPERATORS = frozenset({"eq", "gte", "lte"})
DECISION_ACTIONS = frozenset(
    {
        "block_deploy",
        "rollback",
        "reject_change",
        "open_investigation",
        "capacity_review",
    }
)
MEASUREMENTS = frozenset(
    {
        "aggregate.recall_at_k",
        "aggregate.selected_context_recall_at_5",
        "aggregate.required_fact_coverage",
        "aggregate.citation_resolve_rate",
        "aggregate.abstention_accuracy",
        "aggregate.permission_bypass_count",
        "aggregate.cross_version_contamination_count",
        "latency.latency_ms.p95",
        "operational.cost_per_request",
        "operational.completed_requests_per_hour",
    }
)
ANALYSIS_KINDS = frozenset({"bootstrap", "significant_change"})
CLUSTER_SEVERITIES = frozenset({"P0", "P1", "P2", "P3"})
CLUSTER_DISPOSITIONS = frozenset(
    {"prompt_or_schema_fix", "metric_candidate", "fixture_only", "discard"}
)
IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")


@dataclass(frozen=True)
class ThresholdOrBudget:
    """One scalar gate, budget, or baseline-only observation contract."""

    mode: str
    unit: str
    operator: str | None
    value: float | None

    def to_dict(self) -> dict[str, object]:
        """Return the stable JSON representation used in reports and checksums."""

        output: dict[str, object] = {"mode": self.mode, "unit": self.unit}
        if self.operator is not None:
            output["operator"] = self.operator
        if self.value is not None:
            output["value"] = self.value
        return output


@dataclass(frozen=True)
class MetricDefinition:
    """Validated definition of one candidate, active, or retired metric."""

    metric_id: str
    role: str
    question: str
    source_kind: str
    source_ref: str
    owner: str
    grader_id: str
    grader_version: str
    grader_layer: str
    dataset_segment: str
    measurement: str
    threshold_or_budget: ThresholdOrBudget
    decision_actions: tuple[str, ...]
    cost_class: str
    run_cadence: str
    retirement_condition: str
    status: str
    introduced_at: str
    last_reviewed_at: str
    retirement_reason: str | None

    def to_dict(self) -> dict[str, object]:
        """Return a canonical, serialization-safe registry entry."""

        output: dict[str, object] = {
            "metric_id": self.metric_id,
            "role": self.role,
            "question": self.question,
            "source": {"kind": self.source_kind, "ref": self.source_ref},
            "owner": self.owner,
            "grader_id": self.grader_id,
            "grader_version": self.grader_version,
            "grader_layer": self.grader_layer,
            "dataset_segment": self.dataset_segment,
            "measurement": self.measurement,
            "threshold_or_budget": self.threshold_or_budget.to_dict(),
            "decision_action": list(self.decision_actions),
            "cost_class": self.cost_class,
            "run_cadence": self.run_cadence,
            "retirement_condition": self.retirement_condition,
            "status": self.status,
            "introduced_at": self.introduced_at,
            "last_reviewed_at": self.last_reviewed_at,
        }
        if self.retirement_reason is not None:
            output["retirement_reason"] = self.retirement_reason
        return output


@dataclass(frozen=True)
class MetricRegistry:
    """Validated registry plus its semantic canonical checksum."""

    version: str
    metrics: tuple[MetricDefinition, ...]
    checksum: str

    @property
    def active_metrics(self) -> tuple[MetricDefinition, ...]:
        """Return active definitions in stable metric-id order."""

        return tuple(metric for metric in self.metrics if metric.status == "active")


@dataclass(frozen=True)
class ErrorAnalysisArtifact:
    """Validated human error-analysis sample and cluster summary."""

    analysis_id: str
    analysis_kind: str
    sampled_at: str
    case_count: int
    failed_case_count: int
    cluster_count: int
    checksum: str


def load_metric_registry(path: Path | None = None) -> MetricRegistry:
    """Load a strict metric registry and compute its semantic checksum."""

    resolved = path or DEFAULT_METRIC_REGISTRY_PATH
    payload = _load_json(resolved, label="metric registry")
    root = _strict_mapping(
        payload,
        required={"registry_version", "metrics"},
        optional=set(),
        label="metric registry",
    )
    if root["registry_version"] != REGISTRY_VERSION:
        raise ValueError("metric registry registry_version is incompatible")
    raw_metrics = _strict_list(root["metrics"], label="metric registry metrics")
    if not 1 <= len(raw_metrics) <= 100:
        raise ValueError("metric registry must contain 1 to 100 metrics")
    metrics = tuple(
        sorted((_parse_metric(item) for item in raw_metrics), key=_metric_id)
    )
    metric_ids = [metric.metric_id for metric in metrics]
    if len(metric_ids) != len(set(metric_ids)):
        raise ValueError("metric registry metric_id values must be unique")
    active = [metric for metric in metrics if metric.status == "active"]
    active_roles = {metric.role for metric in active}
    if active_roles != ROLES:
        raise ValueError("active metric registry must cover all three metric roles")
    active_measurements = [metric.measurement for metric in active]
    if len(active_measurements) != len(set(active_measurements)):
        raise ValueError("active metric measurements must be unique")
    canonical = {
        "registry_version": REGISTRY_VERSION,
        "metrics": [metric.to_dict() for metric in metrics],
    }
    return MetricRegistry(
        version=REGISTRY_VERSION,
        metrics=metrics,
        checksum=_checksum(canonical),
    )


def load_error_analysis(
    path: Path,
    metric_registry: MetricRegistry | None = None,
) -> ErrorAnalysisArtifact:
    """Validate one human error-analysis artifact without exposing its notes."""

    payload = _load_json(path, label="error analysis")
    root = _strict_mapping(
        payload,
        required={
            "artifact_version",
            "analysis_id",
            "analysis_kind",
            "change_reference",
            "sampled_at",
            "sampling_strata",
            "reviewer",
            "cases",
            "clusters",
        },
        optional=set(),
        label="error analysis",
    )
    if root["artifact_version"] != ERROR_ANALYSIS_VERSION:
        raise ValueError("error analysis artifact_version is incompatible")
    analysis_id = _identifier(root["analysis_id"], label="analysis_id")
    analysis_kind = _enum_string(
        root["analysis_kind"], ANALYSIS_KINDS, label="analysis_kind"
    )
    _bounded_string(root["change_reference"], label="change_reference", maximum=256)
    sampled_at = _iso_date(root["sampled_at"], label="sampled_at")
    _bounded_string(root["reviewer"], label="reviewer", maximum=128)
    strata = _bounded_string_list(
        root["sampling_strata"], label="sampling_strata", minimum=1, maximum=16
    )
    if len(strata) != len(set(strata)):
        raise ValueError("sampling_strata values must be unique")
    raw_cases = _strict_list(root["cases"], label="error analysis cases")
    minimum = 30 if analysis_kind == "bootstrap" else 1
    maximum = 50 if analysis_kind == "bootstrap" else 100
    if not minimum <= len(raw_cases) <= maximum:
        raise ValueError(
            f"{analysis_kind} error analysis requires {minimum} to {maximum} cases"
        )
    cases = [_parse_analysis_case(item) for item in raw_cases]
    trace_ids = [item["trace_id"] for item in cases]
    if len(trace_ids) != len(set(trace_ids)):
        raise ValueError("error analysis trace_id values must be unique")
    failed_trace_ids = {
        item["trace_id"] for item in cases if item["overall_pass"] is False
    }
    raw_clusters = _strict_list(root["clusters"], label="error analysis clusters")
    clusters = [_parse_cluster(item) for item in raw_clusters]
    category_ids = [item["category_id"] for item in clusters]
    if len(category_ids) != len(set(category_ids)):
        raise ValueError("error analysis category_id values must be unique")
    clustered_trace_ids = [
        trace_id
        for cluster in clusters
        for trace_id in cast(list[str], cluster["trace_ids"])
    ]
    if len(clustered_trace_ids) != len(set(clustered_trace_ids)):
        raise ValueError("failed traces may belong to only one error cluster")
    if set(clustered_trace_ids) != failed_trace_ids:
        raise ValueError("error clusters must cover every and only failed trace")
    registry = metric_registry or load_metric_registry()
    linkable_metric_ids = {
        metric.metric_id
        for metric in registry.metrics
        if metric.status in {"candidate", "active"}
    }
    candidate_metric_ids = {
        cast(str, cluster["metric_id"])
        for cluster in clusters
        if cluster["metric_id"] is not None
    }
    if not candidate_metric_ids.issubset(linkable_metric_ids):
        raise ValueError(
            "metric candidates must link to candidate or active registry metrics"
        )
    canonical = {
        "artifact_version": ERROR_ANALYSIS_VERSION,
        "analysis_id": analysis_id,
        "analysis_kind": analysis_kind,
        "change_reference": root["change_reference"],
        "sampled_at": sampled_at,
        "sampling_strata": strata,
        "reviewer": root["reviewer"],
        "cases": cases,
        "clusters": clusters,
    }
    return ErrorAnalysisArtifact(
        analysis_id=analysis_id,
        analysis_kind=analysis_kind,
        sampled_at=sampled_at,
        case_count=len(cases),
        failed_case_count=len(failed_trace_ids),
        cluster_count=len(clusters),
        checksum=_checksum(canonical),
    )


def _parse_metric(value: object) -> MetricDefinition:
    required = {
        "metric_id",
        "role",
        "question",
        "source",
        "owner",
        "grader_id",
        "grader_version",
        "grader_layer",
        "dataset_segment",
        "measurement",
        "threshold_or_budget",
        "decision_action",
        "cost_class",
        "run_cadence",
        "retirement_condition",
        "status",
        "introduced_at",
        "last_reviewed_at",
    }
    raw = _strict_mapping(
        value,
        required=required,
        optional={"retirement_reason"},
        label="metric registry entry",
    )
    metric_id = _identifier(raw["metric_id"], label="metric_id")
    role = _enum_string(raw["role"], ROLES, label=f"{metric_id}.role")
    source = _strict_mapping(
        raw["source"],
        required={"kind", "ref"},
        optional=set(),
        label=f"{metric_id}.source",
    )
    source_kind = _enum_string(
        source["kind"], SOURCE_KINDS, label=f"{metric_id}.source.kind"
    )
    source_ref = _bounded_string(
        source["ref"], label=f"{metric_id}.source.ref", maximum=256
    )
    threshold = _parse_threshold(raw["threshold_or_budget"], metric_id=metric_id)
    actions = _bounded_string_list(
        raw["decision_action"],
        label=f"{metric_id}.decision_action",
        minimum=1,
        maximum=5,
    )
    if len(actions) != len(set(actions)) or not set(actions).issubset(DECISION_ACTIONS):
        raise ValueError(f"{metric_id}.decision_action values are invalid")
    status = _enum_string(raw["status"], METRIC_STATUSES, label=f"{metric_id}.status")
    retirement_reason_value = raw.get("retirement_reason")
    retirement_reason = (
        None
        if retirement_reason_value is None
        else _bounded_string(
            retirement_reason_value,
            label=f"{metric_id}.retirement_reason",
            maximum=512,
        )
    )
    if status == "retired" and retirement_reason is None:
        raise ValueError(f"{metric_id} retired metric requires retirement_reason")
    if status != "retired" and retirement_reason is not None:
        raise ValueError(f"{metric_id} non-retired metric forbids retirement_reason")
    introduced_at = _iso_date(raw["introduced_at"], label=f"{metric_id}.introduced_at")
    last_reviewed_at = _iso_date(
        raw["last_reviewed_at"], label=f"{metric_id}.last_reviewed_at"
    )
    if last_reviewed_at < introduced_at:
        raise ValueError(f"{metric_id}.last_reviewed_at precedes introduced_at")
    return MetricDefinition(
        metric_id=metric_id,
        role=role,
        question=_bounded_string(
            raw["question"], label=f"{metric_id}.question", maximum=512
        ),
        source_kind=source_kind,
        source_ref=source_ref,
        owner=_bounded_string(raw["owner"], label=f"{metric_id}.owner", maximum=128),
        grader_id=_identifier(raw["grader_id"], label=f"{metric_id}.grader_id"),
        grader_version=_bounded_string(
            raw["grader_version"],
            label=f"{metric_id}.grader_version",
            maximum=64,
        ),
        grader_layer=_enum_string(
            raw["grader_layer"],
            GRADER_LAYERS,
            label=f"{metric_id}.grader_layer",
        ),
        dataset_segment=_enum_string(
            raw["dataset_segment"],
            DATASET_SEGMENTS,
            label=f"{metric_id}.dataset_segment",
        ),
        measurement=_enum_string(
            raw["measurement"], MEASUREMENTS, label=f"{metric_id}.measurement"
        ),
        threshold_or_budget=threshold,
        decision_actions=tuple(actions),
        cost_class=_enum_string(
            raw["cost_class"], COST_CLASSES, label=f"{metric_id}.cost_class"
        ),
        run_cadence=_enum_string(
            raw["run_cadence"], RUN_CADENCES, label=f"{metric_id}.run_cadence"
        ),
        retirement_condition=_bounded_string(
            raw["retirement_condition"],
            label=f"{metric_id}.retirement_condition",
            maximum=256,
        ),
        status=status,
        introduced_at=introduced_at,
        last_reviewed_at=last_reviewed_at,
        retirement_reason=retirement_reason,
    )


def _parse_threshold(value: object, *, metric_id: str) -> ThresholdOrBudget:
    raw = _strict_mapping(
        value,
        required={"mode", "unit"},
        optional={"operator", "value"},
        label=f"{metric_id}.threshold_or_budget",
    )
    mode = _enum_string(
        raw["mode"], THRESHOLD_MODES, label=f"{metric_id}.threshold_or_budget.mode"
    )
    unit = _bounded_string(
        raw["unit"], label=f"{metric_id}.threshold_or_budget.unit", maximum=64
    )
    if mode == "baseline_only":
        if "operator" in raw or "value" in raw:
            raise ValueError(f"{metric_id} baseline_only metric forbids operator/value")
        return ThresholdOrBudget(mode=mode, unit=unit, operator=None, value=None)
    if set(raw) != {"mode", "unit", "operator", "value"}:
        raise ValueError(f"{metric_id} gate/budget requires operator and value")
    operator = _enum_string(
        raw["operator"],
        THRESHOLD_OPERATORS,
        label=f"{metric_id}.threshold_or_budget.operator",
    )
    number = raw["value"]
    if isinstance(number, bool) or not isinstance(number, (int, float)):
        raise ValueError(f"{metric_id}.threshold_or_budget.value must be numeric")
    value_as_float = float(number)
    if not math.isfinite(value_as_float):
        raise ValueError(f"{metric_id}.threshold_or_budget.value must be finite")
    return ThresholdOrBudget(
        mode=mode,
        unit=unit,
        operator=operator,
        value=value_as_float,
    )


def _parse_analysis_case(value: object) -> dict[str, object]:
    raw = _strict_mapping(
        value,
        required={"trace_id", "case_id", "overall_pass", "review_note"},
        optional=set(),
        label="error analysis case",
    )
    if not isinstance(raw["overall_pass"], bool):
        raise ValueError("error analysis overall_pass must be boolean")
    return {
        "trace_id": _identifier(raw["trace_id"], label="trace_id"),
        "case_id": _identifier(raw["case_id"], label="case_id"),
        "overall_pass": raw["overall_pass"],
        "review_note": _bounded_string(
            raw["review_note"], label="review_note", maximum=500
        ),
    }


def _parse_cluster(value: object) -> dict[str, object]:
    raw = _strict_mapping(
        value,
        required={
            "category_id",
            "name",
            "trace_ids",
            "count",
            "severity",
            "generalization_failure",
            "disposition",
            "metric_id",
        },
        optional=set(),
        label="error analysis cluster",
    )
    trace_ids = _bounded_string_list(
        raw["trace_ids"], label="cluster.trace_ids", minimum=1, maximum=100
    )
    if len(trace_ids) != len(set(trace_ids)):
        raise ValueError("cluster.trace_ids values must be unique")
    count = raw["count"]
    if isinstance(count, bool) or not isinstance(count, int) or count != len(trace_ids):
        raise ValueError("error analysis cluster count must match trace_ids")
    if not isinstance(raw["generalization_failure"], bool):
        raise ValueError("cluster.generalization_failure must be boolean")
    generalization_failure = raw["generalization_failure"]
    disposition = _enum_string(
        raw["disposition"], CLUSTER_DISPOSITIONS, label="cluster.disposition"
    )
    metric_id_value = raw["metric_id"]
    metric_id = (
        None
        if metric_id_value is None
        else _identifier(metric_id_value, label="cluster.metric_id")
    )
    if (disposition == "metric_candidate") != (metric_id is not None):
        raise ValueError(
            "metric_candidate clusters require metric_id and other clusters forbid it"
        )
    if disposition == "metric_candidate" and generalization_failure is not True:
        raise ValueError("metric_candidate clusters must be generalization failures")
    return {
        "category_id": _identifier(raw["category_id"], label="category_id"),
        "name": _bounded_string(raw["name"], label="cluster.name", maximum=128),
        "trace_ids": trace_ids,
        "count": count,
        "severity": _enum_string(
            raw["severity"], CLUSTER_SEVERITIES, label="cluster.severity"
        ),
        "generalization_failure": generalization_failure,
        "disposition": disposition,
        "metric_id": metric_id,
    }


def _load_json(path: Path, *, label: str) -> object:
    try:
        return cast(object, json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to load {label} {path}: {exc}") from exc


def _strict_mapping(
    value: object,
    *,
    required: set[str],
    optional: set[str],
    label: str,
) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object")
    output = cast(dict[str, object], value)
    fields = set(output)
    if not required.issubset(fields) or not fields.issubset(required | optional):
        raise ValueError(f"{label} fields are invalid")
    return output


def _strict_list(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return cast(list[object], value)


def _bounded_string(value: object, *, label: str, maximum: int) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError(f"{label} must be a bounded non-empty string")
    return value


def _identifier(value: object, *, label: str) -> str:
    output = _bounded_string(value, label=label, maximum=128)
    if IDENTIFIER_PATTERN.fullmatch(output) is None:
        raise ValueError(f"{label} must be a registered identifier")
    return output


def _enum_string(value: object, allowed: frozenset[str], *, label: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{label} is invalid")
    return value


def _bounded_string_list(
    value: object,
    *,
    label: str,
    minimum: int,
    maximum: int,
) -> list[str]:
    raw = _strict_list(value, label=label)
    if not minimum <= len(raw) <= maximum or any(
        not isinstance(item, str) or not item or len(item) > 128 for item in raw
    ):
        raise ValueError(f"{label} must contain bounded non-empty strings")
    return [cast(str, item) for item in raw]


def _iso_date(value: object, *, label: str) -> str:
    output = _bounded_string(value, label=label, maximum=10)
    try:
        parsed = date.fromisoformat(output)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO date") from exc
    if parsed.isoformat() != output:
        raise ValueError(f"{label} must use YYYY-MM-DD")
    return output


def _checksum(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _metric_id(metric: MetricDefinition) -> str:
    return metric.metric_id


def main(argv: list[str] | None = None) -> int:
    """Validate governance inputs and print bounded metadata summaries."""

    parser = argparse.ArgumentParser(
        description="Validate eval metric-registry and error-analysis artifacts."
    )
    parser.add_argument(
        "--metric-registry",
        type=Path,
        default=DEFAULT_METRIC_REGISTRY_PATH,
    )
    parser.add_argument("--error-analysis", type=Path)
    args = parser.parse_args(argv)

    registry = load_metric_registry(args.metric_registry)
    output: dict[str, object] = {
        "metric_registry": {
            "version": registry.version,
            "checksum": registry.checksum,
            "active_counts": {
                role: sum(
                    1 for metric in registry.active_metrics if metric.role == role
                )
                for role in ("goal", "guardrail", "operational")
            },
        }
    }
    if args.error_analysis is not None:
        analysis = load_error_analysis(args.error_analysis, registry)
        output["error_analysis"] = {
            "version": ERROR_ANALYSIS_VERSION,
            "analysis_id": analysis.analysis_id,
            "analysis_kind": analysis.analysis_kind,
            "sampled_at": analysis.sampled_at,
            "case_count": analysis.case_count,
            "failed_case_count": analysis.failed_case_count,
            "cluster_count": analysis.cluster_count,
            "checksum": analysis.checksum,
        }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
