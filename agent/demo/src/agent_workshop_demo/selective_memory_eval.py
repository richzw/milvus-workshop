"""Executable, payload-free evaluation for selective Memory."""

from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from agent_workshop_demo.selective_memory import (
    LocalSelectiveMemoryStore,
    SelectiveMemoryError,
    SelectiveMemoryService,
)

CASE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
SCENARIOS = {
    "explicit_preference", "ordinary_turn", "correction", "conflict", "replay",
    "stale_filter",
}


class _PostWriteFailureStore(LocalSelectiveMemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    def append_events(self, events: Any) -> int:
        count = super().append_events(events)
        if not self.failed and any(e.event_type == "memory_promoted" for e in events):
            self.failed = True
            raise SelectiveMemoryError("injected post-write failure")
        return count


def evaluate_selective_memory(path: Path) -> dict[str, Any]:
    """Execute registered scenarios against the real local service."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "cases"}
        or payload["schema_version"] != "selective-memory-eval-v2"
        or not isinstance(payload["cases"], list)
        or not 1 <= len(payload["cases"]) <= 100
    ):
        raise ValueError("Invalid selective-Memory eval fixture")
    cases = [_validate_case(item) for item in payload["cases"]]
    if len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Selective-Memory eval case_id values must be unique")
    previous_provider = os.environ.get("EMBEDDING_PROVIDER")
    os.environ["EMBEDDING_PROVIDER"] = "deterministic"
    try:
        observations = [_run_case(case) for case in cases]
    finally:
        if previous_provider is None:
            os.environ.pop("EMBEDDING_PROVIDER", None)
        else:
            os.environ["EMBEDDING_PROVIDER"] = previous_provider
    tp = sum(o["expected_retain"] and o["actual_retained"] for o in observations)
    fp = sum(not o["expected_retain"] and o["actual_retained"] for o in observations)
    fn = sum(o["expected_retain"] and not o["actual_retained"] for o in observations)
    event_classes = sorted({o["event_class"] for o in observations})
    return {
        "schema_version": "selective-memory-eval-report-v2",
        "provenance": {
            "runner": "LocalSelectiveMemoryStore",
            "decay_mode": "application",
            "workflow": "SelectiveMemoryService",
            "ranking_parity": None,
        },
        "case_count": len(observations),
        "selection_precision": _ratio(tp, tp + fp),
        "selection_recall": _ratio(tp, tp + fn),
        "selection_by_event_class": {
            name: {
                "precision": _selection_metric(observations, name, "precision"),
                "recall": _selection_metric(observations, name, "recall"),
            }
            for name in event_classes
        },
        "active_fact_precision": _ratio(
            sum(o["expected_active_fact"] and o["actual_active_fact"] for o in observations),
            sum(o["actual_active_fact"] for o in observations),
        ),
        "correction_accuracy": _mean_expected(
            observations, "expected_correction", "correction_superseded"
        ),
        "conflict_detection_accuracy": _ratio(
            sum(o["expected_conflict"] == o["actual_conflict"] for o in observations),
            len(observations),
        ),
        "relevant_memory_recall_before_decay": _mean(
            o["recall_before"] for o in observations
        ),
        "relevant_memory_recall_after_decay": _mean(
            o["recall_after"] for o in observations
        ),
        "stale_memory_intrusion_rate": _ratio(
            sum(o["stale_intrusions"] for o in observations),
            sum(o["stale_opportunities"] for o in observations),
        ),
        "lineage_coverage": _ratio(
            sum(o["resolved_sources"] for o in observations),
            sum(o["source_count"] for o in observations),
        ),
        "memory_pack_average_records": round(
            sum(o["pack_records"] for o in observations) / len(observations), 4
        ),
        "memory_pack_average_characters": round(
            sum(o["pack_characters"] for o in observations) / len(observations), 4
        ),
        "memory_pack_truncation_rate": _ratio(
            sum(o["truncated_count"] > 0 for o in observations), len(observations)
        ),
        "memory_pack_size_violation_rate": _ratio(
            sum(o["pack_records"] > 12 for o in observations), len(observations)
        ),
        "consolidation_exact_once_accuracy": _ratio(
            sum(o["exact_once"] for o in observations if o["expected_consolidation"]),
            sum(o["expected_consolidation"] for o in observations),
        ),
        "cases": [
            {
                "case_id": o["case_id"],
                "event_class": o["event_class"],
                "actual_retained": o["actual_retained"],
            }
            for o in observations
        ],
    }


def _run_case(case: dict[str, Any]) -> dict[str, Any]:
    scenario = case["scenario"]
    store = (
        _PostWriteFailureStore()
        if scenario == "replay"
        else LocalSelectiveMemoryStore()
    )
    service = SelectiveMemoryService(store)
    turns = {
        "explicit_preference": [("q1", "请记住以后用中文", "以后用中文")],
        "ordinary_turn": [("q1", "今天天气不错", None)],
        "correction": [
            ("q1", "请记住我叫张三", "我叫张三"),
            ("q2", "更正，不是张三是李四", None),
        ],
        "conflict": [
            ("q1", "请记住我叫张三", "我叫张三"),
            ("q2", "请记住我叫李四", "我叫李四"),
        ],
        "replay": [("q1", "请记住以后用中文", "以后用中文")],
        "stale_filter": [("q1", "请记住以后用中文", "以后用中文")],
    }[scenario]
    results = []
    for index, (query_id, query, remembered) in enumerate(turns, 1):
        results.append(
            service.persist_turn(
                session_id="eval_session",
                query_id=query_id,
                query=query,
                answer="evaluated",
                terminal_status="memory_saved",
                remembered_statement=remembered,
                now_ms=index * 1_000,
            )
        )
    if scenario == "replay":
        service.drain_consolidation_outbox("eval_session", now_ms=2_000)
    if scenario == "stale_filter":
        active = next(iter(store.facts.values()))
        store.upsert_facts((replace(active, status="superseded"),))
    before = service.recall(
        "你还记得吗",
        session_id="eval_session",
        now_ms=3_000,
        include_episodes=True,
    )
    after = service.recall(
        "你还记得吗",
        session_id="eval_session",
        now_ms=400 * 86_400_000,
        include_episodes=True,
    )
    facts = list(store.facts.values())
    relevant_value = {
        "explicit_preference": "zh-CN",
        "replay": "zh-CN",
        "correction": "李四",
    }.get(scenario)
    source_ids = [source for fact in facts for source in fact.source_event_ids]
    stale = sum(f.status in {"superseded", "tombstoned"} for f in before.durable_facts)
    return {
        **case,
        "event_class": next(
            event.event_type
            for event in store.events.values()
            if event.query_id == turns[-1][0] and event.parent_event_id is None
        ),
        "actual_retained": results[-1].retention_class != "ephemeral",
        "actual_active_fact": any(f.status == "active" for f in facts),
        "correction_superseded": any(f.status == "superseded" for f in facts),
        "actual_conflict": any(f.status == "disputed" for f in facts),
        "recall_before": (
            (1.0 if any(f.value == relevant_value for f in before.durable_facts) else 0.0)
            if case["expected_active_fact"]
            else None
        ),
        "recall_after": (
            (1.0 if any(f.value == relevant_value for f in after.durable_facts) else 0.0)
            if case["expected_active_fact"]
            else None
        ),
        "stale_intrusions": stale,
        "stale_opportunities": sum(
            fact.status in {"superseded", "tombstoned"} for fact in facts
        ),
        "resolved_sources": sum(source in store.events for source in source_ids),
        "source_count": len(source_ids),
        "pack_records": len(before.durable_facts) + len(before.recent_episodes),
        "pack_characters": len(before.rendered_context),
        "truncated_count": before.truncated_count,
        "expected_consolidation": scenario not in {"ordinary_turn", "stale_filter"},
        "exact_once": (
            not store.list_pending_consolidations("eval_session", limit=20)
            and len(store.facts) == len({fact.memory_id for fact in store.facts.values()})
            and (
                scenario != "replay"
                or (len(facts) == 1 and facts[0].revision == 1)
            )
            and len(
                [
                    event
                    for event in store.events.values()
                    if event.event_type in {"memory_promoted", "memory_reconfirmed"}
                ]
            )
            == sum(result.consolidation_status in {"completed", "pending_retry"} for result in results)
            and service.drain_consolidation_outbox("eval_session", now_ms=9_000) == 0
        ),
    }


def _validate_case(value: object) -> dict[str, Any]:
    required = {
        "case_id", "scenario", "expected_retain", "expected_active_fact",
        "expected_correction", "expected_conflict",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Invalid selective-Memory eval case shape")
    if not isinstance(value["case_id"], str) or not CASE_ID.fullmatch(value["case_id"]):
        raise ValueError("Invalid selective-Memory eval case_id")
    if value["scenario"] not in SCENARIOS:
        raise ValueError("Invalid selective-Memory eval scenario")
    if any(not isinstance(value[field], bool) for field in required if field.startswith("expected_")):
        raise ValueError("Invalid selective-Memory eval expectation")
    return dict(value)


def _ratio(n: int, d: int) -> float | None:
    return None if d == 0 else round(n / d, 4)


def _mean(values: Any) -> float | None:
    items = [value for value in values if value is not None]
    return None if not items else round(sum(items) / len(items), 4)


def _mean_expected(rows: list[dict[str, Any]], expected: str, actual: str) -> float | None:
    selected = [row for row in rows if row[expected]]
    return _ratio(sum(row[actual] for row in selected), len(selected))


def _selection_metric(
    rows: list[dict[str, Any]], event_class: str, metric: str
) -> float | None:
    selected = [row for row in rows if row["event_class"] == event_class]
    tp = sum(row["expected_retain"] and row["actual_retained"] for row in selected)
    if metric == "precision":
        return _ratio(tp, sum(row["actual_retained"] for row in selected))
    return _ratio(tp, sum(row["expected_retain"] for row in selected))


__all__ = ["evaluate_selective_memory"]
