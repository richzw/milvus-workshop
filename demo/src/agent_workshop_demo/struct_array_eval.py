"""Isolated comparative evaluation for StructArray retrieval profiles."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, cast

from agent_workshop_demo.models import KBChunk, SearchResult
from agent_workshop_demo.retrieval import InMemoryHybridRetriever
from agent_workshop_demo.struct_array import (
    ElementPredicate,
    ElementOperator,
    InMemoryStructArrayRetriever,
    ProjectionBuild,
    StructArrayProfile,
    FUSION_RECIPE,
)

STRUCT_ARRAY_EVAL_VERSION: Final = "struct-array-eval-v1"
STRUCT_ARRAY_CASE_VERSION: Final = "struct-array-eval-cases-v1"


def evaluate_struct_array_profiles(
    *,
    cases_path: Path,
    chunks: list[KBChunk],
    projection: ProjectionBuild,
    top_k: int = 20,
    hardware_note: str = "local deterministic workshop runtime",
    profile_retrievers: Mapping[str, object] | None = None,
    execution_mode: str = "local_emulation",
) -> dict[str, Any]:
    """Compare flat and three StructArray profiles over one fixed corpus."""

    if not 1 <= top_k <= 64:
        raise ValueError("StructArray eval top_k must be between 1 and 64")
    cases, fixture_checksum = _load_cases(cases_path)
    flat = InMemoryHybridRetriever(chunks)
    local_profiles: dict[str, object] = {
        "flat_hybrid": flat,
        StructArrayProfile.ELEMENT.value: InMemoryStructArrayRetriever(
            flat, projection, profile=StructArrayProfile.ELEMENT
        ),
        StructArrayProfile.TWO_STAGE.value: InMemoryStructArrayRetriever(
            flat, projection, profile=StructArrayProfile.TWO_STAGE
        ),
        StructArrayProfile.FUSED.value: InMemoryStructArrayRetriever(
            flat, projection, profile=StructArrayProfile.FUSED
        ),
    }
    profiles = dict(profile_retrievers or local_profiles)
    expected_profiles = {
        "flat_hybrid",
        StructArrayProfile.ELEMENT.value,
        StructArrayProfile.TWO_STAGE.value,
        StructArrayProfile.FUSED.value,
    }
    if set(profiles) != expected_profiles:
        raise ValueError("StructArray eval requires exactly four registered profiles")
    corpus_by_id = {item.chunk_id: item for item in chunks}
    passage_by_id = {
        passage.chunk_id: passage
        for parent in projection.parents
        for passage in parent.passages
    }
    unknown_expected = sorted(
        {
            chunk_id
            for case in cases
            for chunk_id in _string_list(
                case["expected_chunk_ids"], "expected_chunk_ids"
            )
            if chunk_id not in corpus_by_id
        }
    )
    if unknown_expected:
        raise ValueError(
            "StructArray eval expected chunks are absent from the corpus: "
            + ", ".join(unknown_expected)
        )
    profile_reports: list[dict[str, Any]] = []
    for profile_name, retriever in profiles.items():
        case_reports: list[dict[str, Any]] = []
        latencies: list[float] = []
        for case in cases:
            started = time.perf_counter()
            groups, parent_count = _run_case(
                profile_name,
                retriever,
                case,
                top_k=top_k,
            )
            latencies.append((time.perf_counter() - started) * 1000)
            results = _dedupe([item for group in groups for item in group], top_k)
            expected = set(
                _string_list(case["expected_chunk_ids"], "expected_chunk_ids")
            )
            retrieved = [item.chunk.chunk_id for item in results]
            hits = expected.intersection(retrieved)
            filters = cast(dict[str, Any], case["filters"])
            expected_docs = {
                (corpus_by_id[item].doc_id, corpus_by_id[item].doc_version)
                for item in expected
                if item in corpus_by_id
            }
            retrieved_docs = {
                (item.chunk.doc_id, item.chunk.doc_version) for item in results
            }
            resolved = [
                item
                for item in results
                if item.chunk.chunk_id in corpus_by_id
                and item.chunk.checksum == corpus_by_id[item.chunk.chunk_id].checksum
            ]
            violations = [
                item for item in results if not _matches_filters(item, filters)
            ]
            offset_failures = [
                item
                for item in results
                if item.struct_field is not None and item.element_offset is None
            ]
            predicates = tuple(
                _predicate_from_mapping(item)
                for item in case.get("element_predicates", [])
            )
            predicate_failures = [
                item
                for item in results
                if predicates
                and (
                    item.chunk.chunk_id not in passage_by_id
                    or not _projected_passage_matches(
                        passage_by_id[item.chunk.chunk_id], predicates
                    )
                )
            ]
            parent_counts = Counter(
                item.document_key or f"{item.chunk.doc_id}\0{item.chunk.doc_version}"
                for item in results
            )
            aspect_hits = [
                bool(expected.intersection(item.chunk.chunk_id for item in group))
                for group in groups
            ]
            case_reports.append(
                {
                    "case_id": case["case_id"],
                    "expected_chunk_ids": sorted(expected),
                    "retrieved_chunk_ids": retrieved,
                    "passage_recall_at_k": round(len(hits) / len(expected), 4),
                    "selected_context_recall_at_5": round(
                        len(expected.intersection(retrieved[:5])) / len(expected), 4
                    ),
                    "document_recall_at_k": round(
                        len(expected_docs.intersection(retrieved_docs))
                        / max(1, len(expected_docs)),
                        4,
                    ),
                    "aspect_resolution_rate": round(
                        sum(aspect_hits) / max(1, len(aspect_hits)), 4
                    ),
                    "citation_resolve_rate": (
                        round(len(resolved) / len(results), 4) if results else None
                    ),
                    "citation_resolution_status": (
                        "measured" if results else "evaluation_incomplete"
                    ),
                    "citation_resolution_reason": (
                        None if results else "no_retrieved_citations"
                    ),
                    "parent_candidate_count": parent_count,
                    "parent_only_evidence_count": sum(
                        item.result_granularity != "passage" for item in results
                    ),
                    "offset_resolution_failures": len(offset_failures),
                    "permission_version_violations": len(violations),
                    "same_element_predicate_failures": len(predicate_failures),
                    "duplicate_parent_distribution": {
                        "parents_with_multiple_hits": sum(
                            count > 1 for count in parent_counts.values()
                        ),
                        "max_hits_per_parent": max(parent_counts.values(), default=0),
                    },
                }
            )
        profile_reports.append(
            {
                "profile": profile_name,
                "support_status": "supported",
                "execution_mode": execution_mode,
                "fusion_recipe": (
                    FUSION_RECIPE
                    if profile_name == StructArrayProfile.FUSED.value
                    else None
                ),
                "mean_passage_recall_at_k": _mean(
                    [float(item["passage_recall_at_k"]) for item in case_reports]
                ),
                "citation_resolve_rate": _mean(
                    [
                        float(item["citation_resolve_rate"])
                        for item in case_reports
                        if item["citation_resolve_rate"] is not None
                    ]
                ),
                "citation_resolution_status": (
                    "measured"
                    if any(
                        item["citation_resolve_rate"] is not None
                        for item in case_reports
                    )
                    else "evaluation_incomplete"
                ),
                "parent_only_evidence_count": sum(
                    int(item["parent_only_evidence_count"]) for item in case_reports
                ),
                "latency_ms": {
                    "p50": _percentile(latencies, 0.50),
                    "p95": _percentile(latencies, 0.95),
                    "samples": len(latencies),
                },
                "cases": case_reports,
            }
        )
    vector_bytes = projection.passage_count * 2 * 1024 * 4
    report = {
        "schema_version": STRUCT_ARRAY_EVAL_VERSION,
        "fixture_checksum": fixture_checksum,
        "corpus_fingerprint": _corpus_fingerprint(chunks),
        "projection_fingerprint": projection.projection_fingerprint,
        "embedding_fingerprint": projection.text_embedding_fingerprint,
        "hardware_runtime_note": hardware_note,
        "configured_profiles": list(profiles),
        "top_k": top_k,
        "build_storage_observations": {
            "parent_count": projection.parent_count,
            "passage_count": projection.passage_count,
            "duplicated_vector_bytes": vector_bytes,
            "native_index_bytes": None,
            "native_build_time_ms": None,
            "native_peak_resource": None,
            "native_observation_status": "evaluation_incomplete",
            "native_observation_reason": (
                "offline_local_emulation"
                if execution_mode == "local_emulation"
                else "read_only_native_mode_does_not_collect_build_or_index_telemetry"
            ),
        },
        "maxsim_reference": {
            "status": "evaluation_incomplete",
            "reason": "synthetic_probe_only",
            "retrieval_ann_ratio": 3.0,
            "emb_list_rerank": True,
        },
        "end_to_end_quality": {
            "status": "evaluation_incomplete",
            "reason": "same_reranker_generator_required_fact_and_final_answer_graders_not_connected",
            "required_fact_recall": None,
            "final_answer_correctness": None,
            "final_answer_citation_correctness": None,
            "end_to_end_latency_ms": None,
        },
        "profiles": profile_reports,
    }
    validate_struct_array_eval_report(report)
    return report


def _run_case(
    profile_name: str,
    retriever: object,
    case: dict[str, Any],
    *,
    top_k: int,
) -> tuple[list[list[SearchResult]], int]:
    queries = _string_list(case["queries"], "queries")
    filters = case.get("filters")
    if not isinstance(filters, dict):
        raise ValueError("StructArray eval filters must be an object")
    if profile_name == "flat_hybrid":
        search = getattr(retriever, "search", None)
        if not callable(search):
            raise TypeError("flat_hybrid retriever lacks search")
        return [
            list(search(query, top_k=top_k, filters=filters)) for query in queries
        ], 0
    search_profile = getattr(retriever, "search_profile", None)
    if not callable(search_profile):
        raise TypeError("StructArray profile retriever lacks search_profile")
    predicates = tuple(
        _predicate_from_mapping(item) for item in case.get("element_predicates", [])
    )
    run = search_profile(
        queries,
        top_k=top_k,
        filters=filters,
        element_predicates=predicates,
    )
    return [list(group) for group in run.results_by_query], len(run.document_candidates)


def _load_cases(path: Path) -> tuple[list[dict[str, Any]], str]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Unable to read StructArray eval cases") from exc
    if not isinstance(payload, Mapping) or set(payload) != {"schema_version", "cases"}:
        raise ValueError("StructArray eval fixture fields do not match the schema")
    if payload.get("schema_version") != STRUCT_ARRAY_CASE_VERSION:
        raise ValueError("Unsupported StructArray eval case schema_version")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("StructArray eval cases must be a non-empty list")
    allowed = {
        "case_id",
        "queries",
        "filters",
        "element_predicates",
        "expected_chunk_ids",
    }
    cases: list[dict[str, Any]] = []
    ids: list[str] = []
    for value in raw_cases:
        if not isinstance(value, dict) or not set(value).issubset(allowed):
            raise ValueError("StructArray eval case has unknown fields")
        case_id = value.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("StructArray eval case_id must be non-empty")
        _string_list(value.get("queries"), "queries")
        _string_list(value.get("expected_chunk_ids"), "expected_chunk_ids")
        ids.append(case_id)
        cases.append(dict(value))
    if len(ids) != len(set(ids)):
        raise ValueError("StructArray eval case ids must be unique")
    return cases, hashlib.sha256(raw).hexdigest()


def _predicate_from_mapping(value: object) -> ElementPredicate:
    if not isinstance(value, Mapping) or set(value) != {"field", "operator", "value"}:
        raise ValueError("StructArray eval element predicate is invalid")
    field = value["field"]
    operator = value["operator"]
    predicate_value = value["value"]
    if not isinstance(field, str) or not isinstance(operator, str):
        raise ValueError("StructArray eval predicate field/operator are invalid")
    if operator not in {"eq", "ne", "gt", "gte", "lt", "lte"}:
        raise ValueError("StructArray eval predicate operator is invalid")
    if isinstance(predicate_value, bool) or not isinstance(predicate_value, (str, int)):
        raise ValueError("StructArray eval predicate value is invalid")
    return ElementPredicate(
        field,
        cast(ElementOperator, operator),
        predicate_value,
    )


def _string_list(value: object, field: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ValueError(f"{field} must be a non-empty string list")
    return list(value)


def _dedupe(results: list[SearchResult], top_k: int) -> list[SearchResult]:
    output: list[SearchResult] = []
    seen: set[str] = set()
    for result in results:
        if result.chunk.chunk_id in seen:
            continue
        seen.add(result.chunk.chunk_id)
        output.append(result)
        if len(output) == top_k:
            break
    return output


def _matches_filters(result: SearchResult, filters: Mapping[str, Any]) -> bool:
    chunk = result.chunk
    for field, expected in filters.items():
        actual = getattr(chunk, field, None)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def _projected_passage_matches(
    passage: object, predicates: tuple[ElementPredicate, ...]
) -> bool:
    operators = {
        "eq": lambda left, right: left == right,
        "ne": lambda left, right: left != right,
        "gt": lambda left, right: left > right,
        "gte": lambda left, right: left >= right,
        "lt": lambda left, right: left < right,
        "lte": lambda left, right: left <= right,
    }
    return all(
        operators[item.operator](getattr(passage, item.field), item.value)
        for item in predicates
    )


def validate_struct_array_eval_report(report: Mapping[str, Any]) -> None:
    """Fail closed on missing, fabricated, or non-finite report fields."""

    required = {
        "schema_version",
        "fixture_checksum",
        "corpus_fingerprint",
        "projection_fingerprint",
        "embedding_fingerprint",
        "hardware_runtime_note",
        "configured_profiles",
        "top_k",
        "build_storage_observations",
        "maxsim_reference",
        "end_to_end_quality",
        "profiles",
    }
    if (
        set(report) != required
        or report.get("schema_version") != STRUCT_ARRAY_EVAL_VERSION
    ):
        raise ValueError("StructArray eval report shape is invalid")
    profiles = report.get("profiles")
    if not isinstance(profiles, list) or len(profiles) != 4:
        raise ValueError("StructArray eval report must contain four profiles")
    for profile in profiles:
        if not isinstance(profile, Mapping):
            raise ValueError("StructArray eval profile is invalid")
        for metric in ("mean_passage_recall_at_k", "citation_resolve_rate"):
            value = profile.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"StructArray eval {metric} must be finite")
        cases = profile.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ValueError("StructArray eval profile cases are missing")
        for case in cases:
            if not isinstance(case, Mapping):
                raise ValueError("StructArray eval case report is invalid")
            for metric in (
                "passage_recall_at_k",
                "selected_context_recall_at_5",
                "document_recall_at_k",
                "aspect_resolution_rate",
                "citation_resolve_rate",
            ):
                value = case.get(metric)
                if metric == "citation_resolve_rate" and value is None:
                    if (
                        case.get("citation_resolution_status")
                        != "evaluation_incomplete"
                    ):
                        raise ValueError(
                            "Missing citation denominator requires incomplete status"
                        )
                    continue
                if (
                    not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or not 0 <= float(value) <= 1
                ):
                    raise ValueError(f"StructArray eval case {metric} is invalid")
    end_to_end = report.get("end_to_end_quality")
    if not isinstance(end_to_end, Mapping) or (
        end_to_end.get("status") != "evaluation_incomplete"
        or not end_to_end.get("reason")
    ):
        raise ValueError("Uncollected end-to-end metrics require an incomplete reason")


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return round(ordered[index], 3)


def _corpus_fingerprint(chunks: list[KBChunk]) -> str:
    payload = [
        [item.chunk_id, item.checksum]
        for item in sorted(chunks, key=lambda item: item.chunk_id)
    ]
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
