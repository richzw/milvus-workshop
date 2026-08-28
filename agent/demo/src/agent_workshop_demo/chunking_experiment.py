"""Repeatable Min-Max chunking comparison over stable retrieval anchors."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agent_workshop_demo.chunking import (
    TOKEN_PATTERN,
    ChunkingConfig,
    count_chunk_tokens,
)
from agent_workshop_demo.config import VECTOR_DIMS
from agent_workshop_demo.embedding import text_embedding_fingerprint
from agent_workshop_demo.image_embedding import (
    DeterministicImageEmbeddingProvider,
    ImageEmbeddingProvider,
)
from agent_workshop_demo.ingestion import ingest_demo_sources
from agent_workshop_demo.models import KBChunk
from agent_workshop_demo.query_transform import IdentityQueryTransformer
from agent_workshop_demo.reranker import RuleBasedReranker
from agent_workshop_demo.retrieval import InMemoryHybridRetriever
from agent_workshop_demo.validation import normalize_filters
from agent_workshop_demo.workflow import AgenticRAGWorkflow

CONFIG_SCHEMA_VERSION = "chunking-experiment-v2"
ANCHOR_SCHEMA_VERSION = "chunking-anchors-v2"
RECOMMENDATION_SCHEMA_VERSION = "chunking-recommendation-v1"
NEAR_DUPLICATE_THRESHOLD = 0.85
QUALITY_GATE_PROFILE = "rag-eval-baseline-2026-08-05-minus-explicit-tolerance-v1"
COMMITTED_QUALITY_GATES: dict[str, float] = {
    "retrieval_recall_at_20": 0.90,
    "selected_context_recall_at_5": 0.90,
    "citation_precision": 0.70,
    "citation_coverage": 0.90,
    "required_fact_coverage": 0.90,
    "abstention_correctness": 1.0,
    "markdown_section_preservation_rate": 1.0,
    "release_boundary_preservation_rate": 1.0,
    "pdf_page_preservation_rate": 1.0,
}


def run_chunking_experiment(
    *,
    configs_path: Path,
    anchors_path: Path,
    local_dir: Path,
    mock_s3_dir: Path,
    image_provider: ImageEmbeddingProvider | None = None,
    clock: Callable[[], float] = time.perf_counter,
    semantic_grader: Callable[[str, str, list[str]], dict[str, float]] | None = None,
    grader_calibrated: bool = False,
    semantic_grader_profile: str | None = None,
    recommendation_path: Path | None = None,
) -> dict[str, Any]:
    """Run isolated end-to-end evaluation for every strict chunk profile."""

    if semantic_grader is not None and not grader_calibrated:
        raise ValueError("semantic grader must be calibrated before use")
    if semantic_grader is not None and (
        semantic_grader_profile is None
        or not semantic_grader_profile.strip()
        or len(semantic_grader_profile) > 200
    ):
        raise ValueError("calibrated semantic grader requires a bounded profile")
    if semantic_grader is None and semantic_grader_profile is not None:
        raise ValueError("semantic grader profile requires a grader")

    configs = load_chunking_configs(configs_path)
    anchors = _load_anchor_cases(anchors_path)
    experiment_fingerprint = _experiment_fingerprint(
        configs_path,
        anchors_path,
        local_dir,
        mock_s3_dir,
    )
    selected_image_provider = (
        image_provider or DeterministicImageEmbeddingProvider()
    )
    fixed_variables = {
        "corpus_question_config_fingerprint": experiment_fingerprint,
        "text_embedding_fingerprint": text_embedding_fingerprint(),
        "image_embedding_fingerprint": selected_image_provider.fingerprint(
            dimensions=VECTOR_DIMS["IMAGE_DIM"]
        ),
        "retrieval": "in-memory-hybrid-v1",
        "retrieval_top_k": 20,
        "reranker": "rule-based-reranker",
        "reranked_top_k": 8,
        "selected_context_top_k": 5,
        "query_transformation": "identity",
        "context_compression": "disabled",
        "generator": "deterministic-answer-generator",
        "semantic_grader_profile": semantic_grader_profile or "not_configured",
        "provider_trial_count": 1,
        "quality_gate_profile": QUALITY_GATE_PROFILE,
        "quality_gates": dict(COMMITTED_QUALITY_GATES),
    }
    reports: list[dict[str, Any]] = []
    for config in configs:
        started = clock()
        ingestion = ingest_demo_sources(
            local_dir,
            mock_s3_dir,
            image_embedding_provider=selected_image_provider,
            chunking_config=config,
        )
        ingestion_ms = max(0.0, (clock() - started) * 1000.0)
        chunks = ingestion.kb_chunks
        query_report = _evaluate_anchors(
            chunks,
            anchors,
            semantic_grader=semantic_grader,
        )
        reports.append(
            {
                "config": {
                    **config.to_dict(),
                    "config_fingerprint": config.fingerprint,
                },
                "corpus": _corpus_metrics(chunks, config=config),
                "retrieval_recall_at_20": query_report[
                    "retrieval_recall_at_20"
                ],
                "reranked_recall_at_8": query_report["reranked_recall_at_8"],
                "selected_context_recall_at_5": query_report[
                    "selected_context_recall_at_5"
                ],
                "citation_precision": query_report["citation_precision"],
                "citation_coverage": query_report["citation_coverage"],
                "citation_granularity": query_report["citation_granularity"],
                "citation_provenance_valid": query_report[
                    "citation_provenance_valid"
                ],
                "required_fact_coverage": query_report[
                    "required_fact_coverage"
                ],
                "abstention_correctness": query_report[
                    "abstention_correctness"
                ],
                "cross_version_contamination_count": query_report[
                    "cross_version_contamination_count"
                ],
                "faithfulness": query_report["faithfulness"],
                "answer_relevancy": query_report["answer_relevancy"],
                "latency_ms": query_report["latency_ms"],
                "usage": query_report["usage"],
                "cases": query_report["cases"],
                "ingestion_time_ms": round(ingestion_ms, 3),
                "index_size_bytes": None,
                "index_size_status": "not_built",
                "isolation": {
                    "chunk_set_id": config.fingerprint,
                    "index_id": None,
                    "index_status": "not_built_local_in_memory",
                },
            }
        )
    passing = [item for item in reports if not _constraint_violations(item)]
    semantic_complete = bool(reports) and all(
        item["faithfulness"] is not None
        and item["answer_relevancy"] is not None
        for item in reports
    )
    evaluation_fingerprint = _evaluation_fingerprint(
        experiment_fingerprint,
        fixed_variables,
        reports,
    )
    frontier = _pareto_frontier(passing) if semantic_complete else passing
    candidate = (
        _rank_frontier(frontier)[0]
        if semantic_complete and len(frontier) == 1
        else None
    )
    reviewed = _load_recommendation(recommendation_path)
    reviewed_candidate = None
    if (
        reviewed is not None
        and reviewed["experiment_fingerprint"] == experiment_fingerprint
        and reviewed["evaluation_fingerprint"] == evaluation_fingerprint
    ):
        reviewed_candidate = next(
            (
                item
                for item in frontier
                if reviewed["config_name"] == item["config"]["name"]
                and reviewed["config_fingerprint"]
                == item["config"]["config_fingerprint"]
            ),
            None,
        )
    recommended_name = (
        reviewed_candidate["config"]["name"]
        if semantic_complete and reviewed_candidate is not None
        else None
    )
    status = (
        "complete"
        if recommended_name is not None
        else (
            "evaluation_incomplete"
            if not semantic_complete
            else "review_required"
        )
    )
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "anchor_schema_version": ANCHOR_SCHEMA_VERSION,
        "num_configs": len(reports),
        "num_anchor_cases": len(anchors),
        "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
        "experiment_fingerprint": experiment_fingerprint,
        "evaluation_fingerprint": evaluation_fingerprint,
        "evaluation_status": status,
        "semantic_grader": {
            "calibrated": grader_calibrated,
            "faithfulness": "independent" if semantic_complete else None,
            "answer_relevancy": "independent" if semantic_complete else None,
        },
        "recommended_config": recommended_name,
        "recommendation": {
            "status": status,
            "candidate_config": (
                candidate["config"]["name"] if candidate else None
            ),
            "pareto_frontier": [
                item["config"]["name"] for item in frontier
            ],
            "rejected": {
                item["config"]["name"]: _constraint_violations(item)
                for item in reports
                if _constraint_violations(item)
            },
            "reviewed_artifact": reviewed,
            "uncertainty": (
                "faithfulness_and_relevancy_unavailable"
                if not semantic_complete
                else (
                    None
                    if reviewed_candidate is not None
                    else "reviewed_artifact_required"
                )
            ),
        },
        "recommendation_policy": "constraint_first_pareto_no_blended_score",
        "quality_gate_profile": QUALITY_GATE_PROFILE,
        "quality_gates": dict(COMMITTED_QUALITY_GATES),
        "fixed_variables": fixed_variables,
        "results": reports,
    }


def load_chunking_configs(path: Path) -> list[ChunkingConfig]:
    """Load at least three unique small/medium/large configurations."""

    payload = _read_json(path, label="chunking config")
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "configs"}
        or payload.get("schema_version") != CONFIG_SCHEMA_VERSION
        or not isinstance(payload.get("configs"), list)
    ):
        raise ValueError(
            f"chunking config must use schema_version {CONFIG_SCHEMA_VERSION}"
        )
    raw_configs = payload["configs"]
    if len(raw_configs) < 3:
        raise ValueError("chunking experiment requires at least three configs")
    configs: list[ChunkingConfig] = []
    for index, raw in enumerate(raw_configs, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"chunking config {index} must be an object")
        if set(raw) != {
            "name",
            "min_tokens",
            "max_tokens",
            "overlap_tokens",
            "boundary_policy",
        }:
            raise ValueError(
                f"chunking config {index} has missing or unknown fields"
            )
        configs.append(
            ChunkingConfig(
                name=raw["name"],
                min_tokens=raw["min_tokens"],
                max_tokens=raw["max_tokens"],
                overlap_tokens=raw["overlap_tokens"],
                boundary_policy=raw["boundary_policy"],
            )
        )
    names = [config.name for config in configs]
    if len(names) != len(set(names)):
        raise ValueError("chunking config names must be unique")
    behaviors = [
        (
            config.min_tokens,
            config.max_tokens,
            config.overlap_tokens,
            config.boundary_policy,
        )
        for config in configs
    ]
    if len(behaviors) != len(set(behaviors)):
        raise ValueError("chunking config behaviors must be unique")
    return configs


def _load_anchor_cases(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path, label="chunking anchor")
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "cases"}
        or payload.get("schema_version") != ANCHOR_SCHEMA_VERSION
        or not isinstance(payload.get("cases"), list)
        or not payload["cases"]
    ):
        raise ValueError(
            f"chunking anchors must use schema_version {ANCHOR_SCHEMA_VERSION}"
        )
    output: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    for index, raw in enumerate(payload["cases"], start=1):
        if not isinstance(raw, dict) or set(raw) != {
            "case_id",
            "question",
            "filters",
            "expected_anchors",
            "should_abstain",
        }:
            raise ValueError(
                f"chunking anchor case {index} has invalid fields"
            )
        case_id = raw["case_id"]
        question = raw["question"]
        filters = raw["filters"]
        expected = raw["expected_anchors"]
        should_abstain = raw["should_abstain"]
        if (
            not isinstance(case_id, str)
            or not case_id.strip()
            or case_id in case_ids
            or not isinstance(question, str)
            or not question.strip()
            or not isinstance(filters, dict)
            or not isinstance(expected, list)
            or not isinstance(should_abstain, bool)
            or (not should_abstain and not expected)
            or (should_abstain and bool(expected))
        ):
            raise ValueError(f"chunking anchor case {index} is invalid")
        normalized_anchors: list[dict[str, Any]] = []
        for anchor_index, anchor in enumerate(expected, start=1):
            if (
                not isinstance(anchor, dict)
                or set(anchor) != {"source_uri", "required_terms"}
                or not isinstance(anchor["source_uri"], str)
                or not anchor["source_uri"].strip()
                or not isinstance(anchor["required_terms"], list)
                or not anchor["required_terms"]
                or any(
                    not isinstance(term, str) or not term.strip()
                    for term in anchor["required_terms"]
                )
            ):
                raise ValueError(
                    f"chunking anchor {case_id!r}/{anchor_index} is invalid"
                )
            normalized_anchors.append(
                {
                    "source_uri": anchor["source_uri"],
                    "required_terms": list(anchor["required_terms"]),
                }
            )
        case_ids.add(case_id)
        normalized_filters = normalize_filters(filters)
        output.append(
            {
                "case_id": case_id,
                "question": question,
                "filters": normalized_filters,
                "expected_anchors": normalized_anchors,
                "should_abstain": should_abstain,
            }
        )
    return output


def _evaluate_anchors(
    chunks: list[KBChunk],
    cases: list[dict[str, Any]],
    *,
    semantic_grader: Callable[[str, str, list[str]], dict[str, float]] | None,
) -> dict[str, Any]:
    """Run the production workflow seams against one isolated chunk set."""

    retriever = InMemoryHybridRetriever(chunks)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    reports: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases, start=1):
        case_started = time.perf_counter()
        workflow = AgenticRAGWorkflow(
            retriever=retriever,
            reranker=RuleBasedReranker(),
            query_transformer=IdentityQueryTransformer(),
            selective_memory_enabled=False,
            response_cache_enabled=False,
        )
        response = workflow.run(
            str(case["question"]),
            filters=dict(case["filters"]),
            session_id=f"chunk_eval_session_{case_index}",
            query_id=f"chunk_eval_query_{case_index}",
        )
        recalled_ids = [
            str(item["chunk_id"])
            for item in response.get("milvus_recalled", [])[:20]
        ]
        reranked_items = list(response.get("reranked", []))[:8]
        reranked_ids = [str(item["chunk_id"]) for item in reranked_items]
        selected_ids = [
            str(item["chunk_id"])
            for item in reranked_items
            if item.get("selected") is True
        ][:5]
        cited_ids = [
            str(item["chunk_id"])
            for item in response.get("citations", [])
        ]
        anchors = list(case["expected_anchors"])
        recalled_hits = _matching_anchor_indexes(
            [chunks_by_id[item] for item in recalled_ids if item in chunks_by_id],
            anchors,
        )
        selected_hits = _matching_anchor_indexes(
            [chunks_by_id[item] for item in selected_ids if item in chunks_by_id],
            anchors,
        )
        reranked_hits = _matching_anchor_indexes(
            [chunks_by_id[item] for item in reranked_ids if item in chunks_by_id],
            anchors,
        )
        cited_hits = _matching_anchor_indexes(
            [chunks_by_id[item] for item in cited_ids if item in chunks_by_id],
            anchors,
        )
        answer = str(response.get("answer", ""))
        relevant_citations = sum(
            1
            for chunk_id in cited_ids
            if chunk_id in chunks_by_id
            and _matching_anchor_indexes([chunks_by_id[chunk_id]], anchors)
        )
        required_terms = list(
            dict.fromkeys(
                str(term)
                for anchor in anchors
                for term in anchor["required_terms"]
            )
        )
        fact_hits = sum(
            1 for term in required_terms if term.casefold() in answer.casefold()
        )
        semantic = (
            _validated_semantic_grade(
                semantic_grader(
                    str(case["question"]),
                    answer,
                    [
                        chunks_by_id[item].text
                        for item in cited_ids
                        if item in chunks_by_id
                    ],
                )
            )
            if semantic_grader is not None
            else {"faithfulness": None, "answer_relevancy": None}
        )
        contamination = _cross_version_contamination(
            [chunks_by_id[item] for item in selected_ids if item in chunks_by_id],
            dict(case["filters"]),
        )
        should_abstain = bool(case["should_abstain"])
        did_abstain = response.get("terminal_status") == "abstained"
        citation_subset_valid = set(cited_ids).issubset(selected_ids)
        metrics = response.get("metrics", {})
        end_to_end_ms = (time.perf_counter() - case_started) * 1000
        reports.append(
            {
                "case_id": case["case_id"],
                "num_expected_anchors": len(anchors),
                "recalled_anchor_indexes": recalled_hits,
                "selected_anchor_indexes": selected_hits,
                "retrieval_recall_at_20": (
                    round(len(recalled_hits) / len(anchors), 4)
                    if anchors
                    else None
                ),
                "reranked_recall_at_8": (
                    round(len(reranked_hits) / len(anchors), 4)
                    if anchors
                    else None
                ),
                "selected_context_recall_at_5": (
                    round(len(selected_hits) / len(anchors), 4)
                    if anchors
                    else None
                ),
                "citation_precision": _ratio(
                    relevant_citations,
                    len(cited_ids),
                ),
                "citation_coverage": round(
                    len(cited_hits) / len(anchors),
                    4,
                ) if anchors else None,
                "citation_granularity": (
                    1.0
                    if citation_subset_valid
                    and all(item in chunks_by_id for item in cited_ids)
                    else 0.0
                ),
                "citation_provenance_valid": citation_subset_valid
                and all(item in chunks_by_id for item in cited_ids),
                "required_fact_coverage": _ratio(
                    fact_hits,
                    len(required_terms),
                ),
                "should_abstain": should_abstain,
                "did_abstain": did_abstain,
                "abstention_correctness": should_abstain == did_abstain,
                "cross_version_contamination_count": contamination,
                "faithfulness": semantic["faithfulness"],
                "answer_relevancy": semantic["answer_relevancy"],
                "latency_ms": {
                    "end_to_end": round(end_to_end_ms, 3),
                    "retrieval": round(float(metrics.get("retrieval_latency_ms", 0.0)), 3),
                    "rerank": round(float(metrics.get("rerank_latency_ms", 0.0)), 3),
                    "generation": round(float(metrics.get("generation_latency_ms", 0.0)), 3),
                },
                "usage": {
                    "provider_call_count": 0,
                    "input_tokens": sum(
                        count_chunk_tokens(chunks_by_id[item].text)
                        for item in selected_ids
                        if item in chunks_by_id
                    ),
                    "output_tokens": count_chunk_tokens(answer),
                    "cost_estimate": 0.0,
                    "cost_profile": "deterministic_offline_non_billable",
                },
            }
        )
    return {
        "retrieval_recall_at_20": _mean_optional(
            reports,
            "retrieval_recall_at_20",
        ),
        "reranked_recall_at_8": _mean_optional(reports, "reranked_recall_at_8"),
        "selected_context_recall_at_5": _mean_optional(
            reports,
            "selected_context_recall_at_5",
        ),
        "citation_precision": _mean_optional(reports, "citation_precision"),
        "citation_coverage": _mean_optional(reports, "citation_coverage"),
        "citation_granularity": _mean_optional(
            reports,
            "citation_granularity",
        ),
        "citation_provenance_valid": all(
            bool(case["citation_provenance_valid"]) for case in reports
        ),
        "required_fact_coverage": _mean_optional(
            reports,
            "required_fact_coverage",
        ),
        "abstention_correctness": _mean_optional(
            reports,
            "abstention_correctness",
        ),
        "cross_version_contamination_count": sum(
            int(case["cross_version_contamination_count"]) for case in reports
        ),
        "faithfulness": _mean_optional(reports, "faithfulness"),
        "answer_relevancy": _mean_optional(reports, "answer_relevancy"),
        "latency_ms": {
            stage: _latency_summary(
                [float(case["latency_ms"][stage]) for case in reports]
            )
            for stage in ("end_to_end", "retrieval", "rerank", "generation")
        },
        "usage": {
            "provider_call_count": sum(
                int(case["usage"]["provider_call_count"]) for case in reports
            ),
            "input_tokens": sum(
                int(case["usage"]["input_tokens"]) for case in reports
            ),
            "output_tokens": sum(
                int(case["usage"]["output_tokens"]) for case in reports
            ),
            "cost_estimate": 0.0,
            "cost_profile": "deterministic_offline_non_billable",
        },
        "cases": reports,
    }


def _matching_anchor_indexes(
    chunks: list[KBChunk],
    anchors: list[dict[str, Any]],
) -> list[int]:
    matches: list[int] = []
    for index, anchor in enumerate(anchors):
        required = [
            str(term).casefold() for term in anchor["required_terms"]
        ]
        if any(
            chunk.source_uri == anchor["source_uri"]
            and all(term in chunk.text.casefold() for term in required)
            for chunk in chunks
        ):
            matches.append(index)
    return matches


def _corpus_metrics(
    chunks: list[KBChunk],
    *,
    config: ChunkingConfig,
) -> dict[str, Any]:
    textual = [
        chunk for chunk in chunks if chunk.record_type != "image"
    ]
    lengths = sorted(count_chunk_tokens(chunk.text) for chunk in textual)
    character_lengths = sorted(len(chunk.text) for chunk in textual)
    duplicate_pairs, comparable_pairs = _near_duplicate_pairs(textual)
    markdown = [chunk for chunk in textual if chunk.doc_type == "markdown"]
    pdf = [chunk for chunk in textual if chunk.doc_type == "pdf"]
    return {
        "total_kb_record_count": len(chunks),
        "textual_chunk_count": len(textual),
        "empty_chunk_count": sum(1 for length in lengths if length == 0),
        "empty_chunk_rate": _ratio(
            sum(1 for length in lengths if length == 0),
            len(lengths),
        ),
        "under_min_chunk_count": sum(
            1 for length in lengths if length < config.min_tokens
        ),
        "over_max_chunk_count": sum(
            1 for length in lengths if length > config.max_tokens
        ),
        "token_length": {
            "min": lengths[0] if lengths else 0,
            "p50": _percentile(lengths, 0.50),
            "p95": _percentile(lengths, 0.95),
            "max": lengths[-1] if lengths else 0,
            "mean": (
                round(sum(lengths) / len(lengths), 3)
                if lengths
                else 0.0
            ),
        },
        "character_length": {
            "min": character_lengths[0] if character_lengths else 0,
            "p50": _percentile(character_lengths, 0.50),
            "p95": _percentile(character_lengths, 0.95),
            "max": character_lengths[-1] if character_lengths else 0,
            "mean": (
                round(sum(character_lengths) / len(character_lengths), 3)
                if character_lengths
                else 0.0
            ),
        },
        "near_duplicate_pair_count": duplicate_pairs,
        "comparable_same_source_pair_count": comparable_pairs,
        "near_duplicate_pair_rate": (
            round(duplicate_pairs / comparable_pairs, 4)
            if comparable_pairs
            else 0.0
        ),
        "markdown_section_preservation_rate": _ratio(
            sum(1 for chunk in markdown if chunk.section),
            len(markdown),
        ),
        "release_boundary_preservation_rate": _ratio(
            sum(
                1
                for chunk in markdown
                if "release" in chunk.doc_id and chunk.section
            ),
            sum(1 for chunk in markdown if "release" in chunk.doc_id),
        ),
        "pdf_page_preservation_rate": _ratio(
            sum(1 for chunk in pdf if chunk.page_no is not None),
            len(pdf),
        ),
    }


def _near_duplicate_pairs(chunks: list[KBChunk]) -> tuple[int, int]:
    duplicate_pairs = 0
    comparable_pairs = 0
    for left_index, left in enumerate(chunks):
        left_tokens = _token_set(left.text)
        for right in chunks[left_index + 1 :]:
            if (
                left.source_uri != right.source_uri
                or left.doc_version != right.doc_version
            ):
                continue
            comparable_pairs += 1
            right_tokens = _token_set(right.text)
            union = left_tokens.union(right_tokens)
            similarity = (
                len(left_tokens.intersection(right_tokens)) / len(union)
                if union
                else 1.0
            )
            if similarity >= NEAR_DUPLICATE_THRESHOLD:
                duplicate_pairs += 1
    return duplicate_pairs, comparable_pairs


def _token_set(text: str) -> set[str]:
    return {token.casefold() for token in TOKEN_PATTERN.findall(text)}


def _percentile(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    index = math.ceil((len(values) - 1) * quantile)
    return values[index]


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else round(numerator / denominator, 4)


def _mean(cases: list[dict[str, Any]], key: str) -> float:
    return round(
        sum(float(case[key]) for case in cases) / len(cases),
        4,
    )


def _mean_optional(
    cases: list[dict[str, Any]],
    key: str,
) -> float | None:
    values = [float(case[key]) for case in cases if case[key] is not None]
    return round(sum(values) / len(values), 4) if values else None


def _latency_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "p95": None}
    ordered = sorted(values)
    return {
        "median": round(statistics.median(ordered), 3),
        "p95": round(float(ordered[math.ceil((len(ordered) - 1) * 0.95)]), 3),
    }


def _validated_semantic_grade(payload: object) -> dict[str, float]:
    if not isinstance(payload, dict) or set(payload) != {
        "faithfulness",
        "answer_relevancy",
    }:
        raise ValueError("semantic grader must return independent dimensions")
    output: dict[str, float] = {}
    for key in ("faithfulness", "answer_relevancy"):
        value = payload[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0 <= float(value) <= 1
        ):
            raise ValueError(f"semantic grader {key} must be in [0, 1]")
        output[key] = round(float(value), 4)
    return output


def _cross_version_contamination(
    chunks: list[KBChunk],
    filters: dict[str, Any],
) -> int:
    requested_current = filters.get("is_current")
    requested_versions = filters.get("doc_version")
    versions = (
        set(str(item) for item in requested_versions)
        if isinstance(requested_versions, list)
        else (
            {str(requested_versions)} if requested_versions is not None else set()
        )
    )
    return sum(
        1
        for chunk in chunks
        if (
            requested_current is True
            and not chunk.is_current
            or versions
            and chunk.doc_version not in versions
        )
    )


def _constraint_violations(report: dict[str, Any]) -> list[str]:
    corpus = report["corpus"]
    violations: list[str] = []
    if int(corpus["empty_chunk_count"]) > 0:
        violations.append("empty_chunks")
    if int(corpus["over_max_chunk_count"]) > 0:
        violations.append("over_max_chunks")
    if float(corpus["near_duplicate_pair_rate"]) > 0.25:
        violations.append("excessive_near_duplicates")
    if int(report["cross_version_contamination_count"]) > 0:
        violations.append("cross_version_contamination")
    if report["citation_provenance_valid"] is not True:
        violations.append("citation_provenance_violation")
    for field in (
        "retrieval_recall_at_20",
        "selected_context_recall_at_5",
        "citation_precision",
        "citation_coverage",
        "required_fact_coverage",
        "abstention_correctness",
    ):
        value = report[field]
        if value is None or float(value) < COMMITTED_QUALITY_GATES[field]:
            violations.append(f"below_gate:{field}")
    for field in (
        "markdown_section_preservation_rate",
        "release_boundary_preservation_rate",
        "pdf_page_preservation_rate",
    ):
        if float(corpus[field]) < COMMITTED_QUALITY_GATES[field]:
            violations.append(f"below_gate:{field}")
    return violations


def _pareto_frontier(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def dimensions(item: dict[str, Any]) -> tuple[float, float, float]:
        return (
            float(item["selected_context_recall_at_5"]),
            float(item["faithfulness"]),
            float(item["answer_relevancy"]),
        )

    frontier: list[dict[str, Any]] = []
    for candidate in reports:
        candidate_values = dimensions(candidate)
        dominated = any(
            other is not candidate
            and all(
                right >= left
                for left, right in zip(candidate_values, dimensions(other))
            )
            and any(
                right > left
                for left, right in zip(candidate_values, dimensions(other))
            )
            for other in reports
        )
        if not dominated:
            frontier.append(candidate)
    return frontier


def _rank_frontier(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        reports,
        key=lambda item: (
            int(item["usage"]["input_tokens"]),
            float(item["latency_ms"]["end_to_end"]["median"] or 0.0),
            int(item["corpus"]["textual_chunk_count"]),
            int(item["config"]["max_tokens"]),
            str(item["config"]["name"]),
        ),
    )


def _load_recommendation(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    payload = _read_json(path, label="chunking recommendation")
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "experiment_fingerprint",
        "evaluation_fingerprint",
        "config_name",
        "config_fingerprint",
        "reviewer",
        "rationale",
    }:
        raise ValueError("chunking recommendation has invalid fields")
    if payload["schema_version"] != RECOMMENDATION_SCHEMA_VERSION:
        raise ValueError(
            "chunking recommendation must use schema_version "
            f"{RECOMMENDATION_SCHEMA_VERSION}"
        )
    for key in (
        "experiment_fingerprint",
        "evaluation_fingerprint",
        "config_name",
        "config_fingerprint",
        "reviewer",
        "rationale",
    ):
        value = payload[key]
        if not isinstance(value, str) or not value.strip() or len(value) > 500:
            raise ValueError(f"chunking recommendation {key} is invalid")
    return {key: str(value) for key, value in payload.items()}


def _evaluation_fingerprint(
    experiment_fingerprint: str,
    fixed_variables: dict[str, Any],
    reports: list[dict[str, Any]],
) -> str:
    """Bind review to stable quality outputs and every configured provider."""

    quality_fields = (
        "retrieval_recall_at_20",
        "reranked_recall_at_8",
        "selected_context_recall_at_5",
        "citation_precision",
        "citation_coverage",
        "citation_granularity",
        "citation_provenance_valid",
        "required_fact_coverage",
        "abstention_correctness",
        "cross_version_contamination_count",
        "faithfulness",
        "answer_relevancy",
    )
    payload = {
        "experiment_fingerprint": experiment_fingerprint,
        "fixed_variables": fixed_variables,
        "results": [
            {
                "config_fingerprint": item["config"]["config_fingerprint"],
                "quality": {field: item[field] for field in quality_fields},
                "corpus": item["corpus"],
            }
            for item in reports
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _experiment_fingerprint(
    configs_path: Path,
    anchors_path: Path,
    local_dir: Path,
    mock_s3_dir: Path,
) -> str:
    digest = hashlib.sha256()
    roots = (
        ("configs", configs_path),
        ("anchors", anchors_path),
    )
    for label, path in roots:
        digest.update(label.encode("utf-8"))
        digest.update(path.read_bytes())
    for label, root in (("local", local_dir), ("mock_s3", mock_s3_dir)):
        if not root.is_dir():
            raise ValueError(f"chunking experiment source root is missing: {label}")
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            digest.update(label.encode("utf-8"))
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
    return f"sha256:{digest.hexdigest()}"


def _read_json(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read {label} fixture: {exc}") from exc
