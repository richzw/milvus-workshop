"""Repeatable Min-Max chunking comparison over stable retrieval anchors."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agent_workshop_demo.chunking import (
    TOKEN_PATTERN,
    ChunkingConfig,
    count_chunk_tokens,
)
from agent_workshop_demo.image_embedding import (
    DeterministicImageEmbeddingProvider,
    ImageEmbeddingProvider,
)
from agent_workshop_demo.ingestion import ingest_demo_sources
from agent_workshop_demo.models import KBChunk
from agent_workshop_demo.reranker import RuleBasedReranker
from agent_workshop_demo.retrieval import InMemoryHybridRetriever
from agent_workshop_demo.validation import normalize_filters

CONFIG_SCHEMA_VERSION = "chunking-experiment-v1"
ANCHOR_SCHEMA_VERSION = "chunking-anchors-v1"
NEAR_DUPLICATE_THRESHOLD = 0.85


def run_chunking_experiment(
    *,
    configs_path: Path,
    anchors_path: Path,
    local_dir: Path,
    mock_s3_dir: Path,
    image_provider: ImageEmbeddingProvider | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Run every strict config over the same corpus and stable query anchors."""

    configs = load_chunking_configs(configs_path)
    anchors = _load_anchor_cases(anchors_path)
    selected_image_provider = (
        image_provider or DeterministicImageEmbeddingProvider()
    )
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
        query_report = _evaluate_anchors(chunks, anchors)
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
                "selected_context_recall_at_5": query_report[
                    "selected_context_recall_at_5"
                ],
                "cases": query_report["cases"],
                "ingestion_time_ms": round(ingestion_ms, 3),
                "index_size_bytes": None,
                "index_size_status": "not_built",
            }
        )
    recommended = sorted(
        reports,
        key=lambda item: (
            -float(item["selected_context_recall_at_5"]),
            -float(item["retrieval_recall_at_20"]),
            float(item["corpus"]["near_duplicate_pair_rate"]),
            int(item["corpus"]["textual_chunk_count"]),
            str(item["config"]["name"]),
        ),
    )[0]
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "anchor_schema_version": ANCHOR_SCHEMA_VERSION,
        "num_configs": len(reports),
        "num_anchor_cases": len(anchors),
        "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
        "recommended_config": recommended["config"]["name"],
        "recommendation_policy": (
            "selected_recall, retrieval_recall, lower_near_duplicate_rate, "
            "fewer_chunks, name"
        ),
        "results": reports,
    }


def load_chunking_configs(path: Path) -> list[ChunkingConfig]:
    """Load at least two unique strict configurations."""

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
    if len(raw_configs) < 2:
        raise ValueError("chunking experiment requires at least two configs")
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
        }:
            raise ValueError(
                f"chunking anchor case {index} has invalid fields"
            )
        case_id = raw["case_id"]
        question = raw["question"]
        filters = raw["filters"]
        expected = raw["expected_anchors"]
        if (
            not isinstance(case_id, str)
            or not case_id.strip()
            or case_id in case_ids
            or not isinstance(question, str)
            or not question.strip()
            or not isinstance(filters, dict)
            or not isinstance(expected, list)
            or not expected
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
            }
        )
    return output


def _evaluate_anchors(
    chunks: list[KBChunk],
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    retriever = InMemoryHybridRetriever(chunks)
    reranker = RuleBasedReranker()
    reports: list[dict[str, Any]] = []
    for case in cases:
        recalled = retriever.search(
            str(case["question"]),
            top_k=20,
            filters=dict(case["filters"]),
        )
        selected = (
            list(
                reranker.rerank(
                    str(case["question"]),
                    recalled,
                    top_k=min(5, len(recalled)),
                ).results
            )
            if recalled
            else []
        )
        anchors = list(case["expected_anchors"])
        recalled_hits = _matching_anchor_indexes(
            [item.chunk for item in recalled],
            anchors,
        )
        selected_hits = _matching_anchor_indexes(
            [item.chunk for item in selected],
            anchors,
        )
        reports.append(
            {
                "case_id": case["case_id"],
                "num_expected_anchors": len(anchors),
                "recalled_anchor_indexes": recalled_hits,
                "selected_anchor_indexes": selected_hits,
                "retrieval_recall_at_20": round(
                    len(recalled_hits) / len(anchors),
                    4,
                ),
                "selected_context_recall_at_5": round(
                    len(selected_hits) / len(anchors),
                    4,
                ),
            }
        )
    return {
        "retrieval_recall_at_20": _mean(
            reports,
            "retrieval_recall_at_20",
        ),
        "selected_context_recall_at_5": _mean(
            reports,
            "selected_context_recall_at_5",
        ),
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
    duplicate_pairs, comparable_pairs = _near_duplicate_pairs(textual)
    markdown = [chunk for chunk in textual if chunk.doc_type == "markdown"]
    pdf = [chunk for chunk in textual if chunk.doc_type == "pdf"]
    return {
        "total_kb_record_count": len(chunks),
        "textual_chunk_count": len(textual),
        "empty_chunk_count": sum(1 for length in lengths if length == 0),
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


def _read_json(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read {label} fixture: {exc}") from exc
