"""Golden-question evaluation for retrieval and citation behavior."""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

from agent_workshop_demo.workflow import AgenticRAGWorkflow


class EvaluationWorkflow(Protocol):
    """Minimal workflow surface consumed by the evaluator."""

    def run(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


def evaluate_questions(
    *,
    questions_path: Path,
    golden_answers_path: Path | None = None,
    workflow: EvaluationWorkflow | None = None,
    top_k: int = 20,
) -> dict[str, Any]:
    """Evaluate retrieval, selection, citations, facts, and abstention."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    if not isinstance(questions, list):
        raise ValueError("questions file must contain a JSON list")
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

    runner = workflow or AgenticRAGWorkflow()
    cases: list[dict[str, Any]] = []

    for question in questions:
        question_id = str(question["question_id"])
        golden = golden_answers.get(question_id, {})
        filters = question.get("metadata_filters") or {}
        response = runner.run(
            question["question"],
            filters=_expand_filters(filters),
        )
        question_sources = set(question.get("expected_sources", []))
        expected = set(golden.get("required_citations", question_sources))
        if golden and expected != question_sources:
            raise ValueError(
                f"Question {question_id!r} expected_sources do not match "
                "golden required_citations"
            )
        recalled = {
            item["chunk_id"]
            for item in response["milvus_recalled"][:top_k]
        }
        reranked = {
            item["chunk_id"] for item in response.get("reranked", [])[:8]
        }
        selected = {
            item["chunk_id"]
            for item in response.get("reranked", [])
            if item.get("selected")
        }
        cited = {item["chunk_id"] for item in response["citations"]}
        expected_recalled = expected.intersection(recalled)
        expected_reranked = expected.intersection(reranked)
        expected_selected = expected.intersection(selected)
        expected_cited = expected.intersection(cited)
        recall = len(expected_recalled) / len(expected) if expected else None
        reranked_recall = (
            len(expected_reranked) / len(expected) if expected else None
        )
        selected_recall = (
            len(expected_selected) / len(expected) if expected else None
        )
        coverage = len(expected_cited) / len(expected) if expected else None
        precision = (
            len(expected_cited) / len(cited)
            if cited
            else (1.0 if not expected else 0.0)
        )
        required_facts = [
            str(value) for value in golden.get("required_facts", [])
        ]
        normalized_answer = _normalize_fact_text(str(response.get("answer", "")))
        matched_facts = [
            fact
            for fact in required_facts
            if _normalize_fact_text(fact) in normalized_answer
        ]
        fact_coverage: float | None = (
            len(matched_facts) / len(required_facts)
            if required_facts
            else None
        )
        should_abstain = bool(question.get("should_abstain", False))
        did_abstain = response.get("terminal_status") == "abstained"
        expected_tools = [
            str(value) for value in question.get("expected_tools", [])
        ]
        selected_tools = [
            str(value) for value in response.get("selected_tools", [])
        ]
        expected_entities = [
            str(value) for value in question.get("expected_entities", [])
        ]
        matched_entities = [
            str(item["entity_id"])
            for item in response.get("matched_entities", [])
        ]
        expected_version_scope = question.get("expected_version_scope")
        expected_doc_versions = sorted(
            str(value)
            for value in question.get("expected_doc_versions", [])
        )
        response_version_scope = response.get("version_scope", {})
        actual_version_scope = response_version_scope.get("mode")
        actual_doc_versions = sorted(
            str(value)
            for value in response_version_scope.get("doc_versions", [])
        )
        contamination_count = _cross_version_contamination_count(response)
        cases.append(
            {
                "question_id": question_id,
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
                "selected_context_recall_at_5": _optional_round(
                    selected_recall
                ),
                "citation_coverage": _optional_round(coverage),
                "citation_precision": round(precision, 4),
                "required_facts": required_facts,
                "matched_facts": matched_facts,
                "required_fact_coverage": _optional_round(fact_coverage),
                "should_abstain": should_abstain,
                "did_abstain": did_abstain,
                "abstention_correct": should_abstain == did_abstain,
                "expected_tools": expected_tools,
                "selected_tools": selected_tools,
                "tool_selection_correct": (
                    not expected_tools
                    or selected_tools == expected_tools
                ),
                "expected_entities": expected_entities,
                "matched_entities": matched_entities,
                "entity_resolution_correct": (
                    None
                    if not expected_entities
                    else matched_entities == expected_entities
                ),
                "expected_version_scope": expected_version_scope,
                "actual_version_scope": actual_version_scope,
                "expected_doc_versions": expected_doc_versions,
                "actual_doc_versions": actual_doc_versions,
                "version_scope_correct": (
                    None
                    if expected_version_scope is None
                    else (
                        actual_version_scope == expected_version_scope
                        and (
                            not expected_doc_versions
                            or actual_doc_versions == expected_doc_versions
                        )
                    )
                ),
                "cross_version_contamination_count": contamination_count,
                "recall_hit": bool(expected_recalled),
                "citation_hit": bool(expected_cited),
                "enough_evidence": response["enough_evidence"],
            }
        )

    total = len(cases) or 1
    return {
        "num_questions": len(cases),
        "recall_at_k": _mean(item["recall_at_k"] for item in cases),
        "reranked_recall_at_8": _mean(
            item["reranked_recall_at_8"] for item in cases
        ),
        "selected_context_recall_at_5": _mean(
            item["selected_context_recall_at_5"] for item in cases
        ),
        "citation_coverage": _mean(
            item["citation_coverage"] for item in cases
        ),
        "citation_precision": _mean(
            item["citation_precision"] for item in cases
        ),
        "required_fact_coverage": _mean(
            item["required_fact_coverage"] for item in cases
        ),
        "abstention_accuracy": round(
            sum(1 for item in cases if item["abstention_correct"]) / total,
            4,
        ),
        "tool_selection_accuracy": round(
            sum(1 for item in cases if item["tool_selection_correct"]) / total,
            4,
        ),
        "entity_resolution_accuracy": _mean(
            (
                1.0
                if item["entity_resolution_correct"]
                else (
                    0.0
                    if item["entity_resolution_correct"] is not None
                    else None
                )
            )
            for item in cases
        ),
        "version_scope_accuracy": _mean(
            (
                1.0
                if item["version_scope_correct"]
                else (
                    0.0
                    if item["version_scope_correct"] is not None
                    else None
                )
            )
            for item in cases
        ),
        "cross_version_contamination_count": sum(
            int(item["cross_version_contamination_count"])
            for item in cases
        ),
        "enough_evidence_rate": round(
            sum(1 for item in cases if item["enough_evidence"]) / total,
            4,
        ),
        "cases": cases,
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
    selected = [
        item
        for item in response.get("reranked", [])
        if item.get("selected")
    ]
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
        if mode == "comparison" and scopes and not any(
            _record_matches_scope(item, scope) for scope in scopes
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
        len(versions) - 1
        for versions in by_doc.values()
        if len(versions) > 1
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


def _mean(values: Iterable[float | None]) -> float:
    included = [value for value in values if value is not None]
    if not included:
        return 0.0
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
            raise ValueError(
                f"Unsupported golden answers YAML at line {index + 1}"
            )
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
                raise ValueError(
                    f"Unsupported golden answers YAML at line {index + 1}"
                )
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

    return {
        key: _validate_golden_case(key, value)
        for key, value in output.items()
    }


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
    required_facts = value.get("required_facts")
    required_citations = value.get("required_citations")
    if not isinstance(required_facts, list) or not all(
        isinstance(item, str) for item in required_facts
    ):
        raise ValueError(
            f"Golden answer {question_id!r} requires a string facts list"
        )
    if not isinstance(required_citations, list) or not all(
        isinstance(item, str) for item in required_citations
    ):
        raise ValueError(
            f"Golden answer {question_id!r} requires a citation ID list"
        )
    return dict(value)
