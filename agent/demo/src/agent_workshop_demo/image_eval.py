"""Independent text-to-image and image-to-image retrieval evaluation."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from agent_workshop_demo.config import VECTOR_DIMS
from agent_workshop_demo.image_embedding import ImageEmbeddingProvider
from agent_workshop_demo.image_retrieval import (
    ImageVectorRetriever,
    image_only_filters,
    search_similar_images,
    validate_image_search_top_k,
)
from agent_workshop_demo.models import SearchResult

IMAGE_EVAL_SCHEMA_VERSION = "image-retrieval-v1"


class MultimodalEvalRetriever(ImageVectorRetriever, Protocol):
    """Text hybrid plus image-vector search required by the eval runner."""

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
        order_by: list[str] | None = None,
    ) -> list[SearchResult]: ...


def evaluate_image_retrieval(
    *,
    cases_path: Path,
    retriever: MultimodalEvalRetriever,
    image_provider: ImageEmbeddingProvider,
    top_k: int = 3,
    assets_root: Path | None = None,
) -> dict[str, Any]:
    """Evaluate both image-retrieval modes against versioned source fixtures."""

    validate_image_search_top_k(top_k)
    cases = _load_cases(cases_path)
    root = (
        assets_root.resolve()
        if assets_root is not None
        else cases_path.parent.parent.resolve()
    )
    reports: list[dict[str, Any]] = []
    for case in cases:
        mode = str(case["mode"])
        expected = set(_string_list(case["expected_sources"]))
        filters = _case_filters(case)
        if mode == "text":
            query = case.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(
                    f"Image eval case {case['case_id']!r} requires query"
                )
            results = retriever.search(
                query,
                top_k=top_k,
                filters=image_only_filters(filters),
            )
            retrieved = [
                item.chunk.source_uri for item in results[:top_k]
            ]
        else:
            image_path = _resolve_image_path(
                case.get("image_path"),
                root=root,
                case_id=str(case["case_id"]),
            )
            image_results = search_similar_images(
                image_path,
                retriever=retriever,
                provider=image_provider,
                top_k=top_k,
                filters=filters,
            )
            retrieved = [
                item.chunk.source_uri
                for item in image_results[:top_k]
            ]
        hits = expected.intersection(retrieved)
        first_rank = next(
            (
                rank
                for rank, source in enumerate(retrieved, start=1)
                if source in expected
            ),
            None,
        )
        reports.append(
            {
                "case_id": case["case_id"],
                "mode": mode,
                "expected_sources": sorted(expected),
                "retrieved_sources": retrieved,
                "recall_at_k": round(len(hits) / len(expected), 4),
                "reciprocal_rank": (
                    0.0
                    if first_rank is None
                    else round(1.0 / first_rank, 4)
                ),
            }
        )

    text_cases = [item for item in reports if item["mode"] == "text"]
    image_cases = [item for item in reports if item["mode"] == "image"]
    fingerprint = image_provider.fingerprint(
        dimensions=VECTOR_DIMS["IMAGE_DIM"]
    )
    return {
        "fixture_schema_version": IMAGE_EVAL_SCHEMA_VERSION,
        "quality_mode": (
            "semantic"
            if image_provider.name == "dinov3"
            else "pipeline_only"
        ),
        "image_embedding_fingerprint": fingerprint,
        "top_k": top_k,
        "num_cases": len(reports),
        "num_text_to_image_cases": len(text_cases),
        "num_image_to_image_cases": len(image_cases),
        "text_to_image_recall_at_k": _mean_metric(
            text_cases,
            "recall_at_k",
        ),
        "text_to_image_mrr": _mean_metric(
            text_cases,
            "reciprocal_rank",
        ),
        "image_to_image_recall_at_k": _mean_metric(
            image_cases,
            "recall_at_k",
        ),
        "image_to_image_mrr": _mean_metric(
            image_cases,
            "reciprocal_rank",
        ),
        "cases": reports,
    }


def _load_cases(path: Path) -> list[dict[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read image eval cases: {exc}") from exc
    if (
        not isinstance(raw, dict)
        or set(raw) != {"schema_version", "cases"}
        or raw.get("schema_version") != IMAGE_EVAL_SCHEMA_VERSION
    ):
        raise ValueError(
            "image eval fixture must use schema_version image-retrieval-v1"
        )
    raw_cases = raw.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("image eval cases must be a non-empty JSON list")
    cases: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, item in enumerate(raw_cases, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"image eval case {index} must be an object")
        case_id = item.get("case_id")
        mode = item.get("mode")
        if (
            not isinstance(case_id, str)
            or not case_id.strip()
            or case_id in ids
        ):
            raise ValueError("image eval case_id values must be unique")
        if mode not in {"text", "image"}:
            raise ValueError(
                f"Image eval case {case_id!r} has invalid mode"
            )
        allowed_fields = {
            "case_id",
            "mode",
            "expected_sources",
            "filters",
            "query" if mode == "text" else "image_path",
        }
        unknown_fields = set(item).difference(allowed_fields)
        if unknown_fields:
            raise ValueError(
                f"Image eval case {case_id!r} has unknown fields: "
                f"{sorted(unknown_fields)}"
            )
        expected = item.get("expected_sources")
        values = _string_list(expected)
        if not values or len(values) != len(set(values)):
            raise ValueError(
                f"Image eval case {case_id!r} requires unique expected_sources"
            )
        ids.add(case_id)
        cases.append(dict(item))
    return cases


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError("expected_sources must be a list of non-empty strings")
    return [str(item) for item in value]


def _case_filters(case: dict[str, Any]) -> dict[str, Any]:
    value = case.get("filters")
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(
            f"Image eval case {case['case_id']!r} filters must be an object"
        )
    return dict(value)


def _resolve_image_path(
    value: Any,
    *,
    root: Path,
    case_id: str,
) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(
            f"Image eval case {case_id!r} requires a safe image_path"
        )
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(
            f"Image eval case {case_id!r} requires a safe image_path"
        )
    relative = PurePosixPath(value)
    if relative.is_absolute():
        raise ValueError(
            f"Image eval case {case_id!r} requires a relative image_path"
        )
    path = root.joinpath(*relative.parts).resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError(
            f"Image eval case {case_id!r} image_path escapes assets_root"
        ) from None
    if not path.is_file():
        raise ValueError(
            f"Image eval case {case_id!r} image_path does not exist"
        )
    return path


def _mean_metric(
    cases: list[dict[str, Any]],
    key: str,
) -> float | None:
    if not cases:
        return None
    return round(
        sum(float(item[key]) for item in cases) / len(cases),
        4,
    )
