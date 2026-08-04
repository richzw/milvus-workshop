"""Shared contracts for safe image-vector retrieval."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Protocol

from agent_workshop_demo.config import VECTOR_DIMS
from agent_workshop_demo.image_embedding import (
    IMAGE_EMBEDDING_FINGERPRINT_KEY,
    ImageEmbeddingProvider,
)
from agent_workshop_demo.models import ImageSearchResult, KBChunk
from agent_workshop_demo.validation import normalize_filters

MAX_IMAGE_SEARCH_TOP_K = 100


class ImageVectorRetriever(Protocol):
    """Low-level image-vector search surface shared by local and Milvus."""

    def search_image_vector(
        self,
        query_vector: list[float],
        *,
        image_fingerprint: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[ImageSearchResult]: ...


def search_similar_images(
    image_path: Path,
    *,
    retriever: ImageVectorRetriever,
    provider: ImageEmbeddingProvider,
    top_k: int,
    filters: dict[str, Any] | None = None,
) -> list[ImageSearchResult]:
    """Embed real query bytes and search the matching persisted image space."""

    validate_image_search_top_k(top_k)
    validated_filters = image_only_filters(filters)
    fingerprint = validate_image_fingerprint(
        provider.fingerprint(dimensions=VECTOR_DIMS["IMAGE_DIM"])
    )
    query_vector = provider.embed(
        image_path,
        dimensions=VECTOR_DIMS["IMAGE_DIM"],
    )
    return retriever.search_image_vector(
        query_vector,
        image_fingerprint=fingerprint,
        top_k=top_k,
        filters=validated_filters,
    )


def validate_image_query_vector(values: list[float]) -> list[float]:
    """Validate an already normalized query without changing vector space."""

    invalid = (
        len(values) != VECTOR_DIMS["IMAGE_DIM"]
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        )
    )
    norm = (
        0.0
        if invalid
        else math.sqrt(sum(float(value) ** 2 for value in values))
    )
    if invalid or not math.isclose(
        norm,
        1.0,
        rel_tol=1e-5,
        abs_tol=1e-5,
    ):
        raise ValueError(
            "image query vector must contain exactly "
            f"{VECTOR_DIMS['IMAGE_DIM']} finite L2-normalized values"
        )
    return [float(value) for value in values]


def image_cosine_score(
    query_vector: list[float],
    candidate_vector: list[float],
) -> float:
    """Return raw COSINE for validated L2 vectors, including negative scores."""

    if len(query_vector) != len(candidate_vector):
        raise ValueError("image vectors must have matching dimensions")
    score = sum(
        float(left) * float(right)
        for left, right in zip(
            query_vector,
            candidate_vector,
            strict=True,
        )
    )
    if not math.isfinite(score):
        raise ValueError("image cosine score must be finite")
    return score


def image_only_filters(
    filters: dict[str, Any] | None,
) -> dict[str, Any]:
    """Validate caller filters and force the non-null image-vector predicate."""

    output = normalize_filters(filters)
    requested = output.get("has_image_vector")
    if requested is False:
        raise ValueError(
            "image search cannot use has_image_vector=false"
        )
    output["has_image_vector"] = True
    return output


def require_image_space(
    chunk: KBChunk,
    *,
    expected_fingerprint: str,
) -> None:
    """Fail closed for non-image or mixed-space search results."""

    validate_image_fingerprint(expected_fingerprint)
    actual = (chunk.metadata or {}).get(
        IMAGE_EMBEDDING_FINGERPRINT_KEY
    )
    if not chunk.has_image_vector or actual != expected_fingerprint:
        raise ValueError(
            "Image search result does not match the query vector space"
        )


def validate_image_search_top_k(top_k: int) -> None:
    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or not 1 <= top_k <= MAX_IMAGE_SEARCH_TOP_K
    ):
        raise ValueError(
            f"image search top_k must be between 1 and {MAX_IMAGE_SEARCH_TOP_K}"
        )


def validate_image_fingerprint(value: str) -> str:
    """Validate vector-space identity before any external search call."""

    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ValueError("image fingerprint must contain 1..512 characters")
    return value
