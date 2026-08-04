"""Deterministic checksum and experimental dedup signature helpers."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from agent_workshop_demo.embedding import tokenize


def normalize_text(text: str) -> str:
    """Normalize text with the same tokens used by local retrieval."""

    return re.sub(r"\s+", " ", " ".join(tokenize(text))).strip()


def checksum(text: str) -> str:
    """Return a full SHA-256 checksum."""

    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def near_duplicate_jaccard(
    left: str,
    right: str,
    *,
    shingle_size: int = 3,
) -> float:
    """Estimate local near-duplicate overlap without persisting a signature."""

    if shingle_size <= 0:
        raise ValueError("shingle_size must be greater than zero")

    def shingles(text: str) -> set[tuple[str, ...]]:
        tokens = tokenize(text)
        if not tokens:
            return set()
        if len(tokens) < shingle_size:
            return {tuple(tokens)}
        return {
            tuple(tokens[index : index + shingle_size])
            for index in range(len(tokens) - shingle_size + 1)
        }

    left_shingles = shingles(left)
    right_shingles = shingles(right)
    if not left_shingles and not right_shingles:
        return 1.0
    union = left_shingles | right_shingles
    return len(left_shingles & right_shingles) / len(union)


def build_dedup_record(
    *,
    doc_id: str,
    chunk_id: str | None,
    source_uri: str,
    source_type: str,
    record_level: str,
    text: str,
    created_at: int,
) -> dict[str, Any]:
    """Build one provisional P2 dedup record."""

    if record_level not in {"doc", "chunk"}:
        raise ValueError("record_level must be 'doc' or 'chunk'")
    if record_level == "doc" and chunk_id is not None:
        raise ValueError("doc-level dedup records require null chunk_id")
    if record_level == "chunk" and not chunk_id:
        raise ValueError("chunk-level dedup records require chunk_id")
    normalized = normalize_text(text)
    return {
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "source_uri": source_uri,
        "source_type": source_type,
        "record_level": record_level,
        "normalized_text": normalized,
        "checksum": checksum(normalized),
        "created_at": created_at,
        "metadata": {
            "shingle_size": 3,
            "num_hashes": 256,
            "parser_version": "milvus-3-dido-v1",
        },
    }
