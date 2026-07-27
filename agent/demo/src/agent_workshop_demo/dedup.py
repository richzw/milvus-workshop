"""Deterministic checksum and experimental dedup signature helpers."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from agent_workshop_demo.config import VECTOR_DIMS
from agent_workshop_demo.embedding import tokenize


def normalize_text(text: str) -> str:
    """Normalize text with the same tokens used by local retrieval."""

    return re.sub(r"\s+", " ", " ".join(tokenize(text))).strip()


def checksum(text: str) -> str:
    """Return a full SHA-256 checksum."""

    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def minhash_signature(
    text: str,
    dim: int = VECTOR_DIMS["MINHASH_DIM"],
) -> list[int]:
    """Return an experimental binary MinHash-style signature."""

    if dim <= 0:
        raise ValueError("dim must be greater than zero")
    tokens = sorted(set(tokenize(text))) or [""]
    signature: list[int] = []
    for index in range(dim):
        values = [
            int.from_bytes(
                hashlib.blake2b(
                    f"{index}:{token}".encode("utf-8"),
                    digest_size=8,
                ).digest(),
                "big",
            )
            for token in tokens
        ]
        signature.append(min(values) & 1)
    return signature


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
        "minhash_signature": minhash_signature(normalized),
        "created_at": created_at,
        "metadata": {
            "shingle_size": 1,
            "num_perm": VECTOR_DIMS["MINHASH_DIM"],
            "parser_version": "local-demo-v1",
            "experimental": True,
        },
    }
