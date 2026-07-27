"""Shared constants and environment loading for the workshop demo."""

from __future__ import annotations

import os
import re
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Final

DEMO_ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE: Final = DEMO_ROOT / ".env"
ENV_KEY_PATTERN: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

COLLECTION_NAMES: Final = {
    "kb_chunks": "kb_chunks",
    "conversation_memory": "conversation_memory",
    "doc_dedup_signatures": "doc_dedup_signatures",
}

VECTOR_DIMS: Final = {
    "TEXT_DIM": 1024,
    "IMAGE_DIM": 768,
    "MINHASH_DIM": 256,
}

TIME_UNIT: Final = "epoch_milliseconds"

DEFAULT_SEARCH_PARAMS: Final[dict[str, Any]] = {
    "max_retry": 3,
    "milvus_top_k": 20,
    "reranker_top_k": 8,
    "answer_context_top_k": 5,
    "search_mode": "hybrid",
    "order_by": ["updated_at desc", "priority desc"],
    "filters": {
        "source_type": ["local", "s3"],
        "doc_type": ["markdown", "pdf", "text", "image"],
        "is_current": True,
    },
}


def load_demo_env(
    path: Path | None = None,
    *,
    environ: MutableMapping[str, str] | None = None,
    override: bool = False,
) -> Path | None:
    """Load ``demo/.env`` without replacing explicit process variables."""

    env_path = DEFAULT_ENV_FILE if path is None else path
    if not env_path.is_file():
        return None

    target = os.environ if environ is None else environ
    for line_number, raw_line in enumerate(
        env_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, raw_value = line.partition("=")
        key = key.strip()
        if not separator or not ENV_KEY_PATTERN.fullmatch(key):
            raise ValueError(
                f"Invalid environment entry at {env_path}:{line_number}"
            )
        value = _parse_env_value(raw_value.strip(), env_path, line_number)
        if override or key not in target:
            target[key] = value
    return env_path


def _parse_env_value(value: str, path: Path, line_number: int) -> str:
    """Parse the small, explicit dotenv syntax used by the demo."""

    if not value or value[0] not in {"'", '"'}:
        return value
    quote = value[0]
    if len(value) < 2 or value[-1] != quote:
        raise ValueError(
            f"Unclosed quoted value at {path}:{line_number}"
        )
    unquoted = value[1:-1]
    if quote == '"':
        return (
            unquoted.replace(r"\n", "\n")
            .replace(r"\t", "\t")
            .replace(r"\"", '"')
            .replace(r"\\", "\\")
        )
    return unquoted
