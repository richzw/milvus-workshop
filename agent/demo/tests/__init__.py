"""Pin the deterministic offline profile before any demo module is imported.

The demo package auto-loads ``demo/.env`` at import time, and real process
variables take precedence over that file. Either source can silently point a
stage at a live provider, which contradicts the documented contract that the
test suite is deterministic and needs no live services. Importing this package
first neutralizes both sources for the whole run.

The skip flag is duplicated here rather than imported from
``agent_workshop_demo.config`` because importing that module would already have
loaded the env file.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping

SKIP_ENV_FILE_VAR = "AGENT_WORKSHOP_SKIP_ENV_FILE"
PROVIDER_SELECTORS = frozenset(
    {
        "ANSWER_GENERATOR",
        "CONTEXT_COMPRESSION_MODE",
        "EMBEDDING_PROVIDER",
        "IMAGE_EMBEDDING_PROVIDER",
        "MEMORY_SELECTOR",
        "QUERY_CLASSIFIER",
        "QUERY_TRANSFORMER",
        "RERANKER",
        "RETRIEVAL_TIER",
        "STRUCT_ARRAY_RETRIEVAL",
    }
)
CREDENTIAL_PREFIXES = ("OPENAI_", "HF_", "MINIO_", "DINOV3_")


def pin_offline_profile(
    environ: MutableMapping[str, str] | None = None,
) -> None:
    """Remove every provider selector and credential from one environment."""

    target = os.environ if environ is None else environ
    target[SKIP_ENV_FILE_VAR] = "1"
    for name in list(target):
        if name in PROVIDER_SELECTORS or name.startswith(CREDENTIAL_PREFIXES):
            del target[name]


pin_offline_profile()
