"""TIMESTAMPTZ codec shared by Milvus lifecycle adapters."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any


def milvus_timestamp(epoch_ms: int) -> str:
    """Encode non-negative UTC epoch milliseconds as canonical ISO-8601."""

    if isinstance(epoch_ms, bool) or not isinstance(epoch_ms, int) or epoch_ms < 0:
        raise ValueError("epoch milliseconds must be a non-negative integer")
    value = datetime.fromtimestamp(epoch_ms / 1000, tz=UTC)
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def timestamp_literal(epoch_ms: int) -> str:
    """Return a safely quoted Milvus TIMESTAMPTZ expression literal."""

    return json.dumps(milvus_timestamp(epoch_ms))


def epoch_ms_from_milvus(value: Any) -> int:
    """Decode a Milvus TIMESTAMPTZ response into the domain epoch-ms form."""

    if isinstance(value, bool):
        raise ValueError("TIMESTAMPTZ cannot be a boolean")
    if isinstance(value, int):
        if value < 0:
            raise ValueError("legacy epoch milliseconds cannot be negative")
        return value
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    else:
        raise ValueError("TIMESTAMPTZ must be an ISO-8601 value")
    if parsed.tzinfo is None:
        raise ValueError("TIMESTAMPTZ must include a timezone")
    return int(parsed.astimezone(UTC).timestamp() * 1000)


def optional_epoch_ms_from_milvus(value: Any) -> int | None:
    """Decode a nullable Milvus TIMESTAMPTZ response."""

    return None if value is None else epoch_ms_from_milvus(value)


def encode_expiry(record: dict[str, Any]) -> dict[str, Any]:
    """Copy one storage record and encode a present expires_at field."""

    output = dict(record)
    if output.get("expires_at") is None:
        output.pop("expires_at", None)
    else:
        output["expires_at"] = milvus_timestamp(output["expires_at"])
    return output
