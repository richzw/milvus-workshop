"""Validation helpers for query identity and supported search filters."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

ALLOWED_FILTER_VALUES = {
    "source_type": frozenset({"local", "s3", "mfs"}),
    "doc_type": frozenset({"markdown", "pdf", "text", "image", "table"}),
    "department": frozenset(
        {"engineering", "product", "hr", "security", "general"}
    ),
}
BOOLEAN_FILTERS = frozenset({"has_image_vector", "is_current"})
FREE_TEXT_FILTERS = frozenset({"doc_version"})


def validate_question(question: str) -> str:
    """Return a trimmed non-empty question."""

    normalized = question.strip()
    if not normalized:
        raise ValueError("question must contain non-whitespace characters")
    return normalized


def validate_identifier(value: str, *, field_name: str) -> str:
    """Validate a bounded query/session identifier."""

    normalized = value.strip()
    if not normalized:
        raise ValueError(
            f"{field_name} must contain non-whitespace characters"
        )
    if len(normalized) > 128:
        raise ValueError(f"{field_name} must be at most 128 characters")
    return normalized


def normalize_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    """Validate and copy the supported scalar/list filter contract."""

    if filters is None:
        return {}

    normalized: dict[str, Any] = {}
    for field, raw_value in filters.items():
        if field in BOOLEAN_FILTERS:
            if not isinstance(raw_value, bool):
                raise ValueError(f"Filter {field!r} must be a boolean")
            normalized[field] = raw_value
            continue

        if field in FREE_TEXT_FILTERS:
            if not isinstance(raw_value, str) or not raw_value.strip():
                raise ValueError(f"Filter {field!r} must be a non-empty string")
            value = raw_value.strip()
            if len(value) > 64:
                raise ValueError(
                    f"Filter {field!r} must be at most 64 characters"
                )
            normalized[field] = value
            continue

        allowed = ALLOWED_FILTER_VALUES.get(field)
        if allowed is None:
            raise ValueError(f"Unsupported search filter: {field}")

        if isinstance(raw_value, str):
            values = [raw_value]
            preserve_scalar = True
        elif _is_value_list(raw_value):
            values = list(raw_value)
            preserve_scalar = False
        else:
            raise ValueError(
                f"Filter {field!r} must be a string or list of strings"
            )

        invalid = [
            value
            for value in values
            if not isinstance(value, str) or value not in allowed
        ]
        if invalid:
            raise ValueError(f"Invalid {field} filter value(s): {invalid}")
        normalized[field] = values[0] if preserve_scalar else values

    return normalized


def _is_value_list(value: object) -> bool:
    return isinstance(value, Iterable) and not isinstance(
        value, (str, bytes, dict)
    )
