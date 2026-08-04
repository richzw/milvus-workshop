"""Deterministic hard-boundary-aware Min-Max chunking primitives."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

TOKENIZER_VERSION = "lexical-cjk-word-punct-v1"
BOUNDARY_POLICY = "paragraph_sentence_v1"
TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]")
CONFIG_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
SENTENCE_ENDINGS = frozenset({".", "!", "?", "。", "！", "？", ";", "；"})


@dataclass(frozen=True)
class ChunkingConfig:
    """One strict, comparable Min-Max/overlap experiment configuration."""

    name: str
    min_tokens: int
    max_tokens: int
    overlap_tokens: int
    boundary_policy: str = BOUNDARY_POLICY

    def __post_init__(self) -> None:
        if not CONFIG_NAME_PATTERN.fullmatch(self.name):
            raise ValueError(
                "chunking config name must match [a-z][a-z0-9_-]{0,63}"
            )
        values = (
            self.min_tokens,
            self.max_tokens,
            self.overlap_tokens,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in values
        ):
            raise ValueError("chunking token limits must be integers")
        if not 1 <= self.min_tokens <= self.max_tokens <= 4096:
            raise ValueError(
                "chunking limits require 1 <= min_tokens <= max_tokens <= 4096"
            )
        if not 0 <= self.overlap_tokens < self.min_tokens:
            raise ValueError(
                "overlap_tokens must be >= 0 and less than min_tokens"
            )
        if self.boundary_policy != BOUNDARY_POLICY:
            raise ValueError(
                f"boundary_policy must be {BOUNDARY_POLICY!r}"
            )

    @property
    def fingerprint(self) -> str:
        """Return a stable identity for persisted chunk metadata."""

        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return f"minmax:{digest}"

    def to_dict(self) -> dict[str, str | int]:
        return {
            "name": self.name,
            "min_tokens": self.min_tokens,
            "max_tokens": self.max_tokens,
            "overlap_tokens": self.overlap_tokens,
            "boundary_policy": self.boundary_policy,
            "tokenizer_version": TOKENIZER_VERSION,
        }


@dataclass(frozen=True)
class ChunkFragment:
    """One bounded text window and its source token coordinates."""

    text: str
    token_count: int
    start_token: int
    end_token: int
    applied_overlap_tokens: int

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("chunk fragment text must be non-empty")
        if (
            self.token_count <= 0
            or self.end_token - self.start_token != self.token_count
            or not 0 <= self.applied_overlap_tokens < self.token_count
        ):
            raise ValueError("chunk fragment token coordinates are invalid")


def count_chunk_tokens(text: str) -> int:
    """Count versioned lexical tokens used by this experiment only."""

    return len(TOKEN_PATTERN.findall(text))


def split_text(
    text: str,
    *,
    config: ChunkingConfig,
) -> list[ChunkFragment]:
    """Split one hard semantic unit into deterministic bounded windows."""

    normalized = text.strip()
    matches = list(TOKEN_PATTERN.finditer(normalized))
    if not matches:
        raise ValueError("chunking input must contain at least one token")
    total = len(matches)
    if total <= config.max_tokens:
        return [
            ChunkFragment(
                text=normalized,
                token_count=total,
                start_token=0,
                end_token=total,
                applied_overlap_tokens=0,
            )
        ]

    boundaries = _preferred_boundaries(normalized, matches)
    fragments: list[ChunkFragment] = []
    start = 0
    previous_end = 0
    while start < total:
        remaining = total - start
        if remaining <= config.max_tokens:
            end = total
        else:
            lower = start + config.min_tokens
            upper = start + config.max_tokens
            feasible_ends = [
                candidate
                for candidate in range(upper, lower - 1, -1)
                if _can_partition(
                    total - candidate + config.overlap_tokens,
                    config=config,
                )
            ]
            feasible_end_set = set(feasible_ends)
            preferred = [
                boundary
                for boundary in boundaries
                if boundary in feasible_end_set
            ]
            if preferred:
                end = preferred[-1]
            elif feasible_ends:
                end = feasible_ends[0]
            else:
                preferred = [
                    boundary
                    for boundary in boundaries
                    if lower <= boundary <= upper
                ]
                end = preferred[-1] if preferred else upper
        fragment_text = normalized[
            matches[start].start() : matches[end - 1].end()
        ].strip()
        overlap = 0 if not fragments else previous_end - start
        fragments.append(
            ChunkFragment(
                text=fragment_text,
                token_count=end - start,
                start_token=start,
                end_token=end,
                applied_overlap_tokens=overlap,
            )
        )
        if end == total:
            break
        previous_end = end
        next_start = end - config.overlap_tokens
        if next_start <= start:
            raise RuntimeError("chunking configuration made no progress")
        start = next_start
    return fragments


def _can_partition(length: int, *, config: ChunkingConfig) -> bool:
    """Return whether a suffix has an all-in-range overlapping partition."""

    if length < config.min_tokens:
        return False
    min_stride = config.min_tokens - config.overlap_tokens
    max_stride = config.max_tokens - config.overlap_tokens
    unique_tokens = length - config.overlap_tokens
    minimum_chunks = max(
        1,
        (unique_tokens + max_stride - 1) // max_stride,
    )
    maximum_chunks = unique_tokens // min_stride
    return minimum_chunks <= maximum_chunks


def _preferred_boundaries(
    text: str,
    matches: list[re.Match[str]],
) -> list[int]:
    boundaries: list[int] = []
    for index, match in enumerate(matches):
        next_start = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text)
        )
        between = text[match.end() : next_start]
        if match.group() in SENTENCE_ENDINGS or "\n\n" in between:
            boundaries.append(index + 1)
    return boundaries
