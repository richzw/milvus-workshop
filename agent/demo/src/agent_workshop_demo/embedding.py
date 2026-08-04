"""Configurable text embeddings and deterministic workshop fallbacks."""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter
from collections.abc import Callable, Mapping
from functools import lru_cache
from importlib import import_module
from typing import Any, Protocol

from agent_workshop_demo.config import VECTOR_DIMS

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+")
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_FINGERPRINT_KEY = "text_embedding_fingerprint"
SYNONYMS = {
    "s3": ["object storage", "minio", "bucket"],
    "sync": ["synchronization", "ingestion", "pipeline"],
    "同步": ["sync", "ingestion", "pipeline"],
    "流程": ["flow", "pipeline", "steps"],
    "milvus": ["vector database", "hybrid search", "bm25"],
    "图片": ["image", "diagram", "dinov3"],
    "架构": ["architecture", "workflow", "design"],
    "权限": ["acl", "security", "filter"],
    "过期": ["ttl", "expires_at", "lifecycle"],
}


class TextEmbeddingProvider(Protocol):
    """Small interface shared by text-embedding implementations."""

    name: str

    def embed(self, text: str, *, dimensions: int) -> list[float]:
        """Return one validated dense text vector."""

    def fingerprint(self, *, dimensions: int) -> str:
        """Identify the vector space used for persisted records."""


class TextEmbeddingError(RuntimeError):
    """Sanitized provider or output failure for embedding operations."""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


class DeterministicTextEmbeddingProvider:
    """Stable hashed provider used only for offline demos and tests."""

    name = "deterministic"

    def embed(self, text: str, *, dimensions: int) -> list[float]:
        """Return the legacy deterministic vector."""

        _validate_embedding_request(text, dimensions)
        return _deterministic_dense_vector(text, dimensions)

    def fingerprint(self, *, dimensions: int) -> str:
        """Identify the deterministic fallback vector space."""

        return f"deterministic:sha256-token-v1:{dimensions}"


class OpenAITextEmbeddingProvider:
    """Dimension-preserving adapter for the OpenAI Embeddings API."""

    name = "openai"

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.client = client
        self.model = model
        self.timeout_seconds = timeout_seconds

    def embed(self, text: str, *, dimensions: int) -> list[float]:
        """Request and validate one OpenAI text embedding."""

        _validate_embedding_request(text, dimensions)
        provider_error_reason: str | None = None
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model,
                dimensions=dimensions,
                encoding_format="float",
                timeout=self.timeout_seconds,
            )
        except Exception as exc:
            provider_error_reason = _provider_reason(exc)
        if provider_error_reason is not None:
            raise TextEmbeddingError(provider_error_reason)

        raw_vector: Any = None
        try:
            raw_vector = response.data[0].embedding
        except (AttributeError, IndexError, TypeError):
            pass
        if (
            not isinstance(raw_vector, list)
            or len(raw_vector) != dimensions
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in raw_vector
            )
        ):
            raise TextEmbeddingError("invalid_provider_output")
        return [float(value) for value in raw_vector]

    def fingerprint(self, *, dimensions: int) -> str:
        """Identify the OpenAI model and requested output dimension."""

        return f"openai:{self.model}:{dimensions}"


def build_text_embedding_provider(
    environ: Mapping[str, str] | None = None,
    *,
    client_factory: Callable[[str], Any] | None = None,
) -> TextEmbeddingProvider:
    """Build the configured provider without making a network request."""

    values = os.environ if environ is None else environ
    mode = values.get("EMBEDDING_PROVIDER", "deterministic").strip().lower()
    if mode not in {"auto", "openai", "deterministic"}:
        raise ValueError(
            "EMBEDDING_PROVIDER must be auto, openai, or deterministic"
        )
    if mode == "deterministic":
        return DeterministicTextEmbeddingProvider()

    api_key = values.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        if mode == "openai":
            raise ValueError(
                "OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai"
            )
        return DeterministicTextEmbeddingProvider()

    model = values.get(
        "OPENAI_EMBEDDING_MODEL",
        DEFAULT_OPENAI_EMBEDDING_MODEL,
    ).strip()
    if not model:
        raise ValueError("OPENAI_EMBEDDING_MODEL must be non-empty")
    timeout_seconds = _positive_timeout(
        values.get("OPENAI_EMBEDDING_TIMEOUT_SECONDS", "30")
    )
    create_client = client_factory or _create_openai_client
    return OpenAITextEmbeddingProvider(
        client=create_client(api_key),
        model=model,
        timeout_seconds=timeout_seconds,
    )


def tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese/English text with deterministic expansions."""

    tokens: list[str] = []
    for match in TOKEN_PATTERN.findall(text.lower()):
        tokens.append(match)
        if re.fullmatch(r"[\u4e00-\u9fff]+", match) and len(match) > 1:
            tokens.extend(
                match[index : index + 2]
                for index in range(len(match) - 1)
            )
    expanded = list(tokens)
    for token in tokens:
        expanded.extend(SYNONYMS.get(token, []))
    return expanded


def sparse_vector(text: str) -> dict[str, float]:
    """Build a normalized term-frequency map."""

    counts = Counter(tokenize(text))
    total = sum(counts.values()) or 1
    return {token: count / total for token, count in counts.items()}


def dense_vector(
    text: str,
    dim: int = VECTOR_DIMS["TEXT_DIM"],
) -> list[float]:
    """Build a dense vector with the process-configured text provider."""

    return _default_text_embedding_provider().embed(text, dimensions=dim)


def text_embedding_fingerprint(
    dimensions: int = VECTOR_DIMS["TEXT_DIM"],
) -> str:
    """Return the configured vector-space identity for persistence checks."""

    return _default_text_embedding_provider().fingerprint(
        dimensions=dimensions
    )


def embedding_metadata(
    metadata: Mapping[str, Any] | None = None,
    *,
    dimensions: int = VECTOR_DIMS["TEXT_DIM"],
) -> dict[str, Any]:
    """Attach the configured text-vector fingerprint to record metadata."""

    output = dict(metadata or {})
    output[EMBEDDING_FINGERPRINT_KEY] = text_embedding_fingerprint(dimensions)
    return output


def _deterministic_dense_vector(text: str, dim: int) -> list[float]:
    """Build the legacy deterministic hashed vector."""

    vector = [0.0] * dim
    for token, weight in sparse_vector(text).items():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign * weight
    return normalize(vector)


@lru_cache(maxsize=1)
def _default_text_embedding_provider() -> TextEmbeddingProvider:
    """Reuse one configured provider and its HTTP connection pool."""

    return build_text_embedding_provider()


def _create_openai_client(api_key: str) -> Any:
    try:
        openai_module = import_module("openai")
    except ImportError as exc:
        raise RuntimeError(
            "Install OpenAI support with `pip install -r "
            "demo/requirements.txt`."
        ) from exc
    return openai_module.OpenAI(api_key=api_key, max_retries=0)


def _validate_embedding_request(text: str, dimensions: int) -> None:
    if not text.strip():
        raise ValueError("Embedding text must be non-empty")
    if dimensions <= 0:
        raise ValueError("Embedding dimensions must be greater than zero")


def _positive_timeout(raw_value: str) -> float:
    try:
        timeout = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            "OPENAI_EMBEDDING_TIMEOUT_SECONDS must be positive"
        ) from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(
            "OPENAI_EMBEDDING_TIMEOUT_SECONDS must be positive"
        )
    return timeout


def _provider_reason(error: Exception) -> str:
    error_name = type(error).__name__
    if isinstance(error, TimeoutError) or error_name == "APITimeoutError":
        return "timeout"
    if error_name == "APIConnectionError":
        return "connection_error"
    if error_name == "AuthenticationError":
        return "authentication_error"
    if error_name == "RateLimitError":
        return "rate_limited"
    return "provider_error"


def normalize(vector: list[float]) -> list[float]:
    """L2-normalize a vector."""

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return non-negative cosine similarity for equal-length vectors."""

    if not left or not right or len(left) != len(right):
        return 0.0
    return max(0.0, sum(a * b for a, b in zip(left, right)))
