"""Image-file embeddings with deterministic and DINOv3 providers."""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Callable, Mapping
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol

from agent_workshop_demo.config import VECTOR_DIMS
from agent_workshop_demo.embedding import normalize

DEFAULT_DINOV3_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DEFAULT_MAX_IMAGE_BYTES = 20 * 1024 * 1024
IMAGE_EMBEDDING_FINGERPRINT_KEY = "image_embedding_fingerprint"
IMAGE_POOLING = "pooler_output"
IMAGE_NORMALIZATION = "l2"
SUPPORTED_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})
IMAGE_ERROR_REASONS = frozenset(
    {
        "missing_source",
        "unsupported_format",
        "oversize",
        "decode_error",
        "dependency_missing",
        "device_unavailable",
        "model_load_error",
        "inference_error",
        "invalid_provider_output",
        "dimension_mismatch",
    }
)


class ImageEmbeddingProvider(Protocol):
    """Small interface shared by image-file embedding implementations."""

    name: str

    def embed(self, image_path: Path, *, dimensions: int) -> list[float]:
        """Return one validated image vector."""

    def fingerprint(self, *, dimensions: int) -> str:
        """Identify the persisted image vector space."""


class DINOv3Runtime(Protocol):
    """Narrow inference seam used by the real provider and fake-runtime tests."""

    def embed(self, image_path: Path) -> list[float]:
        """Return the model's unnormalized global pooled feature."""


class ImageEmbeddingError(RuntimeError):
    """Sanitized image source, dependency, model or output failure."""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = (
            reason_code
            if reason_code in IMAGE_ERROR_REASONS
            else "inference_error"
        )
        super().__init__(self.reason_code)


class DeterministicImageEmbeddingProvider:
    """Stable image-byte hash used only by offline demos and tests."""

    name = "deterministic"

    def __init__(
        self,
        *,
        max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    ) -> None:
        self.max_image_bytes = _validate_max_bytes(max_image_bytes)

    def embed(self, image_path: Path, *, dimensions: int) -> list[float]:
        """Hash validated image bytes into a distinct offline vector space."""

        _validate_dimensions(dimensions)
        content = _read_validated_image(
            image_path,
            max_image_bytes=self.max_image_bytes,
        )
        values: list[float] = []
        counter = 0
        seed = hashlib.sha256(content).digest()
        while len(values) < dimensions:
            block = hashlib.sha256(
                seed + counter.to_bytes(8, "big")
            ).digest()
            values.extend((byte / 127.5) - 1.0 for byte in block)
            counter += 1
        return _validated_normalized_vector(values[:dimensions], dimensions)

    def fingerprint(self, *, dimensions: int) -> str:
        """Identify the deterministic image-byte vector space."""

        _validate_dimensions(dimensions)
        return f"deterministic:sha256-image-bytes-v1:l2:{dimensions}"


class DINOv3ImageEmbeddingProvider:
    """Lazily loaded DINOv3 global image-feature provider."""

    name = "dinov3"

    def __init__(
        self,
        *,
        model: str = DEFAULT_DINOV3_MODEL,
        device: str = "cpu",
        token: str | None = None,
        local_files_only: bool = False,
        max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
        runtime_factory: Callable[
            [str, str, str | None, bool],
            DINOv3Runtime,
        ]
        | None = None,
    ) -> None:
        if not model.strip() or len(model) > 240:
            raise ValueError("DINOv3 model must contain 1..240 characters")
        if device not in {"cpu", "cuda", "mps", "auto"}:
            raise ValueError("DINOV3_DEVICE must be cpu, cuda, mps, or auto")
        self.model = model
        self.device = device
        self.token = token
        self.local_files_only = local_files_only
        self.max_image_bytes = _validate_max_bytes(max_image_bytes)
        self.runtime_factory = runtime_factory or _build_transformers_runtime
        self._runtime: DINOv3Runtime | None = None

    def embed(self, image_path: Path, *, dimensions: int) -> list[float]:
        """Decode one image, run DINOv3 once, and validate its global feature."""

        if dimensions != VECTOR_DIMS["IMAGE_DIM"]:
            raise ImageEmbeddingError("dimension_mismatch")
        _read_validated_image(
            image_path,
            max_image_bytes=self.max_image_bytes,
        )
        if self._runtime is None:
            try:
                self._runtime = self.runtime_factory(
                    self.model,
                    self.device,
                    self.token,
                    self.local_files_only,
                )
            except ImageEmbeddingError:
                raise
            except Exception:
                raise ImageEmbeddingError("model_load_error") from None
        try:
            raw_vector = self._runtime.embed(image_path)
        except ImageEmbeddingError:
            raise
        except Exception:
            raise ImageEmbeddingError("inference_error") from None
        return _validated_normalized_vector(raw_vector, dimensions)

    def fingerprint(self, *, dimensions: int) -> str:
        """Identify model, pooling, normalization and output dimension."""

        if dimensions != VECTOR_DIMS["IMAGE_DIM"]:
            raise ImageEmbeddingError("dimension_mismatch")
        return (
            f"dinov3:{self.model}:{IMAGE_POOLING}:"
            f"{IMAGE_NORMALIZATION}:{dimensions}"
        )


class _TransformersDINOv3Runtime:
    """Actual Pillow + Transformers + PyTorch inference runtime."""

    def __init__(
        self,
        model_id: str,
        device: str,
        token: str | None,
        local_files_only: bool,
        *,
        module_loader: Callable[[str], Any] | None = None,
    ) -> None:
        load_module = module_loader or import_module
        try:
            self.torch = load_module("torch")
            transformers = load_module("transformers")
            self.image_module = load_module("PIL.Image")
        except ImportError:
            raise ImageEmbeddingError("dependency_missing") from None
        self.device = _resolve_device(self.torch, device)
        load_args: dict[str, Any] = {
            "local_files_only": local_files_only,
            "trust_remote_code": False,
        }
        if token:
            load_args["token"] = token
        try:
            self.processor = transformers.AutoImageProcessor.from_pretrained(
                model_id,
                **load_args,
            )
            self.model = transformers.AutoModel.from_pretrained(
                model_id,
                **load_args,
            )
            self.model.to(self.device)
            self.model.eval()
        except Exception:
            raise ImageEmbeddingError("model_load_error") from None

    def embed(self, image_path: Path) -> list[float]:
        """Return the unnormalized `pooler_output` for one RGB image."""

        try:
            with self.image_module.open(image_path) as image:
                inputs = self.processor(
                    images=image.convert("RGB"),
                    return_tensors="pt",
                )
        except Exception:
            raise ImageEmbeddingError("decode_error") from None
        try:
            model_inputs = (
                inputs.to(self.device)
                if hasattr(inputs, "to")
                else inputs
            )
            with self.torch.inference_mode():
                outputs = self.model(**model_inputs)
            pooled = outputs.pooler_output
            values = pooled[0].detach().cpu().tolist()
        except Exception:
            raise ImageEmbeddingError("inference_error") from None
        if not isinstance(values, list):
            raise ImageEmbeddingError("invalid_provider_output")
        return values


def build_image_embedding_provider(
    environ: Mapping[str, str] | None = None,
    *,
    runtime_factory: Callable[
        [str, str, str | None, bool],
        DINOv3Runtime,
    ]
    | None = None,
) -> ImageEmbeddingProvider:
    """Build the explicit image provider without loading model weights."""

    values = os.environ if environ is None else environ
    mode = values.get(
        "IMAGE_EMBEDDING_PROVIDER",
        "deterministic",
    ).strip().lower()
    if mode not in {"deterministic", "dinov3"}:
        raise ValueError(
            "IMAGE_EMBEDDING_PROVIDER must be deterministic or dinov3"
        )
    max_bytes = _positive_int(
        values.get(
            "IMAGE_EMBEDDING_MAX_BYTES",
            str(DEFAULT_MAX_IMAGE_BYTES),
        ),
        name="IMAGE_EMBEDDING_MAX_BYTES",
    )
    if mode == "deterministic":
        return DeterministicImageEmbeddingProvider(
            max_image_bytes=max_bytes
        )
    model = values.get("DINOV3_MODEL", DEFAULT_DINOV3_MODEL).strip()
    if not model:
        raise ValueError("DINOV3_MODEL must be non-empty")
    device = values.get("DINOV3_DEVICE", "cpu").strip().lower()
    local_files_only = _boolean(
        values.get("DINOV3_LOCAL_FILES_ONLY", "false"),
        name="DINOV3_LOCAL_FILES_ONLY",
    )
    return DINOv3ImageEmbeddingProvider(
        model=model,
        device=device,
        token=values.get("HF_TOKEN", "").strip() or None,
        local_files_only=local_files_only,
        max_image_bytes=max_bytes,
        runtime_factory=runtime_factory,
    )


def image_file_vector(
    image_path: Path,
    dimensions: int = VECTOR_DIMS["IMAGE_DIM"],
) -> list[float]:
    """Embed one image with the process-configured provider."""

    return configured_image_embedding_provider().embed(
        image_path,
        dimensions=dimensions,
    )


def image_embedding_fingerprint(
    dimensions: int = VECTOR_DIMS["IMAGE_DIM"],
) -> str:
    """Return the configured image vector-space identity."""

    return configured_image_embedding_provider().fingerprint(
        dimensions=dimensions
    )


def image_embedding_metadata(
    metadata: Mapping[str, Any] | None = None,
    *,
    provider: ImageEmbeddingProvider | None = None,
    dimensions: int = VECTOR_DIMS["IMAGE_DIM"],
) -> dict[str, Any]:
    """Attach image-vector provenance without loading model weights."""

    selected = provider or configured_image_embedding_provider()
    output = dict(metadata or {})
    output[IMAGE_EMBEDDING_FINGERPRINT_KEY] = selected.fingerprint(
        dimensions=dimensions
    )
    output["image_pooling"] = (
        IMAGE_POOLING if selected.name == "dinov3" else "byte_hash"
    )
    output["image_normalization"] = IMAGE_NORMALIZATION
    return output


@lru_cache(maxsize=1)
def configured_image_embedding_provider() -> ImageEmbeddingProvider:
    """Return one process-wide provider so model weights are loaded once."""

    return build_image_embedding_provider()


def _build_transformers_runtime(
    model_id: str,
    device: str,
    token: str | None,
    local_files_only: bool,
) -> DINOv3Runtime:
    return _TransformersDINOv3Runtime(
        model_id,
        device,
        token,
        local_files_only,
    )


def _read_validated_image(
    image_path: Path,
    *,
    max_image_bytes: int,
) -> bytes:
    if (
        image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES
        or not image_path.is_file()
    ):
        reason = (
            "unsupported_format"
            if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES
            else "missing_source"
        )
        raise ImageEmbeddingError(reason)
    try:
        with image_path.open("rb") as handle:
            content = handle.read(max_image_bytes + 1)
    except OSError:
        raise ImageEmbeddingError("missing_source") from None
    if len(content) > max_image_bytes:
        raise ImageEmbeddingError("oversize")
    if not _has_supported_signature(content):
        raise ImageEmbeddingError("unsupported_format")
    return content


def _has_supported_signature(content: bytes) -> bool:
    return (
        content.startswith(b"\x89PNG\r\n\x1a\n")
        or content.startswith(b"\xff\xd8\xff")
        or (
            len(content) >= 12
            and content[:4] == b"RIFF"
            and content[8:12] == b"WEBP"
        )
    )


def _validated_normalized_vector(
    raw_vector: object,
    dimensions: int,
) -> list[float]:
    if (
        not isinstance(raw_vector, list)
        or len(raw_vector) != dimensions
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in raw_vector
        )
    ):
        raise ImageEmbeddingError("invalid_provider_output")
    vector = [float(value) for value in raw_vector]
    normalized = normalize(vector)
    norm = math.sqrt(sum(value * value for value in normalized))
    if not math.isfinite(norm) or not math.isclose(
        norm,
        1.0,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise ImageEmbeddingError("invalid_provider_output")
    return normalized


def _resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        if bool(torch.cuda.is_available()):
            return "cuda"
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        return "mps" if mps and bool(mps.is_available()) else "cpu"
    if requested == "cuda" and not bool(torch.cuda.is_available()):
        raise ImageEmbeddingError("device_unavailable")
    if requested == "mps":
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is None or not bool(mps.is_available()):
            raise ImageEmbeddingError("device_unavailable")
    return requested


def _validate_dimensions(dimensions: int) -> None:
    if dimensions <= 0:
        raise ValueError("Image embedding dimensions must be positive")


def _validate_max_bytes(value: int) -> int:
    if value <= 0 or value > 100 * 1024 * 1024:
        raise ValueError(
            "IMAGE_EMBEDDING_MAX_BYTES must be between 1 and 104857600"
        )
    return value


def _positive_int(raw_value: str, *, name: str) -> int:
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    return _validate_max_bytes(value)


def _boolean(raw_value: str, *, name: str) -> bool:
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")
