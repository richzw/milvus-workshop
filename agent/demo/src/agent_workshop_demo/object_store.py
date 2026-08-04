"""Bounded, injectable MinIO source adapter for offline ingestion."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path, PurePosixPath
from typing import Any, Final, Protocol

DEFAULT_MINIO_MAX_OBJECTS: Final = 1_000
DEFAULT_MINIO_MAX_OBJECT_BYTES: Final = 16 * 1024 * 1024
READ_CHUNK_BYTES: Final = 64 * 1024
BUCKET_PATTERN: Final = re.compile(
    r"^(?=.{3,63}$)[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$"
)


class MinIOAdapterError(RuntimeError):
    """A sanitized MinIO configuration, listing, or download failure."""


class MinIOListedObject(Protocol):
    """Narrow object metadata surface used from the MinIO SDK."""

    object_name: str
    size: int | None
    is_dir: bool


class MinIOResponse(Protocol):
    """Narrow streaming response surface returned by ``get_object``."""

    def read(self, amount: int | None = None) -> bytes:
        """Read at most ``amount`` bytes."""

    def close(self) -> None:
        """Close the response body."""

    def release_conn(self) -> None:
        """Release the underlying HTTP connection."""


class MinIOClient(Protocol):
    """Injectable subset of the MinIO client used by ingestion."""

    def bucket_exists(self, bucket_name: str) -> bool:
        """Return whether ``bucket_name`` exists."""

    def list_objects(
        self,
        bucket_name: str,
        prefix: str | None = None,
        recursive: bool = False,
    ) -> Iterable[MinIOListedObject]:
        """List objects under ``prefix``."""

    def get_object(self, bucket_name: str, object_name: str) -> MinIOResponse:
        """Open one object response."""


@dataclass(frozen=True)
class MinIOConfig:
    """Validated connection and resource bounds for one MinIO exercise."""

    endpoint: str
    bucket: str
    prefix: str = ""
    secure: bool = False
    access_key: str | None = None
    secret_key: str | None = None
    max_objects: int = DEFAULT_MINIO_MAX_OBJECTS
    max_object_bytes: int = DEFAULT_MINIO_MAX_OBJECT_BYTES

    def __post_init__(self) -> None:
        endpoint = self.endpoint.strip()
        bucket = self.bucket.strip()
        prefix = _normalize_prefix(self.prefix)
        if (
            not endpoint
            or len(endpoint) > 255
            or "://" in endpoint
            or "/" in endpoint
            or "@" in endpoint
            or any(character.isspace() for character in endpoint)
        ):
            raise ValueError(
                "MINIO_ENDPOINT must be a host[:port] without scheme, path, or credentials"
            )
        if not BUCKET_PATTERN.fullmatch(bucket):
            raise ValueError("MINIO_BUCKET must be a valid 3..63 character bucket name")
        access_key = _optional_secret(self.access_key)
        secret_key = _optional_secret(self.secret_key)
        if (access_key is None) != (secret_key is None):
            raise ValueError(
                "MINIO_ACCESS_KEY and MINIO_SECRET_KEY must both be set or both be empty"
            )
        if not 1 <= self.max_objects <= 100_000:
            raise ValueError("MINIO_MAX_OBJECTS must be between 1 and 100000")
        if not 1 <= self.max_object_bytes <= 1024 * 1024 * 1024:
            raise ValueError(
                "MINIO_MAX_OBJECT_BYTES must be between 1 and 1073741824"
            )
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "bucket", bucket)
        object.__setattr__(self, "prefix", prefix)
        object.__setattr__(self, "access_key", access_key)
        object.__setattr__(self, "secret_key", secret_key)

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> MinIOConfig:
        """Build validated configuration from environment values."""

        values = os.environ if environ is None else environ
        return cls(
            endpoint=values.get("MINIO_ENDPOINT", "localhost:9000"),
            bucket=values.get("MINIO_BUCKET", "internal-agent-chat-demo"),
            prefix=values.get("MINIO_PREFIX", ""),
            secure=_parse_bool(values.get("MINIO_SECURE", "false"), "MINIO_SECURE"),
            access_key=values.get("MINIO_ACCESS_KEY"),
            secret_key=values.get("MINIO_SECRET_KEY"),
            max_objects=_parse_positive_int(
                values.get("MINIO_MAX_OBJECTS", str(DEFAULT_MINIO_MAX_OBJECTS)),
                "MINIO_MAX_OBJECTS",
            ),
            max_object_bytes=_parse_positive_int(
                values.get(
                    "MINIO_MAX_OBJECT_BYTES",
                    str(DEFAULT_MINIO_MAX_OBJECT_BYTES),
                ),
                "MINIO_MAX_OBJECT_BYTES",
            ),
        )

    @property
    def base_uri(self) -> str:
        """Return the display-safe stable URI prefix for downloaded objects."""

        suffix = f"/{self.prefix}" if self.prefix else ""
        return f"s3://{self.bucket}{suffix}"


@dataclass(frozen=True)
class MinIOSnapshot:
    """Materialized object-store snapshot consumed by the shared parser."""

    root: Path
    base_uri: str
    object_count: int
    total_bytes: int

    def __post_init__(self) -> None:
        if not self.root.is_dir():
            raise ValueError("MinIO snapshot root must be a directory")
        if not self.base_uri.startswith("s3://"):
            raise ValueError("MinIO snapshot base_uri must use s3://")
        if self.object_count <= 0:
            raise ValueError("MinIO snapshot must contain at least one object")
        if self.total_bytes < 0:
            raise ValueError("MinIO snapshot total_bytes cannot be negative")


class MinIOSourceAdapter:
    """Download one deterministic, bounded MinIO prefix into a temporary root."""

    def __init__(self, client: MinIOClient, config: MinIOConfig) -> None:
        self.client = client
        self.config = config

    def download_to(self, target_dir: Path) -> MinIOSnapshot:
        """Download configured objects after validating every storage boundary."""

        target_dir.mkdir(parents=True, exist_ok=True)
        if not target_dir.is_dir():
            raise NotADirectoryError(f"MinIO snapshot target is not a directory: {target_dir}")
        if any(target_dir.iterdir()):
            raise ValueError("MinIO snapshot target directory must be empty")
        try:
            exists = self.client.bucket_exists(self.config.bucket)
        except Exception:
            raise MinIOAdapterError(
                f"Unable to check MinIO bucket {self.config.bucket!r}"
            ) from None
        if not exists:
            raise MinIOAdapterError(
                f"MinIO bucket {self.config.bucket!r} does not exist"
            )
        try:
            listed = self.client.list_objects(
                self.config.bucket,
                prefix=(f"{self.config.prefix}/" if self.config.prefix else None),
                recursive=True,
            )
        except Exception:
            raise MinIOAdapterError(
                f"Unable to list MinIO bucket {self.config.bucket!r}"
            ) from None

        objects: list[tuple[str, PurePosixPath, int | None]] = []
        destinations: set[PurePosixPath] = set()
        try:
            for item in listed:
                if item.is_dir:
                    continue
                if len(objects) >= self.config.max_objects:
                    raise MinIOAdapterError(
                        f"MinIO prefix {self.config.base_uri!r} exceeds "
                        f"{self.config.max_objects} objects"
                    )
                key = item.object_name
                relative = _safe_relative_key(key, prefix=self.config.prefix)
                if relative in destinations:
                    raise MinIOAdapterError(
                        "MinIO returned colliding object destinations"
                    )
                destinations.add(relative)
                size = item.size
                if size is not None and (
                    size < 0 or size > self.config.max_object_bytes
                ):
                    raise MinIOAdapterError(
                        f"MinIO object {self._safe_uri(key)!r} exceeds "
                        f"{self.config.max_object_bytes} bytes"
                    )
                objects.append((key, relative, size))
        except MinIOAdapterError:
            raise
        except Exception:
            raise MinIOAdapterError(
                f"Unable to list MinIO bucket {self.config.bucket!r}"
            ) from None
        objects.sort(key=lambda item: item[0])
        if not objects:
            raise MinIOAdapterError(
                f"MinIO prefix {self.config.base_uri!r} contains no objects"
            )
        staging = Path(
            tempfile.mkdtemp(
                prefix=".minio-staging-",
                dir=target_dir,
            )
        )
        snapshot_root = target_dir / "snapshot"
        total_bytes = 0
        try:
            for key, relative, expected_size in objects:
                destination = staging.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                downloaded = self._download_object(
                    key,
                    destination,
                    expected_size=expected_size,
                )
                total_bytes += downloaded
            staging.replace(snapshot_root)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            if snapshot_root.exists():
                shutil.rmtree(snapshot_root)
            raise
        return MinIOSnapshot(
            root=snapshot_root,
            base_uri=self.config.base_uri,
            object_count=len(objects),
            total_bytes=total_bytes,
        )

    def _download_object(
        self,
        key: str,
        destination: Path,
        *,
        expected_size: int | None,
    ) -> int:
        safe_uri = self._safe_uri(key)
        try:
            response = self.client.get_object(self.config.bucket, key)
        except Exception:
            raise MinIOAdapterError(
                f"Unable to open MinIO object {safe_uri!r}"
            ) from None

        downloaded = 0
        primary_error: Exception | None = None
        cleanup_error: Exception | None = None
        try:
            with destination.open("wb") as handle:
                while True:
                    remaining = self.config.max_object_bytes - downloaded + 1
                    chunk = response.read(min(READ_CHUNK_BYTES, remaining))
                    if not isinstance(chunk, bytes):
                        raise MinIOAdapterError(
                            f"MinIO object {safe_uri!r} returned a non-bytes body"
                        )
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > self.config.max_object_bytes:
                        raise MinIOAdapterError(
                            f"MinIO object {safe_uri!r} exceeds "
                            f"{self.config.max_object_bytes} bytes"
                        )
                    handle.write(chunk)
            if expected_size is not None and downloaded != expected_size:
                raise MinIOAdapterError(
                    f"MinIO object {safe_uri!r} size changed during download"
                )
        except Exception as exc:
            primary_error = exc
        finally:
            try:
                response.close()
            except Exception as exc:
                cleanup_error = exc
            try:
                response.release_conn()
            except Exception as exc:
                if cleanup_error is None:
                    cleanup_error = exc

        if primary_error is not None:
            destination.unlink(missing_ok=True)
            if isinstance(primary_error, MinIOAdapterError):
                raise primary_error
            raise MinIOAdapterError(
                f"Unable to download MinIO object {safe_uri!r}"
            ) from None
        if cleanup_error is not None:
            destination.unlink(missing_ok=True)
            raise MinIOAdapterError(
                f"Unable to close MinIO object {safe_uri!r}"
            ) from None
        return downloaded

    def _safe_uri(self, key: str) -> str:
        return f"s3://{self.config.bucket}/{key}"


def build_minio_source_adapter(
    environ: Mapping[str, str] | None = None,
    *,
    client: MinIOClient | None = None,
) -> MinIOSourceAdapter:
    """Build the real SDK adapter, or use an injected contract-test client."""

    config = MinIOConfig.from_env(environ)
    if client is not None:
        return MinIOSourceAdapter(client, config)
    try:
        minio_module = import_module("minio")
    except ImportError:
        raise MinIOAdapterError(
            "MinIO source mode requires demo/requirements.txt"
        ) from None
    try:
        sdk_client: Any = minio_module.Minio(
            config.endpoint,
            access_key=config.access_key,
            secret_key=config.secret_key,
            secure=config.secure,
        )
    except Exception:
        raise MinIOAdapterError("Unable to initialize the MinIO client") from None
    return MinIOSourceAdapter(sdk_client, config)


def _safe_relative_key(key: str, *, prefix: str) -> PurePosixPath:
    if (
        not isinstance(key, str)
        or not key
        or key.startswith("/")
        or "\\" in key
        or "\x00" in key
    ):
        raise MinIOAdapterError("MinIO returned an unsafe object key")
    key_parts = key.split("/")
    if any(part in {"", ".", ".."} for part in key_parts):
        raise MinIOAdapterError("MinIO returned an unsafe object key")
    if prefix:
        prefix_with_separator = f"{prefix}/"
        if not key.startswith(prefix_with_separator):
            raise MinIOAdapterError("MinIO returned an object outside the configured prefix")
        relative_parts = key_parts[len(prefix.split("/")) :]
    else:
        relative_parts = key_parts
    if not relative_parts:
        raise MinIOAdapterError("MinIO returned an unsafe object key")
    return PurePosixPath(*relative_parts)


def _normalize_prefix(value: str) -> str:
    prefix = value.strip().strip("/")
    if "\\" in prefix or "\x00" in prefix:
        raise ValueError("MINIO_PREFIX must use safe POSIX path segments")
    if prefix and any(part in {"", ".", ".."} for part in prefix.split("/")):
        raise ValueError("MINIO_PREFIX must use safe POSIX path segments")
    return prefix


def _optional_secret(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if len(stripped) > 256:
        raise ValueError("MinIO credential values must be at most 256 characters")
    return stripped


def _parse_bool(value: str, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def _parse_positive_int(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return parsed
