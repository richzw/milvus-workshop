"""Offline local and mock-S3 ingestion for workshop fixtures."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

from agent_workshop_demo.chunking import (
    ChunkFragment,
    ChunkingConfig,
    count_chunk_tokens,
    split_text,
)
from agent_workshop_demo.config import VECTOR_DIMS
from agent_workshop_demo.dedup import build_dedup_record, checksum
from agent_workshop_demo.embedding import (
    dense_vector,
    embedding_metadata,
    sparse_vector,
)
from agent_workshop_demo.image_embedding import (
    ImageEmbeddingError,
    ImageEmbeddingProvider,
    configured_image_embedding_provider,
    image_embedding_metadata,
)
from agent_workshop_demo.models import KBChunk
from agent_workshop_demo.object_store import MinIOSourceAdapter

DEFAULT_NOW_MS = 1782604800000
SUPPORTED_TEXT_SUFFIXES = {".md", ".txt"}
SUPPORTED_PDF_SUFFIX = ".pdf"
ASSET_MANIFEST = "asset_manifest.json"
VERSION_MANIFEST = "document_versions.json"
RETRIEVAL_TEXT_VERSION = "title-heading-text-v1"
MARKDOWN_HEADING_PATTERN = re.compile(
    r"^(?P<marks>#{1,6})[ \t]+(?P<title>.+?)[ \t]*#*[ \t]*$"
)


@dataclass(frozen=True)
class IngestionResult:
    """Validated records produced by one offline ingestion run."""

    kb_chunks: list[KBChunk]
    dedup_signatures: list[dict[str, Any]]


@dataclass(frozen=True)
class MinIOIngestionResult:
    """Shared ingestion output plus bounded MinIO snapshot statistics."""

    ingestion: IngestionResult
    source_base_uri: str
    source_object_count: int
    source_total_bytes: int


@dataclass(frozen=True)
class DocumentVersion:
    """Validated version metadata attached to every source chunk."""

    doc_id: str
    doc_version: str
    is_current: bool


@dataclass(frozen=True)
class TextUnit:
    """One semantic text unit and its inherited Markdown context."""

    text: str
    section: str | None
    heading_path: tuple[str, ...] = ()
    semantic_unit_index: int | None = None
    split_index: int = 1
    token_count: int | None = None
    applied_overlap_tokens: int = 0


def ingest_demo_sources(
    local_dir: Path,
    mock_s3_dir: Path,
    *,
    version_manifest_path: Path | None = None,
    mock_s3_base_uri: str = "s3://internal-agent-chat-demo",
    image_embedding_provider: ImageEmbeddingProvider | None = None,
    chunking_config: ChunkingConfig | None = None,
) -> IngestionResult:
    """Ingest required local and mock-S3 fixture directories."""

    _require_source_directory(local_dir, source_name="local")
    _require_source_directory(mock_s3_dir, source_name="mock S3")
    resolved_version_manifest = (
        local_dir.parent / VERSION_MANIFEST
        if version_manifest_path is None
        else version_manifest_path
    )
    version_manifest = _load_version_manifest(resolved_version_manifest)
    selected_image_provider = (
        image_embedding_provider or configured_image_embedding_provider()
    )
    local_manifest_chunks = _ingest_asset_manifest(
        local_dir,
        manifest_path=local_dir.parent / ASSET_MANIFEST,
        source_type="local",
        base_uri="sample_data/local_docs",
        version_manifest=version_manifest,
        image_embedding_provider=selected_image_provider,
        chunking_config=chunking_config,
    )
    handled_local_paths = {
        Path(str(item.metadata["asset_path"]))
        for item in local_manifest_chunks
        if item.metadata and "asset_path" in item.metadata
    }
    s3_manifest_path = mock_s3_dir / ASSET_MANIFEST
    s3_manifest_chunks = _ingest_asset_manifest(
        mock_s3_dir,
        manifest_path=s3_manifest_path,
        source_type="s3",
        base_uri=mock_s3_base_uri,
        version_manifest=version_manifest,
        image_embedding_provider=selected_image_provider,
        chunking_config=chunking_config,
    )
    handled_s3_paths = {
        Path(str(item.metadata["asset_path"]))
        for item in s3_manifest_chunks
        if item.metadata and "asset_path" in item.metadata
    }
    if s3_manifest_path.exists():
        handled_s3_paths.add(Path(ASSET_MANIFEST))
    chunks = _ingest_tree(
        local_dir,
        source_type="local",
        base_uri="sample_data/local_docs",
        handled_paths=handled_local_paths,
        version_manifest=version_manifest,
        chunking_config=chunking_config,
    )
    chunks.extend(local_manifest_chunks)
    chunks.extend(
        _ingest_tree(
            mock_s3_dir,
            source_type="s3",
            base_uri=mock_s3_base_uri,
            handled_paths=handled_s3_paths,
            version_manifest=version_manifest,
            chunking_config=chunking_config,
        )
    )
    chunks.extend(s3_manifest_chunks)
    identities = [
        (item.doc_id, item.doc_version, item.chunk_id) for item in chunks
    ]
    if len(identities) != len(set(identities)):
        raise ValueError(
            "Ingestion produced duplicate "
            "(doc_id, doc_version, chunk_id) identities"
        )
    chunk_ids = [item.chunk_id for item in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Ingestion produced duplicate global chunk_id values")
    _validate_version_families(chunks)
    dedup_records = [
        build_dedup_record(
            doc_id=item.doc_id,
            chunk_id=item.chunk_id,
            source_uri=item.source_uri,
            source_type=item.source_type,
            record_level="chunk",
            text=item.text,
            created_at=item.created_at or DEFAULT_NOW_MS,
        )
        for item in chunks
    ]
    return IngestionResult(chunks, dedup_records)


def ingest_minio_sources(
    local_dir: Path,
    adapter: MinIOSourceAdapter,
    *,
    version_manifest_path: Path | None = None,
    image_embedding_provider: ImageEmbeddingProvider | None = None,
    chunking_config: ChunkingConfig | None = None,
) -> MinIOIngestionResult:
    """Download a bounded MinIO snapshot and reuse the normal ingestion pipeline."""

    with tempfile.TemporaryDirectory(prefix="agent-workshop-minio-") as temp_dir:
        snapshot = adapter.download_to(Path(temp_dir))
        ingestion = ingest_demo_sources(
            local_dir,
            snapshot.root,
            version_manifest_path=version_manifest_path,
            mock_s3_base_uri=snapshot.base_uri,
            image_embedding_provider=image_embedding_provider,
            chunking_config=chunking_config,
        )
    return MinIOIngestionResult(
        ingestion=ingestion,
        source_base_uri=snapshot.base_uri,
        source_object_count=snapshot.object_count,
        source_total_bytes=snapshot.total_bytes,
    )


def write_ingestion_result(
    result: IngestionResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Write deterministic JSONL outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    kb_path = output_dir / "kb_chunks.jsonl"
    dedup_path = output_dir / "doc_dedup_signatures.jsonl"
    _write_jsonl(kb_path, (item.to_dict() for item in result.kb_chunks))
    _write_jsonl(dedup_path, result.dedup_signatures)
    return {
        "kb_chunks": kb_path,
        "doc_dedup_signatures": dedup_path,
    }


def _ingest_tree(
    root: Path,
    *,
    source_type: str,
    base_uri: str,
    handled_paths: set[Path],
    version_manifest: dict[str, DocumentVersion],
    chunking_config: ChunkingConfig | None,
) -> list[KBChunk]:
    chunks: list[KBChunk] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative in handled_paths:
            continue
        suffix = path.suffix.lower()
        if suffix in SUPPORTED_TEXT_SUFFIXES:
            chunks.extend(
                _chunk_file(
                    path,
                    root=root,
                    source_type=source_type,
                    base_uri=base_uri,
                    version_manifest=version_manifest,
                    chunking_config=chunking_config,
                )
            )
        elif suffix == SUPPORTED_PDF_SUFFIX:
            chunks.extend(
                _chunk_pdf(
                    path,
                    root=root,
                    source_type=source_type,
                    base_uri=base_uri,
                    version_manifest=version_manifest,
                    chunking_config=chunking_config,
                )
            )
        else:
            raise ValueError(
                f"Unsupported document type {suffix or '<none>'!r}: {path}"
            )
    return chunks


def _ingest_asset_manifest(
    root: Path,
    *,
    manifest_path: Path,
    source_type: str,
    base_uri: str,
    version_manifest: dict[str, DocumentVersion],
    image_embedding_provider: ImageEmbeddingProvider,
    chunking_config: ChunkingConfig | None,
) -> list[KBChunk]:
    if not manifest_path.exists():
        return []
    try:
        records = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Unable to read asset manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(records, list):
        raise ValueError(f"Asset manifest {manifest_path} must be a JSON list")

    output: list[KBChunk] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Asset manifest record {index} must be an object")
        try:
            asset_path = _safe_manifest_asset_path(
                record["asset_path"],
                record_index=index,
            )
            text = record["text"]
            doc_type = record["doc_type"]
            record_type = record["record_type"]
            title = record["title"]
        except KeyError as exc:
            raise ValueError(
                f"Asset manifest record {index} is missing {exc.args[0]!r}"
            ) from exc
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Asset manifest record {index} has empty text")
        full_path = root.joinpath(*asset_path.parts)
        try:
            full_path.resolve(strict=False).relative_to(
                root.resolve(strict=True)
            )
        except (OSError, ValueError):
            raise ValueError(
                f"Asset manifest record {index} escapes its source root"
            ) from None
        if not full_path.is_file():
            raise FileNotFoundError(
                f"Asset manifest record {index} references {full_path}"
            )
        relative = Path(*asset_path.parts)
        source_uri = f"{base_uri}/{asset_path.as_posix()}"
        version = _version_for_source(source_uri, version_manifest)
        identity_prefix = _versioned_identity_prefix(version)
        has_image = doc_type == "image"
        section = record.get("section")
        image_embedding: list[float] | None = None
        asset_metadata: dict[str, Any] = {
            "parser": "asset_manifest",
            "asset_path": asset_path.as_posix(),
            "retrieval_text_version": RETRIEVAL_TEXT_VERSION,
            "mime_type": (
                "application/pdf"
                if doc_type == "pdf"
                else _image_mime_type(relative)
            ),
        }
        if has_image:
            try:
                image_embedding = image_embedding_provider.embed(
                    full_path,
                    dimensions=VECTOR_DIMS["IMAGE_DIM"],
                )
            except ImageEmbeddingError as exc:
                raise ValueError(
                    f"Image embedding failed for {source_uri}: "
                    f"{exc.reason_code}"
                ) from None
            try:
                image_checksum = hashlib.sha256(
                    full_path.read_bytes()
                ).hexdigest()
            except OSError:
                raise ValueError(
                    f"Unable to checksum image source {source_uri}"
                ) from None
            asset_metadata["image_checksum"] = image_checksum
            asset_metadata["mime_type"] = _image_mime_type(relative)
            asset_metadata = image_embedding_metadata(
                asset_metadata,
                provider=image_embedding_provider,
            )
        bucket, object_key = _object_store_coordinates(
            source_type=source_type,
            base_uri=base_uri,
            relative=relative,
        )
        token_count = count_chunk_tokens(text)
        fragments = (
            split_text(text, config=chunking_config)
            if chunking_config is not None and not has_image
            else [
                ChunkFragment(
                    text=text,
                    token_count=token_count,
                    start_token=0,
                    end_token=token_count,
                    applied_overlap_tokens=0,
                )
            ]
        )
        for split_index, fragment in enumerate(fragments, start=1):
            fragment_metadata = dict(asset_metadata)
            if not has_image:
                fragment_metadata.update(
                    _chunking_metadata(
                        chunking_config,
                        semantic_unit_index=index,
                        split_index=split_index,
                        token_count=fragment.token_count,
                        applied_overlap_tokens=(
                            fragment.applied_overlap_tokens
                        ),
                    )
                )
            retrieval_text = _retrieval_text(
                title=title,
                section=section,
                heading_path=(),
                text=fragment.text,
            )
            chunk_index = (
                index if chunking_config is None else len(output) + 1
            )
            chunk_id = f"{identity_prefix}_asset_{index:03d}"
            if chunking_config is not None and not has_image:
                chunk_id = f"{chunk_id}_c{split_index:03d}"
            output.append(
                KBChunk(
                    doc_id=version.doc_id,
                    chunk_id=chunk_id,
                    parent_id=None,
                    record_type=record_type,
                    source_type=source_type,
                    source_uri=source_uri,
                    bucket=bucket,
                    object_key=object_key,
                    doc_type=doc_type,
                    title=title,
                    section=section,
                    page_no=record.get("page_no"),
                    chunk_index=chunk_index,
                    text=fragment.text,
                    text_summary=fragment.text[:180],
                    language="mixed",
                    department=record.get(
                        "department",
                        _department_from_path(relative),
                    ),
                    updated_at=DEFAULT_NOW_MS - chunk_index * 86_400_000,
                    created_at=(
                        DEFAULT_NOW_MS
                        - (chunk_index + 30) * 86_400_000
                    ),
                    priority=record.get("priority", 5),
                    doc_version=version.doc_version,
                    is_current=version.is_current,
                    checksum=(
                        str(fragment_metadata["image_checksum"])
                        if has_image
                        else checksum(fragment.text)
                    ),
                    metadata=embedding_metadata(fragment_metadata),
                    has_image_vector=has_image,
                    text_vector=dense_vector(retrieval_text),
                    sparse_vector=sparse_vector(retrieval_text),
                    image_vector=image_embedding,
                )
            )
    return output


def _safe_manifest_asset_path(
    value: Any,
    *,
    record_index: int,
) -> PurePosixPath:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(
            f"Asset manifest record {record_index} has an invalid asset_path"
        )
    segments = value.split("/")
    if any(segment in {"", ".", ".."} for segment in segments):
        raise ValueError(
            f"Asset manifest record {record_index} has an unsafe asset_path"
        )
    path = PurePosixPath(value)
    if path.is_absolute():
        raise ValueError(
            f"Asset manifest record {record_index} has an unsafe asset_path"
        )
    return path


def _image_mime_type(path: Path) -> str:
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "application/octet-stream")


def _chunk_file(
    path: Path,
    *,
    root: Path,
    source_type: str,
    base_uri: str,
    version_manifest: dict[str, DocumentVersion],
    chunking_config: ChunkingConfig | None,
) -> list[KBChunk]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Unable to read source document {path}: {exc}") from exc
    if not text.strip():
        raise ValueError(f"Source document is empty: {path}")
    relative = path.relative_to(root)
    source_uri = f"{base_uri}/{relative.as_posix()}"
    version = _version_for_source(source_uri, version_manifest)
    identity_prefix = _versioned_identity_prefix(version)
    units = (
        _markdown_units(text)
        if path.suffix.lower() == ".md"
        else (
            [TextUnit(text=text.strip(), section=None)]
            if chunking_config is not None
            else _plain_text_units(text)
        )
    )
    units = _apply_chunking_config(units, chunking_config)
    department = _department_from_path(relative)
    doc_type = _doc_type(path)
    bucket, object_key = _object_store_coordinates(
        source_type=source_type,
        base_uri=base_uri,
        relative=relative,
    )
    output: list[KBChunk] = []
    title = _title_from_text(text, path)
    for index, unit in enumerate(units, start=1):
        unit_metadata: dict[str, Any] = {
            "parser": "local_markdown" if doc_type == "markdown" else "plain_text",
            "relative_path": relative.as_posix(),
        }
        if unit.heading_path:
            unit_metadata["heading_path"] = list(unit.heading_path)
        unit_metadata["retrieval_text_version"] = RETRIEVAL_TEXT_VERSION
        unit_metadata.update(
            _chunking_metadata(
                chunking_config,
                semantic_unit_index=unit.semantic_unit_index,
                split_index=unit.split_index,
                token_count=unit.token_count,
                applied_overlap_tokens=unit.applied_overlap_tokens,
            )
        )
        retrieval_text = _retrieval_text(
            title=title,
            section=unit.section,
            heading_path=unit.heading_path,
            text=unit.text,
        )
        output.append(
            KBChunk(
                doc_id=version.doc_id,
                chunk_id=f"{identity_prefix}_c{index:03d}",
                parent_id=None,
                record_type="text_chunk",
                source_type=source_type,
                source_uri=source_uri,
                bucket=bucket,
                object_key=object_key,
                doc_type=doc_type,
                title=title,
                section=unit.section,
                page_no=None,
                chunk_index=index,
                text=unit.text,
                text_summary=unit.text[:180],
                language="mixed",
                department=department,
                updated_at=DEFAULT_NOW_MS - index * 86_400_000,
                created_at=DEFAULT_NOW_MS - (index + 30) * 86_400_000,
                priority=8 if department == "engineering" else 5,
                doc_version=version.doc_version,
                is_current=version.is_current,
                checksum=checksum(unit.text),
                metadata=embedding_metadata(unit_metadata),
                has_image_vector=False,
                text_vector=dense_vector(retrieval_text),
                sparse_vector=sparse_vector(retrieval_text),
                image_vector=None,
            )
        )
    return output


def _plain_text_units(text: str) -> list[TextUnit]:
    return [
        TextUnit(text=part.strip(), section=None)
        for part in text.split("\n\n")
        if part.strip()
    ]


def _markdown_units(text: str) -> list[TextUnit]:
    """Split Markdown at semantic heading boundaries and retain heading ancestry."""

    units: list[TextUnit] = []
    heading_path: list[str] = []
    current_heading_line: str | None = None
    current_section: str | None = None
    current_path: tuple[str, ...] = ()
    body_lines: list[str] = []

    def flush() -> None:
        nonlocal body_lines
        body = "\n".join(body_lines).strip()
        if current_heading_line is not None:
            content = "\n\n".join(
                part for part in (current_heading_line, body) if part
            )
        else:
            content = body
        if content:
            units.append(
                TextUnit(
                    text=content,
                    section=current_section,
                    heading_path=current_path,
                )
            )
        body_lines = []

    for line in text.splitlines():
        match = MARKDOWN_HEADING_PATTERN.match(line.strip())
        if match is None:
            body_lines.append(line)
            continue
        flush()
        level = len(match.group("marks"))
        title = match.group("title").strip()
        heading_path[level - 1 :] = [title]
        current_heading_line = f"{'#' * level} {title}"
        current_section = title
        current_path = tuple(heading_path)
    flush()
    if not units:
        raise ValueError("Markdown source contains no ingestible text")
    return units


def _apply_chunking_config(
    units: list[TextUnit],
    config: ChunkingConfig | None,
) -> list[TextUnit]:
    if config is None:
        return units
    output: list[TextUnit] = []
    for semantic_index, unit in enumerate(units, start=1):
        fragments = split_text(unit.text, config=config)
        output.extend(
            TextUnit(
                text=fragment.text,
                section=unit.section,
                heading_path=unit.heading_path,
                semantic_unit_index=semantic_index,
                split_index=split_index,
                token_count=fragment.token_count,
                applied_overlap_tokens=fragment.applied_overlap_tokens,
            )
            for split_index, fragment in enumerate(fragments, start=1)
        )
    return output


def _chunking_metadata(
    config: ChunkingConfig | None,
    *,
    semantic_unit_index: int | None,
    split_index: int,
    token_count: int | None,
    applied_overlap_tokens: int,
) -> dict[str, Any]:
    if config is None:
        return {}
    if semantic_unit_index is None or token_count is None:
        raise ValueError("configured chunks require complete chunking metadata")
    return {
        "chunking": {
            **config.to_dict(),
            "config_fingerprint": config.fingerprint,
            "semantic_unit_index": semantic_unit_index,
            "split_index": split_index,
            "token_count": token_count,
            "applied_overlap_tokens": applied_overlap_tokens,
        }
    }


def _retrieval_text(
    *,
    title: str,
    section: str | None,
    heading_path: tuple[str, ...],
    text: str,
) -> str:
    """Attach bounded document context to the text used for retrieval vectors."""

    path_text = " > ".join(heading_path)
    parts: list[str] = []
    for value in (title, path_text, section or "", text):
        normalized = value.strip()
        if normalized and normalized not in parts:
            parts.append(normalized)
    return "\n".join(parts)


def _chunk_pdf(
    path: Path,
    *,
    root: Path,
    source_type: str,
    base_uri: str,
    version_manifest: dict[str, DocumentVersion],
    chunking_config: ChunkingConfig | None,
) -> list[KBChunk]:
    """Extract page-addressable PDF chunks through the optional parser."""

    try:
        pypdf = import_module("pypdf")
    except ImportError as exc:
        raise RuntimeError(
            f"PDF source {path} requires pypdf; install demo/requirements.txt"
        ) from exc
    try:
        reader = pypdf.PdfReader(path)
    except Exception as exc:
        raise ValueError(f"Unable to parse PDF source {path}: {exc}") from exc

    relative = path.relative_to(root)
    source_uri = f"{base_uri}/{relative.as_posix()}"
    version = _version_for_source(source_uri, version_manifest)
    identity_prefix = _versioned_identity_prefix(version)
    department = _department_from_path(relative)
    bucket, object_key = _object_store_coordinates(
        source_type=source_type,
        base_uri=base_uri,
        relative=relative,
    )
    metadata_title = getattr(getattr(reader, "metadata", None), "title", None)
    title = (
        str(metadata_title).strip()
        if metadata_title and str(metadata_title).strip()
        else path.stem.replace("_", " ").title()
    )
    output: list[KBChunk] = []
    for page_no, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text()
        except Exception as exc:
            raise ValueError(
                f"Unable to extract PDF source {path} page {page_no}: {exc}"
            ) from exc
        if not isinstance(page_text, str) or not page_text.strip():
            raise ValueError(
                f"PDF source {path} page {page_no} has no extractable text"
            )
        normalized_page = page_text.strip()
        if chunking_config is not None:
            page_fragments = [
                (page_no, split_index, fragment)
                for split_index, fragment in enumerate(
                    split_text(normalized_page, config=chunking_config),
                    start=1,
                )
            ]
        else:
            paragraphs = [
                part.strip()
                for part in normalized_page.split("\n\n")
                if part.strip()
            ]
            page_fragments = []
            for paragraph_index, paragraph in enumerate(
                paragraphs,
                start=1,
            ):
                paragraph_token_count = count_chunk_tokens(paragraph)
                page_fragments.append(
                    (
                        paragraph_index,
                        1,
                        ChunkFragment(
                            text=paragraph,
                            token_count=paragraph_token_count,
                            start_token=0,
                            end_token=paragraph_token_count,
                            applied_overlap_tokens=0,
                        ),
                    )
                )
        for semantic_unit_index, split_index, fragment in page_fragments:
            chunk_index = len(output) + 1
            fragment_section = _section_from_text(fragment.text)
            retrieval_text = _retrieval_text(
                title=title,
                section=fragment_section,
                heading_path=(),
                text=fragment.text,
            )
            chunk_metadata: dict[str, Any] = {
                "parser": "pypdf",
                "relative_path": relative.as_posix(),
                "page_count": len(reader.pages),
                "retrieval_text_version": RETRIEVAL_TEXT_VERSION,
            }
            chunk_metadata.update(
                _chunking_metadata(
                    chunking_config,
                    semantic_unit_index=semantic_unit_index,
                    split_index=split_index,
                    token_count=fragment.token_count,
                    applied_overlap_tokens=(
                        fragment.applied_overlap_tokens
                    ),
                )
            )
            identity_index = (
                semantic_unit_index
                if chunking_config is None
                else split_index
            )
            output.append(
                KBChunk(
                    doc_id=version.doc_id,
                    chunk_id=(
                        f"{identity_prefix}_p{page_no:03d}"
                        f"_c{identity_index:03d}"
                    ),
                    parent_id=None,
                    record_type="pdf_page",
                    source_type=source_type,
                    source_uri=source_uri,
                    bucket=bucket,
                    object_key=object_key,
                    doc_type="pdf",
                    title=title,
                    section=fragment_section,
                    page_no=page_no,
                    chunk_index=chunk_index,
                    text=fragment.text,
                    text_summary=fragment.text[:180],
                    language="mixed",
                    department=department,
                    updated_at=(
                        DEFAULT_NOW_MS - chunk_index * 86_400_000
                    ),
                    created_at=(
                        DEFAULT_NOW_MS
                        - (chunk_index + 30) * 86_400_000
                    ),
                    priority=8 if department == "engineering" else 5,
                    doc_version=version.doc_version,
                    is_current=version.is_current,
                    checksum=checksum(fragment.text),
                    metadata=embedding_metadata(chunk_metadata),
                    has_image_vector=False,
                    text_vector=dense_vector(retrieval_text),
                    sparse_vector=sparse_vector(retrieval_text),
                    image_vector=None,
                )
            )
    if not output:
        raise ValueError(f"PDF source contains no pages: {path}")
    return output


def _write_jsonl(
    path: Path,
    records: Iterable[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _stable_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def _load_version_manifest(path: Path) -> dict[str, DocumentVersion]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read version manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Version manifest {path} must be a JSON object")
    output: dict[str, DocumentVersion] = {}
    for source_uri, raw in payload.items():
        if not isinstance(source_uri, str) or not source_uri.strip():
            raise ValueError("Version manifest source URIs must be non-empty")
        if not isinstance(raw, dict):
            raise ValueError(
                f"Version manifest entry {source_uri!r} must be an object"
            )
        unknown = sorted(
            set(raw).difference({"doc_id", "doc_version", "is_current"})
        )
        if unknown:
            raise ValueError(
                f"Version manifest entry {source_uri!r} has unknown "
                f"fields: {unknown}"
            )
        doc_id = raw.get("doc_id")
        doc_version = raw.get("doc_version")
        is_current = raw.get("is_current")
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError(
                f"Version manifest entry {source_uri!r} needs doc_id"
            )
        if not isinstance(doc_version, str) or not doc_version.strip():
            raise ValueError(
                f"Version manifest entry {source_uri!r} needs doc_version"
            )
        if not isinstance(is_current, bool):
            raise ValueError(
                f"Version manifest entry {source_uri!r} needs boolean "
                "is_current"
            )
        output[source_uri] = DocumentVersion(
            doc_id=doc_id.strip(),
            doc_version=doc_version.strip(),
            is_current=is_current,
        )
    return output


def _version_for_source(
    source_uri: str,
    manifest: dict[str, DocumentVersion],
) -> DocumentVersion:
    configured = manifest.get(source_uri)
    if configured is not None:
        return configured
    return DocumentVersion(
        doc_id=f"doc_{_stable_id(source_uri)}",
        doc_version="unversioned",
        is_current=True,
    )


def _versioned_identity_prefix(version: DocumentVersion) -> str:
    if version.doc_version == "unversioned":
        return version.doc_id
    if version.doc_version.isalnum():
        safe_version = version.doc_version
    else:
        readable = "".join(
            character if character.isalnum() else "_"
            for character in version.doc_version
        ).strip("_")
        safe_version = (
            f"{(readable or 'version')[:32]}_"
            f"{_stable_id(version.doc_version)}"
        )
    if not safe_version:
        raise ValueError("doc_version must contain an alphanumeric character")
    return f"{version.doc_id}_{safe_version}"


def _validate_version_families(chunks: list[KBChunk]) -> None:
    editions: dict[str, dict[str, bool]] = {}
    source_uris: dict[tuple[str, str], set[str]] = {}
    for chunk in chunks:
        family = editions.setdefault(chunk.doc_id, {})
        prior = family.get(chunk.doc_version)
        if prior is not None and prior != chunk.is_current:
            raise ValueError(
                f"Document {chunk.doc_id!r} version "
                f"{chunk.doc_version!r} has conflicting is_current values"
            )
        family[chunk.doc_version] = chunk.is_current
        source_uris.setdefault(
            (chunk.doc_id, chunk.doc_version),
            set(),
        ).add(chunk.source_uri)
    for doc_id, family in editions.items():
        current = [
            version for version, is_current in family.items() if is_current
        ]
        if len(current) != 1:
            raise ValueError(
                f"Document {doc_id!r} must have exactly one current edition"
            )
        if "unversioned" in family and len(family) > 1:
            raise ValueError(
                f"Document {doc_id!r} cannot mix unversioned and versioned "
                "editions"
            )
        if (
            "unversioned" in family
            and len(source_uris[(doc_id, "unversioned")]) > 1
        ):
            raise ValueError(
                f"Document {doc_id!r} has multiple unversioned sources"
            )


def _require_source_directory(path: Path, *, source_name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{source_name} source directory does not exist: {path}"
        )
    if not path.is_dir():
        raise NotADirectoryError(
            f"{source_name} source path is not a directory: {path}"
        )


def _department_from_path(path: Path) -> str:
    first = path.parts[0].lower() if path.parts else ""
    allowed = {"engineering", "product", "hr", "security", "general"}
    return first if first in allowed else "general"


def _object_store_coordinates(
    *,
    source_type: str,
    base_uri: str,
    relative: Path,
) -> tuple[str | None, str | None]:
    if source_type != "s3":
        return None, None
    prefix = "s3://"
    if not base_uri.startswith(prefix):
        raise ValueError("S3 source base URI must start with s3://")
    location = base_uri[len(prefix) :].strip("/")
    bucket, separator, object_prefix = location.partition("/")
    if not bucket:
        raise ValueError("S3 source base URI must include a bucket")
    object_key = (
        f"{object_prefix}/{relative.as_posix()}"
        if separator
        else relative.as_posix()
    )
    return bucket, object_key


def _title_from_text(text: str, path: Path) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or path.stem
    return path.stem.replace("_", " ").title()


def _section_from_text(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or None
    return None


def _doc_type(path: Path) -> str:
    return "markdown" if path.suffix.lower() == ".md" else "text"
