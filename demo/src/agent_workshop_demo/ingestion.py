"""Offline local and mock-S3 ingestion for workshop fixtures."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from agent_workshop_demo.dedup import build_dedup_record, checksum
from agent_workshop_demo.embedding import (
    dense_vector,
    embedding_metadata,
    image_vector,
    sparse_vector,
)
from agent_workshop_demo.models import KBChunk

DEFAULT_NOW_MS = 1782604800000
SUPPORTED_TEXT_SUFFIXES = {".md", ".txt"}
SUPPORTED_PDF_SUFFIX = ".pdf"
ASSET_MANIFEST = "asset_manifest.json"
VERSION_MANIFEST = "document_versions.json"


@dataclass(frozen=True)
class IngestionResult:
    """Validated records produced by one offline ingestion run."""

    kb_chunks: list[KBChunk]
    dedup_signatures: list[dict[str, Any]]


@dataclass(frozen=True)
class DocumentVersion:
    """Validated version metadata attached to every source chunk."""

    doc_id: str
    doc_version: str
    is_current: bool


def ingest_demo_sources(
    local_dir: Path,
    mock_s3_dir: Path,
    *,
    version_manifest_path: Path | None = None,
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
    manifest_chunks = _ingest_asset_manifest(
        local_dir,
        source_type="local",
        base_uri="sample_data/local_docs",
        version_manifest=version_manifest,
    )
    handled_local_paths = {
        Path(str(item.metadata["asset_path"]))
        for item in manifest_chunks
        if item.metadata and "asset_path" in item.metadata
    }
    chunks = _ingest_tree(
        local_dir,
        source_type="local",
        base_uri="sample_data/local_docs",
        handled_paths=handled_local_paths,
        version_manifest=version_manifest,
    )
    chunks.extend(manifest_chunks)
    chunks.extend(
        _ingest_tree(
            mock_s3_dir,
            source_type="s3",
            base_uri="s3://internal-agent-chat-demo",
            handled_paths=set(),
            version_manifest=version_manifest,
        )
    )
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
    source_type: str,
    base_uri: str,
    version_manifest: dict[str, DocumentVersion],
) -> list[KBChunk]:
    manifest_path = root.parent / ASSET_MANIFEST
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
            asset_path = Path(record["asset_path"])
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
        full_path = root / asset_path
        if not full_path.exists():
            raise FileNotFoundError(
                f"Asset manifest record {index} references {full_path}"
            )
        source_uri = f"{base_uri}/{asset_path.as_posix()}"
        version = _version_for_source(source_uri, version_manifest)
        identity_prefix = _versioned_identity_prefix(version)
        has_image = doc_type == "image"
        output.append(
            KBChunk(
                doc_id=version.doc_id,
                chunk_id=f"{identity_prefix}_asset_{index:03d}",
                parent_id=None,
                record_type=record_type,
                source_type=source_type,
                source_uri=source_uri,
                bucket=None,
                object_key=None,
                doc_type=doc_type,
                title=title,
                section=record.get("section"),
                page_no=record.get("page_no"),
                chunk_index=index,
                text=text,
                text_summary=text[:180],
                language="mixed",
                department=record.get(
                    "department",
                    _department_from_path(asset_path),
                ),
                updated_at=DEFAULT_NOW_MS - index * 86_400_000,
                created_at=DEFAULT_NOW_MS - (index + 30) * 86_400_000,
                priority=record.get("priority", 5),
                doc_version=version.doc_version,
                is_current=version.is_current,
                checksum=checksum(text),
                metadata=embedding_metadata({
                    "parser": "asset_manifest",
                    "asset_path": asset_path.as_posix(),
                    "mime_type": (
                        "application/pdf"
                        if doc_type == "pdf"
                        else "image/png"
                    ),
                    "image_model": (
                        "deterministic-placeholder" if has_image else None
                    ),
                }),
                has_image_vector=has_image,
                text_vector=dense_vector(text),
                sparse_vector=sparse_vector(text),
                image_vector=image_vector(text) if has_image else None,
            )
        )
    return output


def _chunk_file(
    path: Path,
    *,
    root: Path,
    source_type: str,
    base_uri: str,
    version_manifest: dict[str, DocumentVersion],
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
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    department = _department_from_path(relative)
    doc_type = _doc_type(path)
    bucket = "internal-agent-chat-demo" if source_type == "s3" else None
    object_key = relative.as_posix() if source_type == "s3" else None
    output: list[KBChunk] = []
    for index, paragraph in enumerate(paragraphs, start=1):
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
                title=_title_from_text(text, path),
                section=_section_from_text(paragraph),
                page_no=None,
                chunk_index=index,
                text=paragraph,
                text_summary=paragraph[:180],
                language="mixed",
                department=department,
                updated_at=DEFAULT_NOW_MS - index * 86_400_000,
                created_at=DEFAULT_NOW_MS - (index + 30) * 86_400_000,
                priority=8 if department == "engineering" else 5,
                doc_version=version.doc_version,
                is_current=version.is_current,
                checksum=checksum(paragraph),
                metadata=embedding_metadata({
                    "parser": "local_markdown",
                    "relative_path": relative.as_posix(),
                }),
                has_image_vector=False,
                text_vector=dense_vector(paragraph),
                sparse_vector=sparse_vector(paragraph),
                image_vector=None,
            )
        )
    return output


def _chunk_pdf(
    path: Path,
    *,
    root: Path,
    source_type: str,
    base_uri: str,
    version_manifest: dict[str, DocumentVersion],
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
    bucket = "internal-agent-chat-demo" if source_type == "s3" else None
    object_key = relative.as_posix() if source_type == "s3" else None
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
        paragraphs = [
            part.strip()
            for part in page_text.split("\n\n")
            if part.strip()
        ]
        for paragraph_index, paragraph in enumerate(paragraphs, start=1):
            chunk_index = len(output) + 1
            output.append(
                KBChunk(
                    doc_id=version.doc_id,
                    chunk_id=(
                        f"{identity_prefix}_p{page_no:03d}"
                        f"_c{paragraph_index:03d}"
                    ),
                    parent_id=None,
                    record_type="pdf_page",
                    source_type=source_type,
                    source_uri=source_uri,
                    bucket=bucket,
                    object_key=object_key,
                    doc_type="pdf",
                    title=title,
                    section=_section_from_text(paragraph),
                    page_no=page_no,
                    chunk_index=chunk_index,
                    text=paragraph,
                    text_summary=paragraph[:180],
                    language="mixed",
                    department=department,
                    updated_at=DEFAULT_NOW_MS - chunk_index * 86_400_000,
                    created_at=(
                        DEFAULT_NOW_MS - (chunk_index + 30) * 86_400_000
                    ),
                    priority=8 if department == "engineering" else 5,
                    doc_version=version.doc_version,
                    is_current=version.is_current,
                    checksum=checksum(paragraph),
                    metadata=embedding_metadata(
                        {
                            "parser": "pypdf",
                            "relative_path": relative.as_posix(),
                            "page_count": len(reader.pages),
                        }
                    ),
                    has_image_vector=False,
                    text_vector=dense_vector(paragraph),
                    sparse_vector=sparse_vector(paragraph),
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
