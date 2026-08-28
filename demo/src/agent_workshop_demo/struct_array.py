"""Validated StructArray projection and retrieval profiles for Milvus 3.0."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from enum import Enum
from importlib import import_module
from pathlib import Path
from typing import Any, Final, Literal, Protocol

from agent_workshop_demo.config import DEMO_ROOT, VECTOR_DIMS
from agent_workshop_demo.embedding import (
    EMBEDDING_FINGERPRINT_KEY,
    cosine_similarity,
    dense_vector,
    text_embedding_fingerprint,
)
from agent_workshop_demo.models import KBChunk, SearchResult
from agent_workshop_demo.retrieval import HybridRetriever
from agent_workshop_demo.schema.collections import KB_DOCUMENTS_COLLECTION
from agent_workshop_demo.validation import normalize_filters, validate_question

PROJECTION_SCHEMA_VERSION: Final = "struct-array-projection-v1"
DEFAULT_PROJECTION_CONFIG: Final = DEMO_ROOT / "config" / "struct_array_projection.json"
MAX_PASSAGES_PER_DOCUMENT: Final = 1024
MAX_SELECTED_DOCUMENTS: Final = 512
STRUCT_FIELD: Final = "passages"
ELEMENT_VECTOR_FIELD: Final = "passages[element_vector]"
EMBEDDING_LIST_FIELD: Final = "passages[embedding_list_vector]"
ELEMENT_INDEX_NAME: Final = "passages_element_cosine_idx"
EMBEDDING_LIST_INDEX_NAME: Final = "passages_embedding_list_maxsim_idx"
FUSION_RECIPE: Final = "struct-rrf-v1"
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_ELEMENT_STRING_FIELDS: Final = frozenset({"section", "record_type", "language"})
_ELEMENT_INTEGER_FIELDS: Final = frozenset({"page_no", "chunk_index"})


class StructArrayProfile(str, Enum):
    """Closed retrieval profile set exposed to workflow configuration."""

    DISABLED = "disabled"
    ELEMENT = "struct_element"
    TWO_STAGE = "struct_two_stage"
    FUSED = "struct_fused"


@dataclass(frozen=True)
class StructArrayRuntimeConfig:
    """Validated environment-derived activation contract."""

    profile: StructArrayProfile
    collection_name: str
    projection_fingerprint: str | None
    parent_top_k: int


@dataclass(frozen=True)
class ProjectionManifest:
    """Reviewed selection of document families for the derived projection."""

    schema_version: str
    selected_doc_ids: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class PassageProjection:
    """One aligned StructArray element without authoritative passage text."""

    chunk_id: str
    checksum: str
    chunk_index: int
    page_no: int
    section: str
    record_type: str
    language: str
    embedding_list_vector: tuple[float, ...]
    element_vector: tuple[float, ...]

    def to_record(self) -> dict[str, object]:
        """Serialize the natural nested insert shape accepted by PyMilvus."""

        data = asdict(self)
        data["embedding_list_vector"] = list(self.embedding_list_vector)
        data["element_vector"] = list(self.element_vector)
        return data


@dataclass(frozen=True)
class DocumentProjection:
    """One immutable parent row for ``kb_documents``."""

    document_key: str
    doc_id: str
    doc_version: str
    source_type: str
    source_uri: str
    doc_type: str
    title: str
    department: str
    is_current: bool
    updated_at: int
    priority: int
    text_embedding_fingerprint: str
    projection_fingerprint: str
    projection_parent_count: int
    projection_passage_count: int
    passage_count: int
    passages: tuple[PassageProjection, ...]

    def to_record(self) -> dict[str, object]:
        """Serialize a complete parent row for natural nested insertion."""

        data = asdict(self)
        data["passages"] = [item.to_record() for item in self.passages]
        return data


@dataclass(frozen=True)
class ProjectionBuild:
    """Complete, fingerprinted projection result returned by the pure builder."""

    parents: tuple[DocumentProjection, ...]
    projection_fingerprint: str
    text_embedding_fingerprint: str
    parent_count: int
    passage_count: int


ElementOperator = Literal["eq", "ne", "gt", "gte", "lt", "lte"]


@dataclass(frozen=True)
class ElementPredicate:
    """One allow-listed same-element scalar predicate."""

    field: str
    operator: ElementOperator
    value: str | int

    def __post_init__(self) -> None:
        allowed = _ELEMENT_STRING_FIELDS | _ELEMENT_INTEGER_FIELDS
        if self.field not in allowed:
            raise ValueError(f"Unsupported StructArray element field: {self.field!r}")
        if self.operator not in {"eq", "ne", "gt", "gte", "lt", "lte"}:
            raise ValueError(f"Unsupported StructArray operator: {self.operator!r}")
        if self.field in _ELEMENT_STRING_FIELDS:
            if self.operator not in {"eq", "ne"}:
                raise ValueError("String element fields accept only eq/ne")
            if not isinstance(self.value, str) or not self.value.strip():
                raise ValueError("String element predicate values must be non-empty")
            if len(self.value) > 2048:
                raise ValueError("Element predicate string exceeds 2048 characters")
        elif isinstance(self.value, bool) or not isinstance(self.value, int):
            raise ValueError("Integer element fields require an integer value")


@dataclass(frozen=True)
class DocumentCandidate:
    """Parent-only routing hint that is never citeable evidence."""

    document_key: str
    doc_id: str
    doc_version: str
    title: str
    rank: int
    score: float

    def to_dict(self) -> dict[str, object]:
        """Serialize bounded document-shortlist provenance."""

        return asdict(self)


@dataclass(frozen=True)
class ProfileSearchRun:
    """Immutable result of one profile search without mutable side channels."""

    configured_profile: StructArrayProfile
    effective_profile: str
    capability_status: str
    results_by_query: tuple[tuple[SearchResult, ...], ...]
    document_candidates: tuple[DocumentCandidate, ...] = ()


class StructArrayFlatRetriever(HybridRetriever, Protocol):
    """Authoritative flat adapter plus the isolated BM25 lane used by fusion."""

    def search_sparse(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
        order_by: list[str] | None = None,
    ) -> list[SearchResult]: ...


def load_projection_manifest(
    path: Path = DEFAULT_PROJECTION_CONFIG,
) -> ProjectionManifest:
    """Load and strictly validate the checked-in projection selection."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(
            f"Unable to read StructArray projection config: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid StructArray projection JSON at line {exc.lineno}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("StructArray projection config must be an object")
    expected = {"schema_version", "selected_doc_ids", "rationale"}
    if set(payload) != expected:
        raise ValueError("StructArray projection config fields do not match the schema")
    schema_version = payload.get("schema_version")
    raw_ids = payload.get("selected_doc_ids")
    rationale = payload.get("rationale")
    if schema_version != PROJECTION_SCHEMA_VERSION:
        raise ValueError("Unsupported StructArray projection schema_version")
    if not isinstance(raw_ids, list) or not 1 <= len(raw_ids) <= MAX_SELECTED_DOCUMENTS:
        raise ValueError("selected_doc_ids must contain 1..512 ids")
    if any(
        not isinstance(item, str) or not item.strip() or len(item) > 128
        for item in raw_ids
    ):
        raise ValueError("selected_doc_ids must contain bounded non-empty strings")
    normalized_ids = tuple(item.strip() for item in raw_ids)
    if len(normalized_ids) != len(set(normalized_ids)):
        raise ValueError("selected_doc_ids must be unique")
    if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 1024:
        raise ValueError("rationale must contain 1..1024 characters")
    return ProjectionManifest(schema_version, normalized_ids, rationale.strip())


def document_key(doc_id: str, doc_version: str) -> str:
    """Return a collision-resistant stable key for one document edition."""

    if not doc_id.strip() or not doc_version.strip():
        raise ValueError("doc_id and doc_version must be non-empty")
    canonical = json.dumps(
        [doc_id, doc_version], ensure_ascii=False, separators=(",", ":")
    )
    return "docv_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_struct_array_projection(
    chunks: Sequence[KBChunk],
    manifest: ProjectionManifest,
) -> ProjectionBuild:
    """Build an all-or-nothing derived projection from authoritative chunks."""

    selected = set(manifest.selected_doc_ids)
    grouped: dict[tuple[str, str], list[KBChunk]] = {}
    seen_chunk_ids: set[str] = set()
    for chunk in chunks:
        if chunk.doc_id not in selected:
            continue
        if chunk.chunk_id in seen_chunk_ids:
            raise ValueError(f"Duplicate projected chunk_id: {chunk.chunk_id}")
        seen_chunk_ids.add(chunk.chunk_id)
        grouped.setdefault((chunk.doc_id, chunk.doc_version), []).append(chunk)
    missing = sorted(selected - {doc_id for doc_id, _ in grouped})
    if missing:
        raise ValueError(
            "Projection selection is missing document ids: " + ", ".join(missing)
        )
    if not grouped:
        raise ValueError("Projection selection produced no document editions")
    if len(grouped) > MAX_SELECTED_DOCUMENTS:
        raise ValueError("Projection contains more than 512 document editions")

    parents_without_identity: list[DocumentProjection] = []
    fingerprints: set[str] = set()
    for (doc_id, version), family in sorted(grouped.items()):
        ordered = sorted(family, key=lambda item: (item.chunk_index, item.chunk_id))
        if len(ordered) > MAX_PASSAGES_PER_DOCUMENT:
            raise ValueError(
                f"Document {doc_id}@{version} exceeds StructArray max_capacity"
            )
        _validate_parent_invariance(ordered)
        passages: list[PassageProjection] = []
        for chunk in ordered:
            checksum = chunk.checksum
            if not isinstance(checksum, str) or not checksum.strip():
                raise ValueError(
                    f"Projected chunk {chunk.chunk_id} requires a checksum"
                )
            fingerprint = _chunk_embedding_fingerprint(chunk)
            fingerprints.add(fingerprint)
            vector = _validated_vector(chunk.text_vector, chunk_id=chunk.chunk_id)
            section = chunk.section or ""
            passages.append(
                PassageProjection(
                    chunk_id=chunk.chunk_id,
                    checksum=checksum,
                    chunk_index=chunk.chunk_index,
                    page_no=chunk.page_no if chunk.page_no is not None else -1,
                    section=section,
                    record_type=chunk.record_type,
                    language=chunk.language,
                    embedding_list_vector=vector,
                    element_vector=vector,
                )
            )
        first = ordered[0]
        parents_without_identity.append(
            DocumentProjection(
                document_key=document_key(doc_id, version),
                doc_id=doc_id,
                doc_version=version,
                source_type=first.source_type,
                source_uri=first.source_uri,
                doc_type=first.doc_type,
                title=first.title,
                department=first.department,
                is_current=first.is_current,
                updated_at=max(item.updated_at for item in ordered),
                priority=max(item.priority for item in ordered),
                text_embedding_fingerprint=_chunk_embedding_fingerprint(first),
                projection_fingerprint="",
                projection_parent_count=0,
                projection_passage_count=0,
                passage_count=len(passages),
                passages=tuple(passages),
            )
        )
    if len(fingerprints) != 1:
        raise ValueError("Projected chunks must share one text embedding fingerprint")
    parent_count = len(parents_without_identity)
    passage_count = sum(parent.passage_count for parent in parents_without_identity)
    projection_fingerprint = _projection_fingerprint(manifest, parents_without_identity)
    parents = tuple(
        replace(
            parent,
            projection_fingerprint=projection_fingerprint,
            projection_parent_count=parent_count,
            projection_passage_count=passage_count,
        )
        for parent in parents_without_identity
    )
    build = ProjectionBuild(
        parents=parents,
        projection_fingerprint=projection_fingerprint,
        text_embedding_fingerprint=next(iter(fingerprints)),
        parent_count=parent_count,
        passage_count=passage_count,
    )
    _validate_projection_build(build)
    return build


def runtime_config_from_mapping(values: Mapping[str, str]) -> StructArrayRuntimeConfig:
    """Validate StructArray runtime environment without performing I/O."""

    raw_profile = values.get("STRUCT_ARRAY_RETRIEVAL", "disabled").strip()
    try:
        profile = StructArrayProfile(raw_profile)
    except ValueError as exc:
        raise ValueError("STRUCT_ARRAY_RETRIEVAL has an unsupported value") from exc
    collection_name = values.get(
        "MILVUS_STRUCT_ARRAY_COLLECTION_NAME", "kb_documents"
    ).strip()
    if not collection_name:
        raise ValueError("MILVUS_STRUCT_ARRAY_COLLECTION_NAME must be non-empty")
    raw_top_k = values.get("STRUCT_ARRAY_PARENT_TOP_K", "8")
    try:
        parent_top_k = int(raw_top_k)
    except ValueError as exc:
        raise ValueError("STRUCT_ARRAY_PARENT_TOP_K must be an integer") from exc
    if not 1 <= parent_top_k <= 64:
        raise ValueError("STRUCT_ARRAY_PARENT_TOP_K must be between 1 and 64")
    fingerprint = values.get("STRUCT_ARRAY_PROJECTION_FINGERPRINT")
    if profile is not StructArrayProfile.DISABLED:
        if fingerprint is None or not _HEX_64.fullmatch(fingerprint):
            raise ValueError(
                "STRUCT_ARRAY_PROJECTION_FINGERPRINT must be 64 lowercase hex characters"
            )
    elif fingerprint is not None and not _HEX_64.fullmatch(fingerprint):
        raise ValueError("STRUCT_ARRAY_PROJECTION_FINGERPRINT is invalid")
    return StructArrayRuntimeConfig(profile, collection_name, fingerprint, parent_top_k)


def element_filter_expression(predicates: Sequence[ElementPredicate]) -> str:
    """Compile bounded AND predicates into Milvus same-element syntax."""

    if len(predicates) > 4:
        raise ValueError("At most four element predicates are allowed")
    operators = {"eq": "==", "ne": "!=", "gt": ">", "gte": ">=", "lt": "<", "lte": "<="}
    clauses = [
        f"$[{item.field}] {operators[item.operator]} "
        + (
            json.dumps(item.value, ensure_ascii=False)
            if isinstance(item.value, str)
            else str(item.value)
        )
        for item in predicates
    ]
    return f"element_filter(passages, {' && '.join(clauses)})" if clauses else ""


def match_any_expression(predicates: Sequence[ElementPredicate]) -> str:
    """Compile a parent-admission MATCH_ANY expression for teaching/tests."""

    element = element_filter_expression(predicates)
    if not element:
        raise ValueError("MATCH_ANY requires at least one element predicate")
    inner = element.removeprefix("element_filter(passages, ").removesuffix(")")
    return f"MATCH_ANY(passages, {inner})"


class MilvusStructArrayStore:
    """Write and activation-check the all-or-nothing derived projection."""

    def __init__(
        self,
        client: Any,
        *,
        collection_name: str = "kb_documents",
        batch_size: int = 32,
    ) -> None:
        if not 1 <= batch_size <= 128:
            raise ValueError("StructArray batch_size must be between 1 and 128")
        self.client = client
        self.collection_name = collection_name
        self.batch_size = batch_size

    def replace_projection(
        self, build: ProjectionBuild, *, retrieval_mode: str
    ) -> dict[str, int | str]:
        """Replace the derived rows only while StructArray retrieval is disabled."""

        if retrieval_mode != StructArrayProfile.DISABLED.value:
            raise RuntimeError(
                "Projection replacement requires STRUCT_ARRAY_RETRIEVAL=disabled"
            )
        if not self.client.has_collection(collection_name=self.collection_name):
            raise RuntimeError("StructArray collection does not exist")
        _validate_projection_build(build)
        _verify_struct_array_schema(
            self.client.describe_collection(collection_name=self.collection_name)
        )
        indexes = set(self.client.list_indexes(collection_name=self.collection_name))
        if not {ELEMENT_INDEX_NAME, EMBEDDING_LIST_INDEX_NAME}.issubset(indexes):
            raise RuntimeError("StructArray vector indexes are incomplete")
        _verify_struct_array_indexes(self.client, self.collection_name)
        self.client.load_collection(collection_name=self.collection_name)
        self.client.delete(
            collection_name=self.collection_name, filter="projection_parent_count >= 0"
        )
        for start in range(0, build.parent_count, self.batch_size):
            batch = [
                parent.to_record()
                for parent in build.parents[start : start + self.batch_size]
            ]
            result = self.client.insert(
                collection_name=self.collection_name, data=batch
            )
            if isinstance(result, Mapping) and int(
                result.get("insert_count", len(batch))
            ) != len(batch):
                raise RuntimeError("Milvus reported an incomplete StructArray insert")
        self.client.flush(collection_name=self.collection_name)
        self.verify_projection(build)
        return {
            "projection_fingerprint": build.projection_fingerprint,
            "parent_count": build.parent_count,
            "passage_count": build.passage_count,
        }

    def verify_projection(self, build: ProjectionBuild) -> None:
        """Verify activation metadata and every projected passage identity."""

        self.ensure_ready(
            expected_projection_fingerprint=build.projection_fingerprint,
            expected_embedding_fingerprint=build.text_embedding_fingerprint,
        )
        self._verify_round_trip(build)

    def ensure_ready(
        self,
        *,
        expected_projection_fingerprint: str,
        expected_embedding_fingerprint: str | None = None,
    ) -> None:
        """Fail closed unless schema indexes and complete-build counts agree."""

        if not _HEX_64.fullmatch(expected_projection_fingerprint):
            raise ValueError("Expected projection fingerprint is invalid")
        if not self.client.has_collection(collection_name=self.collection_name):
            raise RuntimeError("StructArray collection does not exist")
        _verify_struct_array_schema(
            self.client.describe_collection(collection_name=self.collection_name)
        )
        indexes = set(self.client.list_indexes(collection_name=self.collection_name))
        required = {ELEMENT_INDEX_NAME, EMBEDDING_LIST_INDEX_NAME}
        if not required.issubset(indexes):
            raise RuntimeError("StructArray vector indexes are incomplete")
        _verify_struct_array_indexes(self.client, self.collection_name)
        self.client.load_collection(collection_name=self.collection_name)
        rows = self.client.query(
            collection_name=self.collection_name,
            filter=f"projection_fingerprint == {json.dumps(expected_projection_fingerprint)}",
            output_fields=[
                "projection_fingerprint",
                "projection_parent_count",
                "projection_passage_count",
                "passage_count",
                "text_embedding_fingerprint",
            ],
            limit=MAX_SELECTED_DOCUMENTS,
        )
        if not rows:
            raise RuntimeError("StructArray projection fingerprint is not present")
        declared_parents = {int(row["projection_parent_count"]) for row in rows}
        declared_passages = {int(row["projection_passage_count"]) for row in rows}
        fingerprints = {str(row["projection_fingerprint"]) for row in rows}
        embedding_fingerprints = {
            str(row["text_embedding_fingerprint"]) for row in rows
        }
        if (
            fingerprints != {expected_projection_fingerprint}
            or len(declared_parents) != 1
            or len(declared_passages) != 1
            or declared_parents != {len(rows)}
            or declared_passages != {sum(int(row["passage_count"]) for row in rows)}
        ):
            raise RuntimeError("StructArray projection activation counts do not agree")
        if len(embedding_fingerprints) != 1:
            raise RuntimeError("StructArray projection mixes embedding fingerprints")
        expected_embedding = (
            expected_embedding_fingerprint or text_embedding_fingerprint()
        )
        if embedding_fingerprints != {expected_embedding}:
            raise RuntimeError(
                "StructArray embedding fingerprint does not match runtime"
            )
        foreign = self.client.query(
            collection_name=self.collection_name,
            filter=f"projection_fingerprint != {json.dumps(expected_projection_fingerprint)}",
            output_fields=["document_key"],
            limit=1,
        )
        if foreign:
            raise RuntimeError("StructArray collection contains a foreign projection")

    def _verify_round_trip(self, build: ProjectionBuild) -> None:
        rows = self.client.query(
            collection_name=self.collection_name,
            filter=(
                "projection_fingerprint == " + json.dumps(build.projection_fingerprint)
            ),
            output_fields=["document_key", "passages"],
            limit=build.parent_count,
        )
        expected = {
            parent.document_key: [
                (
                    passage.chunk_id,
                    passage.checksum,
                    passage.chunk_index,
                    passage.page_no,
                    passage.section,
                    passage.record_type,
                    passage.language,
                )
                for passage in parent.passages
            ]
            for parent in build.parents
        }
        actual: dict[str, list[tuple[object, ...]]] = {}
        for row in rows:
            key = str(row.get("document_key", ""))
            passages = row.get("passages")
            if not key or not isinstance(passages, list):
                raise RuntimeError("StructArray projection round-trip shape is invalid")
            actual[key] = [
                (
                    passage.get("chunk_id"),
                    passage.get("checksum"),
                    passage.get("chunk_index"),
                    passage.get("page_no"),
                    passage.get("section"),
                    passage.get("record_type"),
                    passage.get("language"),
                )
                for passage in passages
                if isinstance(passage, Mapping)
            ]
        if actual != expected:
            raise RuntimeError("StructArray projection round-trip identities differ")


class InMemoryStructArrayRetriever:
    """Deterministic profile adapter used for offline parity and evaluation."""

    supports_parallel_search = True

    def __init__(
        self,
        flat_retriever: StructArrayFlatRetriever,
        build: ProjectionBuild,
        *,
        profile: StructArrayProfile,
        parent_top_k: int = 8,
    ) -> None:
        if profile is StructArrayProfile.DISABLED:
            raise ValueError("StructArray adapter requires a non-disabled profile")
        self.flat_retriever = flat_retriever
        self.build = build
        self.profile = profile
        self.parent_top_k = parent_top_k
        self._chunk_by_id = {
            chunk.chunk_id: chunk
            for parent in build.parents
            for chunk in _rehydrate_parent(flat_retriever, parent)
        }

    def search_profile(
        self,
        queries: Sequence[str],
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
        order_by: list[str] | None = None,
        element_predicates: Sequence[ElementPredicate] = (),
    ) -> ProfileSearchRun:
        """Execute an isolated local profile with native-equivalent identities."""

        normalized = _validate_profile_request(
            queries, top_k, filters, element_predicates
        )
        if self.profile is StructArrayProfile.TWO_STAGE and len(normalized) not in {
            2,
            3,
        }:
            effective = StructArrayProfile.ELEMENT.value
            results = tuple(
                tuple(
                    replace(item, retrieval_profile=effective)
                    for item in self._element_search(
                        query, top_k, filters, element_predicates
                    )
                )
                for query in normalized
            )
            return ProfileSearchRun(
                self.profile, effective, "fallback_ineligible_group", results
            )
        if self.profile is StructArrayProfile.TWO_STAGE:
            candidates = self._parent_shortlist(normalized, filters)
            allowed = {item.document_key for item in candidates}
            results = tuple(
                tuple(
                    self._element_search(
                        query, top_k, filters, element_predicates, allowed
                    )
                )
                for query in normalized
            )
            return ProfileSearchRun(
                self.profile, self.profile.value, "ready", results, candidates
            )
        results = tuple(
            tuple(self._element_search(query, top_k, filters, element_predicates))
            for query in normalized
        )
        if self.profile is StructArrayProfile.FUSED:
            fused = tuple(
                tuple(
                    fuse_struct_and_bm25(
                        list(items),
                        self.flat_retriever.search_sparse(
                            query, top_k=top_k, filters=filters, order_by=order_by
                        ),
                        top_k=top_k,
                    )
                )
                for query, items in zip(normalized, results, strict=True)
            )
            return ProfileSearchRun(self.profile, self.profile.value, "ready", fused)
        return ProfileSearchRun(self.profile, self.profile.value, "ready", results)

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
        order_by: list[str] | None = None,
    ) -> list[SearchResult]:
        """Provide the existing workflow search seam for one query."""

        return list(
            self.search_profile(
                [query], top_k=top_k, filters=filters, order_by=order_by
            ).results_by_query[0]
        )

    def match_any_parents(
        self,
        predicates: Sequence[ElementPredicate],
        *,
        filters: dict[str, Any] | None = None,
        limit: int = 8,
    ) -> tuple[DocumentCandidate, ...]:
        """Return parent-only MATCH_ANY teaching results with local parity."""

        match_any_expression(predicates)
        if not 1 <= limit <= 64:
            raise ValueError("MATCH_ANY limit must be between 1 and 64")
        normalized_filters = normalize_filters(filters)
        parents = [
            parent
            for parent in self.build.parents
            if _parent_matches(parent, normalized_filters)
            and any(
                _passage_matches(passage, predicates) for passage in parent.passages
            )
        ]
        return tuple(
            DocumentCandidate(
                parent.document_key,
                parent.doc_id,
                parent.doc_version,
                parent.title,
                rank,
                0.0,
            )
            for rank, parent in enumerate(parents[:limit], start=1)
        )

    def aggregations(
        self, results: list[SearchResult], fields: list[str]
    ) -> dict[str, dict[str, int]]:
        """Delegate facets over normalized authoritative chunks."""

        return self.flat_retriever.aggregations(results, fields)

    def fetch_document_chunks(
        self,
        *,
        doc_id: str,
        doc_version: str,
        filters: dict[str, Any] | None = None,
        limit: int,
    ) -> list[KBChunk]:
        """Delegate authoritative sibling expansion."""

        return self.flat_retriever.fetch_document_chunks(
            doc_id=doc_id, doc_version=doc_version, filters=filters, limit=limit
        )

    def fetch_chunks_by_ids(
        self, *, chunk_ids: list[str], filters: dict[str, Any] | None = None
    ) -> list[KBChunk]:
        """Delegate authoritative identity revalidation."""

        return self.flat_retriever.fetch_chunks_by_ids(
            chunk_ids=chunk_ids, filters=filters
        )

    def search_sparse(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
        order_by: list[str] | None = None,
    ) -> list[SearchResult]:
        """Delegate the flat lexical lane."""

        return self.flat_retriever.search_sparse(
            query, top_k=top_k, filters=filters, order_by=order_by
        )

    def _element_search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
        predicates: Sequence[ElementPredicate],
        allowed_keys: set[str] | None = None,
    ) -> list[SearchResult]:
        query_vector = dense_vector(query)
        validated_filters = normalize_filters(filters)
        scored: list[tuple[float, DocumentProjection, int, PassageProjection]] = []
        for parent in self.build.parents:
            if allowed_keys is not None and parent.document_key not in allowed_keys:
                continue
            if not _parent_matches(parent, validated_filters):
                continue
            for offset, passage in enumerate(parent.passages):
                if _passage_matches(passage, predicates):
                    scored.append(
                        (
                            cosine_similarity(
                                query_vector, list(passage.element_vector)
                            ),
                            parent,
                            offset,
                            passage,
                        )
                    )
        ordered = sorted(scored, key=lambda item: (-item[0], item[3].chunk_id))[:top_k]
        output: list[SearchResult] = []
        for rank, (score, parent, offset, passage) in enumerate(ordered, start=1):
            chunk = self._chunk_by_id.get(passage.chunk_id)
            if chunk is None or chunk.checksum != passage.checksum:
                raise RuntimeError(
                    "StructArray projection failed authoritative chunk revalidation"
                )
            output.append(
                _element_result(
                    chunk, rank, score, self.profile.value, parent.document_key, offset
                )
            )
        return output

    def _parent_shortlist(
        self, queries: Sequence[str], filters: dict[str, Any] | None
    ) -> tuple[DocumentCandidate, ...]:
        vectors = [dense_vector(query) for query in queries]
        validated_filters = normalize_filters(filters)
        scored = [
            (
                sum(
                    max(
                        cosine_similarity(vector, list(p.element_vector))
                        for p in parent.passages
                    )
                    for vector in vectors
                ),
                parent,
            )
            for parent in self.build.parents
            if _parent_matches(parent, validated_filters)
        ]
        return tuple(
            DocumentCandidate(
                parent.document_key,
                parent.doc_id,
                parent.doc_version,
                parent.title,
                rank,
                score,
            )
            for rank, (score, parent) in enumerate(
                sorted(scored, key=lambda item: (-item[0], item[1].document_key))[
                    : self.parent_top_k
                ],
                start=1,
            )
        )


class MilvusStructArrayRetriever:
    """Native Milvus element/EmbeddingList adapter with flat evidence revalidation."""

    supports_parallel_search = False

    def __init__(
        self,
        client: Any,
        flat_retriever: StructArrayFlatRetriever,
        config: StructArrayRuntimeConfig,
    ) -> None:
        if (
            config.profile is StructArrayProfile.DISABLED
            or config.projection_fingerprint is None
        ):
            raise ValueError(
                "Native StructArray retriever requires an activated profile"
            )
        self.client = client
        self.flat_retriever = flat_retriever
        self.config = config

    def ensure_ready(self) -> None:
        """Verify the complete projection before admitting any request."""

        MilvusStructArrayStore(
            self.client, collection_name=self.config.collection_name
        ).ensure_ready(
            expected_projection_fingerprint=self.config.projection_fingerprint or ""
        )

    def search_profile(
        self,
        queries: Sequence[str],
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
        order_by: list[str] | None = None,
        element_predicates: Sequence[ElementPredicate] = (),
    ) -> ProfileSearchRun:
        """Execute the configured native profile and return immutable provenance."""

        normalized = _validate_profile_request(
            queries, top_k, filters, element_predicates
        )
        if self.config.profile is StructArrayProfile.TWO_STAGE and len(normalized) in {
            2,
            3,
        }:
            candidates = self._parent_shortlist(normalized, filters)
            keys = [item.document_key for item in candidates]
            if not keys:
                return ProfileSearchRun(
                    self.config.profile,
                    self.config.profile.value,
                    "ready",
                    tuple(() for _ in normalized),
                    candidates,
                )
            results = tuple(
                tuple(
                    self._element_search(
                        query, top_k, filters, element_predicates, keys
                    )
                )
                for query in normalized
            )
            return ProfileSearchRun(
                self.config.profile,
                self.config.profile.value,
                "ready",
                results,
                candidates,
            )
        effective = (
            StructArrayProfile.ELEMENT.value
            if self.config.profile is StructArrayProfile.TWO_STAGE
            else self.config.profile.value
        )
        results = tuple(
            tuple(self._element_search(query, top_k, filters, element_predicates))
            for query in normalized
        )
        if effective != self.config.profile.value:
            results = tuple(
                tuple(replace(item, retrieval_profile=effective) for item in group)
                for group in results
            )
        if self.config.profile is StructArrayProfile.FUSED:
            results = tuple(
                tuple(
                    fuse_struct_and_bm25(
                        list(items),
                        self.flat_retriever.search_sparse(
                            query, top_k=top_k, filters=filters, order_by=order_by
                        ),
                        top_k=top_k,
                    )
                )
                for query, items in zip(normalized, results, strict=True)
            )
        status = (
            "fallback_ineligible_group"
            if effective != self.config.profile.value
            else "ready"
        )
        return ProfileSearchRun(self.config.profile, effective, status, results)

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
        order_by: list[str] | None = None,
    ) -> list[SearchResult]:
        """Provide the existing one-query workflow seam."""

        return list(
            self.search_profile(
                [query], top_k=top_k, filters=filters, order_by=order_by
            ).results_by_query[0]
        )

    def match_any_parents(
        self,
        predicates: Sequence[ElementPredicate],
        *,
        filters: dict[str, Any] | None = None,
        limit: int = 8,
    ) -> tuple[DocumentCandidate, ...]:
        """Execute same-element MATCH_ANY and return non-citeable parents."""

        if not 1 <= limit <= 64:
            raise ValueError("MATCH_ANY limit must be between 1 and 64")
        expression = _parent_expression(
            filters, self.config.projection_fingerprint or ""
        )
        expression += " and " + match_any_expression(predicates)
        rows = self.client.query(
            collection_name=self.config.collection_name,
            filter=expression,
            output_fields=["document_key", "doc_id", "doc_version", "title"],
            limit=limit,
        )
        candidates: list[DocumentCandidate] = []
        for rank, row in enumerate(rows, start=1):
            if not isinstance(row, Mapping):
                raise RuntimeError("MATCH_ANY returned an invalid parent row")
            candidates.append(
                DocumentCandidate(
                    str(row["document_key"]),
                    str(row["doc_id"]),
                    str(row["doc_version"]),
                    str(row["title"]),
                    rank,
                    0.0,
                )
            )
        return tuple(candidates)

    def aggregations(
        self, results: list[SearchResult], fields: list[str]
    ) -> dict[str, dict[str, int]]:
        """Delegate facets to the flat authoritative adapter."""

        return self.flat_retriever.aggregations(results, fields)

    def fetch_document_chunks(
        self,
        *,
        doc_id: str,
        doc_version: str,
        filters: dict[str, Any] | None = None,
        limit: int,
    ) -> list[KBChunk]:
        """Delegate authoritative sibling expansion."""

        return self.flat_retriever.fetch_document_chunks(
            doc_id=doc_id, doc_version=doc_version, filters=filters, limit=limit
        )

    def fetch_chunks_by_ids(
        self, *, chunk_ids: list[str], filters: dict[str, Any] | None = None
    ) -> list[KBChunk]:
        """Delegate authoritative identity revalidation."""

        return self.flat_retriever.fetch_chunks_by_ids(
            chunk_ids=chunk_ids, filters=filters
        )

    def search_sparse(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
        order_by: list[str] | None = None,
    ) -> list[SearchResult]:
        """Delegate the flat lexical lane."""

        return self.flat_retriever.search_sparse(
            query, top_k=top_k, filters=filters, order_by=order_by
        )

    def _element_search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
        predicates: Sequence[ElementPredicate],
        document_keys: Sequence[str] | None = None,
    ) -> list[SearchResult]:
        expression = _parent_expression(
            filters, self.config.projection_fingerprint or "", document_keys
        )
        element_expression = element_filter_expression(predicates)
        if element_expression:
            expression = f"{expression} and {element_expression}"
        raw = self.client.search(
            collection_name=self.config.collection_name,
            data=[dense_vector(query)],
            anns_field=ELEMENT_VECTOR_FIELD,
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            filter=expression,
            limit=top_k,
            output_fields=[
                "document_key",
                "doc_id",
                "doc_version",
                "department",
                "is_current",
                "text_embedding_fingerprint",
                "projection_fingerprint",
                "passages",
            ],
        )
        hits = _first_hits(raw)[:top_k]
        identities: list[tuple[dict[str, Any], int, PassageProjection, float]] = []
        ids: list[str] = []
        for hit in hits:
            entity = _hit_entity(hit)
            offset = hit.get("offset")
            passages = entity.get("passages")
            if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
                raise RuntimeError(
                    "StructArray element hit is missing a citeable offset"
                )
            if not isinstance(passages, list) or offset >= len(passages):
                raise RuntimeError("StructArray element offset is out of range")
            passage = _passage_from_mapping(passages[offset])
            if passage.chunk_id in ids:
                raise RuntimeError(
                    "StructArray element search returned a duplicate chunk_id"
                )
            ids.append(passage.chunk_id)
            identities.append((entity, offset, passage, _finite_hit_score(hit)))
        authoritative = _fetch_authoritative(self.flat_retriever, ids, filters)
        by_id = {item.chunk_id: item for item in authoritative}
        output: list[SearchResult] = []
        for rank, (entity, offset, passage, score) in enumerate(identities, start=1):
            chunk = by_id.get(passage.chunk_id)
            entity_key = str(entity.get("document_key", ""))
            if (
                chunk is None
                or chunk.checksum != passage.checksum
                or chunk.doc_id != str(entity.get("doc_id"))
                or chunk.doc_version != str(entity.get("doc_version"))
                or entity_key != document_key(chunk.doc_id, chunk.doc_version)
                or entity.get("department") != chunk.department
                or entity.get("is_current") is not chunk.is_current
                or entity.get("projection_fingerprint")
                != self.config.projection_fingerprint
                or entity.get("text_embedding_fingerprint")
                != text_embedding_fingerprint()
            ):
                raise RuntimeError(
                    "StructArray element identity failed authoritative revalidation"
                )
            output.append(
                _element_result(
                    chunk,
                    rank,
                    score,
                    self.config.profile.value,
                    entity_key,
                    offset,
                )
            )
        return output

    def _parent_shortlist(
        self, queries: Sequence[str], filters: dict[str, Any] | None
    ) -> tuple[DocumentCandidate, ...]:
        try:
            embedding_module = import_module("pymilvus.client.embedding_list")
        except ImportError as exc:
            raise RuntimeError("PyMilvus EmbeddingList support is unavailable") from exc
        query_list = embedding_module.EmbeddingList()
        for query in queries:
            query_list.add(dense_vector(query))
        expression = _parent_expression(
            filters, self.config.projection_fingerprint or ""
        )
        raw = self.client.search(
            collection_name=self.config.collection_name,
            data=[query_list],
            anns_field=EMBEDDING_LIST_FIELD,
            search_params={
                "metric_type": "MAX_SIM_COSINE",
                "params": {"retrieval_ann_ratio": 3.0, "emb_list_rerank": True},
            },
            filter=expression,
            limit=self.config.parent_top_k,
            output_fields=["document_key", "doc_id", "doc_version", "title"],
        )
        candidates: list[DocumentCandidate] = []
        for rank, hit in enumerate(
            _first_hits(raw)[: self.config.parent_top_k], start=1
        ):
            if "offset" in hit:
                raise RuntimeError(
                    "EmbeddingList unexpectedly returned element identity"
                )
            entity = _hit_entity(hit)
            candidates.append(
                DocumentCandidate(
                    str(entity["document_key"]),
                    str(entity["doc_id"]),
                    str(entity["doc_version"]),
                    str(entity["title"]),
                    rank,
                    _finite_hit_score(hit),
                )
            )
        return tuple(candidates)


def fuse_struct_and_bm25(
    struct_results: list[SearchResult], bm25_results: list[SearchResult], *, top_k: int
) -> list[SearchResult]:
    """Fuse separate element-dense and flat lexical lanes by stable chunk id."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    struct_by_id = {item.chunk.chunk_id: item for item in struct_results}
    bm25_by_id = {item.chunk.chunk_id: item for item in bm25_results}
    scored: list[tuple[float, SearchResult, tuple[str, ...]]] = []
    for chunk_id in sorted(set(struct_by_id) | set(bm25_by_id)):
        struct_item = struct_by_id.get(chunk_id)
        bm25_item = bm25_by_id.get(chunk_id)
        score = (0.65 / (60 + struct_item.rank) if struct_item else 0.0) + (
            0.35 / (60 + bm25_item.rank) if bm25_item else 0.0
        )
        source = struct_item or bm25_item
        if source is None:
            continue
        if struct_item is not None and bm25_item is not None:
            _reject_lane_disagreement(struct_item.chunk, bm25_item.chunk)
        paths = tuple(
            path
            for path, present in (
                ("struct_element", struct_item),
                ("flat_bm25", bm25_item),
            )
            if present is not None
        )
        scored.append((score, source, paths))
    ordered = sorted(scored, key=lambda item: (-item[0], item[1].chunk.chunk_id))[
        :top_k
    ]
    return [
        replace(
            item,
            rank=rank,
            hybrid_score=score,
            retrieval_profile=StructArrayProfile.FUSED.value,
            retrieval_paths=paths,
            fusion_recipe=FUSION_RECIPE,
        )
        for rank, (score, item, paths) in enumerate(ordered, start=1)
    ]


def _reject_lane_disagreement(struct_chunk: KBChunk, bm25_chunk: KBChunk) -> None:
    """Fail closed when two lanes disagree about one passage identity."""

    if (
        struct_chunk.doc_id != bm25_chunk.doc_id
        or struct_chunk.doc_version != bm25_chunk.doc_version
        or struct_chunk.checksum != bm25_chunk.checksum
    ):
        raise RuntimeError(
            "Fused StructArray and BM25 lanes disagree about passage identity"
        )


def _verify_struct_array_schema(description: object) -> None:
    if not isinstance(description, Mapping):
        raise RuntimeError("StructArray schema description is invalid")
    raw_fields = description.get("fields")
    if not isinstance(raw_fields, list):
        raise RuntimeError("StructArray schema fields are missing")
    actual = {
        str(field.get("name")): field
        for field in raw_fields
        if isinstance(field, Mapping)
    }
    expected_fields = KB_DOCUMENTS_COLLECTION["fields"]
    expected_names = {str(field["name"]) for field in expected_fields}
    if set(actual) != expected_names:
        raise RuntimeError("StructArray parent schema fields differ")
    for expected in expected_fields:
        field = actual[str(expected["name"])]
        if _data_type_name(field.get("type")) != _expected_type_name(
            str(expected["type"])
        ):
            raise RuntimeError(
                f"StructArray field {expected['name']} has an unexpected type"
            )
        params = field.get("params", {})
        if not isinstance(params, Mapping):
            raise RuntimeError("StructArray field params are invalid")
        for parameter in ("max_length", "dim", "max_capacity"):
            if parameter in expected and int(params.get(parameter, -1)) != int(
                expected[parameter]
            ):
                raise RuntimeError(
                    f"StructArray field {expected['name']} differs in {parameter}"
                )
        if expected.get("primary_key", False) != bool(field.get("is_primary", False)):
            raise RuntimeError("StructArray primary key does not match the contract")
        if expected["type"] == "Array":
            if _data_type_name(field.get("element_type")) != "STRUCT":
                raise RuntimeError("StructArray element type is not STRUCT")
            raw_subfields = field.get("struct_fields")
            if not isinstance(raw_subfields, list):
                raise RuntimeError("StructArray subfields are missing")
            _verify_struct_subfields(raw_subfields, expected["struct_fields"])


def _verify_struct_subfields(
    raw_subfields: list[object], expected_subfields: object
) -> None:
    if not isinstance(expected_subfields, list):
        raise RuntimeError("StructArray expected subfields are invalid")
    actual = {
        str(field.get("name")): field
        for field in raw_subfields
        if isinstance(field, Mapping)
    }
    if set(actual) != {str(field["name"]) for field in expected_subfields}:
        raise RuntimeError("StructArray subfield names differ")
    for expected in expected_subfields:
        field = actual[str(expected["name"])]
        if _data_type_name(field.get("type")) != _expected_type_name(
            str(expected["type"])
        ):
            raise RuntimeError(
                f"StructArray subfield {expected['name']} has an unexpected type"
            )
        params = field.get("params", {})
        if not isinstance(params, Mapping):
            raise RuntimeError("StructArray subfield params are invalid")
        for parameter in ("max_length", "dim"):
            if parameter in expected and int(params.get(parameter, -1)) != int(
                expected[parameter]
            ):
                raise RuntimeError(
                    f"StructArray subfield {expected['name']} differs in {parameter}"
                )


def _data_type_name(value: object) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name.upper()
    return str(value).rsplit(".", maxsplit=1)[-1].upper()


def _expected_type_name(value: str) -> str:
    return {
        "FloatVector": "FLOAT_VECTOR",
        "SparseFloatVector": "SPARSE_FLOAT_VECTOR",
        "BinaryVector": "BINARY_VECTOR",
        "VarChar": "VARCHAR",
    }.get(value, value.upper())


def _verify_struct_array_indexes(client: Any, collection_name: str) -> None:
    expected = {
        EMBEDDING_LIST_INDEX_NAME: (
            EMBEDDING_LIST_FIELD,
            "HNSW",
            "MAX_SIM_COSINE",
        ),
        ELEMENT_INDEX_NAME: (ELEMENT_VECTOR_FIELD, "HNSW", "COSINE"),
    }
    for index_name, (field_name, index_type, metric_type) in expected.items():
        description = client.describe_index(
            collection_name=collection_name,
            index_name=index_name,
        )
        if not isinstance(description, Mapping):
            raise RuntimeError("StructArray index description is invalid")
        observed = (
            str(description.get("field_name", "")),
            str(description.get("index_type", "")).upper(),
            str(description.get("metric_type", "")).upper(),
        )
        if observed != (field_name, index_type, metric_type):
            raise RuntimeError(f"StructArray index {index_name} differs")
        if (
            index_name == EMBEDDING_LIST_INDEX_NAME
            and "tokenann" not in str(dict(description)).casefold()
        ):
            raise RuntimeError("StructArray EmbeddingList index is not TokenANN")


def _varchar_limits() -> tuple[dict[str, int], dict[str, int]]:
    """Return parent/subfield UTF-8 byte limits from the authoritative schema."""

    parent: dict[str, int] = {}
    passage: dict[str, int] = {}
    for field in KB_DOCUMENTS_COLLECTION["fields"]:
        if field["type"] == "VarChar":
            parent[str(field["name"])] = int(field["max_length"])
        if field["type"] == "Array":
            for subfield in field["struct_fields"]:
                if subfield["type"] == "VarChar":
                    passage[str(subfield["name"])] = int(subfield["max_length"])
    return parent, passage


def _validate_varchar(value: object, *, field: str, limit: int) -> None:
    if not isinstance(value, str):
        raise ValueError(f"StructArray {field} must be a string")
    if len(value.encode("utf-8")) > limit:
        raise ValueError(f"StructArray {field} exceeds {limit} UTF-8 bytes")


def _validate_projection_build(build: ProjectionBuild) -> None:
    """Validate a complete build before any persistent projection mutation."""

    if not _HEX_64.fullmatch(build.projection_fingerprint):
        raise ValueError("StructArray projection fingerprint is invalid")
    if not build.text_embedding_fingerprint:
        raise ValueError("StructArray embedding fingerprint is empty")
    if build.parent_count != len(build.parents) or not build.parents:
        raise ValueError("StructArray projection parent count is inconsistent")
    actual_passages = sum(len(parent.passages) for parent in build.parents)
    if build.passage_count != actual_passages:
        raise ValueError("StructArray projection passage count is inconsistent")
    parent_limits, passage_limits = _varchar_limits()
    seen_keys: set[str] = set()
    seen_chunks: set[str] = set()
    for parent in build.parents:
        if parent.document_key in seen_keys:
            raise ValueError("StructArray projection has duplicate document keys")
        seen_keys.add(parent.document_key)
        if parent.document_key != document_key(parent.doc_id, parent.doc_version):
            raise ValueError("StructArray projection document key is inconsistent")
        if (
            parent.projection_fingerprint != build.projection_fingerprint
            or parent.text_embedding_fingerprint != build.text_embedding_fingerprint
            or parent.projection_parent_count != build.parent_count
            or parent.projection_passage_count != build.passage_count
            or parent.passage_count != len(parent.passages)
            or len(parent.passages) > MAX_PASSAGES_PER_DOCUMENT
        ):
            raise ValueError(
                "StructArray projection activation metadata is inconsistent"
            )
        for field, limit in parent_limits.items():
            _validate_varchar(getattr(parent, field), field=field, limit=limit)
        for passage in parent.passages:
            if passage.chunk_id in seen_chunks:
                raise ValueError("StructArray projection has duplicate chunk ids")
            seen_chunks.add(passage.chunk_id)
            for field, limit in passage_limits.items():
                _validate_varchar(getattr(passage, field), field=field, limit=limit)
            _validated_vector(passage.element_vector, chunk_id=passage.chunk_id)
            _validated_vector(passage.embedding_list_vector, chunk_id=passage.chunk_id)


def _validate_parent_invariance(chunks: Sequence[KBChunk]) -> None:
    fields = (
        "source_type",
        "source_uri",
        "doc_type",
        "title",
        "department",
        "is_current",
    )
    first = chunks[0]
    for chunk in chunks[1:]:
        if any(getattr(chunk, field) != getattr(first, field) for field in fields):
            raise ValueError(
                f"Document {first.doc_id}@{first.doc_version} has mixed parent metadata"
            )


def _chunk_embedding_fingerprint(chunk: KBChunk) -> str:
    value = chunk.metadata.get(EMBEDDING_FINGERPRINT_KEY) if chunk.metadata else None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Projected chunk {chunk.chunk_id} lacks an embedding fingerprint"
        )
    return value


def _validated_vector(values: Sequence[float], *, chunk_id: str) -> tuple[float, ...]:
    if len(values) != VECTOR_DIMS["TEXT_DIM"] or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        raise ValueError(f"Projected chunk {chunk_id} has an invalid text vector")
    return tuple(float(value) for value in values)


def _projection_fingerprint(
    manifest: ProjectionManifest, parents: Sequence[DocumentProjection]
) -> str:
    payload = {
        "schema_version": manifest.schema_version,
        "selected_doc_ids": list(manifest.selected_doc_ids),
        "recipe": "kb-documents-passages-v1",
        "parents": [
            {
                "document_key": parent.document_key,
                "embedding_fingerprint": parent.text_embedding_fingerprint,
                "passages": [
                    [item.chunk_id, item.checksum, item.chunk_index]
                    for item in parent.passages
                ],
            }
            for parent in parents
        ],
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_profile_request(
    queries: Sequence[str],
    top_k: int,
    filters: dict[str, Any] | None,
    predicates: Sequence[ElementPredicate],
) -> tuple[str, ...]:
    if not 1 <= len(queries) <= 3:
        raise ValueError("StructArray profile requires one to three queries")
    if not 1 <= top_k <= 64:
        raise ValueError("StructArray top_k must be between 1 and 64")
    normalize_filters(filters)
    if len(predicates) > 4:
        raise ValueError("At most four element predicates are allowed")
    return tuple(validate_question(query) for query in queries)


def _parent_matches(parent: DocumentProjection, filters: dict[str, Any]) -> bool:
    for field, expected in filters.items():
        if field == "has_image_vector":
            if expected:
                return False
            continue
        actual = getattr(parent, field)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def _passage_matches(
    passage: PassageProjection, predicates: Sequence[ElementPredicate]
) -> bool:
    operations = {
        "eq": lambda left, right: left == right,
        "ne": lambda left, right: left != right,
        "gt": lambda left, right: left > right,
        "gte": lambda left, right: left >= right,
        "lt": lambda left, right: left < right,
        "lte": lambda left, right: left <= right,
    }
    return all(
        operations[item.operator](getattr(passage, item.field), item.value)
        for item in predicates
    )


def _rehydrate_parent(
    retriever: StructArrayFlatRetriever, parent: DocumentProjection
) -> list[KBChunk]:
    ids = [item.chunk_id for item in parent.passages]
    return _fetch_authoritative(retriever, ids, None)


def _fetch_authoritative(
    retriever: StructArrayFlatRetriever,
    ids: Sequence[str],
    filters: dict[str, Any] | None,
) -> list[KBChunk]:
    output: list[KBChunk] = []
    for start in range(0, len(ids), 16):
        batch = list(ids[start : start + 16])
        if batch:
            output.extend(
                retriever.fetch_chunks_by_ids(chunk_ids=batch, filters=filters)
            )
    return output


def _element_result(
    chunk: KBChunk, rank: int, score: float, profile: str, key: str, offset: int
) -> SearchResult:
    if not math.isfinite(score):
        raise RuntimeError("StructArray search score must be finite")
    return SearchResult(
        chunk=chunk,
        rank=rank,
        dense_score=score,
        keyword_score=0.0,
        recency_score=0.0,
        priority_score=0.0,
        hybrid_score=score,
        retrieval_profile=profile,
        document_key=key,
        struct_field=STRUCT_FIELD,
        element_offset=offset,
        retrieval_paths=("struct_element",),
    )


def _parent_expression(
    filters: dict[str, Any] | None,
    projection_fingerprint: str,
    document_keys: Sequence[str] | None = None,
) -> str:
    normalized = normalize_filters(filters)
    if normalized.pop("has_image_vector", False):
        raise ValueError("StructArray projection does not contain image-only records")
    clauses = [f"projection_fingerprint == {json.dumps(projection_fingerprint)}"]
    for field, value in normalized.items():
        if isinstance(value, list):
            clauses.append(f"{field} in {json.dumps(value, ensure_ascii=False)}")
        elif isinstance(value, bool):
            clauses.append(f"{field} == {str(value).lower()}")
        else:
            clauses.append(f"{field} == {json.dumps(value, ensure_ascii=False)}")
    if document_keys is not None:
        if not document_keys:
            clauses.append("document_key in []")
            return " and ".join(clauses)
        clauses.append(f"document_key in {json.dumps(list(document_keys))}")
    return " and ".join(clauses)


def _first_hits(raw: object) -> list[dict[str, Any]]:
    if not raw or not isinstance(raw, list):
        return []
    first = raw[0]
    values = first if isinstance(first, list) else raw
    if any(not isinstance(item, dict) for item in values):
        raise RuntimeError("Milvus StructArray search returned an invalid hit")
    return list(values)


def _hit_entity(hit: Mapping[str, Any]) -> dict[str, Any]:
    entity = hit.get("entity", hit)
    if not isinstance(entity, dict):
        raise RuntimeError("Milvus StructArray hit is missing an entity")
    return entity


def _finite_hit_score(hit: Mapping[str, Any]) -> float:
    value = hit.get("distance", hit.get("score"))
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise RuntimeError("Milvus StructArray hit has an invalid score")
    return float(value)


def _passage_from_mapping(value: object) -> PassageProjection:
    if not isinstance(value, Mapping):
        raise RuntimeError("Milvus StructArray passage output is invalid")
    required = {
        "chunk_id",
        "checksum",
        "chunk_index",
        "page_no",
        "section",
        "record_type",
        "language",
    }
    if not required.issubset(value):
        raise RuntimeError("Milvus StructArray passage output lacks identity fields")
    return PassageProjection(
        chunk_id=str(value["chunk_id"]),
        checksum=str(value["checksum"]),
        chunk_index=int(value["chunk_index"]),
        page_no=int(value["page_no"]),
        section=str(value["section"]),
        record_type=str(value["record_type"]),
        language=str(value["language"]),
        embedding_list_vector=(),
        element_vector=(),
    )
