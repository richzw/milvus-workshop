"""Optional pymilvus schema helpers pending Phase 0 verification."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from importlib import import_module
from typing import Any

from agent_workshop_demo.embedding import (
    EMBEDDING_FINGERPRINT_KEY,
    dense_vector,
    text_embedding_fingerprint,
)
from agent_workshop_demo.image_embedding import (
    IMAGE_EMBEDDING_FINGERPRINT_KEY,
    image_embedding_fingerprint,
)
from agent_workshop_demo.image_retrieval import (
    image_only_filters,
    require_image_space,
    validate_image_fingerprint,
    validate_image_search_top_k,
    validate_image_query_vector,
)
from agent_workshop_demo.models import (
    ImageSearchResult,
    KBChunk,
    SearchResult,
)
from agent_workshop_demo.retrieval import (
    OrderMode,
    order_search_results,
    parse_order_by,
    validate_aggregation_fields,
)
from agent_workshop_demo.schema.collections import (
    CONVERSATION_MEMORY_COLLECTION,
    CONVERSATION_MEMORY_INDEXES,
    DOC_DEDUP_SIGNATURES_COLLECTION,
    DOC_DEDUP_SIGNATURES_INDEXES,
    GROUNDED_RESPONSE_CACHE_COLLECTION,
    GROUNDED_RESPONSE_CACHE_INDEXES,
    KB_CHUNKS_COLLECTION,
    KB_CHUNKS_INDEXES,
    KB_DOCUMENTS_COLLECTION,
    KB_DOCUMENTS_INDEXES,
    MEMORY_EVENTS_COLLECTION,
    MEMORY_EVENTS_INDEXES,
    MEMORY_FACTS_COLLECTION,
    MEMORY_FACTS_INDEXES,
    MEMORY_CONSOLIDATION_JOURNAL_COLLECTION,
    MEMORY_CONSOLIDATION_JOURNAL_INDEXES,
)
from agent_workshop_demo.validation import normalize_filters, validate_question

COLLECTION_DEFINITIONS: list[dict[str, Any]] = [
    KB_CHUNKS_COLLECTION,
    KB_DOCUMENTS_COLLECTION,
    CONVERSATION_MEMORY_COLLECTION,
    MEMORY_EVENTS_COLLECTION,
    MEMORY_FACTS_COLLECTION,
    MEMORY_CONSOLIDATION_JOURNAL_COLLECTION,
    GROUNDED_RESPONSE_CACHE_COLLECTION,
    DOC_DEDUP_SIGNATURES_COLLECTION,
]

INDEX_DEFINITIONS: dict[str, dict[str, Any]] = {
    str(KB_CHUNKS_COLLECTION["collection_name"]): KB_CHUNKS_INDEXES,
    str(KB_DOCUMENTS_COLLECTION["collection_name"]): KB_DOCUMENTS_INDEXES,
    str(CONVERSATION_MEMORY_COLLECTION["collection_name"]): CONVERSATION_MEMORY_INDEXES,
    str(MEMORY_EVENTS_COLLECTION["collection_name"]): MEMORY_EVENTS_INDEXES,
    str(MEMORY_FACTS_COLLECTION["collection_name"]): MEMORY_FACTS_INDEXES,
    str(
        MEMORY_CONSOLIDATION_JOURNAL_COLLECTION["collection_name"]
    ): MEMORY_CONSOLIDATION_JOURNAL_INDEXES,
    str(
        GROUNDED_RESPONSE_CACHE_COLLECTION["collection_name"]
    ): GROUNDED_RESPONSE_CACHE_INDEXES,
    str(
        DOC_DEDUP_SIGNATURES_COLLECTION["collection_name"]
    ): DOC_DEDUP_SIGNATURES_INDEXES,
}
DEMO_COLLECTION_NAMES = tuple(
    str(definition["collection_name"]) for definition in COLLECTION_DEFINITIONS
)
_FIELD_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_LOCAL_BOOL_FACET_FIELDS = frozenset({"has_image_vector", "is_current"})


class MilvusHybridRetriever:
    """Milvus-backed insertion and retrieval adapter for kb chunks."""

    supports_parallel_search = False

    def __init__(
        self,
        client: Any,
        *,
        collection_name: str = "kb_chunks",
        batch_size: int = 100,
        sparse_field: str = "sparse_vector",
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if not _FIELD_NAME.fullmatch(sparse_field):
            raise ValueError("sparse_field must be a valid Milvus field name")
        self.client = client
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.sparse_field = sparse_field

    @classmethod
    def connect(
        cls,
        uri: str,
        token: str | None = None,
        **kwargs: Any,
    ) -> MilvusHybridRetriever:
        """Create an adapter using the installed pymilvus client."""

        try:
            milvus_module = import_module("pymilvus")
        except ImportError as exc:
            raise RuntimeError(
                "Install pymilvus with `pip install -r demo/requirements.txt`."
            ) from exc
        return cls(
            milvus_module.MilvusClient(uri=uri, token=token),
            **kwargs,
        )

    def insert(self, chunks: Iterable[KBChunk]) -> dict[str, int]:
        """Replace matching chunk IDs and insert records in bounded batches."""

        records = [self._record_for_insert(chunk) for chunk in chunks]
        if not self.client.has_collection(collection_name=self.collection_name):
            raise RuntimeError(
                f"Milvus collection {self.collection_name!r} does not exist; "
                "run demo/scripts/create_collections.py first."
            )
        self.client.load_collection(collection_name=self.collection_name)
        self._reject_image_space_conflict(records)
        self._reject_current_family_conflicts(records)
        batch_count = 0
        for start in range(0, len(records), self.batch_size):
            batch = records[start : start + self.batch_size]
            chunk_ids = [record["chunk_id"] for record in batch]
            self.client.delete(
                collection_name=self.collection_name,
                filter=f"chunk_id in {json.dumps(chunk_ids)}",
            )
            result = self.client.insert(
                collection_name=self.collection_name,
                data=batch,
            )
            inserted = _mutation_count(result, "insert_count")
            if inserted is not None and inserted != len(batch):
                raise RuntimeError(
                    "Milvus reported an incomplete insert: "
                    f"expected {len(batch)}, got {inserted}."
                )
            batch_count += 1
        if records:
            self.client.flush(collection_name=self.collection_name)
        return {"insert_count": len(records), "batch_count": batch_count}

    def _reject_image_space_conflict(
        self,
        records: list[dict[str, Any]],
    ) -> None:
        incoming_fingerprints = {
            str(record["metadata"][IMAGE_EMBEDDING_FINGERPRINT_KEY])
            for record in records
            if record["has_image_vector"]
        }
        if not incoming_fingerprints:
            return
        if len(incoming_fingerprints) != 1:
            raise ValueError(
                "Incoming image records must share one embedding fingerprint"
            )
        incoming = next(iter(incoming_fingerprints))
        existing = self.client.query(
            collection_name=self.collection_name,
            filter="has_image_vector == true",
            output_fields=["metadata"],
            limit=1,
        )
        if not existing:
            return
        raw_metadata = existing[0].get("metadata")
        stored = (
            raw_metadata.get(IMAGE_EMBEDDING_FINGERPRINT_KEY)
            if isinstance(raw_metadata, dict)
            else None
        )
        if not isinstance(stored, str) or stored != incoming:
            raise RuntimeError(
                "Incremental image embedding-space replacement is unsafe; "
                "recreate and fully re-ingest kb_chunks before publishing."
            )

    def _reject_current_family_conflicts(
        self,
        records: list[dict[str, Any]],
    ) -> None:
        incoming_current: dict[str, set[str]] = {}
        for record in records:
            if record["is_current"]:
                incoming_current.setdefault(str(record["doc_id"]), set()).add(
                    str(record["doc_version"])
                )
        if any(len(versions) != 1 for versions in incoming_current.values()):
            raise ValueError(
                "Each incoming document family must have one current edition"
            )
        if not incoming_current:
            return
        conflicts: list[tuple[str, str, str]] = []
        for doc_id, versions in sorted(incoming_current.items()):
            incoming_version = next(iter(versions))
            existing = self.client.query(
                collection_name=self.collection_name,
                filter=(
                    "is_current == true and "
                    f"doc_id == {json.dumps(doc_id, ensure_ascii=False)} and "
                    "doc_version != "
                    f"{json.dumps(incoming_version, ensure_ascii=False)}"
                ),
                output_fields=["doc_id", "doc_version"],
                limit=1,
            )
            if existing:
                conflicts.append(
                    (
                        str(existing[0]["doc_id"]),
                        str(existing[0]["doc_version"]),
                        incoming_version,
                    )
                )
        if conflicts:
            raise RuntimeError(
                "Incremental current-edition replacement is unsafe; recreate "
                "and fully re-ingest kb_chunks before publishing. Conflicts: "
                f"{conflicts}"
            )

    def ensure_collection_ready(self) -> None:
        """Fail fast unless the configured collection exists and can be loaded."""

        if not self.client.has_collection(collection_name=self.collection_name):
            raise RuntimeError(
                f"Milvus collection {self.collection_name!r} does not exist; "
                "run demo/scripts/create_collections.py and "
                "demo/scripts/ingest_demo.py first."
            )
        self.client.load_collection(collection_name=self.collection_name)

    def ensure_embedding_space_ready(self, *, sample_size: int = 8) -> str:
        """Fail startup when stored chunks predate the configured vector space.

        Spec 15 § 7 makes the recorded embedding fingerprint the migration gate:
        a provider or model change without a matching re-ingest must stop the
        process instead of serving two silently mixed vector spaces.
        """

        if not 1 <= sample_size <= 64:
            raise ValueError("sample_size must be between 1 and 64")
        expected = text_embedding_fingerprint()
        try:
            rows = self.client.query(
                collection_name=self.collection_name,
                filter="",
                output_fields=["chunk_id", "metadata"],
                limit=sample_size,
            )
        except Exception as exc:
            raise RuntimeError(
                "Unable to read the stored text embedding fingerprint from "
                f"collection {self.collection_name!r}"
            ) from exc
        observed = {
            str(_entity_metadata(row).get(EMBEDDING_FINGERPRINT_KEY))
            for row in rows
        }
        if not observed:
            return expected
        if observed != {expected}:
            raise RuntimeError(
                "Stored chunks were embedded in a different vector space: "
                f"expected {expected!r}, found {sorted(observed)!r}. "
                "Re-ingest demo/scripts/ingest_demo.py after an embedding "
                "provider or model change."
            )
        return expected

    def verify_inserted(self, chunk_ids: Iterable[str]) -> int:
        """Read inserted chunk IDs back from Milvus and require all of them."""

        expected = list(dict.fromkeys(chunk_ids))
        if not expected:
            return 0
        self.client.load_collection(collection_name=self.collection_name)
        found: set[str] = set()
        for start in range(0, len(expected), self.batch_size):
            batch = expected[start : start + self.batch_size]
            rows = self.client.query(
                collection_name=self.collection_name,
                filter=f"chunk_id in {json.dumps(batch)}",
                output_fields=["chunk_id"],
                limit=len(batch),
            )
            found.update(
                str(row["chunk_id"])
                for row in rows
                if isinstance(row, dict) and "chunk_id" in row
            )
        missing = sorted(set(expected) - found)
        if missing:
            preview = ", ".join(missing[:5])
            raise RuntimeError(
                f"Milvus read-back verification failed for "
                f"{len(missing)} chunk(s): {preview}"
            )
        return len(found)

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
        order_by: list[str] | None = None,
        order_mode: OrderMode = "relevance",
    ) -> list[SearchResult]:
        """Run dense and sparse recall, then fuse into workflow results."""

        normalized_query = validate_question(query)
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        expression = _filter_expression(normalize_filters(filters))
        parsed_order = parse_order_by(order_by or [])
        if order_mode not in {"relevance", "scalar"}:
            raise ValueError(f"Unsupported order_mode: {order_mode!r}")
        output_fields = _chunk_output_fields()
        common = {
            "collection_name": self.collection_name,
            "filter": expression,
            "limit": top_k,
            "output_fields": output_fields,
        }
        if order_mode == "scalar" and parsed_order:
            common["order_by_fields"] = [
                {"field": field, "order": direction}
                for field, direction in parsed_order
            ]
        dense_hits = self.client.search(
            **common,
            data=[dense_vector(normalized_query)],
            anns_field="text_vector",
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
        )
        sparse_hits = self.client.search(
            **common,
            data=[normalized_query],
            anns_field=self.sparse_field,
            search_params={"metric_type": "BM25", "params": {}},
        )
        results = _fuse_hits(
            _first_query_hits(dense_hits),
            _first_query_hits(sparse_hits),
            top_k=top_k,
            order_by=order_by or [],
            order_mode=order_mode,
        )
        for result in results:
            _require_matching_embedding_fingerprint(result.chunk)
        return results

    def search_image_vector(
        self,
        query_vector: list[float],
        *,
        image_fingerprint: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[ImageSearchResult]:
        """Search nullable image vectors with an enforced image-only filter."""

        validate_image_search_top_k(top_k)
        validated_query = validate_image_query_vector(query_vector)
        validated_fingerprint = validate_image_fingerprint(image_fingerprint)
        configured_fingerprint = image_embedding_fingerprint()
        if validated_fingerprint != configured_fingerprint:
            raise ValueError(
                "Image query fingerprint does not match the configured "
                "collection vector space"
            )
        expression = _filter_expression(image_only_filters(filters))
        raw_hits = self.client.search(
            collection_name=self.collection_name,
            data=[validated_query],
            anns_field="image_vector",
            search_params={
                "metric_type": "COSINE",
                "params": {"ef": 64},
            },
            filter=expression,
            limit=top_k,
            output_fields=_chunk_output_fields(),
        )
        results: list[ImageSearchResult] = []
        for rank, hit in enumerate(
            _first_query_hits(raw_hits)[:top_k],
            start=1,
        ):
            chunk = _chunk_from_entity(_hit_entity(hit))
            require_image_space(
                chunk,
                expected_fingerprint=validated_fingerprint,
            )
            _require_matching_embedding_fingerprint(chunk)
            results.append(
                ImageSearchResult(
                    chunk=chunk,
                    rank=rank,
                    image_score=_hit_score(hit),
                )
            )
        return results

    def search_sparse(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
        order_by: list[str] | None = None,
    ) -> list[SearchResult]:
        """Run the authoritative BM25 lane without dense recall."""

        normalized_query = validate_question(query)
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        parsed_order = parse_order_by(order_by or [])
        common: dict[str, Any] = {
            "collection_name": self.collection_name,
            "data": [normalized_query],
            "anns_field": self.sparse_field,
            "search_params": {"metric_type": "BM25", "params": {}},
            "filter": _filter_expression(normalize_filters(filters)),
            "limit": top_k,
            "output_fields": _chunk_output_fields(),
        }
        if parsed_order:
            common["order_by_fields"] = [
                {"field": field, "order": direction}
                for field, direction in parsed_order
            ]
        raw_hits = self.client.search(**common)
        results: list[SearchResult] = []
        for rank, hit in enumerate(_first_query_hits(raw_hits)[:top_k], start=1):
            chunk = _chunk_from_entity(_hit_entity(hit))
            _require_matching_embedding_fingerprint(chunk)
            score = _hit_score(hit)
            results.append(
                SearchResult(
                    chunk=chunk,
                    rank=rank,
                    dense_score=0.0,
                    keyword_score=score,
                    recency_score=0.0,
                    priority_score=0.0,
                    hybrid_score=score,
                    retrieval_profile="flat_bm25",
                    retrieval_paths=("flat_bm25",),
                )
            )
        return results

    def aggregations(
        self,
        results: list[SearchResult],
        fields: list[str],
    ) -> dict[str, dict[str, int]]:
        """Aggregate facets over the retained, de-duplicated candidates."""

        requested = validate_aggregation_fields(fields)
        candidates: dict[str, KBChunk] = {}
        for item in results:
            candidates.setdefault(item.chunk.chunk_id, item.chunk)
        chunk_ids = list(candidates)
        if len(chunk_ids) > 64:
            raise ValueError("Aggregation candidate set exceeds 64 chunks")
        output: dict[str, dict[str, int]] = {field: {} for field in requested}
        if not requested or not chunk_ids:
            return output

        grouped_fields = [
            field for field in requested if field not in _LOCAL_BOOL_FACET_FIELDS
        ]
        if grouped_fields:
            rows = self.client.query(
                collection_name=self.collection_name,
                filter="chunk_id in " + json.dumps(chunk_ids, ensure_ascii=False),
                group_by_fields=grouped_fields,
                output_fields=[*grouped_fields, "count(*)"],
                limit=64,
            )
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("Milvus aggregation returned an invalid row")
                raw_count = row.get("count(*)", row.get("count_all"))
                if (
                    isinstance(raw_count, bool)
                    or not isinstance(raw_count, int)
                    or raw_count <= 0
                ):
                    raise ValueError("Milvus aggregation returned an invalid count")
                for field in grouped_fields:
                    if field not in row:
                        raise ValueError("Milvus aggregation row is missing a field")
                    key = str(row[field])
                    output[field][key] = output[field].get(key, 0) + raw_count

        for field in requested:
            if field not in _LOCAL_BOOL_FACET_FIELDS:
                continue
            for chunk in candidates.values():
                key = str(getattr(chunk, field))
                output[field][key] = output[field].get(key, 0) + 1
        if any(sum(counts.values()) != len(chunk_ids) for counts in output.values()):
            raise ValueError("Milvus aggregation counts do not match candidate scope")
        return {field: dict(sorted(counts.items())) for field, counts in output.items()}

    def fetch_document_chunks(
        self,
        *,
        doc_id: str,
        doc_version: str,
        filters: dict[str, Any] | None = None,
        limit: int,
    ) -> list[KBChunk]:
        """Fetch authorized sibling chunks through an exact scalar query."""

        normalized_doc_id = doc_id.strip()
        normalized_doc_version = doc_version.strip()
        if not normalized_doc_id or not normalized_doc_version:
            raise ValueError("doc_id and doc_version must be non-empty")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        base_expression = _filter_expression(normalize_filters(filters))
        family_expression = (
            f"doc_id == {json.dumps(normalized_doc_id, ensure_ascii=False)} "
            "and doc_version == "
            f"{json.dumps(normalized_doc_version, ensure_ascii=False)}"
        )
        expression = (
            f"{base_expression} and {family_expression}"
            if base_expression
            else family_expression
        )
        output_fields = _chunk_output_fields()
        rows = self.client.query(
            collection_name=self.collection_name,
            filter=expression,
            output_fields=output_fields,
            limit=limit,
        )
        chunks = [_chunk_from_entity(row) for row in rows]
        for chunk in chunks:
            _require_matching_embedding_fingerprint(chunk)
        return sorted(
            chunks,
            key=lambda chunk: (
                chunk.chunk_index,
                chunk.page_no or 0,
                chunk.chunk_id,
            ),
        )[:limit]

    def fetch_chunks_by_ids(
        self,
        *,
        chunk_ids: list[str],
        filters: dict[str, Any] | None = None,
    ) -> list[KBChunk]:
        """Fetch bounded authorized chunks for cache freshness validation."""

        expected = list(dict.fromkeys(chunk_ids))
        if not 1 <= len(expected) <= 16 or any(not item.strip() for item in expected):
            raise ValueError("chunk_ids must contain 1..16 non-empty ids")
        base_expression = _filter_expression(normalize_filters(filters))
        ids_expression = "chunk_id in " + json.dumps(expected, ensure_ascii=False)
        expression = (
            f"{base_expression} and {ids_expression}"
            if base_expression
            else ids_expression
        )
        output_fields = _chunk_output_fields()
        try:
            rows = self.client.query(
                collection_name=self.collection_name,
                filter=expression,
                output_fields=output_fields,
                limit=len(expected),
            )
            chunks = [_chunk_from_entity(row) for row in rows]
            for chunk in chunks:
                _require_matching_embedding_fingerprint(chunk)
        except Exception as exc:
            raise RuntimeError("Unable to validate cached KB evidence") from exc
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        return [by_id[item] for item in expected if item in by_id]

    @staticmethod
    def _record_for_insert(chunk: KBChunk) -> dict[str, Any]:
        _require_matching_embedding_fingerprint(chunk)
        record = chunk.to_dict()
        nullable_fields = {
            str(field["name"])
            for field in KB_CHUNKS_COLLECTION["fields"]
            if field.get("nullable", False)
        }
        record = {
            key: value
            for key, value in record.items()
            if value is not None or key not in nullable_fields
        }
        record["retrieval_text"] = _chunk_retrieval_text(chunk)
        record.pop("sparse_vector", None)
        return record


class MilvusDedupStore:
    """Persist raw dedup inputs and let Milvus 3.0 compute MinHash DIDO."""

    def __init__(
        self,
        client: Any,
        *,
        collection_name: str = "doc_dedup_signatures",
        batch_size: int = 100,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self.client = client
        self.collection_name = collection_name
        self.batch_size = batch_size

    def insert(self, records: Iterable[dict[str, Any]]) -> dict[str, int]:
        """Replace chunk-level raw inputs without writing Function output."""

        pending = [self._validated_record(record) for record in records]
        if not self.client.has_collection(collection_name=self.collection_name):
            raise RuntimeError(
                f"Milvus collection {self.collection_name!r} does not exist"
            )
        self.client.load_collection(collection_name=self.collection_name)
        batches = 0
        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            chunk_ids = [item["chunk_id"] for item in batch if item.get("chunk_id")]
            if chunk_ids:
                self.client.delete(
                    collection_name=self.collection_name,
                    filter="chunk_id in " + json.dumps(chunk_ids, ensure_ascii=False),
                )
            result = self.client.insert(
                collection_name=self.collection_name,
                data=batch,
            )
            inserted = _mutation_count(result, "insert_count")
            if inserted is not None and inserted != len(batch):
                raise RuntimeError("Milvus reported an incomplete dedup insert")
            batches += 1
        if pending:
            self.client.flush(collection_name=self.collection_name)
        return {"dedup_insert_count": len(pending), "dedup_batch_count": batches}

    @staticmethod
    def _validated_record(record: dict[str, Any]) -> dict[str, Any]:
        expected = {
            "doc_id",
            "chunk_id",
            "source_uri",
            "source_type",
            "record_level",
            "normalized_text",
            "checksum",
            "created_at",
            "metadata",
        }
        if set(record) != expected or "minhash_signature" in record:
            raise ValueError("Invalid server-side MinHash input record")
        if record["record_level"] == "chunk" and not record["chunk_id"]:
            raise ValueError("Chunk-level dedup input requires chunk_id")
        return dict(record)


def _require_matching_embedding_fingerprint(chunk: KBChunk) -> None:
    expected = text_embedding_fingerprint()
    actual = (chunk.metadata or {}).get(EMBEDDING_FINGERPRINT_KEY)
    if actual != expected:
        raise ValueError(
            "Chunk embedding fingerprint does not match the configured "
            f"vector space: expected {expected!r}, got {actual!r}"
        )
    if chunk.has_image_vector:
        expected_image = image_embedding_fingerprint()
        actual_image = (chunk.metadata or {}).get(IMAGE_EMBEDDING_FINGERPRINT_KEY)
        if actual_image != expected_image:
            raise ValueError(
                "Chunk image embedding fingerprint does not match the "
                "configured vector space: expected "
                f"{expected_image!r}, got {actual_image!r}"
            )


def _chunk_retrieval_text(chunk: KBChunk) -> str:
    """Rebuild the versioned BM25 input without changing citation text."""

    metadata = chunk.metadata or {}
    raw_path = metadata.get("heading_path", [])
    heading_path = (
        tuple(str(item) for item in raw_path) if isinstance(raw_path, list) else ()
    )
    parts: list[str] = []
    for value in (
        chunk.title,
        " > ".join(heading_path),
        chunk.section or "",
        chunk.text,
    ):
        normalized = value.strip()
        if normalized and normalized not in parts:
            parts.append(normalized)
    return "\n".join(parts)


def _filter_expression(filters: dict[str, Any]) -> str:
    clauses: list[str] = []
    for field_name, value in filters.items():
        if isinstance(value, list):
            if not value:
                continue
            clauses.append(f"{field_name} in {json.dumps(value, ensure_ascii=False)}")
        elif isinstance(value, bool):
            clauses.append(f"{field_name} == {str(value).lower()}")
        else:
            clauses.append(f"{field_name} == {json.dumps(value, ensure_ascii=False)}")
    return " and ".join(clauses)


def _first_query_hits(raw_hits: Any) -> list[dict[str, Any]]:
    if not raw_hits:
        return []
    first = raw_hits[0]
    return list(first) if isinstance(first, list) else list(raw_hits)


def _fuse_hits(
    dense_hits: list[dict[str, Any]],
    sparse_hits: list[dict[str, Any]],
    *,
    top_k: int,
    order_by: list[str],
    order_mode: OrderMode = "relevance",
) -> list[SearchResult]:
    entities: dict[str, dict[str, Any]] = {}
    dense_by_id: dict[str, tuple[int, float]] = {}
    sparse_by_id: dict[str, tuple[int, float]] = {}
    for rank, hit in enumerate(dense_hits, start=1):
        entity = _hit_entity(hit)
        chunk_id = str(entity["chunk_id"])
        entities[chunk_id] = entity
        dense_by_id[chunk_id] = (rank, _hit_score(hit))
    for rank, hit in enumerate(sparse_hits, start=1):
        entity = _hit_entity(hit)
        chunk_id = str(entity["chunk_id"])
        entities.setdefault(chunk_id, entity)
        sparse_by_id[chunk_id] = (rank, _hit_score(hit))

    chunks = {key: _chunk_from_entity(value) for key, value in entities.items()}
    max_updated_at = max((item.updated_at for item in chunks.values()), default=1)
    max_priority = max((item.priority for item in chunks.values()), default=1)
    recalled: list[SearchResult] = []
    for chunk_id, chunk in chunks.items():
        dense_rank, dense_score = dense_by_id.get(chunk_id, (0, 0.0))
        sparse_rank, keyword_score = sparse_by_id.get(chunk_id, (0, 0.0))
        dense_rank_score = 1.0 / dense_rank if dense_rank else 0.0
        sparse_rank_score = 1.0 / sparse_rank if sparse_rank else 0.0
        recency_score = chunk.updated_at / max(max_updated_at, 1)
        priority_score = chunk.priority / max(max_priority, 1)
        hybrid_score = (
            0.55 * dense_rank_score
            + 0.35 * sparse_rank_score
            + 0.05 * recency_score
            + 0.05 * priority_score
        )
        recalled.append(
            SearchResult(
                chunk=chunk,
                rank=0,
                dense_score=dense_score,
                keyword_score=keyword_score,
                recency_score=recency_score,
                priority_score=priority_score,
                hybrid_score=hybrid_score,
            )
        )
    ordered = order_search_results(
        recalled,
        order_by,
        order_mode=order_mode,
    )[:top_k]
    return [
        SearchResult(
            chunk=item.chunk,
            rank=rank,
            dense_score=item.dense_score,
            keyword_score=item.keyword_score,
            recency_score=item.recency_score,
            priority_score=item.priority_score,
            hybrid_score=item.hybrid_score,
        )
        for rank, item in enumerate(ordered, start=1)
    ]


def _hit_entity(hit: dict[str, Any]) -> dict[str, Any]:
    entity = hit.get("entity", hit)
    if not isinstance(entity, dict) or "chunk_id" not in entity:
        raise ValueError("Milvus search hit is missing entity.chunk_id")
    return entity


def _chunk_output_fields() -> list[str]:
    """Project stored image vectors because KBChunk validates image records."""

    vector_types = {"FloatVector", "SparseFloatVector", "BinaryVector"}
    return [
        field["name"]
        for field in KB_CHUNKS_COLLECTION["fields"]
        if field["name"] != "id"
        and (field["name"] == "image_vector" or field["type"] not in vector_types)
    ]


def _hit_score(hit: dict[str, Any]) -> float:
    return float(hit.get("distance", hit.get("score", 0.0)))


def _entity_metadata(entity: dict[str, Any]) -> dict[str, Any]:
    raw = entity.get("metadata")
    return dict(raw) if isinstance(raw, dict) else {}


def _chunk_from_entity(entity: dict[str, Any]) -> KBChunk:
    text = str(entity["text"])
    has_image_vector = bool(entity["has_image_vector"])
    raw_image_vector = entity.get("image_vector")
    image_vector = (
        [float(value) for value in raw_image_vector]
        if isinstance(raw_image_vector, list)
        else ([] if has_image_vector else None)
    )
    raw_text_vector = entity.get("text_vector")
    return KBChunk(
        doc_id=str(entity["doc_id"]),
        chunk_id=str(entity["chunk_id"]),
        parent_id=_optional_str(entity.get("parent_id")),
        record_type=str(entity["record_type"]),
        source_type=str(entity["source_type"]),
        source_uri=str(entity["source_uri"]),
        bucket=_optional_str(entity.get("bucket")),
        object_key=_optional_str(entity.get("object_key")),
        doc_type=str(entity["doc_type"]),
        title=str(entity["title"]),
        section=_optional_str(entity.get("section")),
        page_no=_optional_int(entity.get("page_no")),
        chunk_index=int(entity["chunk_index"]),
        text=text,
        text_summary=_optional_str(entity.get("text_summary")),
        language=str(entity["language"]),
        department=str(entity["department"]),
        updated_at=int(entity["updated_at"]),
        created_at=_optional_int(entity.get("created_at")),
        priority=int(entity["priority"]),
        doc_version=str(entity["doc_version"]),
        is_current=bool(entity["is_current"]),
        checksum=_optional_str(entity.get("checksum")),
        metadata=(
            dict(entity["metadata"])
            if isinstance(entity.get("metadata"), dict)
            else None
        ),
        has_image_vector=has_image_vector,
        text_vector=(
            [float(value) for value in raw_text_vector]
            if isinstance(raw_text_vector, list)
            else []
        ),
        sparse_vector={},
        image_vector=image_vector,
    )


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def build_collection_schema(definition: dict[str, Any]) -> Any:
    """Build a pymilvus schema using the installed SDK API."""

    try:
        milvus_module = import_module("pymilvus")
    except ImportError as exc:
        raise RuntimeError(
            "Install pymilvus with `pip install -r demo/requirements.txt`."
        ) from exc

    type_map = {
        "Int64": milvus_module.DataType.INT64,
        "Int32": milvus_module.DataType.INT32,
        "VarChar": milvus_module.DataType.VARCHAR,
        "JSON": milvus_module.DataType.JSON,
        "Bool": milvus_module.DataType.BOOL,
        "Float": milvus_module.DataType.FLOAT,
        "FloatVector": milvus_module.DataType.FLOAT_VECTOR,
        "SparseFloatVector": milvus_module.DataType.SPARSE_FLOAT_VECTOR,
        "BinaryVector": milvus_module.DataType.BINARY_VECTOR,
        "TIMESTAMPTZ": milvus_module.DataType.TIMESTAMPTZ,
        "Array": milvus_module.DataType.ARRAY,
    }
    schema = milvus_module.MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
        description=definition["description"],
    )
    for field in definition["fields"]:
        kwargs = {
            "description": field.get("description", ""),
            "is_primary": field.get("primary_key", False),
            "auto_id": field.get("auto_id", False),
            "nullable": field.get("nullable", False),
        }
        if "max_length" in field:
            kwargs["max_length"] = field["max_length"]
        if "dim" in field:
            kwargs["dim"] = field["dim"]
        if "enable_analyzer" in field:
            kwargs["enable_analyzer"] = field["enable_analyzer"]
        if "analyzer_params" in field:
            kwargs["analyzer_params"] = field["analyzer_params"]
        if field["type"] == "Array":
            if field.get("element_type") != "Struct":
                raise ValueError("Only StructArray definitions are supported")
            struct_schema = milvus_module.MilvusClient.create_struct_field_schema()
            for subfield in field.get("struct_fields", []):
                sub_kwargs = {
                    "field_name": subfield["name"],
                    "datatype": type_map[subfield["type"]],
                }
                if "max_length" in subfield:
                    sub_kwargs["max_length"] = subfield["max_length"]
                if "dim" in subfield:
                    sub_kwargs["dim"] = subfield["dim"]
                struct_schema.add_field(**sub_kwargs)
            kwargs.update(
                {
                    "element_type": milvus_module.DataType.STRUCT,
                    "struct_schema": struct_schema,
                    "max_capacity": field["max_capacity"],
                }
            )
        schema.add_field(
            field_name=field["name"],
            datatype=type_map[field["type"]],
            **kwargs,
        )
    for function in definition.get("functions", []):
        function_type = _required_function_type(milvus_module, function["type"])
        schema.add_function(
            milvus_module.Function(
                name=function["name"],
                function_type=function_type,
                input_field_names=function["input_fields"],
                output_field_names=function["output_fields"],
                params=function.get("params", {}),
            )
        )
    return schema


def _required_function_type(milvus_module: Any, name: str) -> Any:
    """Resolve one server Function type with an actionable SDK diagnostic."""

    function_type = getattr(milvus_module.FunctionType, name, None)
    if function_type is None:
        raise RuntimeError(
            f"The installed pymilvus does not expose FunctionType.{name}. "
            "The demo schema requires the pinned pymilvus==3.0.1 from "
            "demo/requirements.txt."
        )
    return function_type


def create_collections(
    uri: str,
    token: str | None = None,
    *,
    drop_existing: bool = False,
    client: Any | None = None,
) -> dict[str, list[str]]:
    """Create all demo collections and verify them against Milvus."""

    milvus_client = client or _connect_milvus_client(uri, token)
    created: list[str] = []
    existing: list[str] = []
    for definition in COLLECTION_DEFINITIONS:
        name = str(definition["collection_name"])
        if milvus_client.has_collection(collection_name=name):
            if not drop_existing:
                existing.append(name)
                continue
            milvus_client.drop_collection(collection_name=name)
        milvus_client.create_collection(
            collection_name=name,
            schema=build_collection_schema(definition),
            **(
                {"properties": definition["properties"]}
                if "properties" in definition
                else {}
            ),
        )
        created.append(name)
    verified = [
        str(definition["collection_name"])
        for definition in COLLECTION_DEFINITIONS
        if milvus_client.has_collection(
            collection_name=str(definition["collection_name"])
        )
    ]
    expected = {
        str(definition["collection_name"]) for definition in COLLECTION_DEFINITIONS
    }
    missing = sorted(expected - set(verified))
    if missing:
        raise RuntimeError(
            "Milvus collection verification failed: " + ", ".join(missing)
        )
    return {
        "created": created,
        "existing": existing,
        "verified": verified,
    }


def drop_demo_collections(
    uri: str,
    token: str | None = None,
    *,
    client: Any | None = None,
) -> dict[str, list[str]]:
    """Drop only the repository-owned demo collections and verify removal."""

    milvus_client = client or _connect_milvus_client(uri, token)
    dropped: list[str] = []
    absent: list[str] = []
    for name in DEMO_COLLECTION_NAMES:
        if not milvus_client.has_collection(collection_name=name):
            absent.append(name)
            continue
        milvus_client.drop_collection(collection_name=name)
        if milvus_client.has_collection(collection_name=name):
            raise RuntimeError(f"Milvus collection {name!r} still exists after drop")
        dropped.append(name)
    return {
        "targeted": list(DEMO_COLLECTION_NAMES),
        "dropped": dropped,
        "already_absent": absent,
    }


def create_indexes(
    uri: str,
    token: str | None = None,
    *,
    recreate: bool = False,
    client: Any | None = None,
    sparse_compatibility_daat_maxscore: bool = False,
) -> dict[str, dict[str, list[str]]]:
    """Create vector and scalar indexes and verify their server-side names."""

    milvus_client = client or _connect_milvus_client(uri, token)
    report: dict[str, dict[str, list[str]]] = {}
    for collection_name, definitions in INDEX_DEFINITIONS.items():
        if not milvus_client.has_collection(collection_name=collection_name):
            raise RuntimeError(
                f"Milvus collection {collection_name!r} does not exist; "
                "run demo/scripts/create_collections.py first."
            )
        existing = set(milvus_client.list_indexes(collection_name=collection_name))
        requested = _index_requests(
            definitions,
            sparse_compatibility_daat_maxscore=(
                sparse_compatibility_daat_maxscore and collection_name == "kb_chunks"
            ),
        )
        created: list[str] = []
        retained: list[str] = []
        index_params = milvus_client.prepare_index_params()
        for request in requested:
            index_name = str(request["index_name"])
            if index_name in existing:
                if not recreate:
                    retained.append(index_name)
                    continue
                milvus_client.drop_index(
                    collection_name=collection_name,
                    index_name=index_name,
                )
            index_params.add_index(**request)
            created.append(index_name)
        if created:
            milvus_client.create_index(
                collection_name=collection_name,
                index_params=index_params,
                sync=True,
            )
        verified = sorted(milvus_client.list_indexes(collection_name=collection_name))
        expected = {str(item["index_name"]) for item in requested}
        missing = sorted(expected - set(verified))
        if missing:
            raise RuntimeError(
                f"Milvus index verification failed for {collection_name}: "
                + ", ".join(missing)
            )
        for request in requested:
            _verify_server_index(milvus_client, collection_name, request)
        report[collection_name] = {
            "created": created,
            "existing": retained,
            "verified": sorted(expected),
        }
    return report


def _verify_server_index(
    client: Any,
    collection_name: str,
    request: dict[str, Any],
) -> None:
    index_name = str(request["index_name"])
    description = client.describe_index(
        collection_name=collection_name,
        index_name=index_name,
    )
    if not isinstance(description, dict):
        raise RuntimeError(
            f"Milvus index verification returned an invalid {index_name!r} description"
        )
    expected = {
        "field_name": str(request["field_name"]),
        "index_name": index_name,
        "index_type": str(request["index_type"]),
    }
    if "metric_type" in request:
        expected["metric_type"] = str(request["metric_type"])
    mismatches = [
        field
        for field, value in expected.items()
        if str(description.get(field, "")).upper() != value.upper()
    ]
    if mismatches:
        raise RuntimeError(
            f"Milvus index {collection_name}.{index_name} differs in "
            + ", ".join(mismatches)
            + "; rerun with --recreate"
        )


def _connect_milvus_client(uri: str, token: str | None) -> Any:
    try:
        milvus_module = import_module("pymilvus")
    except ImportError as exc:
        raise RuntimeError(
            "Install pymilvus with `pip install -r demo/requirements.txt`."
        ) from exc
    kwargs: dict[str, Any] = {"uri": uri}
    if token:
        kwargs["token"] = token
    return milvus_module.MilvusClient(**kwargs)


def _index_requests(
    definitions: dict[str, Any],
    *,
    sparse_compatibility_daat_maxscore: bool = False,
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for field_name, definition in definitions.items():
        if field_name == "scalar_indexes":
            continue
        params = dict(definition.get("params", {}))
        if field_name == "sparse_vector" and sparse_compatibility_daat_maxscore:
            params["inverted_index_algo"] = "DAAT_MAXSCORE"
        request = {
            "field_name": field_name,
            "index_name": definition.get("index_name", f"{field_name}_idx"),
            "index_type": definition["index_type"],
            "metric_type": definition["metric_type"],
            "params": params,
        }
        requests.append(request)
    for definition in definitions.get("scalar_indexes", []):
        if isinstance(definition, str):
            field_name = definition
            index_type = "INVERTED"
            scalar_params: dict[str, Any] = {}
        elif (
            isinstance(definition, dict)
            and set(definition).issubset({"field_name", "index_type", "params"})
            and isinstance(definition.get("field_name"), str)
            and isinstance(definition.get("index_type"), str)
            and isinstance(definition.get("params", {}), dict)
        ):
            field_name = definition["field_name"]
            index_type = definition["index_type"]
            scalar_params = dict(definition.get("params", {}))
        else:
            raise ValueError("Scalar index definition is invalid")
        requests.append(
            {
                "field_name": field_name,
                "index_name": f"{field_name}_idx",
                "index_type": index_type,
                "params": scalar_params,
            }
        )
    return requests


def _mutation_count(result: Any, key: str) -> int | None:
    if isinstance(result, dict) and key in result:
        return int(result[key])
    value = getattr(result, key, None)
    return None if value is None else int(value)
