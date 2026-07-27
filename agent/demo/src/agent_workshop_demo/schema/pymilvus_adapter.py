"""Optional pymilvus schema helpers pending Phase 0 verification."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from importlib import import_module
from typing import Any

from agent_workshop_demo.embedding import (
    EMBEDDING_FINGERPRINT_KEY,
    dense_vector,
    sparse_vector,
    text_embedding_fingerprint,
)
from agent_workshop_demo.models import KBChunk, SearchResult
from agent_workshop_demo.schema.collections import (
    CONVERSATION_MEMORY_COLLECTION,
    CONVERSATION_MEMORY_INDEXES,
    DOC_DEDUP_SIGNATURES_COLLECTION,
    DOC_DEDUP_SIGNATURES_INDEXES,
    KB_CHUNKS_COLLECTION,
    KB_CHUNKS_INDEXES,
)
from agent_workshop_demo.validation import normalize_filters, validate_question

COLLECTION_DEFINITIONS = [
    KB_CHUNKS_COLLECTION,
    CONVERSATION_MEMORY_COLLECTION,
    DOC_DEDUP_SIGNATURES_COLLECTION,
]

INDEX_DEFINITIONS = {
    str(KB_CHUNKS_COLLECTION["collection_name"]): KB_CHUNKS_INDEXES,
    str(CONVERSATION_MEMORY_COLLECTION["collection_name"]):
        CONVERSATION_MEMORY_INDEXES,
    str(DOC_DEDUP_SIGNATURES_COLLECTION["collection_name"]):
        DOC_DEDUP_SIGNATURES_INDEXES,
}
DEMO_COLLECTION_NAMES = tuple(
    str(definition["collection_name"])
    for definition in COLLECTION_DEFINITIONS
)


class MilvusHybridRetriever:
    """Milvus-backed insertion and retrieval adapter for kb chunks."""

    def __init__(
        self,
        client: Any,
        *,
        collection_name: str = "kb_chunks",
        batch_size: int = 100,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self.client = client
        self.collection_name = collection_name
        self.batch_size = batch_size

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
                "Install pymilvus with `pip install -r "
                "demo/requirements.txt`."
            ) from exc
        return cls(
            milvus_module.MilvusClient(uri=uri, token=token),
            **kwargs,
        )

    def insert(self, chunks: Iterable[KBChunk]) -> dict[str, int]:
        """Replace matching chunk IDs and insert records in bounded batches."""

        records = [self._record_for_insert(chunk) for chunk in chunks]
        if not self.client.has_collection(
            collection_name=self.collection_name
        ):
            raise RuntimeError(
                f"Milvus collection {self.collection_name!r} does not exist; "
                "run demo/scripts/create_collections.py first."
            )
        self.client.load_collection(collection_name=self.collection_name)
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

        if not self.client.has_collection(
            collection_name=self.collection_name
        ):
            raise RuntimeError(
                f"Milvus collection {self.collection_name!r} does not exist; "
                "run demo/scripts/create_collections.py and "
                "demo/scripts/ingest_demo.py first."
            )
        self.client.load_collection(collection_name=self.collection_name)

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
    ) -> list[SearchResult]:
        """Run dense and sparse recall, then fuse into workflow results."""

        normalized_query = validate_question(query)
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        expression = _filter_expression(normalize_filters(filters))
        output_fields = [
            field["name"]
            for field in KB_CHUNKS_COLLECTION["fields"]
            if field["name"] != "id"
            and field["type"]
            not in {"FloatVector", "SparseFloatVector", "BinaryVector"}
        ]
        common = {
            "collection_name": self.collection_name,
            "filter": expression,
            "limit": top_k,
            "output_fields": output_fields,
        }
        dense_hits = self.client.search(
            **common,
            data=[dense_vector(normalized_query)],
            anns_field="text_vector",
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
        )
        sparse_hits = self.client.search(
            **common,
            data=[_milvus_sparse_vector(sparse_vector(normalized_query))],
            anns_field="sparse_vector",
            search_params={"metric_type": "IP", "params": {}},
        )
        results = _fuse_hits(
            _first_query_hits(dense_hits),
            _first_query_hits(sparse_hits),
            top_k=top_k,
            order_by=order_by or [],
        )
        for result in results:
            _require_matching_embedding_fingerprint(result.chunk)
        return results

    @staticmethod
    def aggregations(
        results: list[SearchResult],
        fields: list[str],
    ) -> dict[str, dict[str, int]]:
        """Count public scalar fields over the recalled result set."""

        return {
            field_name: dict(
                sorted(
                    Counter(
                        str(getattr(item.chunk, field_name))
                        for item in results
                    ).items()
                )
            )
            for field_name in fields
        }

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
        record["sparse_vector"] = _milvus_sparse_vector(
            chunk.sparse_vector
        )
        return record


def _require_matching_embedding_fingerprint(chunk: KBChunk) -> None:
    expected = text_embedding_fingerprint()
    actual = (chunk.metadata or {}).get(EMBEDDING_FINGERPRINT_KEY)
    if actual != expected:
        raise ValueError(
            "Chunk embedding fingerprint does not match the configured "
            f"vector space: expected {expected!r}, got {actual!r}"
        )


def _milvus_sparse_vector(
    values: dict[str, float] | dict[int, float],
) -> dict[int, float]:
    """Convert token weights to stable uint32 Milvus sparse dimensions."""

    converted: dict[int, float] = {}
    for token, weight in values.items():
        if isinstance(token, int):
            index = token
        else:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big")
        converted[index] = converted.get(index, 0.0) + float(weight)
    return converted


def _filter_expression(filters: dict[str, Any]) -> str:
    clauses: list[str] = []
    for field_name, value in filters.items():
        if isinstance(value, list):
            if not value:
                continue
            clauses.append(
                f"{field_name} in {json.dumps(value, ensure_ascii=False)}"
            )
        elif isinstance(value, bool):
            clauses.append(f"{field_name} == {str(value).lower()}")
        else:
            clauses.append(
                f"{field_name} == {json.dumps(value, ensure_ascii=False)}"
            )
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
    ordered = _order_results(recalled, order_by)[:top_k]
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


def _hit_score(hit: dict[str, Any]) -> float:
    return float(hit.get("distance", hit.get("score", 0.0)))


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
        sparse_vector=sparse_vector(text),
        image_vector=image_vector,
    )


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _order_results(
    results: list[SearchResult],
    order_by: list[str],
) -> list[SearchResult]:
    supported_fields = {"updated_at", "priority"}
    parsed: list[tuple[str, str]] = []
    for clause in order_by:
        parts = clause.split()
        field_name = parts[0] if parts else ""
        direction = parts[1].lower() if len(parts) > 1 else "asc"
        if (
            field_name not in supported_fields
            or direction not in {"asc", "desc"}
            or len(parts) > 2
        ):
            raise ValueError(f"Unsupported order_by clause: {clause!r}")
        parsed.append((field_name, direction))

    def sort_key(item: SearchResult) -> tuple[float, ...]:
        tie_breakers = [
            float(getattr(item.chunk, field_name))
            * (1 if direction == "asc" else -1)
            for field_name, direction in parsed
        ]
        return (-item.hybrid_score, *tie_breakers)

    return sorted(results, key=sort_key)


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
        "FloatVector": milvus_module.DataType.FLOAT_VECTOR,
        "SparseFloatVector": milvus_module.DataType.SPARSE_FLOAT_VECTOR,
        "BinaryVector": milvus_module.DataType.BINARY_VECTOR,
    }
    fields = []
    for field in definition["fields"]:
        kwargs = {
            "name": field["name"],
            "dtype": type_map[field["type"]],
            "description": field.get("description", ""),
            "is_primary": field.get("primary_key", False),
            "auto_id": field.get("auto_id", False),
            "nullable": field.get("nullable", False),
        }
        if "max_length" in field:
            kwargs["max_length"] = field["max_length"]
        if "dim" in field:
            kwargs["dim"] = field["dim"]
        fields.append(milvus_module.FieldSchema(**kwargs))
    return milvus_module.CollectionSchema(
        fields=fields,
        description=definition["description"],
        enable_dynamic_field=False,
    )


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
        str(definition["collection_name"])
        for definition in COLLECTION_DEFINITIONS
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
            raise RuntimeError(
                f"Milvus collection {name!r} still exists after drop"
            )
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
) -> dict[str, dict[str, list[str]]]:
    """Create vector and scalar indexes and verify their server-side names."""

    milvus_client = client or _connect_milvus_client(uri, token)
    report: dict[str, dict[str, list[str]]] = {}
    for collection_name, definitions in INDEX_DEFINITIONS.items():
        if not milvus_client.has_collection(
            collection_name=collection_name
        ):
            raise RuntimeError(
                f"Milvus collection {collection_name!r} does not exist; "
                "run demo/scripts/create_collections.py first."
            )
        existing = set(
            milvus_client.list_indexes(collection_name=collection_name)
        )
        requested = _index_requests(definitions)
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
        verified = sorted(
            milvus_client.list_indexes(collection_name=collection_name)
        )
        expected = {str(item["index_name"]) for item in requested}
        missing = sorted(expected - set(verified))
        if missing:
            raise RuntimeError(
                f"Milvus index verification failed for {collection_name}: "
                + ", ".join(missing)
            )
        report[collection_name] = {
            "created": created,
            "existing": retained,
            "verified": sorted(expected),
        }
    return report


def _connect_milvus_client(uri: str, token: str | None) -> Any:
    try:
        milvus_module = import_module("pymilvus")
    except ImportError as exc:
        raise RuntimeError(
            "Install pymilvus with `pip install -r "
            "demo/requirements.txt`."
        ) from exc
    kwargs: dict[str, Any] = {"uri": uri}
    if token:
        kwargs["token"] = token
    return milvus_module.MilvusClient(**kwargs)


def _index_requests(
    definitions: dict[str, Any],
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for field_name, definition in definitions.items():
        if field_name == "scalar_indexes":
            continue
        request = {
            "field_name": field_name,
            "index_name": f"{field_name}_idx",
            "index_type": definition["index_type"],
            "metric_type": definition["metric_type"],
            "params": definition.get("params", {}),
        }
        requests.append(request)
    for field_name in definitions.get("scalar_indexes", []):
        requests.append(
            {
                "field_name": field_name,
                "index_name": f"{field_name}_idx",
                "index_type": "INVERTED",
                "params": {},
            }
        )
    return requests


def _mutation_count(result: Any, key: str) -> int | None:
    if isinstance(result, dict) and key in result:
        return int(result[key])
    value = getattr(result, key, None)
    return None if value is None else int(value)
