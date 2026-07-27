"""Session-scoped conversation memory with local and Milvus stores."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol, cast

from agent_workshop_demo.config import VECTOR_DIMS
from agent_workshop_demo.embedding import (
    EMBEDDING_FINGERPRINT_KEY,
    cosine_similarity,
    dense_vector,
    text_embedding_fingerprint,
)
from agent_workshop_demo.validation import validate_identifier

MemoryRole = Literal["user", "assistant", "system", "summary"]
MemoryType = Literal["short_term", "session_summary", "task_state"]

MEMORY_ROLES = frozenset({"user", "assistant", "system", "summary"})
MEMORY_TYPES = frozenset(
    {"short_term", "session_summary", "task_state"}
)
DEFAULT_RECALL_TYPES: tuple[MemoryType, ...] = (
    "session_summary",
    "task_state",
)
MAX_MEMORY_CONTENT_CHARS = 8_192
MAX_MEMORY_SUMMARY_CHARS = 2_048
MAX_MEMORY_METADATA_BYTES = 4_096
MAX_MEMORY_TOP_K = 20
MAX_SESSION_RECORDS = 200
MAX_TURN_RECORDS = 4


class MemoryStoreError(RuntimeError):
    """A sanitized conversation-memory dependency failure."""


@dataclass(frozen=True)
class MemoryRecord:
    """One validated, session-scoped semantic memory record."""

    session_id: str
    turn_id: str
    role: MemoryRole
    content: str
    summary: str | None
    memory_type: MemoryType
    created_at: int
    expires_at: int | None
    metadata: dict[str, Any]
    content_vector: list[float]

    def __post_init__(self) -> None:
        validate_identifier(self.session_id, field_name="session_id")
        validate_identifier(self.turn_id, field_name="turn_id")
        if self.role not in MEMORY_ROLES:
            raise ValueError(f"Unsupported memory role: {self.role}")
        if self.memory_type not in MEMORY_TYPES:
            raise ValueError(
                f"Unsupported memory_type: {self.memory_type}"
            )
        if not self.content.strip():
            raise ValueError("Memory content must be non-empty")
        if len(self.content) > MAX_MEMORY_CONTENT_CHARS:
            raise ValueError("Memory content exceeds the schema limit")
        if (
            self.summary is not None
            and len(self.summary) > MAX_MEMORY_SUMMARY_CHARS
        ):
            raise ValueError("Memory summary exceeds the schema limit")
        if self.created_at < 0:
            raise ValueError("Memory created_at cannot be negative")
        if self.expires_at is not None and self.expires_at < 0:
            raise ValueError("Memory expires_at cannot be negative")
        if not isinstance(self.metadata, dict):
            raise ValueError("Memory metadata must be an object")
        try:
            encoded_metadata = json.dumps(
                self.metadata,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError("Memory metadata must be JSON-serializable") from exc
        if len(encoded_metadata) > MAX_MEMORY_METADATA_BYTES:
            raise ValueError("Memory metadata exceeds the bounded limit")
        if len(self.content_vector) != VECTOR_DIMS["TEXT_DIM"] or any(
            not isinstance(value, (int, float))
            for value in self.content_vector
        ):
            raise ValueError(
                "Memory content_vector does not match the configured dimension"
            )

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        turn_id: str,
        role: MemoryRole,
        content: str,
        memory_type: MemoryType,
        created_at: int,
        expires_at: int | None,
        summary: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        """Build a record and embed its summary or content."""

        if not content.strip():
            raise ValueError("Memory content must be non-empty")
        if len(content) > MAX_MEMORY_CONTENT_CHARS:
            raise ValueError("Memory content exceeds the schema limit")
        if summary is not None and len(summary) > MAX_MEMORY_SUMMARY_CHARS:
            raise ValueError("Memory summary exceeds the schema limit")
        normalized_metadata = {} if metadata is None else dict(metadata)
        fingerprint = text_embedding_fingerprint()
        existing_fingerprint = normalized_metadata.get(
            EMBEDDING_FINGERPRINT_KEY
        )
        if (
            existing_fingerprint is not None
            and existing_fingerprint != fingerprint
        ):
            raise ValueError(
                "Memory embedding fingerprint does not match configuration"
            )
        normalized_metadata[EMBEDDING_FINGERPRINT_KEY] = fingerprint
        return cls(
            session_id=session_id,
            turn_id=turn_id,
            role=role,
            content=content,
            summary=summary,
            memory_type=memory_type,
            created_at=created_at,
            expires_at=expires_at,
            metadata=normalized_metadata,
            content_vector=dense_vector(summary or content),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete record."""

        return asdict(self)

    def presentation_summary(self) -> str:
        """Return bounded session-private text for prompt/UI use."""

        value = self.summary or self.content
        return value[:MAX_MEMORY_SUMMARY_CHARS]


class ConversationMemory(Protocol):
    """Storage contract shared by local and Milvus implementations."""

    def upsert_turn(self, records: list[MemoryRecord]) -> int: ...

    def search(
        self,
        query: str,
        *,
        session_id: str,
        top_k: int,
        now_ms: int,
        memory_types: tuple[MemoryType, ...] = DEFAULT_RECALL_TYPES,
    ) -> list[MemoryRecord]: ...

    def list_session(
        self,
        session_id: str,
        *,
        now_ms: int,
        limit: int = MAX_SESSION_RECORDS,
    ) -> list[MemoryRecord]: ...

    def delete_session(self, session_id: str) -> int: ...


class ConversationMemoryStore:
    """Deterministic in-process implementation for tests and local demos."""

    def __init__(self, now_ms: int = 1_782_604_800_000) -> None:
        if now_ms < 0:
            raise ValueError("now_ms cannot be negative")
        self.now_ms = now_ms
        self.records: list[MemoryRecord] = []

    def add_turn(
        self,
        *,
        session_id: str,
        turn_id: str,
        role: MemoryRole,
        content: str,
        memory_type: MemoryType,
        expires_at: int | None = None,
        summary: str | None = None,
        metadata: dict[str, Any] | None = None,
        created_at: int | None = None,
    ) -> MemoryRecord:
        """Validate and append one record, replacing the same record kind."""

        record = MemoryRecord.create(
            session_id=session_id,
            turn_id=turn_id,
            role=role,
            content=content,
            memory_type=memory_type,
            created_at=self.now_ms if created_at is None else created_at,
            expires_at=expires_at,
            summary=summary,
            metadata=metadata,
        )
        self.records = [
            item
            for item in self.records
            if not (
                item.session_id == session_id
                and item.turn_id == turn_id
                and item.role == role
                and item.memory_type == memory_type
            )
        ]
        self.records.append(record)
        return record

    def upsert_turn(self, records: list[MemoryRecord]) -> int:
        """Replace one complete turn atomically in local memory."""

        session_id, turn_id = _validate_turn_batch(records)
        self.records = [
            item
            for item in self.records
            if not (
                item.session_id == session_id
                and item.turn_id == turn_id
            )
        ]
        self.records.extend(records)
        return len(records)

    def search(
        self,
        query: str,
        *,
        session_id: str,
        top_k: int,
        now_ms: int | None = None,
        memory_types: tuple[MemoryType, ...] = DEFAULT_RECALL_TYPES,
    ) -> list[MemoryRecord]:
        """Return semantically ranked live records from one session."""

        normalized_types = _validate_search(
            query,
            session_id=session_id,
            top_k=top_k,
            memory_types=memory_types,
        )
        current = self.now_ms if now_ms is None else now_ms
        if current < 0:
            raise ValueError("now_ms cannot be negative")
        query_vector = dense_vector(query)
        candidates = [
            item
            for item in self.records
            if item.session_id == session_id
            and item.memory_type in normalized_types
            and (item.expires_at is None or item.expires_at > current)
        ]
        scored = (
            (cosine_similarity(query_vector, item.content_vector), item)
            for item in candidates
        )
        return [
            item
            for _, item in sorted(
                scored,
                key=lambda pair: (
                    pair[0],
                    pair[1].created_at,
                    pair[1].turn_id,
                ),
                reverse=True,
            )[:top_k]
        ]

    def list_session(
        self,
        session_id: str,
        *,
        now_ms: int | None = None,
        limit: int = MAX_SESSION_RECORDS,
    ) -> list[MemoryRecord]:
        """List live records for one session in chronological order."""

        validate_identifier(session_id, field_name="session_id")
        _validate_limit(limit, maximum=MAX_SESSION_RECORDS)
        current = self.now_ms if now_ms is None else now_ms
        if current < 0:
            raise ValueError("now_ms cannot be negative")
        records = [
            item
            for item in self.records
            if item.session_id == session_id
            and (item.expires_at is None or item.expires_at > current)
        ]
        return sorted(
            records,
            key=lambda item: (
                item.created_at,
                item.turn_id,
                item.role,
                item.memory_type,
            ),
        )[:limit]

    def delete_session(self, session_id: str) -> int:
        """Delete only records belonging to the requested session."""

        validate_identifier(session_id, field_name="session_id")
        prior_count = len(self.records)
        self.records = [
            item for item in self.records if item.session_id != session_id
        ]
        return prior_count - len(self.records)


class MilvusConversationMemoryStore:
    """Milvus-backed session Memory using explicit expiry filters."""

    OUTPUT_FIELDS = [
        "session_id",
        "turn_id",
        "role",
        "content",
        "summary",
        "memory_type",
        "created_at",
        "expires_at",
        "metadata",
        "content_vector",
    ]

    def __init__(
        self,
        client: Any,
        *,
        collection_name: str = "conversation_memory",
    ) -> None:
        if not collection_name.strip():
            raise ValueError("Memory collection_name must be non-empty")
        self.client = client
        self.collection_name = collection_name

    def ensure_collection_ready(self) -> None:
        """Require and load the configured Memory collection."""

        try:
            exists = self.client.has_collection(
                collection_name=self.collection_name
            )
            if not exists:
                raise MemoryStoreError(
                    f"Milvus collection {self.collection_name!r} does not "
                    "exist; run demo/scripts/create_collections.py first."
                )
            self.client.load_collection(
                collection_name=self.collection_name
            )
        except MemoryStoreError:
            raise
        except Exception as exc:
            raise MemoryStoreError(
                "Unable to prepare the conversation memory collection"
            ) from exc

    def upsert_turn(self, records: list[MemoryRecord]) -> int:
        """Replace one `(session_id, turn_id)` record set."""

        session_id, turn_id = _validate_turn_batch(records)
        expression = (
            f"session_id == {_quoted(session_id)} and "
            f"turn_id == {_quoted(turn_id)}"
        )
        data = [_record_for_milvus(item) for item in records]
        try:
            self.client.delete(
                collection_name=self.collection_name,
                filter=expression,
            )
            result = self.client.insert(
                collection_name=self.collection_name,
                data=data,
            )
            inserted = _mutation_count(result, "insert_count")
            if inserted is not None and inserted != len(data):
                raise MemoryStoreError(
                    "Milvus reported an incomplete Memory insert"
                )
            self.client.flush(collection_name=self.collection_name)
        except MemoryStoreError:
            raise
        except Exception as exc:
            raise MemoryStoreError(
                "Unable to persist conversation memory"
            ) from exc
        return len(data)

    def search(
        self,
        query: str,
        *,
        session_id: str,
        top_k: int,
        now_ms: int,
        memory_types: tuple[MemoryType, ...] = DEFAULT_RECALL_TYPES,
    ) -> list[MemoryRecord]:
        """Search live, same-session Memory with COSINE similarity."""

        normalized_types = _validate_search(
            query,
            session_id=session_id,
            top_k=top_k,
            memory_types=memory_types,
        )
        if now_ms < 0:
            raise ValueError("now_ms cannot be negative")
        expression = _memory_filter(
            session_id,
            now_ms=now_ms,
            memory_types=normalized_types,
        )
        try:
            raw_hits = self.client.search(
                collection_name=self.collection_name,
                data=[dense_vector(query)],
                anns_field="content_vector",
                search_params={
                    "metric_type": "COSINE",
                    "params": {"ef": 32},
                },
                filter=expression,
                limit=top_k,
                output_fields=self.OUTPUT_FIELDS,
            )
        except Exception as exc:
            raise MemoryStoreError(
                "Unable to recall conversation memory"
            ) from exc
        records = [
            _record_from_milvus(_memory_hit_entity(hit))
            for hit in _first_memory_hits(raw_hits)
        ][:top_k]
        _validate_returned_records(
            records,
            session_id=session_id,
            now_ms=now_ms,
            memory_types=normalized_types,
        )
        return records

    def list_session(
        self,
        session_id: str,
        *,
        now_ms: int,
        limit: int = MAX_SESSION_RECORDS,
    ) -> list[MemoryRecord]:
        """List live records belonging only to one session."""

        validate_identifier(session_id, field_name="session_id")
        _validate_limit(limit, maximum=MAX_SESSION_RECORDS)
        if now_ms < 0:
            raise ValueError("now_ms cannot be negative")
        expression = _memory_filter(session_id, now_ms=now_ms)
        iterator: Any | None = None
        selected: list[MemoryRecord] = []
        try:
            iterator = self.client.query_iterator(
                collection_name=self.collection_name,
                filter=expression,
                output_fields=self.OUTPUT_FIELDS,
                batch_size=MAX_SESSION_RECORDS,
                limit=-1,
            )
            while True:
                rows = iterator.next()
                if not rows:
                    break
                batch = [_record_from_milvus(row) for row in rows]
                _validate_returned_records(
                    batch,
                    session_id=session_id,
                    now_ms=now_ms,
                )
                selected.extend(batch)
                selected = sorted(
                    selected,
                    key=_memory_order_key,
                )[:limit]
        except MemoryStoreError:
            raise
        except Exception as exc:
            raise MemoryStoreError(
                "Unable to list conversation memory"
            ) from exc
        finally:
            if iterator is not None:
                try:
                    iterator.close()
                except Exception as exc:
                    raise MemoryStoreError(
                        "Unable to close the conversation memory listing"
                    ) from exc
        return selected

    def delete_session(self, session_id: str) -> int:
        """Delete only one validated session's Memory."""

        validate_identifier(session_id, field_name="session_id")
        try:
            result = self.client.delete(
                collection_name=self.collection_name,
                filter=f"session_id == {_quoted(session_id)}",
            )
        except Exception as exc:
            raise MemoryStoreError(
                "Unable to delete conversation memory"
            ) from exc
        return _mutation_count(result, "delete_count") or 0


def utc_now_ms() -> int:
    """Return the current UTC epoch time in milliseconds."""

    return int(time.time() * 1000)


def build_turn_records(
    *,
    session_id: str,
    turn_id: str,
    user_content: str,
    assistant_content: str,
    created_at: int,
    expires_at: int,
    remembered_statement: str | None = None,
) -> list[MemoryRecord]:
    """Build the bounded record set persisted for one completed turn."""

    user_content = user_content[:MAX_MEMORY_CONTENT_CHARS]
    assistant_content = assistant_content[:MAX_MEMORY_CONTENT_CHARS]
    summary = (
        f"用户问：{user_content}\n回答：{assistant_content}"
    )[:MAX_MEMORY_SUMMARY_CHARS]
    records = [
        MemoryRecord.create(
            session_id=session_id,
            turn_id=turn_id,
            role="user",
            content=user_content,
            memory_type="short_term",
            created_at=created_at,
            expires_at=expires_at,
        ),
        MemoryRecord.create(
            session_id=session_id,
            turn_id=turn_id,
            role="assistant",
            content=assistant_content,
            memory_type="short_term",
            created_at=created_at,
            expires_at=expires_at,
        ),
        MemoryRecord.create(
            session_id=session_id,
            turn_id=turn_id,
            role="summary",
            content=summary,
            summary=summary,
            memory_type="session_summary",
            created_at=created_at,
            expires_at=expires_at,
        ),
    ]
    if remembered_statement:
        statement = remembered_statement.strip()[:MAX_MEMORY_CONTENT_CHARS]
        if statement:
            records.append(
                MemoryRecord.create(
                    session_id=session_id,
                    turn_id=turn_id,
                    role="user",
                    content=statement,
                    summary=statement[:MAX_MEMORY_SUMMARY_CHARS],
                    memory_type="task_state",
                    created_at=created_at,
                    expires_at=expires_at,
                )
            )
    return records


def _validate_turn_batch(
    records: list[MemoryRecord],
) -> tuple[str, str]:
    if not records:
        raise ValueError("Memory turn batch must be non-empty")
    if len(records) > MAX_TURN_RECORDS:
        raise ValueError("Memory turn batch exceeds the bounded record count")
    identities = {
        (record.session_id, record.turn_id) for record in records
    }
    if len(identities) != 1:
        raise ValueError("Memory turn batch must share session_id and turn_id")
    kinds = {
        (record.role, record.memory_type) for record in records
    }
    if len(kinds) != len(records):
        raise ValueError("Memory turn batch contains duplicate record kinds")
    return next(iter(identities))


def _validate_search(
    query: str,
    *,
    session_id: str,
    top_k: int,
    memory_types: tuple[MemoryType, ...],
) -> tuple[MemoryType, ...]:
    if not query.strip():
        raise ValueError("Memory search query must be non-empty")
    if len(query) > MAX_MEMORY_CONTENT_CHARS:
        raise ValueError("Memory search query exceeds the bounded limit")
    validate_identifier(session_id, field_name="session_id")
    _validate_limit(top_k, maximum=MAX_MEMORY_TOP_K)
    if not memory_types or any(
        item not in MEMORY_TYPES for item in memory_types
    ):
        raise ValueError("Memory search types are invalid")
    return tuple(dict.fromkeys(memory_types))


def _validate_limit(value: int, *, maximum: int) -> None:
    if value <= 0 or value > maximum:
        raise ValueError(f"limit must be between 1 and {maximum}")


def _quoted(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _memory_filter(
    session_id: str,
    *,
    now_ms: int,
    memory_types: tuple[MemoryType, ...] | None = None,
) -> str:
    clauses = [
        f"session_id == {_quoted(session_id)}",
        f"(expires_at is null or expires_at > {now_ms})",
    ]
    if memory_types is not None:
        clauses.append(
            "memory_type in "
            + json.dumps(list(memory_types), ensure_ascii=False)
        )
    return " and ".join(clauses)


def _record_for_milvus(record: MemoryRecord) -> dict[str, Any]:
    data = record.to_dict()
    if data["summary"] is None:
        data.pop("summary")
    if data["expires_at"] is None:
        data.pop("expires_at")
    return data


def _record_from_milvus(data: dict[str, Any]) -> MemoryRecord:
    try:
        role = str(data["role"])
        memory_type = str(data["memory_type"])
        if role not in MEMORY_ROLES or memory_type not in MEMORY_TYPES:
            raise ValueError("Milvus Memory row contains an invalid enum")
        return MemoryRecord(
            session_id=str(data["session_id"]),
            turn_id=str(data["turn_id"]),
            role=cast(MemoryRole, role),
            content=str(data["content"]),
            summary=(
                None
                if data.get("summary") is None
                else str(data["summary"])
            ),
            memory_type=cast(MemoryType, memory_type),
            created_at=int(data["created_at"]),
            expires_at=(
                None
                if data.get("expires_at") is None
                else int(data["expires_at"])
            ),
            metadata=(
                dict(data["metadata"])
                if isinstance(data.get("metadata"), dict)
                else {}
            ),
            content_vector=[
                float(value) for value in data["content_vector"]
            ],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MemoryStoreError("Milvus returned an invalid Memory row") from exc


def _first_memory_hits(raw_hits: Any) -> list[dict[str, Any]]:
    if not raw_hits:
        return []
    first = raw_hits[0]
    return list(first) if isinstance(first, list) else list(raw_hits)


def _memory_hit_entity(hit: dict[str, Any]) -> dict[str, Any]:
    entity = hit.get("entity")
    if isinstance(entity, dict):
        return entity
    return hit


def _mutation_count(result: Any, field: str) -> int | None:
    if isinstance(result, dict):
        value = result.get(field)
        return int(value) if value is not None else None
    value = getattr(result, field, None)
    return int(value) if value is not None else None


def _require_memory_embedding_fingerprint(record: MemoryRecord) -> None:
    expected = text_embedding_fingerprint()
    actual = record.metadata.get(EMBEDDING_FINGERPRINT_KEY)
    if actual != expected:
        raise MemoryStoreError(
            "Memory embedding fingerprint does not match configuration"
        )


def _validate_returned_records(
    records: list[MemoryRecord],
    *,
    session_id: str,
    now_ms: int,
    memory_types: tuple[MemoryType, ...] | None = None,
) -> None:
    """Fail closed when Milvus violates the mandatory Memory filter."""

    for record in records:
        _require_memory_embedding_fingerprint(record)
        if (
            record.session_id != session_id
            or (
                record.expires_at is not None
                and record.expires_at <= now_ms
            )
            or (
                memory_types is not None
                and record.memory_type not in memory_types
            )
        ):
            raise MemoryStoreError(
                "Milvus returned Memory outside the requested scope"
            )


def _memory_order_key(
    record: MemoryRecord,
) -> tuple[int, str, str, str]:
    return (
        record.created_at,
        record.turn_id,
        record.role,
        record.memory_type,
    )
