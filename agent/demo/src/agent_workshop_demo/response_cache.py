"""Session-scoped cache for citation-validated grounded responses."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol

from agent_workshop_demo.config import VECTOR_DIMS
from agent_workshop_demo.embedding import (
    cosine_similarity,
    dense_vector,
    text_embedding_fingerprint,
)
from agent_workshop_demo.milvus_time import (
    encode_expiry,
    epoch_ms_from_milvus,
    timestamp_literal,
)
from agent_workshop_demo.validation import validate_identifier

DEFAULT_RESPONSE_CACHE_TTL_SECONDS = 259_200
DEFAULT_RESPONSE_CACHE_TOP_K = 3
DEFAULT_RESPONSE_CACHE_SIMILARITY_THRESHOLD = 0.92
DEFAULT_KB_REVISION = "demo-v1"
RESPONSE_CACHE_WORKFLOW_VERSION = "grounded-response-cache-v1"
MAX_CACHE_ANSWER_CHARS = 12_000
MAX_CACHE_QUERY_CHARS = 8_192
MAX_CACHE_CITATIONS = 16
MAX_CACHE_JSON_BYTES = 32_768
VERSION_CONSTRAINT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])v?\d+(?:\.\d+)+(?![A-Za-z0-9])",
    re.IGNORECASE,
)
NEGATION_MARKERS = (
    "不支持",
    "不包含",
    "没有",
    "删除",
    "缺少",
    "unsupported",
    "not support",
    "doesn't",
    "removed",
    "missing",
)
CacheMatchType = Literal["exact", "semantic"]


class ResponseCacheError(RuntimeError):
    """Sanitized cache dependency or record failure."""


@dataclass(frozen=True)
class CachedEvidence:
    """One cited KB chunk identity used to validate cache freshness."""

    chunk_id: str
    doc_id: str
    doc_version: str
    checksum: str
    is_current: bool

    def __post_init__(self) -> None:
        for field_name in ("chunk_id", "doc_id", "doc_version", "checksum"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Cached evidence {field_name} must be non-empty"
                )


@dataclass(frozen=True)
class GroundedResponseCacheRecord:
    """Complete validated response and the scope required for safe reuse."""

    cache_id: str
    session_id: str
    source_query_id: str
    normalized_query: str
    query_hash: str
    query_vector: list[float]
    embedding_fingerprint: str
    intent: str
    query_type: str
    retrieval_goal: str
    version_scope: dict[str, Any]
    entity_ids: list[str]
    query_constraints: list[str]
    permission_scope_hash: str
    kb_revision: str
    workflow_version: str
    answer: str
    citations: list[dict[str, Any]]
    evidence: list[CachedEvidence]
    created_at: int
    expires_at: int

    def __post_init__(self) -> None:
        validate_identifier(self.cache_id, field_name="cache_id")
        validate_identifier(self.session_id, field_name="session_id")
        validate_identifier(
            self.source_query_id,
            field_name="source_query_id",
        )
        if (
            not self.normalized_query
            or len(self.normalized_query) > MAX_CACHE_QUERY_CHARS
        ):
            raise ValueError("Cached query is empty or too long")
        if not re.fullmatch(r"[0-9a-f]{64}", self.query_hash):
            raise ValueError("Cached query hash must be SHA-256 hex")
        if (
            len(self.query_vector) != VECTOR_DIMS["TEXT_DIM"]
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in self.query_vector
            )
        ):
            raise ValueError("Cached query vector is invalid")
        for value, field_name in (
            (self.embedding_fingerprint, "embedding_fingerprint"),
            (self.intent, "intent"),
            (self.query_type, "query_type"),
            (self.retrieval_goal, "retrieval_goal"),
            (self.permission_scope_hash, "permission_scope_hash"),
            (self.kb_revision, "kb_revision"),
            (self.workflow_version, "workflow_version"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Cached {field_name} must be non-empty")
        if not self.answer or len(self.answer) > MAX_CACHE_ANSWER_CHARS:
            raise ValueError("Cached answer is empty or too long")
        if not 1 <= len(self.citations) <= MAX_CACHE_CITATIONS:
            raise ValueError("Cached citations are outside the allowed bound")
        if not 1 <= len(self.evidence) <= MAX_CACHE_CITATIONS:
            raise ValueError("Cached evidence is outside the allowed bound")
        evidence_by_id = {item.chunk_id: item for item in self.evidence}
        if len(evidence_by_id) != len(self.evidence):
            raise ValueError("Cached evidence chunk IDs must be unique")
        citation_ids: set[str] = set()
        cited_chunk_ids: set[str] = set()
        for citation in self.citations:
            if not isinstance(citation, dict):
                raise ValueError("Cached citation must be an object")
            citation_id = citation.get("citation_id")
            chunk_id = citation.get("chunk_id")
            doc_id = citation.get("doc_id")
            doc_version = citation.get("doc_version")
            if (
                not isinstance(citation_id, str)
                or re.fullmatch(r"C\d+", citation_id) is None
                or not isinstance(chunk_id, str)
                or not chunk_id
                or not isinstance(doc_id, str)
                or not doc_id
                or not isinstance(doc_version, str)
                or not doc_version
            ):
                raise ValueError("Cached citation identity is invalid")
            snapshot = evidence_by_id.get(chunk_id)
            if (
                snapshot is None
                or snapshot.doc_id != doc_id
                or snapshot.doc_version != doc_version
            ):
                raise ValueError(
                    "Cached citation does not match evidence identity"
                )
            citation_ids.add(citation_id)
            cited_chunk_ids.add(chunk_id)
        answer_markers = set(re.findall(r"\[(C\d+)\]", self.answer))
        if (
            len(citation_ids) != len(self.citations)
            or len(cited_chunk_ids) != len(self.citations)
            or cited_chunk_ids != set(evidence_by_id)
            or answer_markers != citation_ids
        ):
            raise ValueError("Cached citation markers are inconsistent")
        if self.created_at < 0 or self.expires_at <= self.created_at:
            raise ValueError("Cached lifecycle timestamps are invalid")
        if any(
            not isinstance(item, str) or not item.strip()
            for item in self.entity_ids + self.query_constraints
        ):
            raise ValueError("Cached entity/constraint values are invalid")
        for value, field_name in (
            (self.version_scope, "version_scope"),
            (self.citations, "citations"),
            ([asdict(item) for item in self.evidence], "evidence"),
            (self.entity_ids, "entity_ids"),
            (self.query_constraints, "query_constraints"),
        ):
            _validate_json_bound(value, field_name=field_name)

    def to_storage_dict(self) -> dict[str, Any]:
        """Serialize the Milvus row without changing JSON field shapes."""

        data = asdict(self)
        return data


@dataclass(frozen=True)
class ResponseCacheCandidate:
    """One private recalled response with query-match metadata."""

    record: GroundedResponseCacheRecord
    similarity: float
    match_type: CacheMatchType

    def __post_init__(self) -> None:
        if not math.isfinite(self.similarity) or not 0 <= self.similarity <= 1:
            raise ValueError("Cache candidate similarity must be in [0, 1]")


class GroundedResponseCache(Protocol):
    """Storage contract shared by local and Milvus cache implementations."""

    def search(
        self,
        query: str,
        *,
        session_id: str,
        top_k: int,
        now_ms: int,
    ) -> list[ResponseCacheCandidate]: ...

    def upsert(self, record: GroundedResponseCacheRecord) -> int: ...

    def delete_session(self, session_id: str) -> int: ...


class GroundedResponseCacheStore:
    """Deterministic in-process response cache."""

    def __init__(self, now_ms: int | None = None) -> None:
        self.now_ms = utc_now_ms() if now_ms is None else now_ms
        if self.now_ms < 0:
            raise ValueError("now_ms cannot be negative")
        self.records: list[GroundedResponseCacheRecord] = []

    def search(
        self,
        query: str,
        *,
        session_id: str,
        top_k: int,
        now_ms: int | None = None,
    ) -> list[ResponseCacheCandidate]:
        """Return live same-session exact hit or semantic candidates."""

        normalized, digest = normalize_query(query), query_hash(query)
        _validate_search(session_id, top_k=top_k)
        current = self.now_ms if now_ms is None else now_ms
        live = [
            record
            for record in self.records
            if record.session_id == session_id and record.expires_at > current
        ]
        exact = sorted(
            (record for record in live if record.query_hash == digest),
            key=lambda item: item.created_at,
            reverse=True,
        )
        if exact:
            return [
                ResponseCacheCandidate(
                    record=exact[0],
                    similarity=1.0,
                    match_type="exact",
                )
            ]
        query_vector = dense_vector(normalized)
        candidates = [
            ResponseCacheCandidate(
                record=record,
                similarity=max(
                    0.0,
                    min(
                        1.0,
                        cosine_similarity(query_vector, record.query_vector),
                    ),
                ),
                match_type="semantic",
            )
            for record in live
        ]
        return sorted(
            candidates,
            key=lambda item: (
                item.similarity,
                item.record.created_at,
                item.record.cache_id,
            ),
            reverse=True,
        )[:top_k]

    def upsert(self, record: GroundedResponseCacheRecord) -> int:
        """Replace one same-session normalized-query cache identity."""

        self.records = [
            item
            for item in self.records
            if not (
                item.session_id == record.session_id
                and item.query_hash == record.query_hash
            )
        ]
        self.records.append(record)
        return 1

    def delete_session(self, session_id: str) -> int:
        validate_identifier(session_id, field_name="session_id")
        before = len(self.records)
        self.records = [
            record
            for record in self.records
            if record.session_id != session_id
        ]
        return before - len(self.records)


class MilvusGroundedResponseCacheStore:
    """Milvus-backed exact and semantic grounded-response cache."""

    OUTPUT_FIELDS = [
        "cache_id",
        "session_id",
        "source_query_id",
        "normalized_query",
        "query_hash",
        "embedding_fingerprint",
        "intent",
        "query_type",
        "retrieval_goal",
        "version_scope",
        "entity_ids",
        "query_constraints",
        "permission_scope_hash",
        "kb_revision",
        "workflow_version",
        "answer",
        "citations",
        "evidence",
        "created_at",
        "expires_at",
    ]

    def __init__(
        self,
        client: Any,
        *,
        collection_name: str = "grounded_response_cache",
    ) -> None:
        if not collection_name.strip():
            raise ValueError("response cache collection name is required")
        self.client = client
        self.collection_name = collection_name

    def ensure_collection_ready(self) -> None:
        if not self.client.has_collection(
            collection_name=self.collection_name
        ):
            raise RuntimeError(
                f"Milvus collection {self.collection_name!r} does not exist; "
                "run demo/scripts/create_collections.py first."
            )
        self.client.load_collection(collection_name=self.collection_name)

    def search(
        self,
        query: str,
        *,
        session_id: str,
        top_k: int,
        now_ms: int,
    ) -> list[ResponseCacheCandidate]:
        """Prefer an exact hash row, otherwise search live session vectors."""

        normalized = normalize_query(query)
        digest = query_hash(query)
        _validate_search(session_id, top_k=top_k)
        expression = (
            f"session_id == {json.dumps(session_id)} and "
            f"expires_at > {timestamp_literal(now_ms)}"
        )
        exact_expression = (
            f"{expression} and query_hash == {json.dumps(digest)}"
        )
        try:
            exact_rows = self.client.query(
                collection_name=self.collection_name,
                filter=exact_expression,
                output_fields=self.OUTPUT_FIELDS,
                limit=1,
            )
            if exact_rows:
                record = _record_from_storage(exact_rows[0])
                _validate_returned_record(
                    record,
                    session_id=session_id,
                    now_ms=now_ms,
                )
                return [
                    ResponseCacheCandidate(record, 1.0, "exact")
                ]
            hits = self.client.search(
                collection_name=self.collection_name,
                data=[dense_vector(normalized)],
                anns_field="query_vector",
                search_params={
                    "metric_type": "COSINE",
                    "params": {"ef": 32},
                },
                filter=expression,
                limit=top_k,
                output_fields=self.OUTPUT_FIELDS,
            )
        except Exception as exc:
            raise ResponseCacheError(
                "Unable to recall grounded response cache"
            ) from exc
        candidates: list[ResponseCacheCandidate] = []
        for hit in _first_hits(hits)[:top_k]:
            entity = hit.get("entity", hit)
            if not isinstance(entity, dict):
                raise ResponseCacheError(
                    "Grounded response cache returned an invalid row"
                )
            record = _record_from_storage(entity)
            _validate_returned_record(
                record,
                session_id=session_id,
                now_ms=now_ms,
            )
            similarity = float(
                hit.get("distance", hit.get("score", 0.0))
            )
            candidates.append(
                ResponseCacheCandidate(
                    record=record,
                    similarity=max(0.0, min(1.0, similarity)),
                    match_type="semantic",
                )
            )
        return candidates

    def upsert(self, record: GroundedResponseCacheRecord) -> int:
        expression = (
            f"session_id == {json.dumps(record.session_id)} and "
            f"query_hash == {json.dumps(record.query_hash)}"
        )
        try:
            self.client.delete(
                collection_name=self.collection_name,
                filter=expression,
            )
            result = self.client.insert(
                collection_name=self.collection_name,
                data=[encode_expiry(record.to_storage_dict())],
            )
            inserted = _mutation_count(result, "insert_count")
            if inserted is not None and inserted != 1:
                raise ResponseCacheError(
                    "Milvus reported an incomplete response cache insert"
                )
            self.client.flush(collection_name=self.collection_name)
        except ResponseCacheError:
            raise
        except Exception as exc:
            raise ResponseCacheError(
                "Unable to persist grounded response cache"
            ) from exc
        return 1

    def delete_session(self, session_id: str) -> int:
        validate_identifier(session_id, field_name="session_id")
        try:
            result = self.client.delete(
                collection_name=self.collection_name,
                filter=f"session_id == {json.dumps(session_id)}",
            )
        except Exception as exc:
            raise ResponseCacheError(
                "Unable to delete grounded response cache"
            ) from exc
        return _mutation_count(result, "delete_count") or 0


def build_cache_record(
    *,
    session_id: str,
    source_query_id: str,
    user_query: str,
    intent: str,
    query_type: str,
    retrieval_goal: str,
    version_scope: dict[str, Any],
    entity_ids: list[str],
    permission_scope_hash_value: str,
    kb_revision: str,
    answer: str,
    citations: list[dict[str, Any]],
    evidence: list[CachedEvidence],
    created_at: int,
    expires_at: int,
) -> GroundedResponseCacheRecord:
    """Build a validated record using the configured embedding space."""

    normalized = normalize_query(user_query)
    digest = query_hash(user_query)
    cache_digest = hashlib.sha256(
        f"{session_id}\0{digest}".encode("utf-8")
    ).hexdigest()
    return GroundedResponseCacheRecord(
        cache_id=f"cache_{cache_digest}",
        session_id=session_id,
        source_query_id=source_query_id,
        normalized_query=normalized,
        query_hash=digest,
        query_vector=dense_vector(normalized),
        embedding_fingerprint=text_embedding_fingerprint(),
        intent=intent,
        query_type=query_type,
        retrieval_goal=retrieval_goal,
        version_scope=dict(version_scope),
        entity_ids=sorted(dict.fromkeys(entity_ids)),
        query_constraints=query_constraints(user_query),
        permission_scope_hash=permission_scope_hash_value,
        kb_revision=kb_revision,
        workflow_version=RESPONSE_CACHE_WORKFLOW_VERSION,
        answer=answer,
        citations=[dict(item) for item in citations],
        evidence=list(evidence),
        created_at=created_at,
        expires_at=expires_at,
    )


def normalize_query(query: str) -> str:
    """Normalize only stable surface differences for exact cache identity."""

    normalized = unicodedata.normalize("NFKC", query).casefold()
    value = " ".join(normalized.split())
    if not value or len(value) > MAX_CACHE_QUERY_CHARS:
        raise ValueError("Response cache query is empty or too long")
    return value


def query_hash(query: str) -> str:
    return hashlib.sha256(normalize_query(query).encode("utf-8")).hexdigest()


def query_constraints(query: str) -> list[str]:
    """Extract conservative constraints semantic similarity cannot override."""

    normalized = normalize_query(query)
    constraints = {
        f"version:{match.group(0).lower()}"
        for match in VERSION_CONSTRAINT_PATTERN.finditer(normalized)
    }
    if any(marker in normalized for marker in NEGATION_MARKERS):
        constraints.add("polarity:negative")
    else:
        constraints.add("polarity:positive")
    return sorted(constraints)


def permission_scope_hash(permission_decision: dict[str, Any]) -> str:
    """Hash the current checker and allowed departments without exposing them."""

    payload = {
        "checker": str(permission_decision.get("checker", "")),
        "allowed": bool(permission_decision.get("allowed", False)),
        "departments": sorted(
            str(item)
            for item in permission_decision.get(
                "allowed_departments",
                [],
            )
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def utc_now_ms() -> int:
    return int(time.time() * 1000)


def _record_from_storage(data: dict[str, Any]) -> GroundedResponseCacheRecord:
    raw_evidence = data.get("evidence")
    raw_citations = data.get("citations")
    raw_entity_ids = data.get("entity_ids")
    raw_constraints = data.get("query_constraints")
    raw_version_scope = data.get("version_scope")
    if (
        not isinstance(raw_evidence, list)
        or not all(isinstance(item, dict) for item in raw_evidence)
        or not isinstance(raw_citations, list)
        or not all(isinstance(item, dict) for item in raw_citations)
        or not isinstance(raw_entity_ids, list)
        or not all(isinstance(item, str) for item in raw_entity_ids)
        or not isinstance(raw_constraints, list)
        or not all(isinstance(item, str) for item in raw_constraints)
        or not isinstance(raw_version_scope, dict)
    ):
        raise ResponseCacheError(
            "Grounded response cache record shape is invalid"
        )
    try:
        evidence = [
            CachedEvidence(
                chunk_id=_required_string(item, "chunk_id"),
                doc_id=_required_string(item, "doc_id"),
                doc_version=_required_string(item, "doc_version"),
                checksum=_required_string(item, "checksum"),
                is_current=_required_bool(item, "is_current"),
            )
            for item in raw_evidence
        ]
        return GroundedResponseCacheRecord(
            cache_id=_required_string(data, "cache_id"),
            session_id=_required_string(data, "session_id"),
            source_query_id=_required_string(data, "source_query_id"),
            normalized_query=_required_string(data, "normalized_query"),
            query_hash=_required_string(data, "query_hash"),
            query_vector=[0.0] * VECTOR_DIMS["TEXT_DIM"],
            embedding_fingerprint=_required_string(
                data,
                "embedding_fingerprint",
            ),
            intent=_required_string(data, "intent"),
            query_type=_required_string(data, "query_type"),
            retrieval_goal=_required_string(data, "retrieval_goal"),
            version_scope=dict(raw_version_scope),
            entity_ids=[str(item) for item in raw_entity_ids],
            query_constraints=[
                str(item) for item in raw_constraints
            ],
            permission_scope_hash=_required_string(
                data,
                "permission_scope_hash",
            ),
            kb_revision=_required_string(data, "kb_revision"),
            workflow_version=_required_string(data, "workflow_version"),
            answer=_required_string(data, "answer"),
            citations=[dict(item) for item in raw_citations],
            evidence=evidence,
            created_at=int(data["created_at"]),
            expires_at=epoch_ms_from_milvus(data["expires_at"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ResponseCacheError(
            "Grounded response cache returned an invalid record"
        ) from exc


def _required_string(data: dict[str, Any], field_name: str) -> str:
    value = data[field_name]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _required_bool(data: dict[str, Any], field_name: str) -> bool:
    value = data[field_name]
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _validate_returned_record(
    record: GroundedResponseCacheRecord,
    *,
    session_id: str,
    now_ms: int,
) -> None:
    if record.session_id != session_id or record.expires_at <= now_ms:
        raise ResponseCacheError(
            "Grounded response cache returned a record outside scope"
        )


def _validate_search(session_id: str, *, top_k: int) -> None:
    validate_identifier(session_id, field_name="session_id")
    if not 1 <= top_k <= 20:
        raise ValueError("Response cache top_k must be between 1 and 20")


def _validate_json_bound(value: Any, *, field_name: str) -> None:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cached {field_name} must be JSON") from exc
    if len(encoded) > MAX_CACHE_JSON_BYTES:
        raise ValueError(f"Cached {field_name} exceeds its size bound")


def _first_hits(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        return []
    first = raw[0]
    return [item for item in first if isinstance(item, dict)]


def _mutation_count(result: Any, field: str) -> int | None:
    if isinstance(result, dict) and field in result:
        return int(result[field])
    value = getattr(result, field, None)
    return None if value is None else int(value)
