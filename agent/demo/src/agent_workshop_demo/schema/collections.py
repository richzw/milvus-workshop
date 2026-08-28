"""Provisional Milvus collection and index definitions."""

from __future__ import annotations

from typing import Any

from agent_workshop_demo.config import (
    COLLECTION_NAMES as COLLECTION_NAMES,
)
from agent_workshop_demo.config import VECTOR_DIMS


def varchar(
    name: str,
    max_length: int,
    nullable: bool,
    description: str,
) -> dict[str, Any]:
    """Build one VarChar field definition."""

    return {
        "name": name,
        "type": "VarChar",
        "max_length": max_length,
        "nullable": nullable,
        "description": description,
    }


RETRIEVAL_ANALYZER_PARAMS: dict[str, Any] = {
    "tokenizer": "standard",
    "filter": [
        "lowercase",
        {
            "type": "synonym",
            # Milvus/Tantivy requires escaped spaces inside multi-word terms.
            "synonyms": [
                r"object\ storage, s3, minio",
                r"vector\ database, vector\ db",
                r"full\ text, bm25",
            ],
            "expand": True,
        },
    ],
}


KB_CHUNKS_COLLECTION: dict[str, Any] = {
    "collection_name": COLLECTION_NAMES["kb_chunks"],
    "description": "Enterprise knowledge chunks for Agentic RAG demo",
    "fields": [
        {
            "name": "id",
            "type": "Int64",
            "primary_key": True,
            "auto_id": True,
        },
        varchar("doc_id", 128, False, "Stable document ID."),
        varchar("chunk_id", 128, False, "Stable citation ID."),
        varchar("parent_id", 128, True, "Parent chunk ID."),
        varchar("record_type", 32, False, "text_chunk/pdf_page/image."),
        varchar("source_type", 32, False, "local/s3/mfs."),
        varchar("source_uri", 1024, False, "Display-safe source URI."),
        varchar("bucket", 128, True, "MinIO bucket."),
        varchar("object_key", 1024, True, "MinIO object key."),
        varchar("doc_type", 32, False, "markdown/pdf/text/image."),
        varchar("title", 512, False, "Document title."),
        varchar("section", 512, True, "Logical section."),
        {"name": "page_no", "type": "Int32", "nullable": True},
        {"name": "chunk_index", "type": "Int32", "nullable": False},
        varchar("text", 16384, False, "Retrieval and citation text."),
        varchar("retrieval_text", 32768, False, "BM25 Function input.")
        | {
            "enable_analyzer": True,
            "analyzer_params": RETRIEVAL_ANALYZER_PARAMS,
        },
        varchar("text_summary", 2048, True, "Bounded UI snippet."),
        varchar("language", 16, False, "zh/en/mixed."),
        varchar("department", 64, False, "Metadata filter value."),
        {"name": "updated_at", "type": "Int64", "nullable": False},
        {"name": "created_at", "type": "Int64", "nullable": True},
        {"name": "priority", "type": "Int32", "nullable": False},
        varchar("doc_version", 64, False, "Opaque document edition."),
        {
            "name": "is_current",
            "type": "Bool",
            "nullable": False,
            "description": "Current edition marker.",
        },
        varchar("checksum", 128, True, "Content checksum."),
        {"name": "metadata", "type": "JSON", "nullable": True},
        {
            "name": "has_image_vector",
            "type": "Bool",
            "nullable": False,
        },
        {
            "name": "text_vector",
            "type": "FloatVector",
            "dim": VECTOR_DIMS["TEXT_DIM"],
            "nullable": False,
        },
        {
            "name": "sparse_vector",
            "type": "SparseFloatVector",
            "nullable": False,
        },
        {
            "name": "image_vector",
            "type": "FloatVector",
            "dim": VECTOR_DIMS["IMAGE_DIM"],
            "nullable": True,
        },
    ],
    "functions": [
        {
            "name": "bm25_function",
            "type": "BM25",
            "input_fields": ["retrieval_text"],
            "output_fields": ["sparse_vector"],
            "params": {},
        }
    ],
}

KB_ENUMS = {
    "record_type": ["text_chunk", "pdf_page", "image", "table"],
    "source_type": ["local", "s3", "mfs"],
    "doc_type": ["markdown", "pdf", "text", "image", "table"],
    "department": ["engineering", "product", "hr", "security", "general"],
    "language": ["zh", "en", "mixed"],
}

KB_CHUNKS_INDEXES = {
    "text_vector": {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200},
    },
    "sparse_vector": {
        "index_type": "SPARSE_INVERTED_INDEX",
        "metric_type": "BM25",
        "params": {},
    },
    "image_vector": {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200},
        "note": "P2 and provisional until Phase 0 verification.",
    },
    "scalar_indexes": [
        "doc_id",
        "chunk_id",
        "record_type",
        "source_type",
        "doc_type",
        "department",
        "updated_at",
        "priority",
        "doc_version",
        "is_current",
        "has_image_vector",
    ],
}

KB_CHUNKS_SEARCH_DEFAULTS = {
    "search_mode": "hybrid",
    "dense_search": {
        "field": "text_vector",
        "top_k": 20,
        "metric_type": "COSINE",
        "params": {"ef": 64},
    },
    "sparse_search": {
        "field": "sparse_vector",
        "top_k": 20,
        "metric_type": "BM25",
    },
    "hybrid": {
        "milvus_top_k": 20,
        "reranker_top_k": 8,
        "answer_context_top_k": 5,
    },
    "filters": {
        "source_type": ["local", "s3"],
        "doc_type": ["markdown", "pdf", "text", "image"],
        "is_current": True,
    },
    "order_by": ["updated_at desc", "priority desc"],
    "order_mode": "relevance",
}

KB_DOCUMENTS_COLLECTION: dict[str, Any] = {
    "collection_name": COLLECTION_NAMES["kb_documents"],
    "description": "Derived StructArray document/passage retrieval projection",
    "fields": [
        varchar("document_key", 69, False, "Stable document-version key.")
        | {"primary_key": True, "auto_id": False},
        varchar("doc_id", 128, False, "Stable document family ID."),
        varchar("doc_version", 64, False, "Opaque document edition."),
        varchar("source_type", 32, False, "Parent source type."),
        varchar("source_uri", 1024, False, "Display-safe parent URI."),
        varchar("doc_type", 32, False, "Parent document type."),
        varchar("title", 512, False, "Parent document title."),
        varchar("department", 64, False, "Permission/filter boundary."),
        {"name": "is_current", "type": "Bool", "nullable": False},
        {"name": "updated_at", "type": "Int64", "nullable": False},
        {"name": "priority", "type": "Int32", "nullable": False},
        varchar(
            "text_embedding_fingerprint",
            256,
            False,
            "Dense vector-space identity.",
        ),
        varchar(
            "projection_fingerprint",
            64,
            False,
            "Full-build projection identity.",
        ),
        {"name": "projection_parent_count", "type": "Int32", "nullable": False},
        {"name": "projection_passage_count", "type": "Int32", "nullable": False},
        {"name": "passage_count", "type": "Int32", "nullable": False},
        {
            "name": "passages",
            "type": "Array",
            "element_type": "Struct",
            "max_capacity": 1024,
            "nullable": False,
            "struct_fields": [
                varchar("chunk_id", 512, False, "Stable citation identity."),
                varchar("checksum", 128, False, "Authoritative content checksum."),
                {"name": "chunk_index", "type": "Int32"},
                {"name": "page_no", "type": "Int32"},
                varchar("section", 2048, False, "Passage section or empty sentinel."),
                varchar("record_type", 64, False, "Passage record type."),
                varchar("language", 32, False, "Passage language."),
                {
                    "name": "embedding_list_vector",
                    "type": "FloatVector",
                    "dim": VECTOR_DIMS["TEXT_DIM"],
                },
                {
                    "name": "element_vector",
                    "type": "FloatVector",
                    "dim": VECTOR_DIMS["TEXT_DIM"],
                },
            ],
        },
    ],
}

KB_DOCUMENTS_INDEXES: dict[str, Any] = {
    "passages[embedding_list_vector]": {
        "index_name": "passages_embedding_list_maxsim_idx",
        "index_type": "HNSW",
        "metric_type": "MAX_SIM_COSINE",
        "params": {
            "M": 16,
            "efConstruction": 200,
            "emb_list_strategy": "tokenann",
        },
    },
    "passages[element_vector]": {
        "index_name": "passages_element_cosine_idx",
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200},
    },
    "scalar_indexes": [
        "doc_id",
        "doc_version",
        "source_type",
        "doc_type",
        "department",
        "is_current",
        "projection_fingerprint",
    ],
}

CONVERSATION_MEMORY_COLLECTION: dict[str, Any] = {
    "collection_name": COLLECTION_NAMES["conversation_memory"],
    "description": "P2 conversation memory experiment",
    "fields": [
        {
            "name": "id",
            "type": "Int64",
            "primary_key": True,
            "auto_id": True,
        },
        varchar("session_id", 128, False, "Session ID."),
        varchar("turn_id", 128, False, "Turn ID."),
        varchar("role", 32, False, "Message role."),
        varchar("content", 8192, False, "Memory content."),
        varchar("summary", 2048, True, "Optional summary."),
        varchar("memory_type", 32, False, "Memory type."),
        {"name": "created_at", "type": "Int64", "nullable": False},
        {"name": "expires_at", "type": "TIMESTAMPTZ", "nullable": True},
        {"name": "metadata", "type": "JSON", "nullable": True},
        {
            "name": "content_vector",
            "type": "FloatVector",
            "dim": VECTOR_DIMS["TEXT_DIM"],
            "nullable": False,
        },
    ],
    "properties": {"ttl_field": "expires_at"},
}

CONVERSATION_MEMORY_ENUMS = {
    "role": ["user", "assistant", "system", "summary"],
    "memory_type": ["short_term", "session_summary", "task_state"],
}

CONVERSATION_MEMORY_INDEXES = {
    "content_vector": {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200},
    },
    "scalar_indexes": [
        "session_id",
        "turn_id",
        "role",
        "memory_type",
        "created_at",
        {"field_name": "expires_at", "index_type": "STL_SORT"},
    ],
}

MEMORY_EVENTS_COLLECTION: dict[str, Any] = {
    "collection_name": COLLECTION_NAMES["memory_events"],
    "description": "Append-only selective-Memory episode lineage",
    "fields": [
        varchar("event_id", 128, False, "Stable immutable event identity.")
        | {"primary_key": True, "auto_id": False},
        varchar("session_id", 128, False, "Session scope."),
        varchar("query_id", 128, True, "Producing query identity."),
        varchar("turn_id", 128, True, "Producing turn identity."),
        varchar("parent_event_id", 128, True, "Parent lineage event."),
        varchar("branch_id", 128, False, "Event branch; main in M4."),
        varchar("event_type", 64, False, "Registered event type."),
        varchar("content", 8192, False, "Bounded event payload."),
        varchar("summary", 2048, True, "Bounded outcome summary."),
        varchar("outcome", 128, True, "Terminal outcome code."),
        {"name": "event_time", "type": "Int64", "nullable": False},
        {"name": "expires_at", "type": "TIMESTAMPTZ", "nullable": True},
        {"name": "salience_score", "type": "Float", "nullable": False},
        {
            "name": "selection_reason",
            "type": "JSON",
            "nullable": False,
        },
        varchar(
            "retention_class",
            32,
            False,
            "ephemeral/candidate/protected.",
        ),
        varchar("decay_profile", 64, False, "Registered profile name."),
        varchar(
            "selector_name",
            64,
            False,
            "Rule/model selector implementation.",
        ),
        varchar(
            "selector_model",
            120,
            True,
            "Optional bounded selector model name.",
        ),
        varchar(
            "selector_fallback_reason",
            64,
            True,
            "Sanitized selector fallback reason.",
        ),
        varchar(
            "permission_scope_hash",
            64,
            False,
            "Opaque scope digest.",
        ),
        varchar(
            "workflow_version",
            128,
            False,
            "Selection/consolidation contract version.",
        ),
        varchar("checksum", 64, False, "Payload SHA-256."),
        {
            "name": "content_vector",
            "type": "FloatVector",
            "dim": VECTOR_DIMS["TEXT_DIM"],
            "nullable": False,
        },
    ],
    "properties": {"ttl_field": "expires_at"},
}

MEMORY_EVENTS_INDEXES = {
    "content_vector": {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200},
    },
    "scalar_indexes": [
        "session_id",
        "query_id",
        "turn_id",
        "event_type",
        "event_time",
        {"field_name": "expires_at", "index_type": "STL_SORT"},
        "retention_class",
        "decay_profile",
        "permission_scope_hash",
    ],
}

MEMORY_FACTS_COLLECTION: dict[str, Any] = {
    "collection_name": COLLECTION_NAMES["memory_facts"],
    "description": "Versioned selective-Memory fact projection",
    "fields": [
        varchar("memory_id", 128, False, "Stable fact-revision identity.")
        | {"primary_key": True, "auto_id": False},
        varchar("session_id", 128, False, "Session scope."),
        varchar("memory_type", 64, False, "Registered fact type."),
        varchar("subject", 256, False, "Typed fact subject."),
        varchar("predicate", 256, False, "Typed fact predicate."),
        varchar("value", 8192, False, "Bounded fact value."),
        varchar("status", 32, False, "Fact projection status."),
        {"name": "confidence", "type": "Float", "nullable": False},
        {"name": "revision", "type": "Int32", "nullable": False},
        {
            "name": "source_event_ids",
            "type": "JSON",
            "nullable": False,
        },
        varchar(
            "supersedes_memory_id",
            128,
            True,
            "Previous fact revision.",
        ),
        {"name": "valid_from", "type": "Int64", "nullable": False},
        {"name": "valid_to", "type": "Int64", "nullable": True},
        {
            "name": "last_confirmed_at",
            "type": "Int64",
            "nullable": False,
        },
        {"name": "expires_at", "type": "TIMESTAMPTZ", "nullable": True},
        {"name": "salience_score", "type": "Float", "nullable": False},
        varchar(
            "permission_scope_hash",
            64,
            False,
            "Opaque scope digest.",
        ),
        {
            "name": "content_vector",
            "type": "FloatVector",
            "dim": VECTOR_DIMS["TEXT_DIM"],
            "nullable": False,
        },
    ],
    "properties": {"ttl_field": "expires_at"},
}

MEMORY_FACTS_INDEXES = {
    "content_vector": {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200},
    },
    "scalar_indexes": [
        "session_id",
        "memory_type",
        "subject",
        "predicate",
        "status",
        "revision",
        "last_confirmed_at",
        {"field_name": "expires_at", "index_type": "STL_SORT"},
        "permission_scope_hash",
    ],
}

MEMORY_CONSOLIDATION_JOURNAL_COLLECTION: dict[str, Any] = {
    "collection_name": COLLECTION_NAMES["memory_consolidation_journal"],
    "description": "Recoverable selective-Memory consolidation outbox",
    "fields": [
        varchar("operation_id", 128, False, "Stable consolidation identity.")
        | {"primary_key": True, "auto_id": False},
        varchar("session_id", 128, False, "Session scope."),
        varchar("trigger_event_id", 128, False, "Trigger event identity."),
        {"name": "source_event_ids", "type": "JSON", "nullable": False},
        {"name": "plan_metadata", "type": "JSON", "nullable": False},
        {"name": "fact_update_0", "type": "JSON", "nullable": False},
        {"name": "fact_update_1", "type": "JSON", "nullable": True},
        {"name": "fact_update_count", "type": "Int32", "nullable": False},
        varchar("fact_vector_0", 12000, False, "Base64 IEEE-754 float64 vector."),
        varchar("fact_vector_1", 12000, False, "Base64 IEEE-754 float64 vector."),
        {"name": "lifecycle_event", "type": "JSON", "nullable": False},
        varchar(
            "lifecycle_vector",
            12000,
            False,
            "Base64 IEEE-754 float64 vector.",
        ),
        {
            "name": "journal_anchor_vector",
            "type": "FloatVector",
            "dim": 2,
            "nullable": False,
            "description": (
                "Non-semantic Milvus collection anchor; never used for recall."
            ),
        },
        varchar("status", 16, False, "pending/applied."),
        {"name": "attempts", "type": "Int32", "nullable": False},
        {"name": "created_at", "type": "Int64", "nullable": False},
        {"name": "updated_at", "type": "Int64", "nullable": False},
        varchar("last_error_code", 32, True, "Registered failure code."),
    ],
}

MEMORY_CONSOLIDATION_JOURNAL_INDEXES = {
    "journal_anchor_vector": {
        "index_type": "AUTOINDEX",
        "metric_type": "COSINE",
        "params": {},
    },
    "scalar_indexes": [
        "session_id",
        "trigger_event_id",
        "status",
        "created_at",
        "updated_at",
    ],
}

GROUNDED_RESPONSE_CACHE_COLLECTION: dict[str, Any] = {
    "collection_name": COLLECTION_NAMES["grounded_response_cache"],
    "description": "Session-scoped citation-validated response cache",
    "fields": [
        {
            "name": "cache_id",
            "type": "VarChar",
            "max_length": 128,
            "nullable": False,
            "primary_key": True,
            "auto_id": False,
            "description": "Stable cache identity.",
        },
        varchar("session_id", 128, False, "Session scope."),
        varchar("source_query_id", 128, False, "Producing query identity."),
        varchar("normalized_query", 8192, False, "Normalized query."),
        varchar("query_hash", 64, False, "Normalized query SHA-256."),
        varchar(
            "embedding_fingerprint",
            256,
            False,
            "Query vector-space identity.",
        ),
        varchar("intent", 32, False, "Validated intent."),
        varchar("query_type", 32, False, "Validated query topic."),
        varchar("retrieval_goal", 16, False, "focused/exhaustive."),
        {"name": "version_scope", "type": "JSON", "nullable": False},
        {"name": "entity_ids", "type": "JSON", "nullable": False},
        {"name": "query_constraints", "type": "JSON", "nullable": False},
        varchar(
            "permission_scope_hash",
            64,
            False,
            "Current permission-scope digest.",
        ),
        varchar("kb_revision", 128, False, "Corpus publication id."),
        varchar(
            "workflow_version",
            128,
            False,
            "Cache validation contract id.",
        ),
        varchar("answer", 12000, False, "Validated grounded answer."),
        {"name": "citations", "type": "JSON", "nullable": False},
        {"name": "evidence", "type": "JSON", "nullable": False},
        {"name": "created_at", "type": "Int64", "nullable": False},
        {"name": "expires_at", "type": "TIMESTAMPTZ", "nullable": False},
        {
            "name": "query_vector",
            "type": "FloatVector",
            "dim": VECTOR_DIMS["TEXT_DIM"],
            "nullable": False,
        },
    ],
    "properties": {"ttl_field": "expires_at"},
}

GROUNDED_RESPONSE_CACHE_INDEXES = {
    "query_vector": {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200},
    },
    "scalar_indexes": [
        "session_id",
        "query_hash",
        "intent",
        "query_type",
        "retrieval_goal",
        "permission_scope_hash",
        "kb_revision",
        "workflow_version",
        "created_at",
        {"field_name": "expires_at", "index_type": "STL_SORT"},
    ],
}

DOC_DEDUP_SIGNATURES_COLLECTION: dict[str, Any] = {
    "collection_name": COLLECTION_NAMES["doc_dedup_signatures"],
    "description": "P2 experimental dedup signatures",
    "fields": [
        {
            "name": "id",
            "type": "Int64",
            "primary_key": True,
            "auto_id": True,
        },
        varchar("doc_id", 128, False, "Document ID."),
        varchar("chunk_id", 128, True, "Chunk ID."),
        varchar("source_uri", 1024, False, "Source URI."),
        varchar("source_type", 32, False, "local/s3/mfs."),
        varchar("record_level", 32, False, "doc/chunk."),
        varchar("normalized_text", 16384, False, "Normalized text.")
        | {
            "enable_analyzer": True,
            "analyzer_params": {
                "tokenizer": "standard",
                "filter": ["lowercase"],
            },
        },
        varchar("checksum", 128, False, "Exact checksum."),
        {
            "name": "minhash_signature",
            "type": "BinaryVector",
            "dim": 8192,
            "nullable": False,
        },
        {"name": "created_at", "type": "Int64", "nullable": False},
        {"name": "metadata", "type": "JSON", "nullable": True},
    ],
    "functions": [
        {
            "name": "minhash_function",
            "type": "MINHASH",
            "input_fields": ["normalized_text"],
            "output_fields": ["minhash_signature"],
            "params": {
                "num_hashes": 256,
                "shingle_size": 3,
                "seed": 1234,
                "token_level": "word",
            },
        }
    ],
}

DOC_DEDUP_ENUMS = {
    "source_type": ["local", "s3", "mfs"],
    "record_level": ["doc", "chunk"],
}

DOC_DEDUP_SIGNATURES_INDEXES = {
    "minhash_signature": {
        "index_type": "MINHASH_LSH",
        "metric_type": "MHJACCARD",
        "params": {
            "mh_element_bit_width": 32,
            "mh_lsh_band": 128,
            "with_raw_data": True,
        },
    },
    "scalar_indexes": [
        "doc_id",
        "chunk_id",
        "source_type",
        "record_level",
        "checksum",
        "created_at",
    ],
}
