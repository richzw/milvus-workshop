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
        "metric_type": "IP",
        "params": {"inverted_index_algo": "DAAT_MAXSCORE"},
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
        "metric_type": "IP",
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
        {"name": "expires_at", "type": "Int64", "nullable": True},
        {"name": "metadata", "type": "JSON", "nullable": True},
        {
            "name": "content_vector",
            "type": "FloatVector",
            "dim": VECTOR_DIMS["TEXT_DIM"],
            "nullable": False,
        },
    ],
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
        "expires_at",
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
        varchar("normalized_text", 16384, False, "Normalized text."),
        varchar("checksum", 128, False, "Exact checksum."),
        {
            "name": "minhash_signature",
            "type": "BinaryVector",
            "dim": VECTOR_DIMS["MINHASH_DIM"],
            "nullable": False,
        },
        {"name": "created_at", "type": "Int64", "nullable": False},
        {"name": "metadata", "type": "JSON", "nullable": True},
    ],
}

DOC_DEDUP_ENUMS = {
    "source_type": ["local", "s3", "mfs"],
    "record_level": ["doc", "chunk"],
}

DOC_DEDUP_SIGNATURES_INDEXES = {
    "minhash_signature": {
        "index_type": "BIN_FLAT",
        "metric_type": "HAMMING",
        "params": {},
        "note": "P2 experimental binary signature index.",
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
