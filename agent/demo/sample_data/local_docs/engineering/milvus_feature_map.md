# Milvus 3.0 Feature Map

## Retrieval Layer

Milvus handles the retrieval layer for the Agent Chat demo. The primary
collection is `kb_chunks`, which stores one record per text chunk, PDF page,
image, or table-like excerpt. Each record has `text_vector`, `sparse_vector`,
nullable `image_vector`, source metadata, and citation fields.

## Hybrid Search

The demo search path combines dense vector search with sparse/BM25 retrieval.
Dense search helps with semantic matches such as "object storage ingestion".
Sparse retrieval helps exact terms such as `S3`, `MinIO`, `chunk_id`,
`updated_at`, and `image_vector`. The local fallback mirrors this behavior so
the workshop can run without a Milvus server.

## Metadata Filter and Order By

Milvus filters use fields including `source_type`, `doc_type`, `department`,
and `has_image_vector`. Search results are ordered by relevance first, then the
demo exposes `updated_at desc` and `priority desc` to explain freshness and
business priority. This answers "RAG 架构里 Milvus 负责哪一层？" by showing that
Milvus owns vector retrieval, sparse retrieval, metadata filter, order_by, and
aggregation.

## Aggregation and Null Vectors

The UI displays aggregation by `source_type`, `doc_type`, `department`, and
`has_image_vector`. Nullable `image_vector` is important because most text
chunks do not have visual features, while diagram records do. This lets the
same knowledge object model cover Markdown, PDF pages, and images.

## Collections

The workshop defines three Milvus collections:

- `kb_chunks`: online Agent Chat retrieval, citation, filters, order_by, and
  aggregation.
- `conversation_memory`: TTL-aware chat memory and session summaries.
- `doc_dedup_signatures`: offline ingestion dedup with checksum and MinHash
  signatures.
