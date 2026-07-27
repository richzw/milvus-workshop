# S3 Sync Design

## Purpose

The S3 sync pipeline makes object storage usable as an internal knowledge
source for Agent Chat. The workshop uses a mock S3 folder, but the fields mirror
what a MinIO or S3 connector would produce: `bucket`, `object_key`, `source_uri`,
`checksum`, `updated_at`, `department`, `priority`, and document type.

## Ingestion Pipeline

The sync job runs in six steps:

1. Bucket scanning lists candidate objects under department prefixes such as
   `engineering/`, `security/`, `hr/`, and `product/`.
2. Change detection compares object metadata, `updated_at`, and `checksum` so
   unchanged files are skipped during incremental refresh.
3. Document parsing extracts text from Markdown and text files, page text from
   PDF assets, and captions or OCR text for image assets.
4. Chunking splits long documents into stable chunks with `doc_id`,
   `chunk_id`, `section`, `page_no`, and `chunk_index`.
5. Embedding generation creates `text_vector` for every chunk, `sparse_vector`
   for BM25-style retrieval, and optional DINOv3-style `image_vector` for image
   records.
6. Milvus insertion writes records into the `kb_chunks` collection with
   metadata fields used for filter, order_by, aggregation, and citation.

## Retrieval Contract

After insertion, the UI can ask "我们 S3 文档同步流程是怎么设计的？" and expect
an answer that cites this document plus the architecture PDF. Milvus is
responsible for high-recall hybrid search, while the reranker chooses the best
chunks for answer context.

## Operational Notes

The demo does not implement production ACL. It keeps `department` and
`source_type` filters so the workshop can explain how ACL would become a
metadata filter in a production system. Temporary uploads should include
`expires_at` and be filtered out when expired.
