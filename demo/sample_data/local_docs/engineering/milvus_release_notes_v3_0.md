# Milvus 3.0 Release Notes

Workshop summary of the official Milvus 3.0 release notes. The sampled edition is `v3.0-beta`, released on May 9, 2026. Official source: https://milvus.io/docs/release_notes.md

## External Collection

Milvus 3.0 can register object-storage data as an External Collection without first importing it into Milvus-managed storage. This supports querying shared lake data while keeping the external source authoritative.

## Snapshot

Snapshots provide a stable, read-only view of collection data at a selected point in time. They are useful for reproducible evaluation, audit, and isolating readers from later writes.

## Query and Search Order By

Query and search can order results by scalar fields. Agentic RAG can combine relevance retrieval with deterministic business ordering such as freshness or priority when the use case requires it.

## Query Aggregation

Aggregation operations summarize matching entities instead of returning only row-level hits. This enables count and grouped analytics to be exposed as a retrieval tool alongside vector search.

## Null Vector

Vector fields can represent missing vector values. A multimodal collection can therefore keep text-only and image-bearing records in one schema without fabricating placeholder image vectors.

## Custom and Synonym Dictionaries

Text analysis supports custom dictionaries and synonym dictionaries. Domain terminology and equivalent expressions can be normalized closer to the retrieval layer.

## Entity TTL

Entity-level TTL allows individual records to expire according to their own lifecycle. This is useful for temporary knowledge, session memory, and other records with different retention windows.

## MinHash Data-in Data-out

MinHash processing can participate in the Data-in, Data-out workflow so applications can work from source data while Milvus handles the configured transformation used for similarity and deduplication.

## Embedding List and DISKANN

Embedding List data can be searched with DISKANN, extending retrieval support for records that contain a variable list of embeddings while using a disk-oriented ANN index.

## Force Merge

Force Merge gives operators an explicit way to consolidate eligible segments. It is an operational control for storage layout and query efficiency rather than an application-level retrieval feature.

## Storage Format V3

Storage Format V3 advances the physical storage design used by Milvus 3.0. The format is a version-specific capability and should not be mixed into a chunk describing Storage Format V2 from Milvus 2.6.
