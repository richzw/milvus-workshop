# Milvus 2.6 Release Notes

Workshop summary of the official Milvus 2.6.x release notes, focused on the Milvus 2.6.0 feature set released on August 6, 2025. Direct upgrade from releases before 2.6.0 is not supported because of architectural changes. Official source: https://milvus.io/docs/v2.6.x/release_notes.md

## Storage Format V2

Milvus 2.6 introduces an adaptive columnar storage format aimed at efficient point lookup and small-batch retrieval across mixed scalar and vector data. This is the 2.6 storage feature and is distinct from Storage Format V3 in Milvus 3.0.

## JSON Flat Index

JSON Flat Index discovers and indexes nested paths beneath a JSON field without requiring every path and type to be declared in advance. It is intended for evolving JSON schemas.

## Streaming Node

Streaming Node becomes generally available in Milvus 2.6 and centralizes shard-level write-ahead-log operations while also serving query-delegation responsibilities.

## Woodpecker Native WAL

Milvus 2.6 can use the object-storage-oriented Woodpecker WAL instead of requiring an external Kafka or Pulsar deployment, reducing operational components for suitable deployments.

## DataNode and IndexNode Consolidation

Index building, compaction, bulk import, and related scheduled work are consolidated under DataNode, while persistence responsibilities move toward Streaming Node. The change simplifies the deployment topology.

## MixCoord

RootCoord, QueryCoord, and DataCoord responsibilities can be combined in MixCoord. In-process coordination replaces some cross-component communication and reduces distributed-system complexity.

## RaBitQ

RaBitQ provides one-bit vector quantization designed to lower resource cost while retaining strong recall for large-scale approximate nearest-neighbor search.

## JSON Capability Enhancements

Milvus 2.6 expands JSON query and indexing capabilities, including support aimed at dynamic nested content. Related JSON improvements belong together in a feature chunk instead of being split into isolated low-context bullets.

## Data-in Data-out Embedding Functions

Embedding functions let applications insert and query with source text while Milvus invokes a configured embedding provider to produce vectors. This removes a separate vectorization step from the application pipeline.

## Phrase Matching

Text search adds phrase-oriented matching and tokenizer improvements for more precise lexical retrieval across supported languages.

## MinHash LSH

MINHASH_LSH supports scalable approximate Jaccard similarity for identifying near-duplicate text. Applications generate MinHash signatures and use the index for data cleaning and deduplication retrieval.

## Time-aware Decay Functions

Exponential, Gaussian, or linear decay can adjust reranking scores from a timestamp field. Recent information can therefore receive higher weight in feeds, commerce, and agent-memory retrieval.

## Online Schema Evolution

A scalar field can be added to an existing collection schema online, reducing the need to create a replacement collection and migrate all data for compatible schema changes.

## INT8 Vector

Milvus 2.6 adds native INT8 vector data for quantized embeddings, initially with HNSW-family indexes. Applications can avoid dequantizing these embeddings before ingestion.
