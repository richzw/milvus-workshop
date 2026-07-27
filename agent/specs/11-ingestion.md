# 11 — Offline Ingestion

Status: draft · Owner: workshop author · Depends on: [`00-prd.md`](./00-prd.md), [`10-data-model.md`](./10-data-model.md), [`10a-openai-text-embedding.md`](./10a-openai-text-embedding.md)

## 1. Purpose

Offline ingestion turns local and MinIO/mock-S3 documents into deterministic `kb_chunks` records. It owns source discovery, parsing, chunking, embedding and insertion; it does not own online upload, MFS synchronization, answer generation or production scheduling.

## 2. Pipeline and boundaries

```text
┌──────────── Source adapters ────────────┐
│ local filesystem    MinIO/mock S3      │
│ markdown/txt/pdf    same demo corpus   │
└──────────────┬──────────────────────────┘
               ▼
┌──────────── Parse / normalize ──────────┐
│ text + metadata + page/section identity│
│ failure ──▶ contextual error report     │
└──────────────┬──────────────────────────┘
               ▼
┌──────────── Chunk / enrich ─────────────┐
│ stable doc_id/chunk_id                  │
│ doc_version + is_current                │
│ Min-Max chunk policy (Phase 0 tuned)    │
│ optional image caption/OCR              │
└──────────────┬──────────────────────────┘
               ▼
┌──────────── Embed / validate ───────────┐
│ text dense + sparse representation      │
│ optional image embedding                │
│ dimension/nullability checks            │
└──────────────┬──────────────────────────┘
               ▼
┌──────────── Persist / verify ───────────┐
│ kb_chunks                               │
│ optional dedup signatures (P2)          │
│ count + sample round-trip verification  │
└─────────────────────────────────────────┘
```

## 3. Input and output contracts

### 3.1 Supported MVP inputs

- Local Markdown, text and PDF files from a checked-in or generated sample corpus.
- The same class of objects served through MinIO/mock S3.
- Three to five curated image samples may populate `image_vector`; image retrieval remains P2.

### 3.2 Logical interface

```python
class SourceDocument(TypedDict):
    source_uri: str
    source_type: str
    content: bytes
    doc_version: str
    is_current: bool
    metadata: dict

class ParsedUnit(TypedDict):
    text: str
    page_no: int | None
    section: str | None
    metadata: dict

class KnowledgeRecord(TypedDict):
    doc_id: str
    doc_version: str
    is_current: bool
    chunk_id: str
    text: str
    metadata: dict
```

These are contract shapes, not verified project symbols. The implementation may use dataclasses, Pydantic models or TypedDict only after dependencies are verified in the future demo project.

## 4. Chunking and identity

- `doc_id` is stable for the logical source and must not depend on ingestion run time.
- `doc_version` comes from a source manifest or authoritative source metadata. A source with no edition concept uses the explicit sentinel `unversioned`; this fallback is rejected when sibling editions exist for the same `doc_id`.
- `is_current` is explicit source metadata, not inferred by comparing version strings or timestamps. Preflight validation requires exactly one current edition per `doc_id`.
- `chunk_id` is stable for unchanged content and ordering within an edition, includes `doc_version` in its identity input, and appears in citations and eval fixtures.
- Min-Max Chunking is a required experiment, not a preselected library. Phase 0 records the chosen min/max size, overlap and boundary rules against the sample corpus.
- Page boundaries are preserved for PDF citations. Markdown heading context is carried into `section`.
- Re-ingesting unchanged input must not create a second logical `(doc_id, doc_version, chunk_id)` record.
- Publishing a new current edition validates the full family first. The Workshop MVP uses collection recreation plus full-corpus ingestion as its controlled publish operation; the adapter rejects an incremental insert when the collection already exposes a different current edition for that `doc_id`. A failed incremental run therefore cannot leave two current editions. Production shadow-collection swap/rollback remains outside this demo.

## 5. Embeddings

- Dense text vectors use the provider contract in [`10a-openai-text-embedding.md`](./10a-openai-text-embedding.md). Configured runs use OpenAI `text-embedding-3-small` with `dimensions=1024`; ingestion and query paths must share that configuration.
- Deterministic dense vectors remain only as the explicit offline/test mode. A configured OpenAI failure must not silently fall back because mixed vector spaces invalidate retrieval.
- DINOv3 is only a candidate for image embeddings from the legacy idea note; model suitability, output dimension, runtime footprint and text-image interoperability are unverified.
- Model names, dimensions and normalization settings live in configuration and are recorded with the generated dataset.
- An embedding failure identifies the source and chunk; it is not converted into a zero vector.

## 6. Behaviour and failure policy

| Case | Required behaviour |
| --- | --- |
| Missing/unreadable source | stop that source with URI and cause; summary reports failure count |
| Unsupported document type | reject explicitly; do not index raw bytes as text |
| Empty parsed content | skip only with a recorded reason |
| Invalid vector dimension | reject before insert |
| Partial batch insert | report inserted and failed ids; rerun remains idempotent |
| Missing/blank `doc_version` | reject unless the document family is explicitly declared `unversioned` |
| Zero or multiple current editions | reject the document family before retrieval-visible state changes |
| New current edition | recreate the demo collection, ingest the fully validated corpus with prior chunks marked historical, and verify all new current chunks by round trip; unsafe incremental replacement is rejected |
| MinIO unavailable | local-source golden path remains runnable; MinIO exercise reports setup error |
| Duplicate content | exact checksum path is preferred; MinHash is P2 and gated by validation |

Per `AGENTS.md`, errors are descriptive, never swallowed, and adapters are injected so local/MinIO behavior can be tested independently.

## 7. Notebook learning path

1. `01_ingestion_local_s3.ipynb` — adapters, parsing, stable identity.
2. `02_text_image_embedding.ipynb` — text embeddings and optional image experiment.
3. `03_milvus_schema_and_insert.ipynb` — validated schema, indexes and round-trip checks.
4. `04_milvus_hybrid_search.ipynb` — retrieval behavior over the ingested corpus.

Exact filenames are part of the Workshop navigation contract; implementation code should remain importable outside notebooks.

## 8. Cross-references

- ← Depends on: [`10-data-model.md § kb_chunks`](./10-data-model.md#3-kb_chunks--authoritative-knowledge-records)
- ← Depends on: [`10a-openai-text-embedding.md`](./10a-openai-text-embedding.md)
- → Produces input for: [`12-agent-workflow.md`](./12-agent-workflow.md)
- ↔ Evaluated by: [`70-quality-and-evaluation.md`](./70-quality-and-evaluation.md)
- ↔ Source notes: [`archive/Draft.md`](./archive/Draft.md), [`archive/notebook.md`](./archive/notebook.md), [`archive/UIdemo-collection.md`](./archive/UIdemo-collection.md)
