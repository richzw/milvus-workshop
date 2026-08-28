# 11 — Offline Ingestion

Status: draft · Owner: workshop author · Depends on: [`00-prd.md`](./00-prd.md), [`10-data-model.md`](./10-data-model.md), [`10a-openai-text-embedding.md`](./10a-openai-text-embedding.md)

## 1. Purpose

Offline ingestion turns local and MinIO/mock-S3 documents into deterministic `kb_chunks` records and, when the StructArray capability gate is enabled, a complete `kb_documents` projection. It owns source discovery, parsing, chunking, embedding, projection assembly and insertion; it does not own online upload, MFS synchronization, answer generation or production scheduling.

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
│ text dense + server BM25 input          │
│ optional image embedding                │
│ dimension/nullability checks            │
└──────────────┬──────────────────────────┘
               ▼
┌──────────── Persist / verify ───────────┐
│ kb_chunks                               │
│ kb_documents: grouped StructArray view  │
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

class KnowledgeDocumentProjection(TypedDict):
    document_key: str
    doc_id: str
    doc_version: str
    is_current: bool
    passages: list[dict]  # validated Struct elements in chunk_index order
```

These are contract shapes, not verified project symbols. The implementation may use dataclasses, Pydantic models or TypedDict only after dependencies are verified in the future demo project.

### 3.3 MinIO adapter contract

The real MinIO exercise is an explicit alternative to the checked-in
`mock_s3` directory. It never runs implicitly during tests or the local golden
path.

```python
@dataclass(frozen=True)
class MinIOConfig:
    endpoint: str
    bucket: str
    prefix: str
    secure: bool
    access_key: str | None
    secret_key: str | None
    max_objects: int
    max_object_bytes: int

class MinIOSourceAdapter:
    def download_to(self, target_dir: Path) -> MinIOSnapshot: ...
```

`MinIOSnapshot` records the temporary root, stable `s3://bucket/prefix` base
URI, object count and total downloaded bytes. The ingestion CLI selects exactly
one object-store source with `--s3-source mock|minio`; `mock` remains the
default. MinIO configuration comes from `MINIO_ENDPOINT`, `MINIO_BUCKET`,
`MINIO_PREFIX`, `MINIO_SECURE`, `MINIO_ACCESS_KEY` and
`MINIO_SECRET_KEY`. Credentials have no CLI flags and never enter reports,
source URIs or exception messages.

The adapter:

- checks bucket existence before listing and lists recursively under the
  configured prefix;
- sorts object keys before download so record ordering is reproducible;
- rejects absolute keys, `..`, backslashes, NULs and keys outside the prefix
  before writing a temporary snapshot;
- caps one exercise at `MINIO_MAX_OBJECTS` (default `1000`) and each object at
  `MINIO_MAX_OBJECT_BYTES` (default `16 MiB`);
- reads response bodies in bounded chunks and always closes/releases the
  response, including error and oversize paths;
- rejects an empty snapshot and reports bucket/object context without echoing
  endpoint credentials or response bodies;
- preserves the complete object key in `KBChunk.object_key`, the bucket in
  `KBChunk.bucket`, and a stable `s3://` source URI used by version manifests.

The MinIO SDK import is lazy. Offline tests inject a client implementing the
same narrow `bucket_exists/list_objects/get_object` protocol, so the default
test command performs no network I/O.

## 4. Chunking and identity

- `doc_id` is stable for the logical source and must not depend on ingestion run time.
- `doc_version` comes from a source manifest or authoritative source metadata. A source with no edition concept uses the explicit sentinel `unversioned`; this fallback is rejected when sibling editions exist for the same `doc_id`.
- `is_current` is explicit source metadata, not inferred by comparing version strings or timestamps. Preflight validation requires exactly one current edition per `doc_id`.
- `chunk_id` is stable for unchanged content and ordering within an edition, includes `doc_version` in its identity input, and appears in citations and eval fixtures.
- Min-Max Chunking uses a versioned, deterministic lexical tokenizer
  (`CJK code point | alphanumeric word | punctuation`) for the offline
  experiment so it needs no model/network dependency. A configuration has a
  unique name, `min_tokens`, `max_tokens`, `overlap_tokens` and the fixed
  `paragraph_sentence_v1` boundary policy; `1 ≤ min ≤ max ≤ 4096` and
  `0 ≤ overlap < min`.
- Heading/release-edition and PDF page boundaries remain hard. A semantic unit
  at or below `max_tokens` stays intact even when below `min_tokens`; an
  oversized unit prefers the last paragraph/sentence boundary within the
  window, otherwise cuts at `max_tokens`, balances an undersized tail where
  possible, and advances by the configured overlap. Config, tokenizer,
  semantic-unit/split indexes, token count and applied overlap are persisted in
  chunk metadata. Re-running one config over unchanged input is deterministic.
- Page boundaries are preserved for PDF citations. Markdown is parsed by heading rather than by blank line: each heading starts a semantic unit, the full heading path is carried in `metadata.heading_path`, and the nearest heading is copied to `section`.
- Release notes use the release edition as a hard boundary and a feature heading as the preferred chunk boundary. One chunk must never contain content from two release editions. An oversized feature may be split into bounded child chunks while retaining the same heading path and `doc_version`; short bug-fix bullets should be grouped only within the same release and category instead of becoming one low-context chunk per bullet.
- Release-note editions share one logical `doc_id`, carry an opaque branch-level `doc_version` such as `v2.6` or `v3.0`, and declare exactly one `is_current` edition. The concrete patch or prerelease state, release date, and official source URL remain in the chunk text or parser metadata so that branch comparison and source traceability are both possible.
- Re-ingesting unchanged input must not create a second logical `(doc_id, doc_version, chunk_id)` record.
- 切块参数不得靠主观目测成为发布默认值。每个 corpus/document-type profile 必须引用一个已提交的 `chunking-experiment-v2` 报告，该报告在同一语料和问题集上比较至少三个 small/medium/large 配置。用户给出的 `128/256/512` 可作为初始 `max_tokens` 量级；本仓库仍以版本化 lexical token 为唯一分割单位，并在报告中附字符长度分布，不把字符与 token 阈值混用。
- 实验为每个配置重建完整、相互隔离的 chunk set 和检索索引；一次 run 不得在同一候选池混入两个配置。除了 retrieval/citation 指标，每个配置还必须跑相同 answer generator，报告 required-fact coverage 与经校准的 faithfulness/answer-relevancy 维度，完整合同见 [`70-quality-and-evaluation.md § Chunk configuration evaluation`](./70-quality-and-evaluation.md#5-chunk-configuration-evaluation)。
- 发布的 config name/fingerprint、tokenizer/boundary-policy version、corpus checksum、eval fixture checksum 与选择理由写入 dataset manifest。报告只能生成 recommendation artifact，不能自动改变 ingestion 默认值；改变默认值需要显式审查、全量 re-ingestion 和 committed baseline。
- Publishing a new current edition validates the full family first. The Workshop MVP uses collection recreation plus full-corpus ingestion as its controlled publish operation; the adapter rejects an incremental insert when the collection already exposes a different current edition for that `doc_id`. A failed incremental run therefore cannot leave two current editions. Production shadow-collection swap/rollback remains outside this demo.

## 4a. StructArray projection assembly

Projection assembly runs only after the complete `kb_chunks` record set has passed identity, version, checksum, vector-dimension and corpus-completeness validation:

1. load the reviewed projection manifest, group every selected document family by `(doc_id, doc_version)`, require a non-empty checksum on every selected chunk, require parent-level `source_type/source_uri/doc_type/title/department/is_current/text_embedding_fingerprint` to be invariant, and compute `updated_at`/`priority` as deterministic maxima;
2. sort each group by `(chunk_index, chunk_id)` and map every chunk to the fixed `passages` schema in [`10-data-model.md § 3a`](./10-data-model.md#3a-kb_documents--structarray-document-retrieval-projection);
3. write the same validated `text_vector` to `embedding_list_vector` and `element_vector`; no embedding provider call is repeated and no vector space is converted;
4. preflight `max_capacity`, every `VARCHAR` byte bound, vector dimensions, finite values and exact passage coverage before creating or mutating `kb_documents`;
5. build both vector indexes, load the projection, then round-trip every parent identity plus a deterministic sample of offsets, compare `chunk_id/checksum/provenance` with `kb_chunks`, and prove each id can rehydrate the authoritative text;
6. publish one manifest containing selected and flat-only document families, selection rationale, chunk-config, embedding fingerprint, parent count, passage count, maximum passages per parent, schema fingerprint and both index fingerprints.

The projection build is all-or-nothing for the target corpus. An oversize document, mixed parent metadata, missing/duplicate chunk, misaligned offset or index failure leaves the flat `kb_chunks` path usable but disables `STRUCT_ARRAY_RETRIEVAL`; it never publishes a partial StructArray projection. A chunk/config/embedding/schema change recreates the projection. Incremental whole-parent replacement and shadow cutover remain future production work.

The checked-in `demo/config/struct_array_projection.json` is strict JSON with `schema_version="struct-array-projection-v1"`, a non-empty unique `selected_doc_ids` list, and a bounded rationale. `build_struct_array_projection(chunks, manifest)` returns immutable parent records plus one activation manifest; it performs no I/O. `MilvusStructArrayStore.replace_projection()` may clear and rewrite the derived collection only while retrieval remains disabled, then performs exact parent/passage read-back. A failed write can leave disposable physical rows but never an activation result: the runtime requires the expected full-build fingerprint and repeated parent/passage counts, rejects any foreign fingerprint, and therefore cannot expose a partial build.

`demo/scripts/ingest_demo.py --struct-array-projection disabled|build` defaults to `disabled`. `build` runs only after authoritative `kb_chunks` insertion/read-back, requires the fixed `kb_documents` schema and both indexes, writes the projection, and reports its fingerprint/counts without vector or text payloads. `--dry-run --struct-array-projection build` validates and reports the planned projection without connecting or mutating. Operators pass the reported fingerprint as `STRUCT_ARRAY_PROJECTION_FINGERPRINT` when deliberately enabling a non-flat profile.

StructArray subfields cannot own BM25 Function output, so the ingestion path continues to write `text/retrieval_text` only to `kb_chunks`. The projection does not copy passage text, `metadata` JSON, image vectors, sparse vectors or parser-specific values that are not in the fixed passage schema.

## 5. Embeddings

- Dense text vectors use the provider contract in [`10a-openai-text-embedding.md`](./10a-openai-text-embedding.md). Configured runs use OpenAI `text-embedding-3-small` with `dimensions=1024`; ingestion and query paths must share that configuration.
- Dense and sparse retrieval inputs prepend the document title, full Markdown heading path and nearest section to the citation text. The stored `text` remains the original citation-addressable unit; `metadata.retrieval_text_version` identifies the contextualization recipe so a recipe change requires full re-embedding and re-ingestion.
- Deterministic dense vectors remain only as the explicit offline/test mode. A configured OpenAI failure must not silently fall back because mixed vector spaces invalidate retrieval.
- Image embeddings use a separate injectable `ImageEmbeddingProvider`; they
  consume the referenced image file bytes, never the manifest caption. The
  configured real provider is
  `facebook/dinov3-vitb16-pretrain-lvd1689m` through Transformers
  `AutoImageProcessor`/`AutoModel`. It extracts `pooler_output` (the global
  CLS feature), validates exactly 768 finite values and applies L2
  normalization before persistence. The model id, pooling recipe,
  normalization and dimension form
  `metadata.image_embedding_fingerprint`.
- `IMAGE_EMBEDDING_PROVIDER=deterministic|dinov3` is explicit and defaults to
  `deterministic`; there is no ambient-token `auto` mode. The offline provider
  hashes validated image bytes only to preserve test reproducibility and is
  labeled as a different vector space. `dinov3` lazily imports Pillow,
  PyTorch and Transformers and may download gated weights only after the user
  accepts the DINOv3 license and configures Hub access. Missing dependencies,
  gated/model-load failures, decode failures, invalid/zero vectors and
  dimension mismatch stop ingestion with the safe source URI and a bounded
  reason code; they never fall back into another image vector space.
- Image inputs are regular files with PNG/JPEG/WebP signatures and a
  configurable byte cap (default 20 MiB). `DINOV3_DEVICE` defaults to `cpu`;
  `auto`, `cuda` and `mps` are explicit alternatives. The provider instance,
  loaded model and processor are reused for one process.
- Local samples read `asset_manifest.json` beside `local_docs`; object-store
  snapshots read it at the selected prefix root. Every manifest `asset_path`
  is a root-confined POSIX-relative path, and MinIO image records preserve the
  stable `bucket`/`object_key` identity before embedding downloaded bytes.
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
| StructArray parent exceeds `max_capacity` or scalar bound | abort the complete projection build, report safe parent identity and keep StructArray retrieval disabled; never truncate or shard silently |
| StructArray parent metadata differs across passages | exclude no individual passage; fail the complete projection preflight and continue only on the flat `kb_chunks` route |
| StructArray vector/index/round-trip mismatch | do not activate the projection; preserve `kb_chunks` as the available baseline |
| Missing/blank `doc_version` | reject unless the document family is explicitly declared `unversioned` |
| Zero or multiple current editions | reject the document family before retrieval-visible state changes |
| New current edition | recreate the demo collection, ingest the fully validated corpus with prior chunks marked historical, and verify all new current chunks by round trip; unsafe incremental replacement is rejected |
| MinIO unavailable | local-source golden path remains runnable; MinIO exercise reports setup error |
| Unsafe/oversize MinIO object | reject before parsing or embedding; close the response and report only the safe `s3://bucket/key` identity |
| Empty MinIO prefix | reject the exercise instead of silently ingesting only local records |
| Image decode/model/output failure | stop the image source with safe URI and bounded reason; do not store a placeholder/zero vector or mix providers |
| Duplicate content | exact checksum path is preferred; MinHash is P2 and gated by validation |

Per `AGENTS.md`, errors are descriptive, never swallowed, and adapters are injected so local/MinIO behavior can be tested independently.

MinIO acceptance requires independent tests for configuration validation,
deterministic listing/download, stable URI/bucket/object-key propagation,
missing bucket, empty prefix, traversal keys, object/count bounds, response
cleanup and sanitized SDK failures. One fake-client integration test must pass
the downloaded snapshot through the normal parse/chunk/embed/validation
pipeline; parser logic must not be duplicated inside the adapter.

## 7. Notebook learning path

1. `01_ingestion_local_s3.ipynb` — adapters, parsing, stable identity.
2. `02_text_image_embedding.ipynb` — text embeddings and optional image experiment.
3. `03_milvus_schema_and_insert.ipynb` — validated flat and StructArray schemas, indexes and round-trip checks.
4. `04_milvus_hybrid_search.ipynb` — flat hybrid, element-level, EmbeddingList and application-fusion behavior over the ingested corpus.

Exact filenames are part of the Workshop navigation contract; implementation code should remain importable outside notebooks.

## 8. Cross-references

- Milvus 3.0 BM25、StructArray、DIDO、snapshot 与 schema evolution contract: [`14-milvus-3-native-capabilities.md`](./14-milvus-3-native-capabilities.md)

- ← Depends on: [`10-data-model.md § kb_chunks`](./10-data-model.md#3-kb_chunks--authoritative-knowledge-records)
- ← Depends on: [`10a-openai-text-embedding.md`](./10a-openai-text-embedding.md)
- → Produces input for: [`12-agent-workflow.md`](./12-agent-workflow.md)
- ↔ Evaluated by: [`70-quality-and-evaluation.md`](./70-quality-and-evaluation.md)
- ↔ Decision: [`99-key-decisions.md § D44`](./99-key-decisions.md#d44--chunk-configuration-is-selected-by-isolated-end-to-end-evaluation)
- ↔ Source notes: [`archive/Draft.md`](./archive/Draft.md), [`archive/notebook.md`](./archive/notebook.md), [`archive/UIdemo-collection.md`](./archive/UIdemo-collection.md)
