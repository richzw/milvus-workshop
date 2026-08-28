# 10a — OpenAI Text Embedding

Status: draft · Owner: workshop author · Depends on: [`10-data-model.md`](./10-data-model.md)

## 1. Purpose

Replace the deterministic dense text-vector placeholder with a real OpenAI embedding while preserving the current 1024-dimensional Milvus schema and existing ingestion/retrieval call sites. This component owns text-vector generation and provider configuration. It does not change sparse retrieval, the separately specified image-vector provider, chunking, Milvus insertion, or answer generation.

## 2. Interface and data flow

The existing `dense_vector(text, dim=1024)` function remains the application seam. Ingestion, in-memory retrieval, Milvus retrieval, and conversation memory therefore use the same configured vector space without direct OpenAI SDK dependencies.

```text
┌──────────────────────── Existing application call sites ────────────────────────┐
│ ingestion chunks   Milvus query   in-memory query   conversation memory         │
└───────────────────────────────────┬───────────────────────────────────────────────┘
                                    │ dense_vector(text, dim)
                                    ▼
┌──────────────────────── Embedding provider boundary ─────────────────────────────┐
│ EMBEDDING_PROVIDER=deterministic|openai|auto                                    │
│                                                                                  │
│ configured OpenAI ──▶ text-embedding-3-small, dimensions=1024 ──▶ validate      │
│ no key in auto ─────▶ deterministic hashed provider (offline/tests only)         │
│ provider failure ───▶ typed error; never silently mix vector spaces              │
└───────────────────────────────────┬───────────────────────────────────────────────┘
                                    ▼
┌──────────────────────────── Storage boundary ─────────────────────────────────────┐
│ KBChunk.text_vector ──▶ Milvus kb_chunks.text_vector FloatVector(dim=1024)       │
└───────────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Provider contract

```python
class TextEmbeddingProvider(Protocol):
    name: str

    def embed(self, text: str, *, dimensions: int) -> list[float]: ...

    def fingerprint(self, *, dimensions: int) -> str: ...
```

The protocol shape is binding. Exact private helper names are implementation details.

## 3. Configuration

| Variable | Contract |
| --- | --- |
| `EMBEDDING_PROVIDER` | `deterministic` (default), `openai`, or `auto`; other values fail configuration |
| `OPENAI_API_KEY` | Secret passed only to the OpenAI SDK; never logged |
| `OPENAI_EMBEDDING_MODEL` | Defaults to `text-embedding-3-small` |
| `OPENAI_EMBEDDING_TIMEOUT_SECONDS` | Positive request timeout; defaults to `30` |

`deterministic` is the default and never constructs an OpenAI client, even when an ambient API key exists. `openai` requires the key and fails fast when it is absent. Optional `auto` selects OpenAI when `OPENAI_API_KEY` is non-empty and otherwise selects the deterministic offline provider.

The adapter requests `dimensions=1024`, matching `VECTOR_DIMS["TEXT_DIM"]`. A model override must support the dimensions parameter and return exactly that length. Changing the dimension is a schema migration and is outside this change.

## 4. Invariants

1. Document ingestion and query embedding use the same configured provider, model, and dimension.
2. OpenAI output contains exactly 1024 finite numeric values before it reaches a `KBChunk` or Milvus request.
3. Every persisted chunk records `text_embedding_fingerprint=<provider>:<model-or-algorithm>:<dimension>` in the existing JSON metadata; Milvus insert and returned-result validation reject missing or mismatched fingerprints.
4. Empty text is rejected locally and is never sent to OpenAI.
5. A configured OpenAI request failure raises a contextual embedding error; it never falls back to deterministic vectors within the same process.
6. Sparse vectors remain deterministic term-frequency maps. Image vectors are
   out of this component's scope entirely: they use the separate 768-dimensional
   `ImageEmbeddingProvider` fixed by [`11-ingestion.md § Embeddings`](./11-ingestion.md#5-embeddings)
   and [`99-key-decisions.md § D23`](./99-key-decisions.md#d23--image-embeddings-use-a-fingerprinted-dinov3-vit-b-vector-space);
   the two vector spaces are never mixed or substituted for each other.
7. API keys, source text, provider response bodies, and raw provider exception causes are absent from errors and logs.

## 5. Behaviour and failure policy

| Case | Required behaviour |
| --- | --- |
| mode omitted, with or without ambient key | use deterministic provider; never perform implicit network I/O |
| `auto` without API key | use deterministic provider for offline demo/tests |
| `openai` without API key | fail configuration before network I/O |
| invalid mode or non-positive timeout | fail configuration with the variable name |
| timeout/auth/rate limit/connection failure | raise a typed reason code with no secret/body |
| empty/malformed/wrong-dimension output | reject before insertion/search |
| alternate OpenAI model | allowed only if it supports the configured 1024-dimensional response |

The OpenAI client is cached per process so repeated chunks reuse connection pooling. Streamlit also caches the provider-aware workflow across reruns. Tests inject a fake client and perform no network I/O.

## 6. Minimal-change boundary

- Keep the public `dense_vector()` signature and all existing ingestion, retrieval, memory, and Milvus adapter call sites.
- Centralize provider implementation in `embedding.py`; limit other changes to metadata wiring, validation, and UI resource caching; reuse the already-declared `openai>=2,<3` dependency.
- Do not alter `KBChunk`, collection definitions, indexes, or stored dimensions; use its existing JSON metadata for the fingerprint.
- Do not touch the image path from this seam. `image_vector` is owned by the
  explicit `IMAGE_EMBEDDING_PROVIDER=deterministic|dinov3` contract in
  [`11-ingestion.md § Embeddings`](./11-ingestion.md#5-embeddings); its offline mode hashes validated
  image bytes, never a caption, and it carries its own
  `metadata.image_embedding_fingerprint`.
- A provider or model change is therefore a full corpus re-ingest. Its cost model, rebuild window and the fingerprint startup gate are recorded in [`15-retrieval-tier-selection.md § 7`](./15-retrieval-tier-selection.md#7-embedding-model-lifecycle); no change ships without that migration plan.

## 7. Tests and acceptance

- Fake-client tests assert model, input, dimensions, float encoding, and timeout passed to `client.embeddings.create`.
- Configuration tests cover all three modes and invalid/missing values.
- Validation tests reject empty input, provider exceptions, non-finite values, and wrong dimensions.
- Existing ingestion, retrieval, schema, workflow, and generation tests remain green without credentials.
- An opt-in real smoke test may be run with a user-provided key, but success is not claimed until actually executed.

## 8. AGENTS.md compliance

- Error Handling: typed, contextual failures; no swallowed provider errors.
- Interfaces over singletons: call sites depend on the provider protocol; the default provider cache is an internal construction detail.
- Testability: fake-client injection pins the SDK boundary; deterministic mode keeps CI offline.
- Security: secrets and document content are not exposed in errors.
- Performance: one cached SDK client; batching is deferred because preserving the current single-text seam is the explicit minimal-change constraint.

## 9. Cross-references

- ← Depends on: [`10-data-model.md § kb_chunks`](./10-data-model.md#3-kb_chunks--authoritative-knowledge-records)
- → Consumed by: [`11-ingestion.md`](./11-ingestion.md), [`12-agent-workflow.md`](./12-agent-workflow.md)
- ↔ Evaluated by: [`70-quality-and-evaluation.md`](./70-quality-and-evaluation.md)
- ↔ Decisions: [`99-key-decisions.md § D12`](./99-key-decisions.md#d12--openai-text-embedding-preserves-the-existing-vector-contract), [`99-key-decisions.md § D51`](./99-key-decisions.md#d51--embedding-model-migration-is-planned-and-fingerprint-gated)
