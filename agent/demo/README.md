# Agent Workshop Demo

This demo implements the first Agent Chat MVP described by
[`specs/index.md`](../specs/index.md). It prefers LangGraph when installed and
keeps a deterministic local retrieval/reranking fallback so the workshop and
tests remain reproducible without external services.

The fallback uses hashed token vectors and a rule-based reranker. It is a
teaching implementation, not evidence that native Milvus 3.0 hybrid search,
DINOv3, or a configured reranker model's quality has been benchmarked. Those
capability checks remain tracked in
[`specs/93-improvements-review.md`](../specs/93-improvements-review.md).

## Runtime Modes

The repository currently provides two local interfaces:

- **CLI**: runs one query against the checked-in sample corpus and prints its
  answer, citations, and trace.
- **Streamlit**: runs the three-tab teaching UI against the configured Milvus
  `kb_chunks` collection. Users enter only a question; the Agent selects tools
  and tool-owned metadata filters.

Streamlit connects directly to Milvus at startup and fails fast when the
configured collection is missing or cannot be loaded. The CLI and deterministic
eval keep their offline `InMemoryHybridRetriever` path.

## Local Setup

Use Python 3.10 or newer. This setup was validated with Python 3.13;
Python 3.11–3.13 is the recommended range. Run every command below from the
repository root, the directory containing `demo/`, `specs/`, and `readme.md`.

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r demo/requirements.txt
cp demo/.env.example demo/.env
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r demo\requirements.txt
Copy-Item demo\.env.example demo\.env
```

The requirements file installs the local package in editable mode, so
`PYTHONPATH` does not need to be configured. At startup, every demo entrypoint
automatically loads `demo/.env`. Values already present in the process
environment take precedence over the file. Keep the virtual environment active
while running the demo. Prefer `python -m streamlit` over the bare executable
name so the command always uses packages from this environment.

## Quick Start: Deterministic Offline Demo

No API key, Milvus server, or object-storage service is required for this path.
The default embedding and answer providers are deterministic and make no
network calls.

Run the CLI:

```bash
python -m agent_workshop_demo.cli \
  "我们 S3 文档同步流程是怎么设计的？"
```

PowerShell accepts the same command on one line:

```powershell
python -m agent_workshop_demo.cli "我们 S3 文档同步流程是怎么设计的？"
```

Run the deterministic tests:

```bash
python -m unittest discover demo/tests -v
```

The offline corpus also demonstrates two Agentic RAG safeguards:

- Reviewed terminology from `demo/config/predefined_entities.yaml` is resolved
  before query rewriting. For example, `跳转按钮` and `领取按钮` resolve to
  `GO按钮`; unresolved domain ambiguity returns a clarification instead of
  retrieving with a guessed meaning.
- Every chunk carries a stable `doc_id`, required `doc_version`, and
  `is_current`. Unqualified questions search only current editions, explicit
  version questions search only that edition, and comparisons keep evidence
  partitioned and label each version in the answer.

## OpenAI Text Embeddings

Dense text vectors are generated behind the existing `dense_vector()` seam,
so ingestion and Milvus queries always use the same configured provider. Edit
these values in `demo/.env` before ingesting documents or starting the API/UI:

```dotenv
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=your-api-key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_EMBEDDING_TIMEOUT_SECONDS=30
```

The adapter requests 1024 dimensions to match the existing
`kb_chunks.text_vector` schema, so no collection migration is required. If a
collection already contains deterministic placeholder vectors, re-ingest all
of its chunks before querying it with OpenAI embeddings; vectors from different
providers must not be mixed. Each chunk records a provider/model/dimension
fingerprint in its existing JSON metadata. Milvus insert and returned-result
validation reject a fingerprint that differs from the current configuration.

Ingestion contextualizes dense and sparse retrieval inputs with the document
title, Markdown heading path, and section while preserving the original chunk
text for citations. The recipe is recorded as
`metadata.retrieval_text_version`. Existing `kb_chunks` collections created
before this recipe must be fully re-ingested; updating application code alone
does not rewrite stored vectors.

`EMBEDDING_PROVIDER=deterministic` is the safe default, so an ambient API key
cannot make tests or offline commands perform network I/O. Optional `auto` mode
selects OpenAI when `OPENAI_API_KEY` is present. Once OpenAI is selected,
provider errors fail the embedding operation instead of silently falling back
to a different vector space. The Streamlit app caches its configured workflow
across reruns, so it does not re-embed the sample corpus on every UI event.

## DINOv3 Image Embeddings

Image records in `asset_manifest.json` are embedded from their referenced PNG
files, not from captions. The offline default hashes validated image bytes in a
clearly fingerprinted test-only vector space. To generate real visual features,
first accept Meta's DINOv3 license and obtain access to the gated model, then
install the optional runtime:

For MinIO ingestion, place `asset_manifest.json` at the configured prefix root
and keep each `asset_path` relative to that root. The adapter embeds downloaded
image bytes while preserving their `s3://` URI, bucket, and object key.

```bash
python -m pip install -r demo/requirements-image.txt
```

Configure the 768-dimensional ViT-B/16 provider:

```dotenv
IMAGE_EMBEDDING_PROVIDER=dinov3
DINOV3_MODEL=facebook/dinov3-vitb16-pretrain-lvd1689m
DINOV3_DEVICE=cpu
DINOV3_LOCAL_FILES_ONLY=false
HF_TOKEN=your-hugging-face-token
```

`HF_TOKEN` is passed only to the model loader and is never written to records or
reports. The provider lazily loads `AutoImageProcessor` and `AutoModel`, takes
the global `pooler_output`, validates 768 finite values, and L2-normalizes the
vector. PNG, JPEG, and WebP inputs are bounded by
`IMAGE_EMBEDDING_MAX_BYTES` (20 MiB by default). Model access, decode,
dependency, inference, zero-vector, or dimension failures stop ingestion with
a bounded reason code; they never fall back into the deterministic image
space.

After accepting the gated model license, run the opt-in real-checkpoint smoke
over all five curated PNG fixtures:

```bash
RUN_DINOV3_SMOKE=1 PYTHONPATH=demo/src \
  python3 -m unittest demo.tests.test_image_embedding.ImageEmbeddingTests.test_real_dinov3_checkpoint_embeds_all_curated_images -v
```

Each image record stores an `image_embedding_fingerprint` containing provider,
model, pooling, normalization, and dimension. Recreate and fully re-ingest a
collection when changing image providers or models. Image similarity search is
an independent lab with two deliberately separate paths:

- text-to-image uses normal dense+sparse title/caption search with
  `has_image_vector=true`;
- image-to-image embeds query bytes with the configured image provider and runs
  COSINE search on `image_vector`.

DINOv3 is not a text-image model, so text vectors are never sent to the image
index. Run the independent 10-case image eval:

```bash
EMBEDDING_PROVIDER=deterministic IMAGE_EMBEDDING_PROVIDER=deterministic \
  PYTHONPATH=demo/src python3 demo/scripts/run_image_eval.py
```

The report includes Recall@K and MRR for both modes and omits raw vectors. The
deterministic provider labels its result `pipeline_only`; configure `dinov3` to
measure real visual similarity. Local and Milvus adapters share the same
dimension, L2, fingerprint, image-only filter, and public-output contract.

## Query Classification

`classify_query` uses an injectable classifier instead of workflow-embedded
keywords. The configured CLI/Streamlit path can use OpenAI Structured Outputs
for fixed `intent`, `query_type`, and `retrieval_goal` enums:

```dotenv
QUERY_CLASSIFIER=openai
OPENAI_API_KEY=your-api-key
OPENAI_CLASSIFIER_MODEL=your-enabled-model-id
OPENAI_CLASSIFIER_TIMEOUT_SECONDS=10
```

`OPENAI_CLASSIFIER_MODEL` may be omitted when `OPENAI_MODEL` is configured.
`auto` selects the LLM only when the required key and model exist; otherwise it
records `not_configured` and uses `RuleBasedQueryClassifier`. Explicit Memory
write/recall and mutation requests always use the deterministic rules fast
path. Provider errors, timeouts, and schema-invalid model output also fall back
with a sanitized trace reason. An LLM-only `conversation` result cannot disable
KB retrieval; it falls back with `unsafe_no_retrieval_intent`. Direct
`AgenticRAGWorkflow()` construction stays rule-based for reproducible offline
tests.

Force the offline path with:

```dotenv
QUERY_CLASSIFIER=rule_based
```

## Model Reranking

Configured CLI and Streamlit workflows can precision-rank the complete bounded
recall pool with one OpenAI Responses request:

```dotenv
RERANKER=openai
OPENAI_API_KEY=your-api-key
OPENAI_RERANKER_MODEL=your-enabled-model-id
OPENAI_RERANKER_TIMEOUT_SECONDS=10
```

`OPENAI_RERANKER_MODEL` may be omitted when `OPENAI_MODEL` is configured. The
adapter uses strict JSON-schema output and accepts a run only when every input
`chunk_id` appears exactly once with a finite relevance score in `[0, 1]`.
Missing, duplicate, invented or malformed results, as well as sanitized
provider failures, rerank the whole candidate batch with
`RuleBasedReranker`; partial model scores are never mixed with fallback
scores. Up to 120 merged/expanded candidates share one bounded 96,000-character
input budget. The trace records the implementation and model that actually
produced the ranking plus a bounded fallback reason; paths that never rank
evidence report `not_run`.

`auto` uses the model only when its key and model are configured. Direct
`AgenticRAGWorkflow()` construction remains rule-based, and the reproducible
offline path can be forced with:

```dotenv
RERANKER=rule_based
```

## OpenAI Answer Generation

The CLI and Streamlit app use `ANSWER_GENERATOR=auto` by default. Configure
both the API key and an explicit model in `demo/.env` to synthesize an answer
from the selected chunks through the OpenAI Responses API:

```dotenv
ANSWER_GENERATOR=openai
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=your-enabled-model-id
OPENAI_TIMEOUT_SECONDS=30
```

No model name is guessed by the code. In `auto` mode, missing OpenAI
configuration selects the deterministic generator and records
`not_configured` in the query trace. Runtime provider errors, timeouts, rate
limits, authentication failures, and citation-invalid model output also use
the deterministic fallback with a sanitized reason code.

Only the current question and at most five selected chunks enter generation.
The complete model response is checked against the request-local `[C1]`…`[Cn]`
citation map before the existing answer-chunk interface exposes it. This is
validated buffering, not token-real-time provider streaming.

Force the reproducible offline path when desired:

```dotenv
ANSWER_GENERATOR=deterministic
```

## Milvus Provisioning and Ingestion

Milvus is required for Streamlit, but not for the default CLI or eval commands.
The following scripts provision and populate the server-backed query path.
Install `demo/requirements.txt` so the scripts and UI use the validated
`pymilvus==3.0.1` client.

Start Milvus first, then configure its endpoint in `demo/.env`. The default URI
already targets a local standalone server:

```dotenv
MILVUS_URI=http://localhost:19530
MILVUS_TOKEN=
MILVUS_COLLECTION_NAME=kb_chunks
MILVUS_MEMORY_COLLECTION_NAME=conversation_memory
MILVUS_RESPONSE_CACHE_COLLECTION_NAME=grounded_response_cache
MEMORY_TOP_K=3
MEMORY_TTL_SECONDS=86400
RESPONSE_CACHE_ENABLED=true
RESPONSE_CACHE_TOP_K=3
RESPONSE_CACHE_TTL_SECONDS=259200
RESPONSE_CACHE_SIMILARITY_THRESHOLD=0.92
KB_REVISION=demo-v1
```

Run the three operations in order:

```bash
python demo/scripts/create_collections.py
python demo/scripts/create_indexes.py
python demo/scripts/ingest_demo.py
```

Provisioning creates Milvus 3.0 BM25 and MinHash functions. BM25 uses a small
reviewed technical synonym dictionary; the sparse index omits the legacy
algorithm parameter so Milvus selects SINDI. Use
`create_indexes.py --sparse-compatibility-daat-maxscore` only for an explicit
legacy compatibility exercise. Ingestion writes raw `retrieval_text` and
normalized dedup text while Milvus owns both function output vectors.

Lifecycle collections now store `TIMESTAMPTZ expires_at` and configure
`ttl_field=expires_at`. An older Int64 field cannot be changed in place, so
that upgrade uses the previewed cleanup and full rebuild below.

The version fields change the `kb_chunks` schema. When upgrading a collection
created before this feature, first preview the fixed deletion scope:

```bash
python demo/scripts/cleanup_milvus.py
```

The preview never connects to Milvus. After confirming that only the seven
repository-owned demo collections are listed, perform the
destructive cleanup and full-corpus rebuild:

```bash
python demo/scripts/cleanup_milvus.py --confirm-drop-demo-data
python demo/scripts/create_collections.py
python demo/scripts/create_indexes.py
python demo/scripts/ingest_demo.py
```

This is required because replacing known `chunk_id` values cannot remove all
legacy records that lack the new version identity. The adapter also rejects an
incremental insert that would introduce a different current edition alongside
the one already visible; publish such an edition through this full rebuild.
The cleanup script never targets collections outside those seven fixed demo
names and verifies that each dropped collection is absent.

For additive nullable vectors, preview and then explicitly apply a migration:

```bash
python demo/scripts/evolve_schema.py add-retrieval-text --apply
python demo/scripts/evolve_schema.py backfill-retrieval-text --records /path/to/id-retrieval-text.jsonl --apply
python demo/scripts/evolve_schema.py add-bm25 --field-name sparse_vector_v2 --apply
python demo/scripts/evolve_schema.py add-embedding --field-name new_embedding --dim 1024
python demo/scripts/evolve_schema.py add-embedding --field-name new_embedding --dim 1024 --apply
python demo/scripts/evolve_schema.py backfill-embedding --field-name new_embedding --dim 1024 --records /path/to/id-vectors.jsonl --apply
```

Embedding backfill accepts only primary `id` plus the named finite vector and
uses bounded `partial_update` batches. Drop, rename, type changes and dynamic
fields are not supported. The BM25 migration sets the Milvus 3 physical-backfill
protobuf flag and revalidates the field, Function and index identities. After
validating the backfill, switch readers explicitly with
`MILVUS_SPARSE_FIELD=sparse_vector_v2`; leaving it unset continues to query
`sparse_vector`, which is also the rollback path.

Snapshot-backed evaluation requires all snapshot arguments together. The
default `run_eval.py` remains offline and makes no Milvus connection:

```bash
python demo/scripts/run_eval.py \
  --milvus-uri http://localhost:19530 \
  --snapshot-name workshop_eval_v1 \
  --source-collection kb_chunks \
  --target-collection workshop_eval_v1_kb \
  --sparse-field sparse_vector_v2
```

`--sparse-field` defaults to `MILVUS_SPARSE_FIELD` and then
`sparse_vector`, so snapshot evaluation follows the same validated reader
cutover as the serving workflow.

The collection script creates and verifies all seven demo collections. The index
script creates named vector and scalar indexes, waits for completion, and
verifies them with `list_indexes`. Both commands are idempotent. Use
`--drop-existing` or `--recreate` only when intentional because those options
remove server-side objects.

`grounded_response_cache` stores complete validated answers, citations, cited
chunk version/checksum snapshots, permission-scope hashes, and `KB_REVISION`
for up to three days. Exact or semantically similar same-session questions can
reuse a response only after classification, permission, revision, expiry, and
live cited-chunk validation all pass. Cache failures fall back to normal RAG.
Increment `KB_REVISION` whenever a new KB publication should invalidate prior
responses.

Selective Memory uses deterministic rules by default. To enable the optional
LLM ambiguity selector, configure:

```dotenv
MEMORY_SELECTOR=auto
MEMORY_SELECTOR_AMBIGUITY_MIN=0.40
MEMORY_SELECTOR_AMBIGUITY_MAX=0.60
OPENAI_MEMORY_SELECTOR_MODEL=your-model
OPENAI_MEMORY_SELECTOR_TIMEOUT_SECONDS=5
```

Rules always run first. Explicit remember/correction and every score outside
the configured `0.40..0.60` band make no selector model call. An in-band call
returns only `ephemeral` or `promote_candidate`; invalid output, timeout, or
missing provider configuration keeps the exact rule decision. The selector
receives no assistant answer, recalled Memory, KB chunks, or tool bodies.
The registered signal requires paired future-uncertainty and reuse/need
semantics—for example “下次可能还会复用” or `might reuse`—and produces the
narrow `0.40` ambiguity used by the demo. A future phrase alone does not
qualify, and the signal is not treated as an explicit remember request.
Selector name/model and a sanitized fallback reason are stored with each
`memory_events` record. Recreate an older workshop `memory_events` collection
before enabling this version because those metadata fields extend its schema.

`ingest_demo.py` parses the local and mock-S3 fixtures, replaces matching
`chunk_id` records in `kb_chunks`, flushes the collection, and queries the IDs
back. Success output includes matching `insert_count` and `verified_count`
values. Add `--output-dir` to also retain the generated JSONL:

```bash
python demo/scripts/ingest_demo.py \
  --output-dir /tmp/agent-workshop-ingest
```

To run the same parser/chunker against a real MinIO bucket, configure the
bounded source adapter and select it explicitly:

```dotenv
MINIO_ENDPOINT=localhost:9000
MINIO_BUCKET=internal-agent-chat-demo
MINIO_PREFIX=
MINIO_SECURE=false
MINIO_ACCESS_KEY=minio-access-key
MINIO_SECRET_KEY=minio-secret-key
MINIO_MAX_OBJECTS=1000
MINIO_MAX_OBJECT_BYTES=16777216
```

```bash
python demo/scripts/ingest_demo.py --s3-source minio --dry-run
```

The adapter recursively snapshots the configured prefix, then reuses the
normal ingestion pipeline. Object keys are sorted and traversal-checked;
object count and per-object byte bounds are enforced. Responses are always
closed and released. Credentials are read only from the environment and are
never placed in `source_uri`, reports, or error messages. The checked-in mock
directory remains the default and performs no MinIO network calls.

Use an explicit URI when needed:

```bash
python demo/scripts/create_collections.py --uri http://localhost:19530
python demo/scripts/create_indexes.py --uri http://localhost:19530
python demo/scripts/ingest_demo.py --uri http://localhost:19530
```

To inspect generated definitions or records without connecting to Milvus, pass
`--dry-run` to any script. With ingestion, `--output-dir` remains optional:

```bash
python demo/scripts/create_collections.py --dry-run
python demo/scripts/create_indexes.py --dry-run
python demo/scripts/ingest_demo.py --dry-run \
  --output-dir /tmp/agent-workshop-ingest
```

For Zilliz Cloud, set `MILVUS_URI` to the cluster public endpoint and
`MILVUS_TOKEN` to its API token. When an application container shares the
Docker Compose network with Milvus, use the Milvus service name instead of
`localhost`, for example `http://standalone:19530`.

## Ingestion Inputs and JSONL Output

The ingestion script reads:

- `sample_data/local_docs/`: local workshop documents.
- `sample_data/mock_s3/`: mock S3 object storage documents.
- `sample_data/asset_manifest.json`: captions and metadata for PDF/image assets.
- `sample_data/document_versions.json`: stable document families, edition
  labels, and the single current edition for each versioned source.

Use `--version-manifest PATH` to validate and ingest against another manifest.
Sources absent from the manifest are explicitly stored as the
`unversioned` current edition.

Manifest-covered PDF/image fixtures use their curated, deterministic text.
Additional PDFs are parsed page by page with `pypdf`; unsupported file types
fail with the source path instead of being silently skipped.

PowerShell:

```powershell
python demo\scripts\ingest_demo.py `
  --uri "http://localhost:19530" `
  --output-dir "$env:TEMP\agent-workshop-ingest"
```

Expected generated files:

- `/tmp/agent-workshop-ingest/kb_chunks.jsonl`: generated `kb_chunks` records with text vectors, sparse vectors, metadata, source fields, and nullable `image_vector`.
- `/tmp/agent-workshop-ingest/doc_dedup_signatures.jsonl`: generated dedup records for `doc_dedup_signatures`.

On Windows, the same files are created under
`$env:TEMP\agent-workshop-ingest`.

Run golden QA evaluation:

```bash
python demo/scripts/run_eval.py
```

This validates matching question IDs across `eval/questions.json` and
`eval/golden_answers.yaml`, then reports retrieval Recall@20, reranked
Recall@8, selected-context Recall@5, citation precision/coverage,
required-fact coverage, abstention accuracy, tool-selection accuracy,
entity-resolution accuracy, version-scope accuracy, and cross-version
contamination count.

Compare the checked-in Min-Max Chunking configurations over the same corpus
and stable source/term anchors:

```bash
OPENAI_API_KEY='' \
EMBEDDING_PROVIDER=deterministic \
IMAGE_EMBEDDING_PROVIDER=deterministic \
python demo/scripts/run_chunking_experiment.py
```

The runner reads `eval/chunking_configs.json` and
`eval/chunking_anchors.json`, then reports lexical token-size distributions,
under/over-limit counts, same-source near-duplicate rate, Markdown/PDF
boundary preservation, Recall@20, selected-context Recall@5, and ingestion
time. It compares at least two strict configurations and emits a deterministic
recommendation; it does not change the production ingestion default. Index
size is explicitly `null`/`not_built` because this offline experiment does not
create a Milvus index.

## Streamlit UI

Streamlit runs the Milvus-backed workflow directly. Provision, index, and
ingest the configured collection first, then start the UI from the repository
root. This local command listens on the loopback interface by default:

```bash
python -m streamlit run demo/src/agent_workshop_demo/streamlit_app.py
```

Open the URL Streamlit prints, usually `http://localhost:8501`. Try these demo
questions:

- `我们 S3 文档同步流程是怎么设计的？`
- `RAG 架构里 Milvus 负责哪一层？`
- `UI demo 会展示哪些 Milvus 3.0 能力？`
- `临时上传和会话记忆如何用 TTL 演示？`
- `本季度客户最关心的问题有没有被产品路线图覆盖？`
- `领取按钮会把用户带到哪里？`
- `GO按钮 v1 和 v2 有什么区别？`
- `请记住我叫张三`，然后问 `你还记得我叫什么吗？`
- `我们 S3 文档同步流程是怎么设计的？`，然后问 `它有哪些步骤？`

The UI has no manual search controls. It shows four tabs: Chat, Evidence,
Agent Trace, and Memory. Chat retains the current session's visible turns.
The Memory tab shows bounded recalled/live records and provides an
active-session-only clear action. Records use explicit `expires_at` filtering;
the default TTL is 24 hours.

While the workflow runs, a compact status panel streams
presentation-safe stage, tool, and retry events. It automatically collapses
after completion. Answer text starts only after generation output passes the
citation self-check; this is validated-buffered answer streaming rather than
raw provider-token streaming.

The Agent Trace tab replays the same event timeline and shows metrics. The full
terminal trace remains available in the Advanced expander for workshop
inspection.
Prompts, document text, Memory content, rewritten queries, filters, secrets,
and raw dependency errors are never included in the presentation event stream.

## Remote Server Access

You are correct: direct access from another computer requires the process to
listen on a network interface, not only `127.0.0.1`. On an EC2 instance, bind
the interface to `0.0.0.0`.

Streamlit:

```bash
python -m streamlit run demo/src/agent_workshop_demo/streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port 8501
```

From your local browser, use the instance address—not `0.0.0.0`:

```text
http://EC2_PUBLIC_IPV4_OR_DNS:8501
```

The EC2 security group and any host firewall must allow TCP port 8501.
Restrict the inbound source to your public IP address or trusted network, for
example `YOUR_PUBLIC_IP/32`; do not expose port 8501 to `0.0.0.0/0`. This demo
has no authentication or ACL and must contain only the checked-in synthetic
Workshop data.

For private access without opening an application port, keep the service bound
to its default loopback address and use an SSH tunnel from your local PC:

```bash
ssh -L 8501:127.0.0.1:8501 ec2-user@EC2_PUBLIC_IPV4_OR_DNS
```

Then open `http://127.0.0.1:8501` locally.

## Server-side MinHash / Dedup

The dedup step is included in offline ingestion. For each generated chunk, the
demo:

1. Normalizes text with the same tokenizer used by retrieval.
2. Computes a stable `sha256` checksum for exact duplicate checks.
3. Sends normalized text to the Milvus 3.0 MINHASH Function; ingestion never
   constructs or persists the function output.
4. Writes records using the P2 contract in
   [`specs/10-data-model.md`](../specs/10-data-model.md): `doc_id`, `chunk_id`,
   `source_uri`, `source_type`, `record_level`, `normalized_text`, `checksum`,
   `created_at`, and `metadata`. Milvus materializes `minhash_signature` and
   indexes it with `MINHASH_LSH/MHJACCARD`.

The local fallback retains exact checksums for deterministic tests; it does not
claim bit-level parity with the server-produced signature.

Inspect the generated dedup output:

```bash
head -n 2 /tmp/agent-workshop-ingest/doc_dedup_signatures.jsonl
```

PowerShell:

```powershell
Get-Content "$env:TEMP\agent-workshop-ingest\doc_dedup_signatures.jsonl" `
  -TotalCount 2
```

## Troubleshooting

- **`No module named agent_workshop_demo` from CLI, UI, or scripts**: activate
  the virtual environment and rerun
  `python -m pip install -r demo/requirements.txt` from the repository root.
- **Wrong Python or package version**: activate `.venv`, then use
  `python -m pip` and `python -m streamlit`.
- **OpenAI is not called**: set both the provider mode and its required model/key
  variables before starting a new process.
- **Milvus connection fails**: confirm that a compatible server is already
  running. The deterministic demo does not start Milvus automatically.
- **PowerShell blocks activation**: follow your organization's approved
  execution-policy process; do not disable security controls globally.

## Contents

- `src/agent_workshop_demo/schema/collections.py`: Milvus collection constants.
- `src/agent_workshop_demo/schema/pymilvus_adapter.py`: pymilvus schema/client helpers.
- `src/agent_workshop_demo/langgraph_workflow.py`: LangGraph node graph wrapper.
- `src/agent_workshop_demo/workflow.py`: agentic RAG node sequence.
- `src/agent_workshop_demo/retrieval.py`: local hybrid retriever fallback.
- `src/agent_workshop_demo/reranker.py`: strict model reranker and whole-batch rule fallback.
- `src/agent_workshop_demo/image_embedding.py`: image-byte and DINOv3 embedding providers.
- `src/agent_workshop_demo/image_retrieval.py`: shared local/Milvus image search contract.
- `src/agent_workshop_demo/image_eval.py`: independent text/image retrieval metrics.
- `src/agent_workshop_demo/ingestion.py`: offline local/mock-S3 ingestion.
- `src/agent_workshop_demo/object_store.py`: bounded real MinIO source adapter.
- `src/agent_workshop_demo/dedup.py`: checksum and MinHash-style signatures.
- `src/agent_workshop_demo/memory.py`: TTL-aware conversation memory demo.
- `src/agent_workshop_demo/eval_runner.py`: golden-question evaluation.
- `src/agent_workshop_demo/streamlit_app.py`: query-only UI demo.
- `scripts/`: schema/index helper scripts.
- `eval/`: golden QA answers plus image-retrieval fixtures.
- `notebooks/`: workshop notebook sequence; a notebook runtime is not included
  in `demo/requirements.txt`.
