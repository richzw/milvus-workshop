# Agent Workshop Demo

This demo implements the first Agent Chat MVP described by
[`specs/index.md`](../specs/index.md). It prefers LangGraph when installed and
keeps a deterministic local retrieval/reranking fallback so the workshop and
tests remain reproducible without external services.

The fallback uses hashed token vectors and a rule-based reranker. It is a
teaching implementation, not evidence that native Milvus 3.0 hybrid search,
DINOv3, or a model reranker has been validated. Those capability checks remain
tracked in [`specs/93-improvements-review.md`](../specs/93-improvements-review.md).

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

`EMBEDDING_PROVIDER=deterministic` is the safe default, so an ambient API key
cannot make tests or offline commands perform network I/O. Optional `auto` mode
selects OpenAI when `OPENAI_API_KEY` is present. Once OpenAI is selected,
provider errors fail the embedding operation instead of silently falling back
to a different vector space. The Streamlit app caches its configured workflow
across reruns, so it does not re-embed the sample corpus on every UI event.

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
`pymilvus==3.0.0` client.

Start Milvus first, then configure its endpoint in `demo/.env`. The default URI
already targets a local standalone server:

```dotenv
MILVUS_URI=http://localhost:19530
MILVUS_TOKEN=
MILVUS_COLLECTION_NAME=kb_chunks
MILVUS_MEMORY_COLLECTION_NAME=conversation_memory
MEMORY_TOP_K=3
MEMORY_TTL_SECONDS=86400
```

Run the three operations in order:

```bash
python demo/scripts/create_collections.py
python demo/scripts/create_indexes.py
python demo/scripts/ingest_demo.py
```

The version fields change the `kb_chunks` schema. When upgrading a collection
created before this feature, first preview the fixed deletion scope:

```bash
python demo/scripts/cleanup_milvus.py
```

The preview never connects to Milvus. After confirming that only `kb_chunks`,
`conversation_memory`, and `doc_dedup_signatures` are listed, perform the
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
The cleanup script never targets collections outside those three fixed demo
names and verifies that each dropped collection is absent.

The collection script creates `kb_chunks`, `conversation_memory`, and
`doc_dedup_signatures`, then verifies each with `has_collection`. The index
script creates named vector and scalar indexes, waits for completion, and
verifies them with `list_indexes`. Both commands are idempotent. Use
`--drop-existing` or `--recreate` only when intentional because those options
remove server-side objects.

`ingest_demo.py` parses the local and mock-S3 fixtures, replaces matching
`chunk_id` records in `kb_chunks`, flushes the collection, and queries the IDs
back. Success output includes matching `insert_count` and `verified_count`
values. Add `--output-dir` to also retain the generated JSONL:

```bash
python demo/scripts/ingest_demo.py \
  --output-dir /tmp/agent-workshop-ingest
```

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

## Offline MinHash / Dedup

The dedup step is included in offline ingestion. For each generated chunk, the
demo:

1. Normalizes text with the same tokenizer used by retrieval.
2. Computes a stable `sha256` checksum for exact duplicate checks.
3. Builds a deterministic MinHash-style binary signature for near-duplicate demos.
4. Writes records using the provisional P2 contract in
   [`specs/10-data-model.md`](../specs/10-data-model.md): `doc_id`, `chunk_id`,
   `source_uri`, `source_type`, `record_level`, `normalized_text`, `checksum`,
   `minhash_signature`, `created_at`, and `metadata`.

The MinHash-style binary representation is experimental and is not used by the
MVP query path.

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
- `src/agent_workshop_demo/reranker.py`: rule-based reranker fallback.
- `src/agent_workshop_demo/ingestion.py`: offline local/mock-S3 ingestion.
- `src/agent_workshop_demo/dedup.py`: checksum and MinHash-style signatures.
- `src/agent_workshop_demo/memory.py`: TTL-aware conversation memory demo.
- `src/agent_workshop_demo/eval_runner.py`: golden-question evaluation.
- `src/agent_workshop_demo/streamlit_app.py`: query-only UI demo.
- `scripts/`: schema/index helper scripts.
- `eval/`: golden questions and expected answers.
- `notebooks/`: workshop notebook sequence; a notebook runtime is not included
  in `demo/requirements.txt`.
