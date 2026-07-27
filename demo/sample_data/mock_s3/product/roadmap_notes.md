# Product Roadmap Notes

## UI Demo Priority

The first workshop version focuses on query-only Agent Chat. Online ingestion,
production ACL, and full image retrieval are advanced follow-up items.

The P0 user experience is a working Streamlit demo: ask a question, see a
streaming answer, inspect citations, compare Milvus recall with reranked
evidence, and review the trace. The UI should feel like a developer workbench,
not a marketing landing page.

## Customer Concern Coverage

产品路线图已经覆盖 S3 data freshness 展示、citation source cards 和
Agent trace。细粒度 permission 与 production ACL 尚未进入当前路线图，
仍属于生产化阶段的未覆盖项。

## Workshop Audience

The demo supports three audiences: users who want an immediate UI, developers
who want notebooks and code, and vibe-coding users who want step-by-step prompts.

## Developer Path

Developers can run notebooks for ingestion, embedding, schema creation, hybrid
search, Agentic RAG, and Streamlit UI. The local fallback does not require a
Milvus server, while pymilvus scripts show how to create real collections.

## Vibe Coding Path

Vibe-coding participants can ask their coding agent to scaffold ingestion,
generate Milvus schema, implement hybrid retrieval, add evidence grading, and
build the Streamlit UI. The sample data gives those prompts concrete filenames,
fields, and expected answers.
