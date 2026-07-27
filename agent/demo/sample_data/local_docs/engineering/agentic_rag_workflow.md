# Agentic RAG Workflow

## LangGraph Nodes

The demo workflow is classify intent, decide whether retrieval is needed,
check permission, select tools, rewrite or decompose the question, run one or
more retrieval calls, rerank, grade evidence, perform targeted supplementary
retrieval, generate an answer, and verify citations. The trace records each
decision so the UI can explain why an answer was produced.

## Query Classification

`classify_query` separates intent from topic. Private queries run a permission
gate, select bounded knowledge tools, and create one to three subqueries.
Most workshop questions need retrieval. General greetings do not.
Architecture questions include S3 sync, Milvus retrieval, embedding,
image_vector, and Agentic RAG workflow topics.

## Query Rewrite

The query planner expands short Chinese questions into search-friendly phrases
and binds each subquery to a registered knowledge tool.
For example, "S3 文档同步流程" becomes "S3 document sync pipeline", "MinIO
bucket scanning metadata extraction chunking embeddings Milvus insertion", and
"object storage ingestion architecture". This gives hybrid retrieval both
semantic and keyword hooks.

## Retrieve, Rerank, Grade

Each search tool owns its metadata filters and recalls candidates using vector
search plus sparse terms. Results from multiple tools are merged before the
reranker scores them. The evidence grader checks coverage for every planned
aspect and identifies missing evidence.

## Retry Policy

The maximum supplementary retrieval count is 3. Each retry targets a missing
aspect and preserves earlier evidence. If coverage is still insufficient, the
answer identifies the unresolved evidence instead of inventing specifics.

## Streaming Answer

Answer chunks are exposed only after retrieval, reranking, grading, generation,
and citation self-check finish. The UI then displays the final tool plan,
recalled candidates, reranked evidence, metrics, and citations.
