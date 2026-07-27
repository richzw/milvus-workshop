# Agent UI MVP

## Demo UI

The Streamlit Chat UI is natural-language-only. Users submit questions; the
Agent selects knowledge tools and internal metadata filters. A stable generated
session id supports bounded multi-turn Memory. The UI shows answer text,
chunk/page citations, source cards, merged tool recall, reranked candidates,
coverage, Agent trace, and session Memory.

## Chat Tab

The Chat tab accepts a user question and streams the final answer. It shows
source cards for each citation, including title, `source_uri`, page number when
available, section, and `chunk_id`. A good demo question is "我们 S3 文档同步流程是怎么设计的？"
because it exercises S3 docs, PDF context, image examples, and Milvus features.

## Evidence Tab

The Evidence tab shows two tables. The first table is the Milvus recall list
with rank, hybrid score, source, chunk, updated time, and snippet. The second
table shows reranked results with old rank, rerank score, selected flag, and
source details. This demonstrates that Milvus handles high recall and the
reranker handles final precision.

## Memory Tab

The Memory tab shows the latest recall/write status, bounded summaries used by
the current follow-up, and live records from only the active session. Explicit
`expires_at` filtering removes expired records consistently in local and
Milvus paths. Users can clear the active conversation and its Memory without
dropping collections or affecting another session. Memory helps resolve
follow-up references but never acts as KB evidence or a citation source.

## Trace Tab

The assistant message streams a compact execution timeline while the Agent is
working: intent, terminology and version resolution, permission, selected
tools, retrieval, reranking, evidence grading, retries, and citation
self-check. The panel collapses after completion, and the Agent Trace tab
replays the same safe events.

Answer text is displayed only after citation self-check succeeds. Raw prompts,
rewritten queries, document text, metadata filters, secrets, and dependency
errors are excluded from presentation events. The full terminal trace remains
available in an Advanced expander for workshop inspection. Metadata filters
are tool implementation details and are not editable UI controls.
