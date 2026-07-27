# TTL Memory Policy

## Temporary Uploads

Temporary uploaded documents and short-term conversation memory use `expires_at`
for TTL demos. Expired records should be filtered out before retrieval.

Temporary files should be treated as bounded knowledge. A user can upload a
draft policy or design note for one session, but the record should expire after
the configured TTL. In Milvus this is represented with scalar fields such as
`created_at` and `expires_at`, plus a metadata flag describing the source.

## Bounded Memory

The conversation_memory collection stores user, assistant, system, and summary
records. Session summaries can live longer than short-term turns.

## Retrieval Rules

Memory retrieval first filters by `session_id`, then removes records where
`expires_at` is older than the current time. Only live records are considered
for vector search. This prevents stale temporary uploads from being cited in
Agent Chat answers.

## Demo Scope

The workshop does not implement production authentication or ACL. It keeps the
TTL fields visible so participants can see how lifecycle governance would fit
beside metadata filters such as department, source type, and document type.
