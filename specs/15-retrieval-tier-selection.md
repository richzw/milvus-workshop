# 15 — Retrieval Tier Selection and Cost Model

Status: draft · Last updated: 2026-08-27

## 1. Purpose

Every other spec answers *how* a retrieval capability works. This one answers
*whether this project should be paying for it at all*, and what the cheaper
rung below it is. It defines a retrieval complexity ladder, the inputs that
select a rung, the per-query and per-corpus cost model, and the migration cost
of the embedding decisions made in [`10a-openai-text-embedding.md`](./10a-openai-text-embedding.md).

The workshop demo deliberately starts near the top of the ladder because its
teaching goal is Milvus 3.0 hybrid/StructArray retrieval. That is a teaching
choice, not an engineering default, and this spec makes the difference explicit
so participants do not copy the architecture without copying the reasoning.

## 2. Selection inputs

A tier is selected from five measured inputs, never from architectural taste.

| Input | What to measure | Pushes down the ladder | Pushes up the ladder |
| --- | --- | --- | --- |
| Data freshness | Time from source edit to retrievable | Answers must reflect edits within minutes | Hours-to-days staleness is acceptable |
| Corpus churn | Fraction of documents added/changed per day and per month | > 10% daily churn | < 5% monthly churn |
| Query patterns | Share of exact-term/identifier queries vs. paraphrased or conversational ones | Keyword, product name, error code, invoice-id shaped | Synonym-heavy, conversational, "find alternatives to X" |
| Scale and latency | Queries/day and the accepted P95 budget | < 1K queries/day, generous budget | > 10K queries/day, sub-50ms budget |
| Team capability | Who owns chunking, embedding fingerprints, and eval | No owner for re-embedding or chunk eval | Dedicated owner and an existing eval harness |

The workshop corpus is a synthetic, effectively static demo set queried by a
handful of workshop participants. On these inputs alone it is a **T0/T1
corpus**; it runs at T2+ only to demonstrate Milvus 3.0 capabilities, and each
of those capabilities remains gated by [`70-quality-and-evaluation.md`](./70-quality-and-evaluation.md).

## 3. The tier ladder

| Tier | Name | Mechanism | Chunking required | Freshness | Model-switch cost | Status in this repo |
| --- | --- | --- | --- | --- | --- | --- |
| T0 | Lexical only | Milvus BM25 Function over `kb_chunks`, no dense lane | No | Perfect | None | Implemented as `RETRIEVAL_TIER=lexical_only`; reference baseline arm (§ 5) |
| T1 | Lexical + bounded query transformation | T0 plus `identity/rewrite/step_back/decompose` and the entity catalog | No | Perfect | None (rule policy, or prompt-only if the optional transformer is configured) | Implemented as `RETRIEVAL_TIER=lexical_rewrite`; see [`12-agent-workflow.md § 5.3`](./12-agent-workflow.md#53-plan_retrieval) |
| T2 | Hybrid: pre-embedded dense + BM25, then rerank | `flat_hybrid` profile, `reranker_top_k` selection | Yes | Stale until re-ingest | Full re-ingest | **Current default** (`RETRIEVAL_TIER=hybrid_dense`) |
| T3 | On-the-fly embedding | Embed the shortlist at query time, no stored dense vectors | Yes (for the shortlist) | Perfect | One call-site change | Not implemented; recorded option (§ 7) |
| T4 | Hot/cold embedding tiers | Pre-embed the frequently accessed subset, embed the tail on demand | Yes | Mixed | Re-embed only the hot subset | Not implemented; recorded option (§ 7) |
| T5 | Full pre-embedding at scale | Whole corpus pre-embedded, ANN-only serving | Yes | Stale | Full re-embed plus downtime | Out of MVP scope |

Two properties of the ladder matter more than the individual rungs:

- **T0/T1 need no chunking decision at all.** Chunk size, overlap and boundary
  policy—the entire cost of [`70-quality-and-evaluation.md § 5`](./70-quality-and-evaluation.md#5-chunk-configuration-evaluation)—is introduced by T2, not by RAG as such.
- **Model-switch cost rises monotonically.** T0/T1 are immune to embedding
  model deprecation; T2 and T5 pay a full re-ingest for it (§ 7).

T1's transformation defaults to the deterministic rule policy
(`QUERY_TRANSFORMER=rule_based`), so the measured arm in § 5 makes zero provider
calls and costs nothing per query. The § 6 rewrite-call anchor applies only when
the optional model-backed transformer is explicitly configured.

`struct_element`, `struct_two_stage` and `struct_fused` in
[`12-agent-workflow.md § 4`](./12-agent-workflow.md#4-tool-catalog) are refinements *inside* T2, not additional
tiers. They change how dense candidates are produced, not whether the corpus is
pre-embedded.

## 4. Escalation rule

A tier is escalated only by an observed failure, and only after the cheaper rung
has been measured on the same golden set:

1. Run T0. If the golden set passes the [`70-quality-and-evaluation.md § 4.1`](./70-quality-and-evaluation.md#41-correctness-gates) correctness gates, stop.
2. Failure mode *"the document exists but retrieval never finds it"* → T1.
   Bounded transformation and the entity catalog are prompt-level changes with
   no re-indexing, so this step is reversible within one deploy.
3. Failure mode *"the right documents rank below the wrong ones"* → T2, and
   only if the added P95 latency is inside the accepted budget.
4. Inside T2, corpus properties select the variant: high churn favours T3,
   a clear access-pattern skew favours T4, and large stable scale favours T5.

Escalation without a recorded failure mode and a comparative report is a spec
violation, not an optimization.

## 5. Lexical-only baseline arm

Because T2 is the default, its *incremental* value must stay visible. Every
comparative retrieval report defined in [`70-quality-and-evaluation.md § 4.2d`](./70-quality-and-evaluation.md#42d-retrieval-tier-comparison)
carries a `lexical_only` arm alongside the dense/hybrid arms, using the same
corpus, questions, permission and version scopes, reranker and generator. The
arm is not expected to win; it is the denominator that tells the workshop what
the embedding pipeline actually bought.

`demo/scripts/run_tier_eval.py` executes the three arms and writes one strict
`retrieval-tier-eval-v1` report. A lexical arm wraps the flat adapter so only
its BM25 lane is reachable — no dense recall, no StructArray profile and no
image lane — and `lexical_only` additionally runs with an empty entity catalog
and the identity transformer, because T0 is defined as retrieval without either.
Permission, version scope, reranker, generator and the golden fixtures are
identical across arms.

Measured on the synthetic corpus (deterministic providers, 15 golden
questions):

| Arm | Recall@20 | Selected Recall@5 | Required-fact coverage | Abstention accuracy | Cases passed |
| --- | --- | --- | --- | --- | --- |
| `lexical_only` (T0) | 0.67 | 0.67 | 0.65 | 0.73 | 8/15 |
| `lexical_rewrite` (T1) | 1.00 | 1.00 | 0.98 | 1.00 | 14/15 |
| `hybrid_dense` (T2) | 1.00 | 1.00 | 0.98 | 1.00 | 14/15 |

The T0 → T1 step carries the entire measured gain; T2 adds **zero** quality
delta over T1 on this corpus, so the report's `default_tier_justification`
resolves to `teaching_goal_only`. Local latency numbers come from the in-memory
adapter and are not evidence about a Milvus deployment.

## 6. Cost and latency model

Costs are parametric. The absolute unit prices below come from the source memo
(§ 10) and are illustrative anchors for the workshop discussion, not repo
targets and not a live price list.

```text
query_embed_cost   = 1 query × avg_query_tokens × price_per_token
shortlist_cost(T3) = shortlist_k × avg_chunk_tokens × price_per_token
corpus_cost(T2/T5) = corpus_chunks × avg_chunk_tokens × price_per_token   # per re-ingest
storage_bytes      = stored_vectors × 1024 dims × 4 bytes                 # see 10a
```

Worked anchors (source memo, `text-embedding-3-small` at $0.02/1M tokens):

| Scenario | Formula | Result |
| --- | --- | --- |
| T3 shortlist, 50 chunks × 500 tokens | 25,000 tokens × $0.02/1M | ≈ $0.0005 per query |
| T3 at 1K queries/day | above × 30 days | ≈ $15/month, $0 storage |
| T2/T5 corpus of 1M chunks × 500 tokens | 500M tokens × $0.02/1M | ≈ $10 per full re-ingest |
| T5 storage at 1M vectors (source memo's 1536 dims, **not** this repo's 1024) | — | ≈ 6GB; the same corpus at 1024 dims is ≈ 4GB |
| T1 rewrite call, **only** with `QUERY_TRANSFORMER=auto\|openai` | one small-model call | ≈ $0.001 per query |

Latency bands, to be replaced by this repo's own measurements under the
[`70-quality-and-evaluation.md § 4.3`](./70-quality-and-evaluation.md#43-performance) hardware profile:

| Tier | Retrieval-stage band |
| --- | --- |
| T0 / T1 | < 50ms |
| T2 | 100–500ms |
| T3 | 200–500ms |
| T4 | 50–100ms on the hot path |
| T5 | < 50ms |

The decisive observation for this project: at T2/T3 scale the embedding *spend*
is small; the **latency** is the real constraint, and every tier above T1 also
buys a permanent chunking and re-ingest obligation. Cost, latency and
throughput remain operational metrics under § 4.3 and are never folded into a
quality score.

## 7. Embedding-model lifecycle

[`10a-openai-text-embedding.md § 6`](./10a-openai-text-embedding.md#6-minimal-change-boundary) fixes 1024 dimensions and forbids mixing
providers inside one collection, so a provider or model change is a full
re-ingest. That is a T2 property, and its cost must be stated rather than
discovered during an incident:

| Path | Re-embedding scope | Serving impact |
| --- | --- | --- |
| T2/T5 full re-ingest | Entire corpus | Rebuild + reindex window; fingerprint mismatch blocks startup until complete |
| T4 hot/cold | Hot subset only (≈ 20% of documents under a Pareto access pattern) | Cold tail degrades to on-demand embedding, no hard downtime |
| T3 on-the-fly | None | Change the model id at one call site |

Required behaviour today, at T2:

- The embedding fingerprint already recorded in chunk metadata is the migration
  gate. A model change without a matching re-ingest fails startup rather than
  silently mixing vector spaces: the flat retriever samples stored chunks during
  startup readiness and refuses any fingerprint other than the configured one,
  alongside the per-result check that already guards every search and lookup.
  Migration itself stays inside the allow-list of
  [`14-milvus-3-native-capabilities.md § 6`](./14-milvus-3-native-capabilities.md#6-schema-evolution-and-backfill). This is what makes the cost above
  visible instead of latent.
- A migration plan names the target model, the corpus size, the estimated
  re-embed cost from § 6, and the rebuild window before the change is applied.
- T3 and T4 stay out of MVP scope. They are recorded here so that a workshop
  participant facing high churn or a deprecated model has a documented exit
  that does not require re-architecting.

## 8. Sub-query tier routing (proposed, eval-gated)

`decompose` already splits a question into two or three plan items with distinct
roles. Today every item executes the same registered retrieval profile. The
recorded extension is to annotate each plan item with the cheapest sufficient
tier — an exact product-name aspect runs T0/T1, a paraphrased aspect runs T2 —
so a decomposed query pays dense cost only for the aspects that need it, and
parallel execution makes the query's latency the maximum of its items rather
than their sum.

Constraints if this is implemented:

- The tier is chosen by the deterministic planner policy, never named by the
  model, exactly as the retrieval profile is chosen today.
- Tier annotation cannot widen permission, tool allow-list or version scope, and
  cannot change `candidate_pool_fingerprint` semantics.
- Mixed-tier candidates merge on `chunk_id` like any other multi-lane result and
  carry their tier in trace provenance.
- Adoption requires a § 4.2d comparative report showing no quality regression
  against uniform-T2 routing.

## 9. Acceptance criteria

- The default tier, its selection inputs and its escalation history are stated
  in this spec and reflected in [`99-key-decisions.md § D50`](./99-key-decisions.md#d50--retrieval-complexity-is-a-measured-ladder-not-a-default).
- Every comparative retrieval report carries a `lexical_only` arm.
- No tier escalation ships without a recorded failure mode plus a comparative
  report on the same golden set.
- Any embedding model or provider change carries a migration plan with the
  re-embed scope, estimated cost and rebuild window; fingerprint mismatch fails
  startup.
- T3/T4/T5 remain labelled as unimplemented options; no spec may cite them as
  available behaviour.

## 10. Cross-references

- ← Depends on: [`10a-openai-text-embedding.md`](./10a-openai-text-embedding.md), [`11-ingestion.md`](./11-ingestion.md)
- → Constrains: [`12-agent-workflow.md § 4`](./12-agent-workflow.md#4-tool-catalog), [`14-milvus-3-native-capabilities.md`](./14-milvus-3-native-capabilities.md)
- ↔ Evaluated by: [`70-quality-and-evaluation.md § 4.2d`](./70-quality-and-evaluation.md#42d-retrieval-tier-comparison), [`70-quality-and-evaluation.md § 4.3`](./70-quality-and-evaluation.md#43-performance)
- ↔ Decisions: [`99-key-decisions.md § D50`](./99-key-decisions.md#d50--retrieval-complexity-is-a-measured-ladder-not-a-default), [`99-key-decisions.md § D51`](./99-key-decisions.md#d51--embedding-model-migration-is-planned-and-fingerprint-gated), [`99-key-decisions.md § D52`](./99-key-decisions.md#d52--sub-query-tier-routing-is-planner-owned-and-eval-gated)
- Source: Rafael Pierre, ["RAG Is Simpler Than You Think"](https://www.lighthousenewsletter.com/p/rag-is-simpler-than-you-think), Lighthouse AI newsletter, 2026-06-10
