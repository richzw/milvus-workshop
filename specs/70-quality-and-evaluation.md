# 70 — Quality, Evaluation, and Demo Safety

Status: draft · Owner: workshop author · Applies to: all specs · Last updated: 2026-08-27

## 1. Purpose

本文把“demo 能跑”升级为可验证合同：数据质量、检索质量、回答忠实度、流程不变量、复现性和本地 demo 安全边界。性能数字在实际栈尚未验证前只记录基线，不伪造 SLA。

质量流程遵循 eval-driven development（EDD），并把 metric set 当作需要持续维护的产品资产。指标候选来自两类证据：真实 trace 中观察到的失败，以及产品目标与 day-one hard constraints。active set 必须同时覆盖 goal、guardrail、operational 三种角色，但保持能支撑决策的最小规模；每个维度绑定独立 grader 和明确动作，报告永远不输出单一混合分数。循环细节见 [§ 2a](#2a-eval-driven-development-loop)，准入与退休合同见 [§ 4.0](#40-metric-portfolio-contract)。

## 2. Test layers

| Layer | Scope | Required examples |
| --- | --- | --- |
| Contract/unit | pure transforms and state transitions | stable ids, rule-based classification, entity/domain resolution, version-scope filters, permission gate, plan bounds, retry cap, citation subset |
| Component | adapters, generators and stores | local/MinIO read, schema validation, insert/search round trip, LLM classifier, reranker, context-compressor and answer-generator adapters; the model-backed compression paths § 3 keeps out of the golden set live here |
| Workflow | LangGraph terminal behavior | direct answer, permission denial, single-tool retrieval, multi-tool comparison, multi-hop supplement, retry exhausted, self-check failure |
| UI | rendered result consistency | no manual metadata controls; every rendered tab shares one `query_id`; citations match evidence; fallback/error labels |
| Offline RAG eval | seeded corpus + golden fixtures | retrieval recall, citation correctness, required-fact coverage, abstention |
| Workshop smoke | clean setup path | ingest seed corpus, ask one golden question, receive answer-or-abstain + trace |

Tests follow `AGENTS.md`: deterministic, behavior-oriented, clear names, and no disabled checks.

## 2a. Eval-driven development loop

评测是随实现共同演进的 living artifact，按以下循环维护；每一条都是硬性流程要求：

1. **分清候选来源**。产品目标和 permission、citation、version/session isolation、输出协议等 hard constraints 在实现前登记为 goal 或 guardrail 候选；除此之外的行为质量维度默认只从真实 output/trace 的 observed failure 派生。通用 metric catalog 只能用于探索，不能替代产品内的失败证据。
2. **从零时先做 error analysis**。尚无失败 taxonomy 时，先只标两个字段：一条自由文本 `review_note`，说明发生了什么、哪里不对；一个 `overall_pass` boolean。人工阅读 30–50 条有代表性的 traces，按 note 聚类出命名失败类别，再为重复出现、需要持续跟踪的类别建立单维度 boolean evaluator。后续可对高风险变更扩大样本，但不得用更大的固定数字替代这一步。
3. **先修一次性规格缺口**。若错误来自未写清楚的 JSON/date/plain-text/AI disclosure 等简单格式要求，先修 prompt 或 schema 并加 contract test。只有跨输入仍存在、简单修改无法可靠泛化的 failure mode，或本来就是 hard constraint 的格式合同，才进入长期 metric registry。
4. **metric 必须改变决策**。每个候选在启用前写明数值变化后谁会做什么，例如 block deploy、回滚 prompt/model/index/chunk config、打开归因调查或调整容量。无法改变任何动作的字段只保留为 trace diagnostic，不得成为 active metric。conversation length 这类一值多义的 proxy 不得单独 gate。
5. **失败即 fixture**。每个新观察到的失败（实现期、review 或 Workshop 现场）必须在修复它的同一变更内进入 `demo/eval/` fixture；只修复不加 fixture 视为未完成。fixture 的目标是复现行为，metric 的目标是支持持续决策，两者不可混为一谈。
6. **一次只动一个变量**。比较 prompt、model、serving/index 配置时固定其余变量分次评测；同一次 run 同时改多个变量得出的结论无效。
7. **每次 run 后读 transcript**。失败先区分 capability gap 与 task-quality 问题（歧义题目、错误 golden label、不可能任务、可利用的捷径），并排除 scaffold entanglement（tool 命名、context 组装、prompt 措辞）之后，才允许归因于模型能力。prompt rewrite、model swap、新 feature 或 retrieval architecture 变化后必须重跑 error analysis，因为 failure distribution 已可能改变。
8. **防止饱和与 Goodhart**。持续优化的 metric 必须定期用新 human labels 复核，避免围绕固定 evaluator 过拟合。非 guardrail metric 连续约三个月保持 100%、没有捕获新失败且不再改变决策时，从 active registry 退休；仍有回归价值的 deterministic fixtures 降级为 contract/regression tests 保留。guardrail 即使长期全绿也不因“没有信息量”删除。

## 3. Golden dataset contract

`eval/questions.json` and `eval/golden_answers.yaml` are versioned with the sample corpus. Each question includes category, expected sources and optional filters; each golden answer includes required facts and citations.

Golden set 必须包含失败与负例（abstention、permission denial、clarification 已在下方最小集内）——只有成功样例的 golden set 无法测试 grader 的辨别力，不合格。若人工标注者对某条 golden label 或 abstain 判断存在分歧，先解决分歧（澄清 rubric 或改写题目）再入集：人都无法一致标注的维度，不允许交给任何自动 grader。

Minimum fixture set covers:

- local Markdown retrieval;
- MinIO/S3-source retrieval;
- PDF page citation;
- bilingual query rewrite;
- query-transformation fixtures for identity, colloquial rewrite, one
  step-back background+primary pair and bounded multi-aspect decomposition;
- product/game terminology resolution covering `GO按钮`, `跳转按钮` and `领取按钮`, plus one same-spelling cross-domain ambiguity;
- two editions of one logical document, with current-only, exact-version and explicit version-comparison questions;
- automatic tool routing to a policy/product/engineering domain;
- comparison requiring at least two tools;
- multi-hop retrieval where second query depends on first-hop evidence;
- permission denial before retrieval;
- low-evidence retry then success;
- an abstention whose retrieval stopped early (`no_progress`);
- standalone and same-session follow-up explanations of `Milvus 3.0 Force
  Merge`, both citing only the live Force Merge chunk;
- one weak or indirect focused chunk that must still abstain;
- reranker fallback via `scenario.reranker=fallback`;
- a cache-hit case whose `prelude` re-asks the same question in one session;
- a Memory state-change case whose `prelude` stores an explicit preference;
- one nullable-image-vector record without enabling image retrieval.
- one long-document StructArray family with repeated parent hits, same-element scalar predicates and stable passage ids; fixtures cover element search, a parallel-array false positive, a two-aspect EmbeddingList shortlist followed by citeable element resolution, and an entity-only hit that must not become evidence.

以下两项**不属于** golden set，由 contract 层拥有：selective compression 去掉无关
文本而不改变 required facts/citations，以及非法压缩输出整体回退原 context。两者都
只能由 model-backed compressor 产生，而离线 golden run 按 [§ 4.6](#46-rag-eval-report-contract)
强制 deterministic provider 且不访问网络；在 CLI 里塞 fake client 会让报告声称一个
没有任何模型产生过的 projection。因此它们按 [§ 6](#6-external-capability-verification-matrix)
的既定分工，以 fake-client contract test 验证，退休条件与 golden fixture 相同：行为
变化必须在同一变更内更新那些 test。

同理，`retry_exhausted` 与 `duplicate_retry_query` 的终止路径需要受控的检索轮次，
由 transition-parity 与 workflow contract test 拥有；golden set 只保留 `no_progress`
这一条可由真实语料自然产生的提前终止。

需要受控前置状态的 case 使用 strict `scenario` object。v2 允许三个可选 key：

- `permission`（`allow|deny`，默认 `allow`）：注入 permission checker；
- `reranker`（`rule_based|fallback`，默认 `rule_based`）：`fallback` 即普通的
  `RERANKER=auto` 无凭据构建——configured wrapper 降级到确定性 rule reranker 并
  报告 `not_configured`，因此离线 CLI 里不出现任何 test double；
- `prelude`：1–3 条 bounded 问题，在被评分的这一轮之前，用**同一个 workflow
  实例和同一个 `session_id`** 依次跑完并丢弃结果。

`prelude` 通过正常 `stream()` 路径建立前置状态，不伪造 cache record 或 Memory
记录，因此 cache hit、Memory state change 与 same-session follow-up 都由生产写
入路径本身产生。声明了非默认 `permission` 的 fixture 必须使用
`scenario_workflow_factory` 注入依赖，普通 factory 不得忽略 setup 后继续运行；
只声明 `prelude` 的 fixture 不需要注入，因为它不改变任何依赖。声明了依赖但被
factory 忽略的 case 会由对应的 `expected_*` 断言失败，而不是静默通过。

`prelude` 轮次不计入被评分轮的 latency，也不产生自己的 case 结果。

## 4. Metrics and gates

### 4.0 Metric portfolio contract

active metric set 是一个受版本控制、可审查、可退休的 registry，不是 runner 能输出多少字段的列表。三类角色缺一不可：

| Role | 回答的问题 | 默认来源 | 数值变化后的典型动作 |
| --- | --- | --- | --- |
| Goal | 正在为之构建的质量是否改善？ | product goal、error analysis 中反复出现的 generalization failure | 接受或拒绝实验，回滚 prompt/model/retrieval/chunk config，或打开归因调查 |
| Guardrail | 绝不能破坏的合同是否发生回归？ | requirement、compliance/safety constraint、past incident | block deploy；已发布版本触发 rollback 或 incident review |
| Operational | 一次请求要花多少资源、系统能处理多快？ | trace 自动采集 | 调整 capacity、provider、预算或优化优先级；没有已批准 budget 时不伪装成 quality gate |

```text
┌──────────────────── Candidate sources ─────────────────────┐
│  30–50 trace error analysis   Product goals / constraints  │
└───────────────────────┬─────────────────────────────────────┘
                        ▼
              ┌───────────────────────┐
              │ Candidate review      │
              │ repeated/generalizes? │
              │ decision-bound?       │
              │ evaluator affordable? │
              └──────┬─────────┬──────┘
                     │ yes     │ one-time omission / no action
                     ▼         ▼
       ┌────────────────────┐  ┌─────────────────────────────┐
       │ Active registry    │  │ Prompt/schema fix + test,   │
       │ goal / guardrail / │  │ fixture-only or discard     │
       │ operational       │  └─────────────────────────────┘
       └─────────┬──────────┘
                 ▼
       ┌────────────────────┐       ┌─────────────────────────┐
       │ Eval run + report  │──────▶│ Deploy/rollback/        │
       │ value + delta      │       │ investigate/capacity    │
       └─────────┬──────────┘       └─────────────────────────┘
                 ▼
       ┌────────────────────┐
       │ Monthly/quarterly  │───▶ keep / modify / retire
       │ metric review      │     (guardrails persist)
       └────────────────────┘
```

metric candidate 只有同时满足以下条件才可进入 active registry：

1. 有证据来源：观察到的 failure cluster，或明确的 product goal/hard constraint；
2. 只回答一个可解释的问题，不能把 unrelated dimensions 混成 overall score；
3. 写明 owner、目标/阈值或观察预算，以及越界后执行的具体动作；
4. 有独立、可版本化的 grader 和 dataset segment，且错误归因不依赖另一个 metric 猜测；
5. 预计运行频率与成本可接受。等价信号优先使用 L1 code evaluator；低重要性且只能靠昂贵 L2/L3 维护的候选应删除；
6. 写明 review cadence 与 retirement 条件。`candidate → active → retired` 状态变化必须经过 review，不允许静默增删。

Canonical registry 是 strict JSON `demo/eval/metric_registry.json`，schema version 为 `eval-metric-registry-v1`。每项至少包含 `metric_id`、`role`、`question`、`source`、`owner`、`grader_id`、`grader_version`、`grader_layer`、`dataset_segment`、registered `measurement`、`threshold_or_budget`、`decision_action`、`cost_class`、`run_cadence`、`retirement_condition`、`status`、`introduced_at`、`last_reviewed_at` 和可选 `retirement_reason`。报告必须嵌入对 validated、按 `metric_id` 排序后 canonical JSON 的 SHA-256 checksum；whitespace 和 object-key order 不改变 checksum，registry 不兼容时不得与旧 baseline 计算 delta。

Binding enums are `role=goal|guardrail|operational`、`source.kind=observed_failure|product_goal|hard_constraint|incident`、`grader_layer=L1_programmatic|L2_judge|L3_human`、`cost_class=near_zero|metered|manual`、`run_cadence=per_pr|nightly|release|monthly|incident`、`status=candidate|active|retired`。`threshold_or_budget.mode` 是 `gate|budget|baseline_only`；gate/budget 还必须给出 `operator=eq|gte|lte`、finite `value` 和 `unit`。`decision_action` 至少选择一个 `block_deploy|rollback|reject_change|open_investigation|capacity_review`，不得使用 `none` 或自由文本代替。日期使用 ISO `YYYY-MM-DD`；unknown fields、重复 `metric_id`、retired metric 缺少 reason、active metric 缺少任一必填字段均 fail closed。

v1 只注册 `dataset_segment=rag_core`，它对应本报告的完整 golden-question case set；拼错或未实现的 segment fail closed。后续新增 segment 必须同时实现 case selector、registry allowlist 与隔离测试，不能只增加标签。

`measurement` 不接受任意 JSONPath，只能是 runner 注册的 scalar key。v1 初始 keys 为 `aggregate.recall_at_k`、`aggregate.selected_context_recall_at_5`、`aggregate.required_fact_coverage`、`aggregate.citation_resolve_rate`、`aggregate.abstention_accuracy`、`aggregate.permission_bypass_count`、`aggregate.cross_version_contamination_count`、`latency.latency_ms.p95`、`operational.cost_per_request` 和 `operational.completed_requests_per_hour`。新增 key 需要 code + spec + registry 同一变更，防止配置读取未审查的 report 字段。

### 4.0a Initial active registry

第一版只保留下列能直接驱动发布或调查的 core metrics。§ 4.2、§ 4.2a、§ 4.2b 和 § 5 的其余字段是 feature scorecard 或 diagnostic；只有通过 § 4.0 准入后才成为长期 active metric。

| Metric id | Role | Signal | Gate / budget | Decision action |
| --- | --- | --- | --- | --- |
| `goal.retrieval_recall_at_20` | Goal | expected sources 的 Recall@20 | `≥ 0.90` | 拒绝并调查 embedding/index/retrieval/corpus change |
| `goal.selected_context_recall_at_5` | Goal | generation context 对 required sources 的 Recall@5 | `≥ 0.90` | 回滚 reranker/query-transform/chunk-config change |
| `goal.required_fact_coverage` | Goal | required facts 的覆盖率 | `≥ 0.90` | 回滚 evidence-selection/generation change 并 review transcript |
| `guardrail.citation_resolve_rate` | Guardrail | citation 是否全部解析到同 query selected context | `= 1.0` | block deploy；已发布则 rollback |
| `guardrail.abstention_correctness` | Guardrail | insufficient-evidence case 是否拒答 | `= 1.0` | block deploy 并检查 grader/generator 边界 |
| `guardrail.permission_bypass_count` | Guardrail | permission denial 前发生的 private retrieval 次数 | 有 permission-denial case 时 `= 0`；无适用 case 时必须为 `evaluation_incomplete` | block deploy；按安全回归处理 |
| `guardrail.cross_version_contamination_count` | Guardrail | 非 comparison case 混入其他版本的次数 | `= 0` | block deploy 并回滚 filter/version-resolution change |
| `operational.end_to_end_latency_p95` | Operational | end-to-end latency P95 | Phase 0 baseline；budget 未批准前不 gate | 超预算时做 stage attribution 与 capacity/provider 调查 |
| `operational.cost_per_request` | Operational | provider calls、input/output tokens、observed/estimated cost | 按 provider profile 单列 budget | 调整 provider/model/context budget；不得与质量混分 |
| `operational.completed_requests_per_hour` | Operational | 固定并发与硬件 profile 下的完成吞吐 | Phase 0 baseline；profile 不同不可比较 | 评估 capacity 或定位 latency/backpressure |

Goal threshold 采用已提交的 teaching profile，只能通过 reviewed spec/baseline change 调整。Operational metric 必须固定 runtime、provider、concurrency 和 dataset profile；否则只记录当前 observation，不计算误导性 delta。像 conversation length、raw tool count、compression ratio 这类一值多义或只能解释机制的字段默认是 diagnostic，不能独立触发发布动作。

任何需要适用样例或 denominator 的 metric 都不得把“没有样例”折算为 0 或 100%。以 permission guardrail 为例，report 同时输出 `permission_denial_case_count`；该值为 0 时，`permission_bypass_count` 为 `null`，decision status 为 `evaluation_incomplete`。

### 4.0b Grader layers and evaluator budget

每个质量维度绑定唯一 grader；禁止单一 judge 输出混合“总分”（god evaluator）。Grader 按成本分层，低层为高层过滤：

- **L1 — Programmatic**（默认且当前唯一的 gating 层）：§ 4.1 全部 correctness gate 与 § 4.2 的检索/引用指标都是确定性代码检查——schema/枚举校验、citation-resolves、required-fact 字符串覆盖、tool-invocation 断言。可复现、可调试，离线 suite 不做网络调用。
- **L2 — LLM-as-judge**（可选，遵循本仓库 provider 约定显式 opt-in，且不进入默认 deterministic suite）：一个 judge 只评一个维度（如 faithfulness、answer relevancy），必须有书面 rubric、简短判定理由、evidence references 与 few-shot anchors。未校准的 judge 分数只能作参考信息，不得 gate 任何 phase 或 milestone。
- **L3 — Human**：golden label 的 ground truth、L2 校准基准和 L1/L2 分歧的仲裁者；由 workshop author 承担仲裁。

L2 judge 的校准合同：先由 L3 标注 ≥50 条包含失败样例的行；judge 与人工一致率（经 chance-agreement 修正）达到高 80%–90% 区间、且分歧样本已逐条分析归档后，该 judge 的分数才可用于 gate。rubric 或失败模式变化后必须重新校准。校准数字、judge token/cost budget 与分歧分析随 fixture 一起提交。相同发布决策能由 L1 回答时不得同时运行 L2；昂贵 evaluator 只运行在 L1 无法判断且 registry 标为 active 的 segment。

### 4.1 Correctness gates

本节是不可违反的 deterministic guardrail tests，不等同于要求把每一条都画在 metric dashboard 上。runner 可以输出这些检查的 registered reason code 用于定位；只有列入 § 4.0a 或后续通过准入 review 的信号，才参与长期 trend、baseline delta 和发布决策。

- Retry progress and cap: 100% of workflow tests terminate with
  `retry_count ≤ 3`; an unchanged evidence-state fingerprint terminates before
  another rerank/grade call, while a new provenance edge counts as progress.
  Supplementary `(tool, normalized query, version scope)` fingerprints are
  unique; a duplicate terminates with `duplicate_retry_query` before plan
  append or tool execution and does not increment `retry_count`.
- Focused single-evidence validity: a single chunk answers only at or above the
  ranking reranker's declared `strong_single_evidence_threshold` (both shipped
  implementations declare `0.80`; an undeclared or out-of-range value fails
  closed), with exact normalized section-name coverage, one authorized tool,
  focused/non-comparison intent, one requested aspect family and matching
  isolated version scope. Weak, indirect, multi-aspect, exhaustive, comparison
  and multi-tool single-chunk cases abstain.
- Evidence diagnostics: every grade exposes a registered `evidence_basis` and
  actionable `missing_aspects`; generic citation/document placeholders are
  absent from trace and retry planning.
- Citation validity: 100% of emitted citations resolve to selected context from the same query.
- Fixture integrity: every golden citation exists in seeded `kb_chunks`.
- StructArray projection integrity: every projected passage maps bijectively to one `kb_chunks` identity with equal parent/version/checksum/vector fingerprint and rehydrates the authoritative text; order is deterministic and an oversize/mixed-parent corpus publishes no partial projection.
- StructArray filter correctness: `MATCH_ANY` and `element_filter` bind every scalar condition to one offset; the parallel-array false-positive fixture is rejected.
- StructArray evidence identity: element hits resolve bounded offsets to stable live `chunk_id`; out-of-range/mismatched offsets fail closed, while EmbeddingList, MATCH-qualified parent and collapsed entity hits contribute zero evidence/citations until element resolution.
- StructArray search-mode validity: `MAX_SIM*` is accepted only for the EmbeddingList subfield and regular `COSINE` only for the element subfield; one subfield/index is never reused across metric families.
- StructArray hybrid identity: same-parent element-only fusion may preserve offset; mixed-granularity hybrid is entity-level and cannot be presented as passage evidence. Collapse tests prove strategy/metric/topk validation and that sub-search `limit` bounds the hits available to collapse.
- Fallback-corpus version integrity: every curated offline record whose title
  names an allow-listed product version carries the same normalized
  `doc_version`; the minimal `Milvus 3.0` fallback document has at least two
  version-matched sibling sections so exact exhaustive queries exercise the
  normal multi-evidence rule instead of silently falling back to `current`.
- Error honesty: dependency failures never produce a normal grounded-answer status.
- Idempotency: ingesting the unchanged seed corpus twice leaves the same logical `(doc_id, doc_version, chunk_id)` set.
- LLM grounding: every model-generated citation marker is a subset of selected context; invalid output activates traced fallback.
- Provider isolation: deterministic tests and fallback paths make no external API calls.
- Image-embedding validity: every manifest image is embedded from the referenced
  image file, has exactly 768 finite L2-normalized values and carries the
  configured image-space fingerprint; caption placeholders, zero vectors,
  provider mixing and silent fallback are rejected before insert.
- Image-provider isolation: the default suite never imports Torch,
  Transformers or Pillow and never downloads weights; injected runtime tests
  cover processor/model input, pooling, normalization, dimension/output
  validation and sanitized load/inference failures.
- Image-retrieval validity: local and Milvus adapters accept the same bounded
  normalized query, force the image-only predicate, use COSINE scores, preserve
  caller filters, return no vectors publicly and fail closed on vector-space
  mismatch.
- MinIO isolation: the default suite never constructs the SDK client; injected
  contract tests prove recursive deterministic listing, bounded response
  cleanup, safe object keys and stable `s3://bucket/key` identities.
- Classification validity: every classifier result uses fixed intent/topic/retrieval-goal enums; malformed or provider-failed LLM output activates a traced rule fallback.
- Classification safety: untrusted Memory cannot trigger explicit memory/operation/sensitive/exhaustive routes, and classifier output cannot grant permission, choose arbitrary tools or construct filters.
- Recall-detector parity: every registered explicit/recent-question phrase produces the same action in the workflow gate and RuleBased classifier; “查找下我最近的三个问题是什么” deterministically bypasses KB retrieval.
- Reranker validity: model output must contain exactly the complete input
  `chunk_id` set once with finite scores in `[0, 1]`; invented, missing,
  duplicate, malformed or provider-failed output activates one whole-batch
  rule fallback with a sanitized trace reason.
- Reranker isolation: rule-based mode and fake-client model tests make no
  network calls; direct workflow construction remains deterministic, while
  configured builders expose the implementation/model that actually produced
  each query ranking.
- Reranker query fallback: after one registered primary failure, later rounds
  in the same query use deterministic fallback without another primary call;
  a separate query attempts the primary again.
- Reranker bounds: up to 120 merged candidates, including two or more
  exhaustive expansion sides, are ranked as one complete batch under the
  96,000-character input cap; a pre-retrieval terminal path reports
  `reranker_name=not_run`.
- Response-cache correctness: exact/semantic hits require current permission, compatible query constraints and live KB revision/version/checksum evidence; a hit preserves citation validity and makes zero tool/rerank/generation calls.
- Response-cache fail-closed: expiry, another session, low similarity, version/negation/scope mismatch, permission change, missing checksum or dependency failure all continue through normal RAG without exposing cached content.
- Response-cache routing: direct, Memory, operation, clarification and
  permission-denied paths make zero grounded-cache search calls; an allowed
  grounded-retrieval path makes at most one lookup inside
  `try_grounded_cache`, and authorized experience recall runs only after a
  cache miss.
- Typed stage outcomes: `classify_and_route` returns only `direct|retrieval`,
  `plan_retrieval` returns a non-empty bounded plan for every retrieval route,
  and `evaluate_evidence` returns exactly one of
  `answer|retry|abstain`; local and LangGraph expose no separate
  decide/select/rewrite/grade/retry-planning workflow nodes.
- Transition parity: table-driven tests cover every shared transition branch
  and impossible state combination; the local dispatcher follows the returned
  `next_node` rather than a parallel hard-coded order, and a compiled-graph
  harness proves local/LangGraph produce the same ordered composite stages,
  terminal status and retry count for direct, clarification, denial, cache hit,
  grounded answer, no-progress and retry-exhausted paths.
- Capability-gated parallelism: two or more independent ready retrieval items
  run concurrently only for an adapter declaring
  `supports_parallel_search=true`; merge/tool-call order remains plan-stable,
  dependent plans and unproven adapters remain sequential, and worker failure
  is attributed to `execute_tool_plan`. Persistence sink tests assert the
  current sequential order until a separate write-safety capability and
  deterministic failure aggregator exist.
- Tool authority: UI never supplies metadata filters; every search filter is produced by a registered tool and intersected with the permission decision.
- Plan bounds: at most three initial subqueries, three supplementary rounds and one registered tool per call.
- Retry fidelity: every supplementary query contains the normalized original
  product/feature/version surface forms; the `Milvus 3.0 Force Merge` fixture
  never rewrites to an unrelated S3 ingestion template.
- Query-transformation validity: every retrieval plan declares exactly one of
  `identity|rewrite|step_back|decompose`, produces at most three unique items,
  preserves named product/feature/version/negation/constraint terms, and never
  widens selected tools, permission, entity or version scope. Provider failure
  returns a labeled deterministic identity/rule result.
- Step-back grounding: a step-back plan always contains an original-retaining
  primary query; background-only evidence cannot satisfy a concrete named
  feature/version aspect or produce a grounded answer.
- Multi-source coverage: comparison answers either cover every planned side or explicitly abstain/report uncovered sides.
- Answer self-check: 100% of grounded terminal answers have `answer_validation.valid=true`.
- Context-compression provenance: every selective span matches the original
  source checksum, offset, quote and order; every summary/extraction unit maps
  to exact support spans. Selected source ids, citation map, version sides and
  evidence grade are unchanged by compression.
- Context-compression fallback: one unknown id, missing required side, empty or
  non-exact support, provider error or malformed output falls back to the
  complete original selected-context set. Derived summary/extraction wording
  is discarded and never becomes generation evidence; partial compressed/
  source mixtures are forbidden.
- Entity resolution: every configured terminology fixture records the expected `entity_id`; unresolved cross-domain collisions request clarification before retrieval.
- Version isolation: current/exact queries have zero cross-version contamination in recalled, selected and cited chunks; explicit comparisons keep citations partitioned and visibly labeled by `doc_version`.
- Product-version resolution: allow-listed `Milvus N.N` surface forms normalize
  to exact stored `vN.N`; unqualified decimals remain non-version text.
- Streaming order: trace-event sequence is contiguous and query-local; tool/retry paths emit their corresponding events; exactly one `final` terminates the stream.
- Streaming safety: grounded answer deltas occur only after a successful verification event; trace events contain no prompts, document bodies, credentials or raw exception text.
- UI progress: the primary Agent Trace presentation is a readable timeline driven by live events, while raw JSON is available only in a collapsed advanced view.
- Selective-Memory UI: bounded distributions cover retention classes,
  registered selection reasons, decay profiles and fact statuses; the full
  same-session lineage view resolves opaque source/supersession/parent ids
  without exposing content, values, vectors or selector prompts.
- Memory isolation: every recall/list/delete result belongs to the active `session_id`; another session and an expired/current-turn record have zero visibility.
- Memory grounding: Memory may resolve a follow-up but never creates a KB citation or turns insufficient KB evidence into `enough_evidence=true`.
- Memory chronological recall: requested recent questions are live `short_term/user` records from only the active session, ordered newest first, bounded to 20, exclude the current command, and never contain assistant/summary/selective values.
- Memory trace honesty: a skipped Conversation Memory lookup is distinguishable from a searched-but-empty lookup; mode, reason, bounded requested count and actual memory types contain no Memory payload.
- Memory lifecycle: after all answer deltas are consumed, requesting `final` idempotently persists the valid terminal turn; incomplete/cancelled streams write nothing; explicit clear removes only the active session.
- Memory degradation: typed recall/write failures preserve an otherwise valid answer, emit only bounded status/count metadata and never expose raw dependency text.
- Memory selection: ordinary turns cannot silently become durable user facts; explicit remember, correction, task transition and repeated operational outcomes produce their registered retention classes.
- Memory lineage: every durable fact resolves to same-session source events; correction creates a higher revision and supersedes rather than overwrites the prior fact.
- Memory consolidation recovery: an injected failure after journal enqueue, fact write or lifecycle-event write leaves one bounded pending entry; replay applies the exact plan once, marks it applied and creates no extra fact revision or lifecycle event.
- Memory conflict safety: disputed facts never enter active working state without deterministic resolution or explicit confirmation.
- Memory forgetting: decay changes rank only; expired, superseded and tombstoned records have zero visibility regardless of vector or decay score.
- Memory physical cleanup: pages are bounded to 100 examined primary keys,
  cursors are HMAC-authenticated and session/snapshot-bound, pending
  consolidation produces zero mutation, exact-id deletes never cross sessions,
  and retained facts keep resolvable source events.
- Memory anti-feedback: recall/display alone never updates `event_time`, `last_confirmed_at` or TTL; reconfirmation appends an event.
- Memory/cache separation: selection, consolidation and decay never create or refresh a grounded-response cache entry, and cache hit never promotes a memory.

### 4.2 RAG evaluation

每次 compatible golden run 计算 § 4.0a 中适用的 core goal/guardrail metrics。下列其余信号按 feature scorecard 或 diagnostic 独立记录，用于定位与候选发现；它们不会因为出现在报告里就自动成为 active metric：

- retrieval Recall@20 against expected sources;
- reranked Recall@8 and selected-context Recall@5;
- citation precision/coverage;
- required-fact coverage;
- abstention correctness for insufficient-evidence questions;
- query-transformation strategy accuracy, protected-term retention and
  step-back primary-coverage rate;
- context-compression provenance-valid rate, context character reduction,
  required-fact retention and compression fallback rate;
- entity-resolution accuracy and cross-version contamination count (target `0` outside explicit comparisons).
- for the P2 Selective Memory fixture set: selection precision/recall, active-fact precision, correction accuracy, relevant-memory recall, stale-memory intrusion, conflict detection, MemoryPack size/truncation and local/Milvus ranking parity.

Numeric pass thresholds are set only after Phase 0 establishes a baseline on the curated corpus. The baseline, chosen thresholds and rationale must be committed; silently changing thresholds to make a run pass is forbidden.

Feature scorecard 只在对应 subsystem 正在开发、发生相关 incident 或 error analysis 发现其 failure cluster 时进入发布 review。若某 signal 没有独立动作，只能作为 active metric 的 drill-down。若它反复捕获新的 generalization failure，按 § 4.0 完成 owner/action/cost review 后再升格；若长期饱和，则按 § 2a 退休 metric、保留必要 regression fixture。

### 4.2a Image retrieval evaluation

Use a separate versioned JSON fixture and report per-case retrieved source URIs,
Recall@K and reciprocal rank for both modes:

- text-to-image: hybrid title/caption search restricted to image records;
- image-to-image: configured image bytes → provider vector → COSINE search over
  non-null `image_vector`.

The aggregate report contains case counts, Recall@K and MRR per mode plus the
image-space fingerprint, but never raw vectors. The deterministic byte-hash
provider can prove exact-image pipeline integrity only and must label the report
`pipeline_only`; semantic image-quality claims require the explicit DINOv3
provider and its gated-model smoke/eval run. Malformed fixtures, missing images,
invalid vectors, conflicting `has_image_vector` filters and fingerprint
mismatches fail closed.

### 4.2b Selective-Memory evaluation

`run_selective_memory_eval.py` consumes at most 100 strict
`selective-memory-eval-v2` registered scenarios containing only bounded case
ids, scenario enums and expectations—never Memory text, hand-authored
`actual_*` values or vectors. It executes the real `SelectiveMemoryService`
with `LocalSelectiveMemoryStore` and reports runner/decay provenance, selection
precision/recall, active-fact precision, correction accuracy,
relevant-memory recall, stale-memory intrusion rate, conflict accuracy,
lineage coverage, MemoryPack size violation/truncation rates and consolidation
exact-once accuracy, including per-event-class selection, before/after-decay
recall, and average MemoryPack records/characters. Empty denominators are
`null`, not silently treated as perfect. Case shapes reject unknown fields,
duplicate or free-form ids and unknown scenarios; reports expose only case ids,
enums, booleans and aggregate metrics. `ranking_parity=null` explicitly means
this default local run makes no Milvus parity claim; a real-Milvus observation
is required before setting it.

### 4.2c StructArray retrieval evaluation

StructArray is a retrieval-architecture change and must use an isolated comparative report before `STRUCT_ARRAY_RETRIEVAL` becomes a Workshop default. The same corpus, chunk configuration, embedding vectors, authorization/version scopes, questions, reranker and generator run under four named profiles: `flat_hybrid`, `struct_element`, `struct_two_stage` and `struct_fused`. Unsupported profiles are `evaluation_incomplete`, not silently removed from the denominator.

The fixture segment includes short/simple documents, long documents with local-only answers, multiple relevant passages under one parent, multi-aspect questions, same-element scalar constraints, current/exact/comparison versions, and negative entity-only results. Each profile reports separately:

- passage Recall@20, selected-context Recall@5, required-fact coverage and citation resolve rate;
- document Recall@K for parent-shortlist modes, plus per-aspect element-resolution rate and parent-hit-without-evidence rate;
- offset-resolution failures, duplicate-parent hit distribution, cross-version/permission violations and same-element predicate accuracy;
- for EmbeddingList, exact-MaxSim reference recall/nDCG, first-stage candidate count, `retrieval_ann_ratio`, rerank toggle and query/entity list-length distributions;
- P50/P95 retrieval and end-to-end latency, index bytes, duplicated-vector bytes, build time and peak build resource observation;
- final answer dimensions under the existing independent graders; no blended “StructArray score”.

TokenANN is the quality-first EmbeddingList baseline. MUVERA is compared only after a measured TokenANN resource-budget violation, with all other variables fixed. LEMUR remains out of scope until a separate training/data-drift spec exists. Candidate ratio, index strategy, collapse rule, query-list construction and fusion weights are changed one at a time.

Adoption requires all existing citation, permission, version and abstention guardrails to remain green; every entity-only hit to remain non-citeable; local/native normalized-result parity; and a reviewed improvement in long-document retrieval quality or user-visible local-evidence explanation that justifies measured storage/latency cost. No improvement means the feature remains an opt-in teaching lab. Short/simple-document regressions keep those records on `flat_hybrid` rather than forcing one corpus-wide profile.

`demo/scripts/run_struct_array_eval.py` writes one strict `struct-array-eval-v1` JSON artifact. It records the corpus/projection/embedding fingerprints, fixed hardware/runtime note, fixture checksum, configured profiles, per-profile support status, case results, quality metrics above, latency samples and build/storage observations. Unknown fields and duplicate case ids are rejected. Missing native service, index-size telemetry or MaxSim reference measurements are represented as `evaluation_incomplete` with a registered reason; they are never zero-filled or omitted. The command is offline and non-mutating by default. Native execution requires explicit URI plus the expected projection fingerprint and reads the already activated projection.

### 4.2d Retrieval tier comparison

检索复杂度本身是一个必须被评测的选择，而不是默认值。[`15-retrieval-tier-selection.md`](./15-retrieval-tier-selection.md) 定义的 tier ladder 要求：任何跨 tier 的升级（T0 → T1 → T2，或 T2 内部切换到 T3/T4/T5）都必须先在同一 golden set 上给出记录在案的 failure mode 和对照报告，不得凭架构偏好直接采用。

对照报告的固定 arm 集合：

- `lexical_only`（T0）：仅 `kb_chunks` BM25 Function，无 dense lane、无 reranker 之外的语义步骤；
- `lexical_rewrite`（T1）：T0 加受限 query transformation 与 entity catalog，不引入向量；
- `hybrid_dense`（T2，当前默认）：`flat_hybrid` profile 的 dense + BM25 与既有 reranker。

三个 arm 共用同一 corpus、chunk configuration、问题集、permission/version scope、reranker 与 generator；`lexical_only` 与 `lexical_rewrite` 不消费任何 chunk-embedding 产物。§ 4.2c 的 StructArray profiles 是 T2 内部的细化 arm，与本节 arm 同表报告但不替代 `lexical_only` 基线。

每个 arm 独立报告 retrieval Recall@20、selected-context Recall@5、required-fact coverage、citation resolve rate、abstention rate、P50/P95 retrieval 与 end-to-end latency，以及 § 4.3 口径的 per-query provider call count 与成本观测。不输出跨 arm 的混合总分。

判定规则：

- `lexical_only` 是分母，不是竞争者。它的作用是量化 dense lane 在当前语料上的增量，而不是证明谁更强。
- T2 保持默认需要相对 `lexical_rewrite` 有可复核的质量增量，并且新增 P95 latency 在已接受预算内；两者都不成立时，报告必须显式记录“默认 tier 由教学目标而非质量证据支撑”。
- 缺少某个 arm 的运行是 `evaluation_incomplete`，不得把该 arm 从分母中删除。

`demo/scripts/run_tier_eval.py` 输出一份严格的 `retrieval-tier-eval-v1` JSON artifact：记录 provider/dataset profile、每个 arm 的上述 metric、latency 采样、per-request provider call 与成本观测、`case_pass_count`，以及 `default_tier_justification`（`quality_evidence` / `teaching_goal_only` / `evaluation_incomplete` 与其 reason codes）。runner 默认离线、不改动任何 collection；`--arms` 只跑部分 arm 时，未跑的 arm 保留为 `evaluation_incomplete`；`--latency-budget-ms` 未提供时不得声称 latency 证据。

当前运行结果（deterministic providers、15 条 golden questions、in-memory 语料）：T0 recall@20 0.67 / required-fact coverage 0.65 / 8 题通过；T1 与 T2 均为 recall@20 1.00 / coverage 0.98 / 14 题通过。全部质量增量来自 T0 → T1，T2 相对 T1 的 delta 为 0，因此报告判定为 `teaching_goal_only`。本地 latency 来自 in-memory adapter，不构成对 Milvus 部署的证据。

本节的 arm 结果不改变 citation、permission、version 与 abstention guardrail；任何 tier 都必须先通过 § 4.1 的 correctness gates 才能进入质量比较。

### 4.3 Performance

Capture end-to-end, query-transformation, retrieval, StructArray parent-shortlist/element-resolution, rerank,
context-compression and generation latency, plus time-to-first-validated-delta
(request start to the first released `answer_delta`), for the
documented local hardware profile. Report provider-call count, prompt/input
characters or tokens where the provider exposes them, and estimated/observed
cost separately. 另在固定 hardware、runtime、provider、dataset、concurrency
和 warm-up policy 下记录 completed requests/hour；缺少这些 profile 字段的
throughput 不可与 baseline 比较。Phase 0 publishes median and P95 over a
repeatable query set；subsequent milestones must not regress beyond an
explicitly accepted budget. Legacy example latency values are illustrative,
not targets. Latency、cost 与 throughput 是 operational metrics，不与 quality
dimensions 混分；预算尚未审批时只产生 observation 和 investigation signal。 成本
观测按 retrieval tier 与 stage 归因（query embedding、shortlist/corpus
embedding、transformation call、reranker call、generation call），使
[`15-retrieval-tier-selection.md § 6`](./15-retrieval-tier-selection.md#6-cost-and-latency-model)
的参数化模型可以用真实数字替换其示例锚点；该 spec 引用的单价来自外部来源，
是讨论锚点而非本仓库目标。

### 4.4 Agent-layer attribution

Agent 评测报告按三层归因，不得合并为单一分数（outcomes ≠ outputs：答案文本正确不代表状态变更正确）：

- **Trajectory layer**：composite stage 顺序与 terminal path 合法性，由 trace events 重建；§ 4.1 的 transition-parity、streaming-order 与 response-cache-routing gate 属于此层。
- **Tool layer**：invocation accuracy（该调/不该调，含 recall-detector bypass 与 cache-hit 零调用）、selection accuracy（`expected_tools` 匹配）、structural accuracy（registered filter 与 version scope 参数合法）；§ 4.1 的 tool-authority 与 plan-bounds gate 属于此层。
- **Outcome layer**：terminal status、citation 有效性、required-fact coverage，以及 Memory/cache 的最终状态变更（persist 幂等、session isolation、cache 未被污染）。

每条失败 case 的报告须指明最先失败的层，使回归可以定位到 classification、planning、retrieval 或 generation，而不是笼统的“答案变差了”。

### 4.5 Reliability: pass@k and pass^k

默认 deterministic 路径（无 live provider）设计上单次运行即可复现，单 trial 是充分的。任一 stage 启用 live LLM provider（`QUERY_CLASSIFIER=openai`、`QUERY_TRANSFORMER=openai`、`RERANKER=openai`、非 disabled 的 model-backed context compression、`ANSWER_GENERATOR=openai` 等）后，对 golden set 的评测必须对每题独立运行 k ≥ 3 个 trial——每个 trial 使用干净的 session 与 Memory 状态，避免跨 trial 状态污染——并同时报告两个数：

- **pass@k**：≥1 个 trial 通过。度量能力上限，随 k 上升。
- **pass^k**：全部 k 个 trial 通过。度量用户实际体验到的可靠性，随 k 急剧下降（单 trial 75% 的通过率意味着 pass@4 ≈ 99% 但 pass^4 ≈ 32%）。

只报告 pass@k 的运行不得用于声明 milestone 达成；gate 判定以 pass^k 为准。单 trial 的 live-provider 分数只能作为探索性参考。

### 4.6 RAG eval report contract

当前 implementation 输出 `rag-eval-v3`；`rag-eval-v2` baseline 明确不兼容，
迁移时必须提交首个 reviewed v3 baseline。v3 不输出单一混合总分，顶层
`metric_portfolio` 嵌入 `eval-metric-registry-v1` 的 version/checksum，并按
`goal`、`guardrail`、`operational` 分组列出 active metrics。每项记录
`grader_id/version`、本次值、上一 committed baseline 值、delta、gate/budget、
`decision_status=pass|fail|observational|evaluation_incomplete` 和注册动作。
没有 baseline 或可用分母时使用 `null`，不得把缺失值伪装成零或满分。

`dimensions` 继续按 `trajectory`、`tool`、`outcome` 保存 case attribution 和
feature diagnostics，但它不是第二套 KPI dashboard。只有 registry 中 active 的
signal 才进入 `metric_portfolio`、计算 decision status 并驱动发布。v2 与 v3 不跨
版本拼接 delta。

每个 case/trial 都运行在新的 workflow 实例及唯一 `session_id/query_id` 下，并
记录以下独立判定：

- `trajectory`：stream event sequence 连续且 query-local、恰有一个末尾
  `final`、terminal/trace 状态一致、terminal status 对应的 required/forbidden
  stage path 合法、query-transformation strategy/item roles 与 fixture
  一致、`prepare_generation_context` 只在 evidence-sufficient 路径出现、
  retry 不超过合同上限；
- `tool`：invocation 与 `expected_tools` 精确匹配，工具和 query plan 均为
  registered/bounded，调用 department scope 同时受 registered tool domain 与
  permission decision 约束，version scope 与 fixture 一致；
- `outcome`：abstention 与 fixture 一致，citation 只来自 selected context，
  focused generation context 不超过 5 条（exhaustive sibling 不超过
  16 条），required citation/fact 全覆盖，compression provenance 合法
  或已整体回退原 context，且 grounded answer self-check 有效。

case pass 是所有适用的上述判定为真；`first_failure_layer` 必须按
`trajectory → tool → outcome` 返回最先失败层及 registered reason codes，不能只
返回自由文本。`transcript_review` 汇总失败层与 reason-code 计数，供人工执行
capability-gap / task-quality / scaffold-entanglement 归因；runner 不替人工猜测
根因。人工结论来自 strict `rag-eval-review-v1` fixture（case、layer、reason、
attribution、owner）；stale review fail closed，未归因失败在报告中显式列出。

默认 CLI 强制 deterministic embedding/classifier/reranker/generator，运行一次且
不访问网络。只有显式 `--live-providers` 才读取已配置 provider；此模式要求
`--trials >= 3`，报告每题及总体的 `pass_at_k`、`pass_power_k` 作为 reliability
diagnostic；每个 active quality metric 的 gate 必须让全部适用 trials 逐一满足自身
threshold，不能用 aggregate mean 掩盖单次失败。`--baseline` 指向 report/registry 版本相同的 committed aggregate report；fixture
ID/content checksum、top-k、run mode、trial count、configured 与 observed
provider/model/fallback、Memory selector、vector-space profile、offline corpus
checksum、Milvus immutable snapshot/restore identity、registry checksum 或 report/grader version
不兼容时 fail closed，不计算误导性 delta。API caller 使用
baseline 时必须显式提供 provider 与 dataset profile，不能用通用默认值比较两个
未知 workflow。

trajectory/tool/outcome 是诊断维度，不生成 baseline delta。Goal/guardrail active
metrics 与硬件无关，跨机器仍按 compatible registry、provider、dataset、fixture 和
grader profile 计算 delta；committed baseline 必须在任何干净 checkout 上可用。
Operational active metrics 只有在 runtime/hardware、concurrency、measurement 与
cost profile 全部一致时才计算 delta。否则其 `baseline`/`delta` 为 `null`，报告在
`baseline.operational_delta_skipped_reason` 记录 `runtime_profile_mismatch` 或
`operational_profile_mismatch`，运行本身不失败。
延迟独立报告 end-to-end、retrieval、rerank、generation 与从 trial 启动到首个
validated `answer_delta` 的 time-to-first-validated-delta median/P95；答案是
validated-buffered 释放，因此该指标不得改称 time-to-first-token（见
[`13-llm-answer-generation.md § 5`](./13-llm-answer-generation.md#5-prompt-and-output-contract)）；仅当 runtime/hardware
profile 与 baseline 完全一致时才对 committed baseline 给出 latency delta。延迟
永远不混入质量分数。

### 4.7 Error-analysis and metric-review artifacts

Canonical error-analysis artifact 存放在 `demo/eval/error_analysis/<date>-<change>.json`。
`eval-error-analysis-v1` 记录一次人工 trace sample，顶层固定 artifact version、
change reference、sample timestamp、sampling strata 与 reviewer。bootstrap sample
包含 30–50 条去重、覆盖主要 route/terminal status 的 case，每条只有 opaque
trace/case id、`overall_pass` 与最多 500 个 Unicode code points 的 `review_note`；
不得复制 prompt、文档正文、Memory payload 或 credential。cluster review 另记录
registered category id/name、`trace_ids`、count、severity、
是否为 observed generalization failure，以及 `prompt_or_schema_fix | metric_candidate |
fixture_only | discard` disposition。metric candidate 必须反向链接 registry review，
必须标为 observed generalization failure，并指向同次加载 registry 中的 candidate 或
active `metric_id`；拼错、未知或 retired metric fail closed。runner 不得根据 note
自动创建 evaluator。

每次 significant change 后追加新的 error-analysis artifact；active registry 至少每月
快速检查一次 owner/action/cost，每季度执行一次 keep/modify/retire review。review
记录 metric 的近期命中数、是否改变过决策、evaluator 运行成本、human disagreement、
Goodhart/overfit 风险和最终状态。Guardrail 只能因约束被正式移除而 retire；goal 和
operational metric 可按 § 2a 的无信息量条件退休。退休 metric 的 historical report
保持可读，active dashboard 不继续运行其 evaluator，必要的 deterministic regression
fixture 仍由 test suite 持有。

## 5. Chunk configuration evaluation

Chunk size is an offline corpus-build decision, not an online Agent branch.
Compare at least three small/medium/large Min-Max configurations over the same
corpus and questions. `max_tokens` values around `128/256/512` are the initial
teaching sweep; each candidate also declares `min_tokens`, `overlap_tokens`
and the fixed hard-boundary policy. The versioned lexical tokenizer remains
the splitting unit. Reports additionally publish character-length
distributions so character-based source proposals can be interpreted without
silently treating characters and tokens as equivalent.

Each configuration gets a complete isolated chunk set and, for online runs, a
separate collection/index or restored target. One run fixes every other
variable: corpus revision, questions/goldens, embedding fingerprint, retrieval
and index parameters, reranker, query transformation, compression mode,
generator, grader versions and provider trial count. Mixing two chunking
profiles in one candidate pool or changing multiple variables invalidates the
comparison.

The versioned `chunking-experiment-v2` report records per configuration:

- chunk count; lexical-token and character percentiles; under-min/over-max and
  empty rates; same-source near-duplicate rate; Markdown heading, release and
  PDF page boundary preservation;
- ingestion time, index size when built, retrieval Recall@20, reranked Recall@8
  and selected-context Recall@5 against stable source-term anchors;
- citation precision/coverage/granularity, required-fact coverage, abstention
  correctness and cross-version contamination;
- faithfulness and answer relevancy as two independent dimensions, never a
  combined judge score;
- end-to-end/retrieval/rerank/generation median and P95, provider-call count,
  input/output token usage when available and cost estimate with price/profile
  provenance.

Faithfulness asks whether every answer claim is supported by its cited source
context. Answer relevancy asks whether the response directly and usefully
answers the question. L1 citation/fact checks always run. A calibrated
single-dimension L2 judge or L3 human labels are required for the semantic
faithfulness/relevancy fields; before the § 4.0b calibration contract is met,
these values are `null`, the report status is `evaluation_incomplete`, and no
new production default may be recommended. Live generators follow § 4.5 with
clean-state `k >= 3` trials and selection uses `pass^k`, not best-trial
`pass@k`.

Selection is constraint-first and contains no blended overall score:

The committed teaching gate profile is
`rag-eval-baseline-2026-08-05-minus-explicit-tolerance-v1`: retrieval Recall@20
and selected-context Recall@5 must be at least `0.90`, citation coverage at
least `0.90`, citation precision at least `0.70`, required-fact coverage at
least `0.90`, abstention correctness exactly `1.0`, and Markdown/release/PDF
boundary preservation exactly `1.0`. The report records these thresholds and
binds them into its evaluation fingerprint. Updating a threshold requires a
reviewed spec/baseline change; a reviewer artifact cannot waive a failed gate.

1. reject any candidate with citation/version/boundary violation, regression
   below committed correctness gates or excessive empty/near-duplicate chunks;
2. retain the Pareto frontier over selected-context recall, faithfulness and
   answer relevancy;
3. among statistically/tolerance-equivalent frontier candidates, prefer lower
   prompt tokens/cost and latency, then fewer chunks and the simpler config;
4. record the winner, non-winners, uncertainty and rationale in a reviewed
   recommendation artifact. The artifact must match both the immutable input
   experiment fingerprint and a stable evaluation fingerprint over provider/
   grader profiles plus non-latency corpus and quality metrics. The runner
   never mutates ingestion defaults.

Per-document-type profiles are allowed only when each type has enough fixtures
to run this complete protocol and ingestion can route the profile
deterministically. Otherwise publish one corpus profile rather than overfit a
small subset. Any chosen profile change requires full re-ingestion, a new
dataset/config fingerprint and a fresh committed RAG baseline.

## 6. External capability verification matrix

Before implementation depends on an external Milvus or OpenAI capability, Phase 0 must run a minimal test against the exact documented version:

| Capability | Evidence required | Fallback if unsupported |
| --- | --- | --- |
| dense + sparse/BM25 hybrid | executable query and stable normalized result shape | use the simplest supported hybrid composition and document it |
| scalar metadata filter | filtered query test | block P0 because source filtering is a core contract |
| ordering with hybrid results | fake/real-client behavior tests for explicit `relevance` and `scalar` modes plus local parity | keep relevance-first local tie-break and disable scalar mode |
| aggregation/grouping | candidate-ID bounded Query Aggregation request and exact local parity | compute the same retained-candidate facets locally and label fallback |
| nullable `image_vector` | mixed null/non-null insert and query | split image examples or defer panel to P2 |
| lifecycle TTL | TIMESTAMPTZ codec/property contract plus real-server expiration test | explicit `expires_at` predicates remain mandatory; block native-TTL claim |
| Milvus Decay Ranker | startup read-only acceptance probe for standard COSINE search plus an exact target service/SDK disposable-collection test for `gauss`/`exp`/`linear`, millisecond units, offset/scale/decay points, one-numeric-field/grouping restrictions, hybrid composition and returned ordering; contract tests cover request shape, fail-closed startup, scope/expiry filters, native/application parity and `no_time_decay` bypass | deterministic application-side decay after a larger bounded candidate recall; do not claim native decay |
| BM25 Function + SINDI | schema/function request, raw-text search, synonym and default-index-param tests | local token sparse vector; never mix vector spaces in one collection |
| MinHash representation | MINHASH DIDO/function/index contract plus known duplicate/near/non-duplicate smoke | keep checksum-only local dedup; do not persist client signatures |
| snapshots/schema evolution | bounded restore-state and additive-field/partial-update fake-client tests, then disposable real collection | default offline eval and full recreation for incompatible fields |
| OpenAI text embedding | fake-client contract plus opt-in 1024-dimension smoke; ingestion/query parity test | deterministic offline provider, clearly labeled and never mixed in one collection |
| OpenAI Responses generation | opt-in configured smoke test with non-empty validated citations | deterministic extractive generator with traced reason code |
| OpenAI Responses classification | fake-client strict JSON-schema contract plus opt-in configured smoke | `RuleBasedQueryClassifier` with traced safe reason code |
| OpenAI Responses reranking | fake-client strict JSON-schema contract plus opt-in configured complete-candidate smoke | whole-batch `RuleBasedReranker` with traced safe reason code |
| OpenAI query transformation | fake-client strict strategy/item schema, protected-term/scope validation and one-request bound; opt-in step-back/decompose smoke | deterministic identity/rule transformation with traced safe reason code |
| OpenAI context compression | fake-client strict source-id/support-span schema, exact source validation, one-batch bound and opt-in long-context smoke | complete original selected context; never mix partial compression |
| DINOv3 image embedding | fake-runtime contract plus opt-in gated-model smoke on all curated PNG fixtures; verify ViT-B/16 `pooler_output`, 768 dimensions and L2 norm | deterministic image-byte vectors in a distinct fingerprinted offline space; never caption vectors |
| grounded response semantic cache | local/Milvus parity tests for COSINE + session/expiry filters and evidence scalar validation | exact hash cache only, or disabled cache |
| StructArray + EmbeddingList | completed Milvus 3.0.0/PyMilvus 3.0.1 probe memo plus schema/index/insert/search tests for same-element filters, offsets, two metric families, exact synthetic MaxSim and hybrid collapse identity; isolated `struct-array-eval-v1` remains the adoption gate | keep `STRUCT_ARRAY_RETRIEVAL=disabled` and use `kb_chunks` flat hybrid when startup/eval gates are absent; local emulation is labelled and makes no native-performance claim |

## 7. Demo security boundary

MVP has no production auth or ACL, so it is restricted to synthetic/curated sample data and local Workshop use. `get_user_permission` is a deterministic teaching gate that demonstrates ordering and allowed-domain intersection; it must never be described as production authorization. Do not ingest real corporate documents or personal data into Memory. Credentials, including `OPENAI_API_KEY`, come from environment/configuration, never notebooks, traces or source URIs. Logs redact prompts, provider error bodies, document bodies, Memory events/facts and rendered MemoryPack by default. Retrieved text and recalled Memory are untrusted prompt data and cannot authorize tools, alter tool filters or create sources. Append-only lineage does not override an authorized erase request. Production authorization, consent and retention remain separate production design work.

## 8. Definition of done

- Relevant tests, formatter and linter pass in the future runnable demo project.
- The documented clean setup and smoke test have been run in the current implementation session before success is claimed.
- No unverified Milvus capability is presented as native functionality.
- StructArray is activated only after the exact target server/SDK passes the § 6 matrix and the isolated § 4.2c report; parent-only results never masquerade as citation evidence.
- All failures include stage/source/query context and preserve the original cause where the language supports it.
- Eval 报告按维度输出并包含与上一次 committed baseline 的 delta；启用 live provider 的运行同时给出 pass@k 与 pass^k（§ 4.5）。
- 本次变更中观察到的每个新失败模式已按 § 2a 进入 fixture，或被显式记录为带 owner 的待办；transcript review 的 capability-gap / task-quality 归因已随报告记录。
- `eval-metric-registry-v1` 的 active set 同时包含 goal、guardrail、operational，且每项都能回答“这个值变化后谁做什么”；无动作的 diagnostics 没有进入发布 decision surface。
- Significant change 已按 § 4.7 重跑 error analysis；metric review 记录新增、保留与退休理由，L2/L3 运行成本未超过 registry budget。

## 9. Cross-references

- ← Contracts: [`10-data-model.md`](./10-data-model.md), [`10a-openai-text-embedding.md`](./10a-openai-text-embedding.md), [`10b-conversation-memory.md`](./10b-conversation-memory.md), [`11-ingestion.md`](./11-ingestion.md), [`12-agent-workflow.md`](./12-agent-workflow.md), [`13-llm-answer-generation.md`](./13-llm-answer-generation.md), [`14-milvus-3-native-capabilities.md`](./14-milvus-3-native-capabilities.md), [`15-retrieval-tier-selection.md`](./15-retrieval-tier-selection.md), [`20-ui-demo.md`](./20-ui-demo.md)
- → Gates delivery in: [`90-roadmap.md`](./90-roadmap.md), [`91-impl-plan.md`](./91-impl-plan.md)
- ↔ Source notes: [`archive/Optimize.md`](./archive/Optimize.md), [`archive/UIdemo-collection.md`](./archive/UIdemo-collection.md)
- ↔ Decisions: [`99-key-decisions.md § D53`](./99-key-decisions.md#d53--model-only-verification-stays-at-the-contract-layer-not-the-golden-set), [`99-key-decisions.md § D43`](./99-key-decisions.md#d43--evaluation-follows-eval-driven-development-with-layered-calibrated-graders), [`99-key-decisions.md § D44`](./99-key-decisions.md#d44--chunk-configuration-is-selected-by-isolated-end-to-end-evaluation), [`99-key-decisions.md § D45`](./99-key-decisions.md#d45--query-transformation-is-bounded-and-scope-preserving), [`99-key-decisions.md § D46`](./99-key-decisions.md#d46--context-compression-is-a-provenance-preserving-generation-projection), [`99-key-decisions.md § D47`](./99-key-decisions.md#d47--active-eval-metrics-form-a-minimal-decision-bound-portfolio), [`99-key-decisions.md § D48`](./99-key-decisions.md#d48--structarray-is-a-derived-document-projection-chunk-identity-remains-authoritative), [`99-key-decisions.md § D49`](./99-key-decisions.md#d49--search-granularity-is-explicit-and-entity-hits-are-not-citation-evidence), [`99-key-decisions.md § D50`](./99-key-decisions.md#d50--retrieval-complexity-is-a-measured-ladder-not-a-default)
