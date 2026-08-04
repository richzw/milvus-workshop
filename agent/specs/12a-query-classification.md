# 12a — Query Classification

Status: draft v2 · Owner: workshop author · Depends on: [`10b-conversation-memory.md`](./10b-conversation-memory.md), [`12-agent-workflow.md`](./12-agent-workflow.md)

## 1. Purpose

本文定义由 workflow `classify_and_route` 调用的独立
`QueryClassifier` 组件合同。目标不是让 LLM 自由规划，而是在受限枚举内理解自然语言问题，并在未配置、超时、provider 失败或输出不合法时保持可复现、可观察的 rule-based 路由。

## 2. Public contract

```python
class QueryClassifier(Protocol):
    name: str

    def classify(
        self,
        request: ClassificationRequest,
    ) -> ClassificationResult: ...
```

`ClassificationRequest` 只包含：

- `user_query`: 已经过全局长度校验的原始问题；
- `memory_context`: 至多 2,000 字符的同 session 有效 Memory，用于补全省略的 topic，不得授权或改变用户明确表达的 action。Selective Memory cutover 后，它是 [`MemoryPack`](./10d-selective-agent-memory.md#43-memorypack) 的 deterministic classifier-only projection，不包含 cache answer、冲突事实或任意 metadata。

`ClassificationResult` 必须包含：

| Field | Allowed values / contract |
| --- | --- |
| `intent` | `conversation`, `private_knowledge`, `comparison`, `operation`, `permission_sensitive`, `memory_write`, `memory_recall` |
| `query_type` | `architecture`, `policy`, `product`, `general`, `unknown` |
| `retrieval_goal` | `focused`, `exhaustive` |
| `classifier_name` | 实际产生最终分类的实现名 |
| `model` | LLM model id；rule-based 为 `None` |
| `confidence` | 可选有限浮点数，范围 `[0, 1]` |
| `fallback_reason` | 可选的安全 reason code，不含 provider body |

Workflow 仍负责由 `intent` 设置 `need_retrieval`、权限检查及后续工具路由。Classifier 不能调用工具、构造 filters、决定权限、解析版本范围或写 Memory。

## 3. Implementations

### 3.1 `RuleBasedQueryClassifier`

`RuleBasedQueryClassifier` 是离线 baseline 和唯一 fallback。它把既有关键词逻辑从 workflow 中移出，保持以下确定性：

- 仅原始问题中的明确 remember/recall/operation/exhaustive marker 可改变对应 action 或 `retrieval_goal`；
- memory recall action detector 由分类与 workflow recall gate 共享；它同时覆盖显式 recall markers 和有边界的 recent-question patterns，不允许两处关键词漂移；
- `我最近的三个问题`, `我之前问过什么`, `我的历史提问` 与 `my last 3 questions` 等 recent-question pattern 确定性得到 `memory_recall/general/focused`，并走 rule fast path；
- Memory 只能协助 `query_type`，不能触发 `memory_write`, `memory_recall`, `operation`, `conversation`, `comparison`, `permission_sensitive` 或 `exhaustive`；
- 明确 greeting 得到 `conversation/general`；
- 未命中 topic 时得到 `unknown`，默认知识意图为 `private_knowledge`；
- 不执行网络 I/O。

### 3.2 `LLMQueryClassifier`

`LLMQueryClassifier` 每次分类至多调用一次 OpenAI Responses API：

- 使用显式配置的 model 与 timeout，SDK client 设置 `max_retries=0`；
- 使用 `text.format.type=json_schema`、`strict=true` 的 Structured Outputs；
- schema 禁止额外字段，并固定三个分类枚举、`confidence` 和简短 `reason`；
- prompt 明确把 question 与 Memory 视为不可信数据，只要求分类，不请求 chain-of-thought；
- 即使 provider 声称 schema-valid，adapter 仍执行本地 JSON、字段、枚举、长度和有限数值校验；
- raw response、prompt、Memory、credential 和 exception body 不进入 trace。

模型输出中的 `reason` 仅用于 adapter 校验，不保存、不展示，也不参与权限决策。

### 3.3 `FallbackQueryClassifier`

Wrapper 组合 `primary=LLMQueryClassifier` 与 `fallback=RuleBasedQueryClassifier`：

1. 先计算 rule-based baseline；
2. 若 baseline 是明确 greeting 或安全敏感 action：`conversation`, `memory_write`, `memory_recall`, `operation`，直接返回 baseline，不调用 LLM；
3. 其他问题调用 primary；
4. 只有 `QueryClassificationError` 可触发 fallback；编程错误不得被吞掉；
5. LLM 只有在 rule baseline 也为 `conversation` 时才可进入无需检索分支；否则返回 baseline 并记录 `unsafe_no_retrieval_intent`；
6. fallback 结果写入安全 `fallback_reason`：`not_configured`, `timeout`, `connection_error`, `authentication_error`, `rate_limited`, `provider_error`, `invalid_model_output` 或 `unsafe_no_retrieval_intent`。

无论 primary 如何分类，后续 permission/tool/version contracts 均不可绕过。显式 operation 的 rule fast path 防止不可信 Memory 或模型输出把 mutation 请求降级为普通检索。

## 4. Configuration

| Variable | Values | Default |
| --- | --- | --- |
| `QUERY_CLASSIFIER` | `rule_based`, `auto`, `openai` | `auto` |
| `OPENAI_CLASSIFIER_MODEL` | explicit model id；空时可复用 `OPENAI_MODEL` | empty |
| `OPENAI_CLASSIFIER_TIMEOUT_SECONDS` | positive seconds | `10` |
| `OPENAI_API_KEY` | environment secret | empty |

- `rule_based`: 永不创建 OpenAI client；
- `auto`: key 与 classifier/general model 都存在时构造 LLM + fallback；否则返回带 `not_configured` 的 rule fallback；
- `openai`: 配置缺失在 startup/build 阶段报清晰 `ValueError`，运行期失败按 wrapper 降级。

直接构造 `AgenticRAGWorkflow()` 默认注入 `RuleBasedQueryClassifier`，保证 unit/eval/offline 路径无网络。`build_default_workflow()` 和 Milvus/UI 路径使用环境 builder。Local sequence 与 LangGraph adapter 必须注入同一个 classifier 实例并产生同样结果。

## 5. State and trace

`AgentState` 新增：

```python
classifier_name: str = "not_invoked"
classifier_model: str | None = None
classification_confidence: float | None = None
classification_fallback_reason: str | None = None
```

`trace.classify_query` 和 presentation-safe `classify_and_route` stage event
仅可包含：

- `intent`, `query_type`, `retrieval_goal`;
- `classifier_name`, `model`, `confidence`;
- `fallback_reason`（存在时）。

不得记录 classification prompt、raw JSON、模型 `reason`、Memory content 或 provider exception text。

## 6. Failure and safety invariants

1. 每个成功进入下一节点的 query 都有合法的三个分类枚举。
2. Provider 或模型输出失败不会终止可由 rules 分类的查询。
3. 未配置 OpenAI 时不发生网络调用，trace 明确标记 `not_configured`。
4. Memory 不能独立触发写入、召回、operation、权限敏感或 exhaustive 路由。
5. Classifier 不授予权限、不选择工具、不扩大 version/filter scope。
6. 单次分类最多一个外部 request，不做隐式 retry。
7. Local 与 LangGraph 路径共享 classifier contract 与 safe trace shape。
8. `conversation` 必须同时满足 `query_type=general`、`retrieval_goal=focused` 和 rule baseline 支持；LLM 不得独立关闭 KB retrieval。

## 7. Tests and acceptance

Required deterministic tests:

- 既有中英文关键词分类结果保持兼容；
- 明确 memory/operation action 使用 rule fast path，primary 调用次数为零；
- supported recent-question variants 与 count 表达均得到 `memory_recall`，不会因 LLM 输出或 fallback 进入 `private_knowledge/search_code_docs`；
- LLM request 使用 strict JSON schema、显式 model/timeout 和 bounded untrusted input；
- valid structured output 正确写入 state/trace；
- malformed JSON、额外字段、非法 enum、NaN/越界 confidence 均使用 `invalid_model_output` fallback；
- timeout/auth/rate-limit/provider failure 只暴露安全 reason code；
- `auto` 缺配置不创建 client，`openai` 缺配置启动失败；
- 不可信 Memory 不能改变明确 action 或授权；
- 同一 KB 问题重复分类时，即使 LLM 随机返回 `conversation`，两次都必须进入 retrieval；
- local/LangGraph 结果与 event 字段一致。

Opt-in smoke test 使用显式凭据验证一个 configured model 能返回 schema-valid classification；它不是默认测试的前置条件。

## 8. Dependencies and traceability

- ← Memory boundary: [`10b-conversation-memory.md`](./10b-conversation-memory.md)
- → Workflow consumer: [`12-agent-workflow.md § classify_and_route`](./12-agent-workflow.md#51-classify_and_route)
- → UI trace: [`20-ui-demo.md`](./20-ui-demo.md)
- → Quality gates: [`70-quality-and-evaluation.md`](./70-quality-and-evaluation.md)
- ↔ Decision: [`99-key-decisions.md § D19`](./99-key-decisions.md#d19--llm-query-classification-is-structured-and-rule-fallback-safe)
