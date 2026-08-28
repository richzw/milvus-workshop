---
name: agent-eval
description: Build, calibrate, and iterate evaluations for LLM and agent systems using eval-driven development — failure-mode discovery, golden sets, layered graders (programmatic → LLM-as-judge → human), judge calibration, trajectory/tool/outcome layers for agents, pass@k vs pass^k reliability metrics, and living-benchmark maintenance. Use whenever the user says "build evals", "evaluate the agent", "add an eval for X", "the answers look wrong — how do we measure this", "write an LLM-as-judge", "golden set", "regression-test the prompts", "is the new model/prompt actually better", "measure retrieval/citation quality", or wants quality gates before shipping an AI feature. Trigger even when the user does not say "eval" if they ask how to tell whether an LLM-powered feature works, compare two prompts/models/configurations, or set up monitoring for AI output quality.
---

# Agent Eval

Treat evaluation as a first-class engineering discipline, not a post-launch afterthought. Non-deterministic systems fail in three quiet ways: false confidence in generic metrics, regressions on unmeasured dimensions, and effort spent optimizing things that don't matter. Eval-driven development (EDD) prevents all three: define the gates before building, derive metrics from *observed* failures, and keep a small calibrated grader set that the team actually trusts.

Distilled from Airbnb's eval-driven-development practice and Cameron Wolfe's agent-evals survey; adapted to this repo's harness (`demo/scripts/run_eval.py`, `specs/70-quality-and-evaluation.md`).

## When this fires

- "build / extend the evals for X" · "add a golden question for Y"
- "how do we know the agent is actually better after this change?"
- "write an LLM-as-judge for faithfulness / tone / relevance"
- "the outputs look off — set up something to measure it"
- "gate this phase on retrieval recall / citation correctness"
- The impl skill needs observable exit criteria for an AI-behaviour phase and none exist yet

## Principles (load-bearing)

1. **Look at your data first.** Read real outputs and transcripts before building any metric. Metrics invented from theory measure the wrong things; metrics derived from observed failures measure what actually breaks.
2. **One grader per dimension — no god evaluators.** A single judge scoring "overall quality" hides which dimension regressed. 3–5 well-calibrated judges beat 20–30 noisy ones.
3. **An uncalibrated judge is worse than no judge.** It manufactures false confidence at scale. Calibrate against human labels to high-80s–90s % agreement before trusting it.
4. **Outcomes ≠ outputs.** Agents change environment state (files written, records inserted, reservations made). Grade the final state, not just the final text.
5. **One trial is not signal.** Agents are stochastic; reliability only shows up across repeated trials (see pass@k vs pass^k below).
6. **Fix one variable at a time.** Vary the prompt with the model fixed, then the model with the prompt fixed, then serving config. Changing several at once makes the eval unreadable.
7. **Evals are living artifacts.** Every new production failure becomes a task; graders get re-calibrated as failure modes evolve; saturated tasks get harder replacements while a regression set keeps the old ones.

## What to produce

Depending on maturity, some subset of:

- **An eval plan** (a short section in the relevant spec — in this repo, `specs/70-quality-and-evaluation.md`): dimensions measured, grader per dimension, gate thresholds, and what traffic/tasks feed it.
- **A task / golden set**: versioned fixtures with inputs, expected outcomes, and labels — including *failures*, not just successes (in this repo: `demo/eval/questions.json` + `demo/eval/golden_answers.yaml`).
- **Graders**: programmatic checks and/or judge prompts, each pinned to one dimension.
- **A calibration report**: judge-vs-human agreement numbers and the disagreement analysis (see [references/judge-calibration.md](references/judge-calibration.md)).
- **An eval report**: per-dimension scores with deltas vs the previous run, plus the transcript-review findings — never a single blended number.

## Workflow

### 1. Define success and gates upfront

Before touching graders, write down: the dimensions that matter for *this* product (faithfulness, citation correctness, abstention, tone, latency, cost…), the ship gate per dimension, and who arbitrates disagreements. Prefer **outcome goals** (the record exists, the citation resolves, the test passes) over process goals — they are objective and don't over-constrain the agent's path. Add process goals only where the trajectory itself is the contract (e.g. "permission gate runs before retrieval").

### 2. Explore and discover failure modes

Run a batch of realistic inputs through the current system — ~100 for a single-turn feature, 10–20 tasks for an agent — and **read every output and transcript**. Categorize failures by hand: e.g. "15 faithfulness, 8 conciseness, 5 over-refusal, 3 format". These categories become the eval dimensions; their frequencies set the priority. Do not skip this step to jump to a metric library — generic metrics score well while missing your product's real failure modes.

### 3. Build the golden set

50–100 hand-labeled rows, built with whoever owns the product judgment. Must include bad examples — a golden set of only successes cannot test a grader's discernment. If human experts disagree on a label, **stop and resolve the disagreement first**: human disagreement means the rubric or the domain is ambiguous, and no automated judge can do better than the humans it imitates. Version the set with the corpus it runs against.

### 4. Layer the graders

Cheapest first; each layer filters for the next:

- **L1 — Programmatic** (deterministic code, no LLM): schema/format validity, length bounds, required-fact string checks, citation-resolves checks, test-case execution, tool-invocation assertions. Efficient, reproducible, debuggable — but reference-bound and blind to nuance.
- **L2 — LLM-as-judge**: a stronger model scoring one dimension against an explicit rubric, with chain-of-thought and few-shot anchors. Use direct scoring, pairwise comparison, or reference-guided scoring as fits. Rubric design and calibration are their own craft — read [references/judge-calibration.md](references/judge-calibration.md) before writing a judge.
- **L3 — Human**: ground truth for high-stakes calls, judge calibration, and resolving L1/L2 disagreement. Start with 20–100 expert-labeled rows; scale to an annotation workforce only after the rubric is stable and volume is the bottleneck.

### 5. Agent-specific layers

For multi-step agents, grade three layers, not one:

- **Trajectory layer** — were the right sub-agents / workflow nodes invoked, in a legal order? Reconstruct paths from traces/spans.
- **Tool layer** — invocation accuracy (called vs. avoided correctly), selection accuracy (right tool), structural accuracy (valid parameters), and where the sequence is the contract, trajectory accuracy.
- **Outcome layer** — the final environment state and answer.

Harness requirements: fresh environment per trial (no cross-task state pollution); the agent runs on the **same scaffold and tools as production**; complete transcripts captured (reasoning, tool calls + parameters, tool results, state changes, recovery attempts); graders run over both transcript and outcome; results aggregate across tasks × trials. Attach an oracle solution + deterministic verification to each task where possible — it proves the task is solvable and catches grader drift.

### 6. Measure reliability, not just capability

Run k independent trials per task (τ-bench uses 4–5, varying user phrasing / conditions):

- **pass@k** — succeeds in ≥1 of k trials: the agent *can* do it. Rises with k.
- **pass^k** — succeeds in *all* k trials: the agent *reliably* does it. Falls sharply with k, and is what users actually experience. A 75 %-per-trial agent has pass@4 ≈ 99 % but pass^4 ≈ 32 %.

Report both. A benchmark that only reports pass@k systematically overstates production readiness.

### 7. Review transcripts after every run

Failures split into capability gaps vs. task-quality issues (ambiguous instructions, wrong ground truth, impossible specs, exploitable shortcuts). Only transcript reading tells them apart. Also check for scaffold entanglement — poor scores may come from tool naming, context management, or prompting rather than the model; do not conclude "the model can't" until the scaffold is ruled out.

### 8. Scale and monitor

Once graders are calibrated, scale the offline set (hundreds–thousands of rows). In production, mirror the offline setup: sample a few percent of de-identified traffic daily, run L1 + L2 continuously, surface flagged outputs for weekly human review, and feed every new failure mode back into step 2. A/B tests, user feedback signals, and cost/latency metrics are complementary layers — overlapping imperfect methods ("Swiss cheese") beat any single one.

## Quality bar

- Every eval dimension traces to an *observed* failure mode or an explicit product gate — none exist "because the metric library had it".
- Every judge has a written rubric, a calibration number against human labels (target high-80s–90s % agreement, κ-corrected), and a recorded disagreement analysis.
- The golden set contains failures and is versioned alongside the corpus/fixtures it runs against.
- Agent evals report per-layer results (trajectory / tool / outcome) and both pass@k and pass^k.
- The eval runs deterministically from a clean checkout (this repo: `PYTHONPATH=demo/src python3 demo/scripts/run_eval.py`), and its report shows deltas vs. the previous run.
- Expect evaluation to consume a meaningful share of total project effort. That is not overhead; it is how the product gets trustworthy.

## Anti-patterns

- **God evaluator** — one judge, one blended score. Which dimension regressed? Nobody knows.
- **Metric theater** — shipping BLEU/ROUGE/generic-helpfulness numbers that never caught a real bug.
- **Trusting an uncalibrated judge** — automation of noise; strictly worse than reading 50 outputs by hand.
- **Golden set of successes only** — the judge is never tested on discernment, so it learns to say yes.
- **Single-trial agent scores** — brittleness is invisible at n=1; pass^k exists for a reason.
- **Grading outputs while ignoring outcomes** — the answer text looked right; the database write was wrong.
- **Frozen benchmark** — saturated tasks, stale rubrics, no intake path for production failures.
- **Changing model + prompt + retrieval at once** and asking the eval which change helped.

## Hand-off

When the eval lands, report: the dimensions and their gates; per-dimension scores with deltas; judge calibration numbers; transcript-review findings (capability vs. task-quality); and which gate now blocks or clears the next impl phase.

## Cross-references

- The **impl** skill's phase exit criteria should cite eval gates from this skill's plan; a phase touching AI behaviour without an eval gate is underspecified.
- The **spec** skill's quality cross-cut (this repo: `specs/70-quality-and-evaluation.md`) is where the eval plan lives; `specs/99-key-decisions.md` records grader-design decisions.
- The **research** skill handles vendoring an eval harness or benchmark (τ-bench, Terminal-Bench) for study before adopting its patterns.
- This repo's existing seam: `demo/scripts/run_eval.py`, `demo/eval/questions.json`, `demo/eval/golden_answers.yaml`, and the fixture contract in `specs/70-quality-and-evaluation.md` § 3.
