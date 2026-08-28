---
name: impl
description: Polish and expand the relevant feature specs first, then implement one phase of the project impl plan end-to-end with high quality bars (correctness, elegance, performance), and run an independent code review against the polished specs before declaring done. Use whenever the user says "build phase N", "implement the next phase", "land M0/M1/M2/M3", "follow the impl plan", "ship phase X entirely", "based on specs think hard and build phase X", or asks for a phase-shaped slice of the spec set. Trigger even when the user does not say "impl" if they reference an impl plan / roadmap milestone and ask for it to be built.
---

# Impl

Land one phase from the project impl plan to a publishable bar — no TODOs, no half-finished modules, no quality-gate bypasses — then run a thorough independent review against the polished specs and fix every valid finding before claiming done. Before writing implementation code, expand and polish the relevant feature specs until they are detailed, concrete, internally consistent, and executable by another engineer. The phase is the unit of completion; partial phases create drift the spec set is meant to prevent.

## When this fires

- "build phase N entirely" / "implement phase N" / "land M<n>"
- "based on the impl plan and other specs in `./specs`, think hard, build phase X"
- "previous phases are done — continue with the next one"
- "ship the spec; one phase at a time"
- The user names a milestone (M0/M1/M2/M3) or a Phase-N task and asks for the code

## What this skill is *not*

- Not for one-off bug fixes or feature requests outside a planned phase. Use direct edits.
- Not for greenfield design work. If `./specs/` is empty or the relevant phase is not specified, hand off to the **spec** skill first.
- Not for prototyping. The quality bar here is "publishable"; throwaway code lives somewhere else.

## Diagram expectation

When an implementation phase changes architecture, data flow, lifecycle/state transitions, or build/dependency order, update the corresponding spec or research diagrams in the same phase. Use fenced ` ```text ` ASCII diagrams with nested boxes or grouped lanes, matching the spec/research skill standard (terminal-safe box-drawing characters, labelled arrows, failure/shutdown paths shown; sequence lifelines with numbered steps for request/protocol flows). Diagrams must show the real components, channels, storage, external systems, and state transitions that changed — never leave them as stale prose-adjacent decorations.

## Workflow

### 1. Bind the scope

Resolve which phase to build, exactly:

- Read the project impl plan — in this repository, `./specs/91-impl-plan.md` — and find the requested phase. If the user named a milestone (M0/M1), translate it via the roadmap (`./specs/90-roadmap.md`) — milestones and phases pair 1:1 but are numbered differently.
- Read every spec section the phase tasks cite. The impl plan's task table has a "Spec" column for a reason.
- Read `./docs/research/` memos referenced by those specs (if the directory exists). Their decisions bind the implementation.
- Read project `AGENTS.md` and `CLAUDE.md`. Engineering norms (error handling, async, type design, security, logging) apply unconditionally.
- Read `./vendors/` references the spec or research cites — for prior art and exact API shapes.

If a previous phase is *not* fully landed (per its exit criteria), say so and offer to land it first. Do not paper over a gap by starting later.

### 2. Polish the specs first

Before implementation, make the specs good enough that the phase can be built without guessing. This is a hard gate: do not write production code until the relevant feature specs and impl plan are executable.

- Expand or refine the cited specs under `./specs` before code. If a new spec file is needed, follow the project naming and index rules from `AGENTS.md`, then update `./specs/index.md`.
- Bring the feature spec to implementation depth: intended behavior, public API/CLI/schema/protocol shape, domain invariants, validation rules, error model, persistence/state changes, lifecycle/shutdown behavior, concurrency model, security and trust boundaries, observability/audit/metrics events, performance budgets, compatibility/migration concerns, tests, and exit criteria.
- Keep the impl plan in sync with the polished specs: task rows, dependencies, spec links, quality gates, and phase exit criteria must point at the current design, not stale placeholders.
- Update diagrams whenever architecture, data flow, lifecycle/state transitions, dependency order, or failure paths matter. A diagram that cannot guide implementation is not polished.
- Record new or changed architectural decisions in `./specs/99-key-decisions.md`.
- If the spec gap requires a product or architecture decision that cannot be inferred from existing specs, research, or code, ask the user before writing code.
- If the requested phase is already well specified, state that briefly and proceed; do not rewrite specs for churn.

Spec polish is complete only when another engineer could implement the phase from the docs without this conversation. If that is not true, keep polishing.

### 3. Plan the phase

Before code:

- Create a task (TaskCreate) per row in the phase's task table. Status starts pending; mark in_progress one at a time.
- Identify any task that still has unresolved dependencies on specs or research after the polish pass. If anything is unclear, ask the user **before writing code**, not after.
- Check the phase's exit criteria. Those are the conditions for "done"; if you cannot articulate them now, you cannot meet them later.

### 4. Implement, task by task

- Work through tasks in the order the impl plan lists them. The order is dependency-correct; deviating without reason invites retrofits.
- For each task: smallest reasonable PR-shaped commit; passing tests local to that change; no `TODO` / `raise NotImplementedError` stubs / `pass  # placeholder` bodies / `# noqa` or `# type: ignore` suppressions introduced to silence a gate.
- **Match the polished specs exactly.** If you find yourself diverging — wrong API name, different invariant, different envelope shape — stop. Either the spec is wrong (update the spec first when the correction is in-phase and unambiguous; otherwise record it in the deferred-findings backlog — see below — and get the user's call), or your reading is. Drift kills spec sets.
- Make illegal states unrepresentable. If the spec lists invariants, encode them in types — frozen `@dataclass` value objects, `NewType`, `Enum` / `Literal` unions, `Protocol` interfaces, `Final` constants, and validated constructors (`__post_init__` or pydantic models at trust boundaries) that refuse to build invalid values.
- Performance budgets in the spec are not aspirational. If the phase task table cites a budget, write the bench and run it before claiming the task complete.

### 4a. Engineering norms (binding)

Project `AGENTS.md` and `CLAUDE.md` define the binding norms for this codebase — error model, async/concurrency patterns, type design, safety/security rules, serialization shapes, testing conventions, observability, performance, dependencies, code style. **Read both before writing code in this phase** and apply every applicable section unconditionally; they are not aspirational.

If a spec for this phase silently relaxes one of those rules, the spec is wrong: record it in the deferred-findings backlog and raise it before writing code. If you genuinely need to deviate at a specific call site (e.g. a single `# type: ignore[override]` or `# noqa: <rule>`), the suppression must carry the specific error code, and the commit message must name the `file:line` and the reason — reviewers will check. Bare, code-less suppressions are never acceptable.

### 5. Run the standard quality gates

After the spec polish pass, run text checks relevant to the changed docs (`git diff --check`, targeted link/index checks). After each meaningful implementation task and again before claiming the phase complete, run the project's documented gates from the **repo root**. In this repository (per `CLAUDE.md` — there is no CI, so these are the only automated checks):

```bash
PYTHONPATH=demo/src python3 -m unittest discover demo/tests -v   # deterministic test suite
PYTHONPATH=demo/src python3 demo/scripts/run_eval.py             # golden-question retrieval/citation eval
ruff check demo/src demo/scripts demo/tests                      # lint
mypy --strict demo/src demo/tests demo/scripts                   # strict type check
```

Strict type checking catches API drift and `None`-handling bugs — cheap to enforce, easy to let rot if you skip it. In another project, use that project's documented gates (`Makefile` targets, `make check` / `make ci` where wired) — never invent commands the project does not document.

**Never** bypass a gate (`--no-verify`, blanket `# noqa` / `# type: ignore`, skip-marks added to make the suite green, deleting a failing test). If a gate fails, fix the underlying cause.

### 6. Verify exit criteria

The phase has explicit exit criteria in the impl plan. Each one is observable: a test passes, a bench fits a budget, a behaviour can be demonstrated. Show evidence for each — paste the green output, the bench number, or a one-line repro. "Looks done" is not done.

If a phase exit criterion is *blocked* by something the user must decide (a credential, a third-party endpoint), say so explicitly and stop. Do not claim done.

### 7. Commit

Stage with named paths (never `git add -A`). One commit, or a small ordered series; follow the project's commit convention (this repository uses Conventional Commits with scope, per `CLAUDE.md`). The message names the phase and the milestone:

```
feat(<scope>): phase <N> — <one-line summary>

<paragraph: what landed; which spec sections; which exit criteria are met>

<paragraph: known follow-ups, deferred items, links to research memos>
```

### 8. Independent code review

This is the load-bearing step. The phase is **not done** until reviewed against the spec and the valid findings fixed.

- Spawn a code-review subagent (`Agent` tool, `subagent_type: "general-purpose"`). Brief the agent like a colleague who hasn't seen this conversation:

  > Review the diff for phase `<N>` against the polished specs under `./specs/<relevant paths>` and `./docs/research/<relevant memos>`. The phase is supposed to deliver `<exit criteria>`. Expect: spec adherence (concrete, correct, elegant, performant); AGENTS.md/CLAUDE.md compliance (error handling, async, type design, safety/security); no TODOs / dead code / silent fallbacks; matching invariants between polished spec and code; tests covering the phase's exit criteria. Cite findings as `path:LINE` with severity P0/P1/P2/P3 and a recommended fix shape. Do not propose redesigns; defer those to the deferred-findings backlog.

- The agent runs read-only and produces a finding list. Read it carefully.

- Categorise findings:
  - **Valid + in-phase** — fix in this phase before claiming done.
  - **Valid + out-of-phase** — append to the deferred-findings backlog (see below) with severity, file:line, and fix shape. Do not silently inflate scope.
  - **Invalid** — note why in the response so the user can sanity-check the call.

- Fix the in-phase findings. Re-run quality gates. If a fix is non-trivial, commit separately ("phase N review: fix <P-id>") so history shows the review pass.

- If a finding reveals a **spec defect** (the spec is wrong, not the code), record it in the deferred-findings backlog and surface it to the user before patching either side. Spec drift here is exactly what the spec set exists to prevent.

#### Deferred-findings backlog

Out-of-phase findings, deferred items, and surfaced spec defects need a single home so they don't get lost. In this repository that is `./specs/93-improvements-review.md`; in another project, any single canonical Markdown file under `./specs/` (or wherever its `AGENTS.md` directs). If the file does not yet exist, create it and note that in the commit message; if it does, append. Each entry includes severity (P0/P1/P2/P3), `file:line` citation, and a one-line fix shape so the next phase can pick it up without re-deriving the context.

### 9. Hand off

Final report to the user, in this shape:

- **Phase**: N — `<one-line description>`.
- **Specs polished/covered**: `<list of spec paths and sections updated or verified as implementation-ready>`.
- **Exit criteria**: each criterion with `✅` + evidence (test name, bench number, command output).
- **Files changed**: high-level summary, not a file list.
- **Review**: number of findings, P0/P1 fixed in this phase, P2/P3 deferred to the backlog with citations.
- **Next phase**: which phase is unlocked, what its first task is.

## Quality bar

- Spec adherence is binary, not "mostly". Either the API matches and the invariants hold, or you stop and reconcile spec ↔ code in writing.
- Specs are part of the deliverable. A phase cannot be publishable if the relevant specs are vague, stale, or missing exit criteria.
- No `TODO`, `raise NotImplementedError("later")`, `pass  # stub`, or `...` placeholder bodies in production code. If a piece of work cannot be completed in this phase, it does not belong in this phase — defer via a backlog entry.
- No dead code or blanket suppressions. If something is unused, remove it; if a suppression is unavoidable, it carries a specific error code and a `file:line` justification in the commit.
- Tests are part of the deliverable, not an afterthought. Each public surface introduced has at least one happy-path test and one error-path test; load-bearing invariants get property tests (`hypothesis`) where the shape allows and the dependency is already available.
- Bench harnesses ship alongside any task with a perf budget.
- Every public module, class, and function has a docstring; the module has a top-level docstring.
- Every function signature on a public surface is fully type-annotated (the `mypy --strict` gate enforces this); `Any` at a boundary is a finding, not a convenience.

## Common failure modes (avoid)

- **"Phase done" with the review skipped.** The review is the load-bearing checkpoint. Always run it.
- **Starting code from underspecified docs.** Polish the specs first. If the implementation requires guessing, the spec is not ready.
- **Refactor smear.** Touching files outside the phase's scope. Resist; defer to the backlog and keep the diff focused.
- **`except Exception: pass` in a "non-critical" path.** All paths reachable from external input are critical. Catch the narrowest exception type, handle it or re-raise with context (`raise NewError(...) from exc`); never swallow.
- **Mutable default arguments and shared module-level state.** `def f(items=[])` and module-global caches are latent bugs; use `None` sentinels, factories, or explicit dependency injection.
- **Blocking calls inside `async def`.** No synchronous I/O, `time.sleep`, or CPU-heavy loops on the event loop; use the async client, `asyncio.to_thread`, or a worker pool.
- **Adding features the spec did not request.** If it's not in the spec for this phase, it is out of scope. Either update the spec first or land later.
- **Skipping the lint and strict-type gates.** An unformatted or untyped diff is not reviewable.
- **`git reset --hard` to recover from confusion.** Never. Investigate; ask the user; preserve work. The git reflog is your friend.

## Cross-references

- The **spec** skill produces and refines the spec set; this skill polishes the relevant specs before consuming them for implementation.
- The **research** skill produces `./docs/research/<spike|study>-*.md`; this skill respects their decisions.
- `./specs/93-improvements-review.md` is the single home for findings deferred out of the current phase.
- `./specs/99-key-decisions.md` is the canonical record of *why*; if your code conflicts with a decision there, escalate to the user before writing.
