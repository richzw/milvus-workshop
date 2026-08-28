# Spec document templates

Read the template for the document you are about to write. Adapt section content to the system at hand; keep the section skeleton so spec sets stay navigable across projects.

## PRD template (`00-prd.md`)

```markdown
# PRD — <product name>

Status: <draft v1> · Owner: <team> · Last updated: <YYYY-MM-DD>

## 1. Problem

What is broken today, with concrete evidence (incidents, costs, missing capability). Avoid abstractions; name the failure mode users actually hit.

## 2. Vision

What "good" looks like, in one paragraph plus one concrete code / UX example. The example is load-bearing — it pins the ergonomic contract.

## 3. Goals

| #  | Goal | Measure |
| -- | ---- | ------- |
| G1 | …    | …       |

Each goal must have a *measurable* success criterion. "Better DX" is not a goal; "≤ 60 s from install to first result on stdout" is.

## 4. Non-goals

Explicit list. Non-goals prevent scope creep more than goals do.

## 5. Users

Primary, secondary, anti-personas. What each persona is doing when they reach for the product.

## 6. Success metrics

What we will measure post-launch to know we shipped the right thing.

## 7. Naming conventions (binding)

Public namespaces, prefixes, file layouts that the rest of the spec set must honour. Lock early; renames are expensive.
```

## Component design template (`NN-<name>.md`)

```markdown
# <NN>-<name>: <subsystem>

Status: <draft|stable> · Owner: <team> · Depends on: <list of NN specs>

## 1. Purpose

One paragraph: what this subsystem owns, what it does not own, why it exists separately.

## 2. Interface

The public types / interfaces / functions / wire shapes. Code-shaped where possible.

## 2a. Architecture / flow diagrams

Concise boxed ASCII diagrams for any non-trivial component boundary, data flow, processing pipeline, state transition, or dependency relationship this subsystem owns. Nested boxes for components and ownership boundaries, labelled arrows for message/data movement, side branches for failures, retries, shutdown, backpressure. Sequence-style lifelines for ordered request/protocol flows where the exact step order is the contract.

## 3. Invariants

The properties that must hold at every observable point. Each invariant has a test or lint that pins it.

## 4. Behaviour

The non-trivial cases — error paths, edge cases, concurrent / async behaviour, cleanup order, failure policy, cancellation.

## 5. Cross-references

- ← Depends on: <links>
- → Consumed by: <links>
- ↔ Related research: <links to ./docs/research/...>
```

## Roadmap template (`90-roadmap.md`)

(4-backtick outer fence so the inner triple-backtick blocks render when copied.)

````markdown
# Roadmap — Incremental Delivery

## 0. Principles

- **Always shippable.** Every milestone leaves the workspace green on the standard quality gates.
- **Type-safety / contract-safety first.** Each milestone may defer features but never relaxes guarantees.
- **Honest calibration.** Estimates are realistic; pad explicitly for review/on-call/meeting overhead.

## 1. Build-order graph

```text
┌──────────┐    ┌────────────────┐    ┌────────────────────┐
│ 00 PRD   │───▶│ 10 Data Model  │───▶│ 11 Runtime Core    │
│ goals    │    │ invariants     │    │ lifecycle / traits │
└──────────┘    └───────┬────────┘    └─────────┬──────────┘
                        │                       │
                        ▼                       ▼
                ┌────────────────┐    ┌────────────────────┐
                │ 12 Foundation  │───▶│ 20 Integration     │
                │ contracts      │    │ transport / sink   │
                └───────┬────────┘    └─────────┬──────────┘
                        │                       │
                        ▼                       ▼
                ┌────────────────┐    ┌────────────────────┐
                │ 60/61 DX       │    │ 70/71/72 Gates     │
                │ pkgs/features  │    │ security/perf/test │
                └────────────────┘    └────────────────────┘
```

## 2. Milestones

### M0 — <user-visible feature>

**Specs touched**: 00, 10, 11, 12.
**Exit criteria**: a fresh user can <do thing> in <time>; <invariant> holds; <test> passes.

### M1 — …

…
````

## Impl-plan template (`91-impl-plan.md`)

````markdown
# Implementation Plan — Dependency-Ordered Build

## 0. Readiness assessment

What is ready, what isn't, and what blocks Phase 1 today. Be honest; missing specs and unvalidated assumptions go here.

## 1. Why dependency order ≠ feature order

Two or three concrete examples where the dependency-correct order differs from the user-feature order, with the *why*. This justifies the rest of the document.

## 2. Estimated total effort

Calendar weeks for one developer, with assumptions. Note where parallelism collapses the schedule.

## 3. Phase 0 — risk retirement

| #  | Deliverable | Lands in | Effort |
| -- | ----------- | -------- | ------ |

Each spike memo from `./docs/research/` listed; missing specs called out; no production code yet.

**Exit gate**: every spike memo committed; specs updated to reflect findings.

## 4. Phase 1 — foundation (weeks N–M)

The spine in strict dependency order. Each row blocks everything underneath.

| #   | Task | Spec | Effort |
| --- | ---- | ---- | ------ |
| 1.1 | …    | …    | …      |

**Exit criteria**: <test> passes; <bench> is within budget; <invariant> verified.

## 5+ Phase 2 … Phase N

Same shape per phase. Cross-reference roadmap milestones explicitly: "Phase 3 closes M2 and starts M3."

## N. What makes this order *correct*, not just plausible

Two or three principles that drove the ordering. State them so a reviewer can challenge the order on its own terms instead of arguing tasks line by line.
````

## Key-decisions template (`99-key-decisions.md`)

```markdown
# Key Decisions

Each decision is permanent; supersede with a new D-id rather than editing in place.

## D1 — <one-line decision>

- **Context**: where this applies
- **Alternatives considered**: A, B, C — with the trade-offs that ruled them out
- **Decision**: the chosen path, in one sentence
- **Why**: load-bearing reasoning
- **Pinned by**: <links to spec sections that depend on this>
- **Date**: <YYYY-MM-DD>
```
