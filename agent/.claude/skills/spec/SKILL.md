---
name: spec
description: Turn a feature idea or rough requirement into a complete, dependency-ordered spec set under ./specs — PRD, component designs, glossary, cross-cuts, key-decisions log, stakeholder roadmap, and engineer-facing implementation plan — cross-referenced with prior-art memos in ./docs/research and vendored code in ./vendors. Use whenever the user says "write the spec", "design this", "let's plan X", "produce a PRD / impl plan / roadmap", "restructure the specs", "review and re-organise the design", "think hard and split this into phases", or describes a non-trivial system that needs a written design before code. Trigger even when the user does not say "spec" if they ask for phased delivery, milestone exit criteria, or a build-order graph.
---

# Spec

Turn requirements into a load-bearing spec set: numbered, cross-linked, dependency-ordered, with a stakeholder-facing roadmap and an engineer-facing implementation plan that pair 1:1. The spec set is the contract between intent and code; if it is wrong or incoherent, every downstream phase pays for it.

## When this fires

- "write the spec / PRD / design / impl plan / roadmap for X"
- "restructure the specs so it can be built incrementally"
- "review the design and add the missing specs"
- "phase this delivery — what lands when?"
- The user describes a system non-trivial enough that ad-hoc coding will produce drift, missing invariants, or unbounded scope
- The research skill has produced memos and the next move is to commit decisions to a spec

## Output shape

A directory of numbered Markdown files under `./specs/`. The numbering is the **build order** — reading top-to-bottom matches the milestone progression in the roadmap. Update `./specs/index.md` so a fresh reader can navigate.

Document templates (PRD, component design, roadmap, impl-plan, key-decisions) live in [references/templates.md](references/templates.md) — read that file when you start writing the corresponding document.

### Diagram expectation

Use fenced ` ```text ` ASCII diagrams whenever they materially improve understanding of architecture, data flow, lifecycle/state transitions, or build/dependency order. Diagrams are part of the spec contract: precise labels, explicit directionality, placed near the decision they clarify. Prefer terminal-safe box-drawing characters (`┌─┐│└┘`, `▼`, `▲`) over Mermaid so specs stay readable in terminals and code review.

- For non-trivial systems, the diagram must be structurally rich enough to review: nested boxes, lanes, or grouped boundaries showing component ownership, trust/runtime boundaries, queues/channels, storage, external services, error paths, and fan-in/fan-out. A bare `A -> B -> C` chain is acceptable only for a short linear ordering with no meaningful boundary.
- For ordered protocols, login flows, request lifecycles, retries, or multi-party handshakes, use a sequence diagram: actors as columns, time top-to-bottom, numbered steps, with durable state changes, validation, persistence, and failure branches labelled.
- Do not add decorative diagrams that merely restate simple prose.

(The roadmap template in `references/templates.md` contains a worked build-order-graph example of the expected style.)

### Right-size the spec set to the problem

**The number of files is a function of system complexity, not a template to fill.** Do not generate every slot below just because the layout shows it. A small library might ship as `00-prd.md` + `10-design.md` + `90-roadmap.md` and nothing else; a large multi-component platform might need every slot plus a few more. Pick the smallest set that captures the load-bearing decisions for *this* system.

Heuristics for sizing:

- **Tiny (1–3 files)** — a single-purpose module, one or two integration points, no novel invariants. Often: PRD + one design doc + a short roadmap. Skip the cross-cuts; their content fits in the design doc.
- **Medium (5–10 files)** — multiple components with non-trivial contracts between them, more than one integration point, real performance or security constraints. PRD + per-component designs + roadmap + impl-plan + glossary if any term is overloaded. Cross-cuts only when their content does not fit cleanly in the component designs.
- **Large (15+ files)** — a platform / SDK with many components, multiple consumers, long-lived contracts, freeze windows, public RFCs. The full canonical layout below earns its keep.

A good test before adding a file: *what specific question does this file answer that nothing else answers?* If you cannot name one, fold it into the nearest related spec.

### Canonical layout (illustrative — pick what applies)

Treat this as a menu of named slots so files can grow in cleanly later — not a checklist. Omit any slot that does not earn its keep for the current scope; keep the numbering scheme so future additions slot in without renumbering. (This repository's own `specs/` directory follows this scheme — mirror its existing naming when extending it.)

```
specs/
├── index.md                       — table of every spec + reading order + build-order graph
├── 00-prd.md                      — product requirements (vision, users, goals, non-goals, success metrics)
├── 10-data-model.md               — wire shapes, envelope / message / record types, naming conventions
├── 11-runtime-core.md             — engine: lifecycle, interfaces, threading, failure policy
├── 12-<component>.md              — additional foundation designs, in dependency order
├── 20-<integration>.md            — outward-facing integrations (transports, sinks, exporters)
├── 30-<bridge>.md                 — interop with neighbouring ecosystems
├── 40-<middleware>.md             — middleware / framework adapters
├── 50-cli.md                      — CLI surface, if any
├── 60-dev-ergonomics.md           — what using the system feels like; concrete examples; quickstart
├── 61-packages-and-features.md    — workspace layout, dependency graph, feature flags
├── 70-security.md                 — threat model, classification, redaction, secrets
├── 71-performance-budgets.md      — P50/P99 targets, bench harness, CI gates
├── 72-testing-strategy.md         — test pyramid, fixtures, integration mocks
├── 80-glossary.md                 — disambiguate overloaded terms
├── 90-roadmap.md                  — STAKEHOLDER-facing: milestones M0…Mn, exit criteria, calendar shape
├── 91-impl-plan.md                — ENGINEER-facing: dependency-ordered phases, effort estimates
├── 92-rfc-<v>.md                  — public-comment summary at freeze (when applicable)
├── 93-improvements-review.md      — deferred-findings backlog; the impl skill appends to it
└── 99-key-decisions.md            — D1…Dn, the *why* behind each load-bearing choice
```

Several of these are conditional even on a large system: `50-cli.md` only if there is a CLI; `30-…` / `40-…` only if there are real interop/middleware surfaces; `92-rfc-*` only at a public freeze; the deferred-findings backlog is created on demand by the impl skill, not preemptively.

Two rules about the roadmap / impl-plan split — they are different documents on purpose:

- **`90-roadmap.md`** is organised by *user-visible feature*. M0 = "hello world emit", M1 = "schema-first authoring". Stakeholders read this to plan calendars.
- **`91-impl-plan.md`** is organised by *dependency order*. Phase 1 = the spine that nothing else can be built without. Engineers read this to know what to write next, and *why*.

The two pair 1:1 against milestones but the order and grouping differ. Earlier drafts conflate them; do not.

## Workflow

1. **Capture intent** — restate the user's requirement in 1–2 paragraphs: the problem and the vision. Mirror it back before generating files. If the requirement is fuzzy ("we want better observability"), force it concrete (users, top job-to-be-done, the one metric of success). A PRD with no measurable success looks fine and ages badly.

2. **Read what exists** — before adding files:
   - `./specs/index.md` (if present) — what's already designed, what naming is used.
   - `./docs/research/` — every memo. Cite their decisions; do not re-litigate.
   - `./vendors/` — for prior art the design should align with or deliberately diverge from. Reference `vendors/<repo>/path:LINE` directly in the spec.
   - **Project `AGENTS.md` and `CLAUDE.md`** — engineering norms the spec **must** encode into design decisions, not just respect in spirit. The spec is allowed to set *tighter* rules; it must not silently relax them. See 2a.

### 2a. Bind project engineering norms

The spec commits the project to specific engineering patterns up front; the impl skill will then match those patterns line-for-line. Before drafting any component design, read the project's `AGENTS.md` / `CLAUDE.md` and encode their norms into the spec text — not as a footnote, not "TBD per coding standards", but as concrete shapes (error types, async surfaces, validation points, lint/type-check gates, doc requirements) that a reviewer can mechanically check.

Do not restate the norms in the spec; **reference them** ("Errors: per AGENTS.md § Error Handling"). If a component genuinely needs to deviate, the spec must say "deviates from AGENTS.md § X because …" so reviewers can challenge it. If the norms are silent on a question the spec must answer, the spec sets the rule and `99-key-decisions.md` records why.

A component design is not done until each norms section relevant to it (error handling, concurrency, type design, safety/security, serialization, testing, observability, performance, documentation) is either pinned by reference or marked "N/A — <one-line reason>".

3. **Run the research skill if prior art is missing** — if the design hinges on an assumption that has not been validated (a library works under the target runtime; an API actually composes; a perf budget is achievable), invoke the research skill first. Do not bake unvalidated assumptions into a spec.

4. **Think hard about phasing** — before writing a single design doc, sketch:
   - What is the smallest end-to-end slice a user can run? That is M0.
   - What does each subsequent milestone *unlock*? Name it from the user's POV.
   - What blocks what? Draw the build-order graph.
   - Where architecture, data flow, or subsystem relationships are load-bearing, draw the boxed ASCII diagram before writing the prose.

   Two principles that separate good phasing from plausible phasing:

   - **Land contracts before consumers.** If every sink consumes a shared schema interface, the schema registry lands in the foundation, not alongside the first sink. Otherwise the contract is provisional and gets retrofitted.
   - **Pay design costs once, in the foundation.** Multi-tenant resolution, security classification, error envelopes, identity / context propagation — adding any of these later is a refactor of every call site. Settle them in the spine even if M0 only uses the trivial case.

5. **Write the PRD first** (`00-prd.md`). Vision, users, goals (with measurable criteria), non-goals (explicit — non-goals prevent scope creep more than goals do), success metrics, naming conventions that will bind the rest of the spec set.

6. **Write the data model** (`10-data-model.md`). The wire shape every downstream component sees. Naming, types, invariants, envelope vs payload distinction. Lock this early — drift here cascades.

7. **Write component designs in build order** (`11-…`, `12-…`, `20-…`, etc.). Each spec ends with a "Cross-references" section pointing to the specs it depends on and the specs that depend on it. Use `[NN-name.md § X](./NN-name.md#x)` link form so jumps work in any markdown viewer.

8. **Write the cross-cuts** (`60`, `61`, `70`, `71`, `72`). These are read alongside the build-order specs, not in sequence. Make them small and concrete; do not let them become philosophy essays.

9. **Write the glossary** (`80-glossary.md`). Every overloaded term gets a one-paragraph disambiguation. Cheap, prevents weeks of arguing.

10. **Write the roadmap and impl-plan together** (`90`, `91`):
    - Roadmap has milestones, exit criteria, and a calendar estimate. Calibrate honestly: if the spec's earlier estimate was off by 2×, say so and adjust.
    - Impl-plan has Phase 0 (risk retirement), Phase 1 (spine), … Phase N (hardening). Each phase has a numbered task table with spec citations and effort estimates. Each phase has explicit *exit criteria* — a test or invariant that must hold before the next phase starts.

11. **Write `99-key-decisions.md`** as the spec set stabilises. Each entry: D-id, decision, alternatives considered, the *why*, and a reverse pointer to the spec sections that depend on it. When a future reviewer asks "why this?" — point them here, not to a chat scrollback.

12. **Update `index.md`** — table of every spec with type + purpose, plus a reading-order list and the build-order graph. The index is the entry point; spend real effort on it.

## Quality bar

A spec set is **done** when:

- A new engineer can read `index.md` → `00-prd.md` → the build-order specs in order, and write code without asking "what did the author mean?"
- Every cross-reference resolves; every "Sink does X" has exactly one authoritative location and others link to it.
- Every load-bearing decision has an entry in `99-key-decisions.md`.
- The roadmap exit criteria are *observable* (a test, a bench, a user-visible behaviour) — not "Phase X is done when we finish Phase X".
- Estimates have been calibrated against `./docs/research/` findings, not napkin guessed.

## Anti-patterns

- **PRD that conflates problem and solution** — the problem section reads like an API spec. Force the problem to stand alone; if you can't write it without naming the solution, the requirement isn't understood yet.
- **Roadmap = impl-plan with different headings.** They are different documents. If 90 and 91 have the same row order, one of them is wrong.
- **Specs that ban features ("we will never do X") without saying why.** Future you will violate the ban for good reasons; preserve the *why* now.
- **Glossary that defines obvious terms.** Define only the overloaded ones — the words that two readers used differently in the same conversation.
- **`99-key-decisions.md` as a changelog.** It is not. It records load-bearing *design* choices, with alternatives and rationale, for future reviewers.

## Hand-off

When the spec set lands:

- Commit with a single message that names the milestone shape: "specs: PRD + foundation designs + roadmap (M0–M2)".
- Tell the user the entry points: `index.md`, the PRD, and the impl-plan's Phase 0.
- If Phase 0 has open spikes, hand off to the **research** skill before any code is written.
- If Phase 0 is closed, the **impl** skill takes over with `91-impl-plan.md` Phase 1.
