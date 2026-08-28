# LLM-as-judge: rubric design and calibration

Read this before writing or modifying any judge prompt. A judge is a measurement instrument; an instrument nobody calibrated produces confident nonsense at scale.

## Rubric design

- **One dimension per judge.** "Faithfulness" and "conciseness" are two judges, not two bullet points in one prompt. A blended score cannot tell you which dimension regressed.
- **Eliminate ambiguity.** If two human experts cannot apply the rubric consistently, an LLM cannot either. Every criterion must be checkable: "professional tone" is ambiguous; "no exclamation marks; sentences end with periods; no marketing superlatives" is checkable.
- **Decompose, don't vibe.** Many small itemized checks outperform one holistic judgment. Turn "is this readable?" into: sentence length bound, jargon list, structural requirements, tone violations.
- **Include explicit negative examples.** Show the judge what scores 0 and *why*. Rubrics that only describe success teach the judge to say yes.
- **Anchor with few-shot examples** spanning the score range, including near-miss cases (the accurate paraphrase that is *not* unfaithful; the terse answer that is *not* incomplete).
- **Ask for chain-of-thought before the verdict**, and use a separate (ideally stronger) model than the one being evaluated.

Scoring setups, by fit:

| Setup | Use when |
| --- | --- |
| Direct assessment (score against rubric) | absolute quality gates, monitoring |
| Pairwise comparison ("which is better?") | A/B of prompts/models; more stable than absolute scores |
| Reference-guided (judge sees golden answer) | correctness-flavoured dimensions with known ground truth |

## Calibration loop

1. **Golden set**: 50–100 human-labeled examples for this dimension, including failures. Built with subject-matter experts; if the experts disagree, fix the rubric or the label — do not proceed.
2. **Run the judge** over the golden set.
3. **Measure agreement** with the human labels. Target high-80s–90s %. Use a chance-corrected statistic (Cohen's kappa / Krippendorff's alpha), not raw accuracy — on an imbalanced set, "always pass" scores deceptively well raw.
4. **Read every disagreement.** Classify: judge wrong (fix prompt / add few-shot anchor), label wrong (fix golden set), rubric ambiguous (fix rubric, relabel). A real example of the loop: a faithfulness judge at 78 % agreement was penalizing accurate paraphrases as unfaithful; adding a paraphrase-is-faithful rule plus two anchors raised agreement to 88 %.
5. **Iterate** steps 2–4 until the target holds, then freeze the judge version alongside the golden set version.
6. **Recalibrate periodically** — on model upgrades, rubric edits, or when new failure modes appear in production. Judge drift is silent.

## Known judge biases to monitor

- **Verbosity bias** — longer answers score higher regardless of content. Counter: length-invariance instruction + a short-correct few-shot anchor.
- **Position bias** (pairwise) — the first (or last) option wins too often. Counter: score both orderings, average or flag flips.
- **Self-preference** — a model rates its own family's outputs higher. Counter: judge from a different model family than the system under test.
- **Sycophancy toward confident tone** — assertive wrong answers outscore hedged right ones. Counter: rubric line separating correctness from confidence.
- **Score compression** — everything gets 4/5. Counter: pairwise setups, or forced rubric anchors per score level.

## When to use which evaluator (decision matrix)

| Situation | Evaluator |
| --- | --- |
| High volume + clear, stable rubric | scaled annotation workforce or calibrated judge |
| High stakes + ambiguous rubric | human experts only — do not automate yet |
| Iterative development loop | subject-matter experts on a small set, fast turnaround |
| Production monitoring | layered: programmatic + calibrated judges + sampled human review |

## Judge prompt skeleton

```text
You are evaluating <dimension> of an answer produced by <system>.

Rubric (apply every item):
1. <checkable criterion>
2. <checkable criterion>
3. <criterion with explicit negative example: "X scores 0 because …">

Not violations (do NOT penalize):
- <near-miss that must pass, e.g. accurate paraphrase of the source>

Examples:
<few-shot: input, answer, reasoning, verdict — spanning the score range>

Now evaluate.
<input, answer, (optional) reference>

First reason step by step through each rubric item, then output:
{"verdict": <pass|fail or 1–5>, "violations": [<rubric item numbers>], "reasoning": "<one paragraph>"}
```

Emit the verdict as structured output (JSON schema) so the L1 programmatic layer can parse and aggregate it — never regex a free-text verdict.
