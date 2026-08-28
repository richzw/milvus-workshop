"""Run the comparative retrieval-tier evaluation (spec 70 § 4.2d)."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import cast

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from agent_workshop_demo.embedding import text_embedding_fingerprint
from agent_workshop_demo.eval_runner import (
    EvalScenarioPermissionChecker,
    EvaluationWorkflow,
)
from agent_workshop_demo.retrieval_tier import TIER_CODES, RetrievalTier
from agent_workshop_demo.retrieval_tier_eval import (
    TIER_ARMS,
    ArmFactory,
    evaluate_retrieval_tiers,
)
from agent_workshop_demo.workflow import AgenticRAGWorkflow

DEFAULT_GOLDEN = Path("demo/eval/golden_answers.yaml")
DEFAULT_METRIC_REGISTRY = Path("demo/eval/metric_registry.json")
DEFAULT_QUESTIONS = Path("demo/eval/questions.json")
DETERMINISTIC_PROVIDERS = {
    "EMBEDDING_PROVIDER": "deterministic",
    "QUERY_CLASSIFIER": "rule_based",
    "RERANKER": "rule_based",
    "ANSWER_GENERATOR": "deterministic",
    "MEMORY_SELECTOR": "rule_based",
    "QUERY_TRANSFORMER": "rule_based",
}


def _content_identity(paths: list[tuple[str, Path]]) -> str:
    digest = hashlib.sha256()
    for label, path in paths:
        payload = path.read_bytes()
        digest.update(label.encode("utf-8"))
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return f"sha256:{digest.hexdigest()}"


def _arm_factory(arm: str) -> ArmFactory:
    """Build one tier arm that differs only in retrieval mechanics."""

    def factory(scenario: dict[str, str]) -> EvaluationWorkflow:
        return cast(
            EvaluationWorkflow,
            AgenticRAGWorkflow(
                permission_checker=EvalScenarioPermissionChecker(
                    allowed=scenario["permission"] == "allow"
                ),
                retrieval_tier=RetrievalTier(arm),
            ),
        )

    return factory


def main(argv: list[str] | None = None) -> int:
    """Compare the lexical baselines against the hybrid default."""

    parser = argparse.ArgumentParser(
        description="Run the retrieval tier comparison evaluation.",
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--golden-answers", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--metric-registry",
        type=Path,
        default=DEFAULT_METRIC_REGISTRY,
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=TIER_ARMS,
        default=list(TIER_ARMS),
        help="Arms to execute; omitted arms stay evaluation_incomplete.",
    )
    parser.add_argument(
        "--latency-budget-ms",
        type=float,
        help=(
            "Accepted end-to-end P95 budget; without it the default tier "
            "cannot claim latency evidence."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the JSON report to this file instead of stdout.",
    )
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        parser.error("--top-k must be greater than zero")
    os.environ.update(DETERMINISTIC_PROVIDERS)
    fixture_identity = _content_identity(
        [
            ("questions", args.questions),
            ("golden_answers", args.golden_answers),
        ]
    )
    report = evaluate_retrieval_tiers(
        questions_path=args.questions,
        golden_answers_path=args.golden_answers,
        arm_workflow_factories={arm: _arm_factory(arm) for arm in args.arms},
        top_k=args.top_k,
        metric_registry_path=args.metric_registry,
        provider_profile={
            "kind": "deterministic",
            "embedding": os.environ["EMBEDDING_PROVIDER"],
            "classifier": os.environ["QUERY_CLASSIFIER"],
            "reranker": os.environ["RERANKER"],
            "generator": os.environ["ANSWER_GENERATOR"],
            "query_transformer": os.environ["QUERY_TRANSFORMER"],
            "text_vector_space": text_embedding_fingerprint(),
        },
        dataset_profile={
            "kind": "offline_seed",
            "fixture_identity": fixture_identity,
            "corpus_identity": _content_identity(
                [
                    (
                        "sample_data",
                        SOURCE_ROOT / "agent_workshop_demo/sample_data.py",
                    )
                ]
            ),
        },
        latency_budget_ms=args.latency_budget_ms,
    )
    report["evaluation"]["tier_codes"] = {
        tier.value: TIER_CODES[tier] for tier in RetrievalTier
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is None:
        print(serialized)
    else:
        try:
            args.output.write_text(f"{serialized}\n", encoding="utf-8")
        except OSError as exc:
            parser.error(f"unable to write tier eval report {args.output}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
