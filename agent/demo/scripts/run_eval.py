"""Run the deterministic golden-question evaluation."""

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

from agent_workshop_demo.eval_runner import (
    EvalScenarioPermissionChecker,
    EvaluationWorkflow,
    evaluate_questions,
)
from agent_workshop_demo.eval_snapshot import MilvusEvalSnapshot
from agent_workshop_demo.embedding import text_embedding_fingerprint
from agent_workshop_demo.langgraph_workflow import build_default_workflow
from agent_workshop_demo.reranker import build_reranker
from agent_workshop_demo.schema.pymilvus_adapter import MilvusHybridRetriever
from agent_workshop_demo.workflow import AgenticRAGWorkflow

DEFAULT_BASELINE = Path("demo/eval/rag_eval_baseline.json")
DEFAULT_GOLDEN = Path("demo/eval/golden_answers.yaml")
DEFAULT_METRIC_REGISTRY = Path("demo/eval/metric_registry.json")
DEFAULT_QUESTIONS = Path("demo/eval/questions.json")
DEFAULT_REVIEW = Path("demo/eval/rag_eval_review.json")
DETERMINISTIC_PROVIDERS = {
    "EMBEDDING_PROVIDER": "deterministic",
    "QUERY_CLASSIFIER": "rule_based",
    "RERANKER": "rule_based",
    "ANSWER_GENERATOR": "deterministic",
    "MEMORY_SELECTOR": "rule_based",
}


def _content_identity(paths: list[tuple[str, Path]]) -> str:
    digest = hashlib.sha256()
    for label, path in paths:
        payload = path.read_bytes()
        digest.update(label.encode("utf-8"))
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return f"sha256:{digest.hexdigest()}"


def main(argv: list[str] | None = None) -> int:
    """Evaluate the configured questions file."""

    parser = argparse.ArgumentParser(description="Run golden QA evaluation.")
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS,
    )
    parser.add_argument(
        "--golden-answers",
        type=Path,
        default=DEFAULT_GOLDEN,
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--metric-registry",
        type=Path,
        default=DEFAULT_METRIC_REGISTRY,
        help="Strict eval-metric-registry-v1 decision portfolio.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the JSON report to this file instead of stdout.",
    )
    parser.add_argument(
        "--live-providers",
        action="store_true",
        help="Use explicitly configured providers; requires at least 3 trials.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        help="Independent trials per question (default: 1 offline, 3 live).",
    )
    baseline_group = parser.add_mutually_exclusive_group()
    baseline_group.add_argument(
        "--baseline",
        type=Path,
        help=(
            "Committed rag-eval-v3 report used for active-metric deltas "
            "(defaults to the repository baseline only in deterministic mode)."
        ),
    )
    baseline_group.add_argument(
        "--no-baseline",
        action="store_true",
        help="Do not load a baseline (intended only when creating a new one).",
    )
    review_group = parser.add_mutually_exclusive_group()
    review_group.add_argument(
        "--review",
        type=Path,
        help="Strict rag-eval-review-v1 human transcript attribution fixture.",
    )
    review_group.add_argument(
        "--no-review",
        action="store_true",
        help="Report every observed failure as awaiting human attribution.",
    )
    parser.add_argument(
        "--milvus-uri",
        help="Enable snapshot-backed eval; intentionally has no env-only default.",
    )
    parser.add_argument("--milvus-token", default=os.getenv("MILVUS_TOKEN"))
    parser.add_argument("--snapshot-name")
    parser.add_argument("--source-collection")
    parser.add_argument("--target-collection")
    parser.add_argument(
        "--sparse-field",
        default=os.getenv("MILVUS_SPARSE_FIELD", "sparse_vector"),
        help="BM25 output field; defaults to MILVUS_SPARSE_FIELD.",
    )
    parser.add_argument("--snapshot-timeout", type=float, default=60.0)
    args = parser.parse_args(argv)
    trials = (
        args.trials if args.trials is not None else (3 if args.live_providers else 1)
    )
    if args.live_providers and trials < 3:
        parser.error("--live-providers requires --trials >= 3")
    if not args.live_providers and trials != 1:
        parser.error("deterministic evaluation requires --trials 1")
    if not args.live_providers:
        os.environ.update(DETERMINISTIC_PROVIDERS)
    snapshot_args = (
        args.milvus_uri,
        args.snapshot_name,
        args.source_collection,
        args.target_collection,
    )
    if any(snapshot_args) and not all(snapshot_args):
        parser.error(
            "--milvus-uri, --snapshot-name, --source-collection and "
            "--target-collection must be provided together"
        )
    retriever = None
    provenance = None
    if all(snapshot_args):
        retriever = MilvusHybridRetriever.connect(
            args.milvus_uri,
            args.milvus_token,
            collection_name=args.target_collection,
            sparse_field=args.sparse_field,
        )
        provenance = MilvusEvalSnapshot(retriever.client).pin(
            source_collection=args.source_collection,
            snapshot_name=args.snapshot_name,
            target_collection=args.target_collection,
            timeout_seconds=args.snapshot_timeout,
        )

    def workflow_factory(scenario: dict[str, str]) -> EvaluationWorkflow:
        permission_checker = EvalScenarioPermissionChecker(
            allowed=scenario["permission"] == "allow"
        )
        if args.live_providers:
            return cast(
                EvaluationWorkflow,
                build_default_workflow(
                    retriever=retriever,
                    permission_checker=permission_checker,
                ),
            )
        # `fallback` is the ordinary `RERANKER=auto` build with no credentials:
        # the configured wrapper degrades to the deterministic rule reranker and
        # reports `not_configured`. No test double enters the offline CLI.
        reranker = (
            build_reranker({"RERANKER": "auto"})
            if scenario.get("reranker") == "fallback"
            else None
        )
        return cast(
            EvaluationWorkflow,
            AgenticRAGWorkflow(
                retriever=retriever,
                permission_checker=permission_checker,
                reranker=reranker,
            ),
        )

    provider_profile = {
        "kind": "configured_live" if args.live_providers else "deterministic",
        "embedding": os.getenv("EMBEDDING_PROVIDER", "deterministic"),
        "classifier": os.getenv("QUERY_CLASSIFIER", "rule_based"),
        "reranker": os.getenv("RERANKER", "rule_based"),
        "generator": os.getenv("ANSWER_GENERATOR", "deterministic"),
        "memory_selector": os.getenv("MEMORY_SELECTOR", "rule_based"),
        "query_transformer": os.getenv("QUERY_TRANSFORMER", "rule_based"),
        "context_compression": os.getenv(
            "CONTEXT_COMPRESSION_MODE",
            "disabled",
        ),
        "text_vector_space": text_embedding_fingerprint(),
    }
    if args.live_providers:
        for name in (
            "OPENAI_EMBEDDING_MODEL",
            "OPENAI_CLASSIFIER_MODEL",
            "OPENAI_RERANKER_MODEL",
            "OPENAI_QUERY_TRANSFORMER_MODEL",
            "OPENAI_CONTEXT_COMPRESSOR_MODEL",
            "OPENAI_MODEL",
        ):
            value = os.getenv(name, "").strip()
            if value:
                provider_profile[name.casefold()] = value
    fixture_identity = _content_identity(
        [
            ("questions", args.questions),
            ("golden_answers", args.golden_answers),
        ]
    )
    dataset_profile = (
        {
            "kind": "milvus_snapshot",
            "fixture_identity": fixture_identity,
            "snapshot_name": str(args.snapshot_name),
            "source_collection": str(args.source_collection),
            "target_collection": str(args.target_collection),
            "sparse_field": str(args.sparse_field),
            "restore_job_id": str(
                getattr(provenance, "restore_job_id", "not_reported")
            ),
        }
        if provenance is not None
        else {
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
        }
    )

    report = evaluate_questions(
        questions_path=args.questions,
        golden_answers_path=args.golden_answers,
        top_k=args.top_k,
        scenario_workflow_factory=workflow_factory,
        trials=trials,
        live_providers=args.live_providers,
        baseline_path=(
            None
            if args.no_baseline
            else (
                args.baseline
                or (
                    DEFAULT_BASELINE
                    if not args.live_providers
                    and args.questions == DEFAULT_QUESTIONS
                    and args.golden_answers == DEFAULT_GOLDEN
                    else None
                )
            )
        ),
        review_path=(
            None
            if args.no_review
            else (
                args.review
                or (
                    DEFAULT_REVIEW
                    if not args.live_providers
                    and args.questions == DEFAULT_QUESTIONS
                    and args.golden_answers == DEFAULT_GOLDEN
                    else None
                )
            )
        ),
        metric_registry_path=args.metric_registry,
        provider_profile=provider_profile,
        dataset_profile=dataset_profile,
    )
    if provenance is not None:
        report["dataset_snapshot"] = provenance.to_dict()
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is None:
        print(serialized)
    else:
        try:
            args.output.write_text(f"{serialized}\n", encoding="utf-8")
        except OSError as exc:
            parser.error(f"unable to write eval report {args.output}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
