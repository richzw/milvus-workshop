"""Run the deterministic golden-question evaluation."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from agent_workshop_demo.eval_runner import evaluate_questions
from agent_workshop_demo.eval_snapshot import MilvusEvalSnapshot
from agent_workshop_demo.schema.pymilvus_adapter import MilvusHybridRetriever
from agent_workshop_demo.workflow import AgenticRAGWorkflow


def main(argv: list[str] | None = None) -> int:
    """Evaluate the configured questions file."""

    parser = argparse.ArgumentParser(description="Run golden QA evaluation.")
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("demo/eval/questions.json"),
    )
    parser.add_argument(
        "--golden-answers",
        type=Path,
        default=Path("demo/eval/golden_answers.yaml"),
    )
    parser.add_argument("--top-k", type=int, default=20)
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
    workflow = None
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
        workflow = AgenticRAGWorkflow(retriever=retriever)
    report = evaluate_questions(
        questions_path=args.questions,
        golden_answers_path=args.golden_answers,
        top_k=args.top_k,
        workflow=workflow,
    )
    if provenance is not None:
        report["dataset_snapshot"] = provenance.to_dict()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
