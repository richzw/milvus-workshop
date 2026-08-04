"""Run the independent text/image retrieval evaluation."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from agent_workshop_demo.image_embedding import (
    configured_image_embedding_provider,
)
from agent_workshop_demo.image_eval import evaluate_image_retrieval
from agent_workshop_demo.ingestion import ingest_demo_sources
from agent_workshop_demo.retrieval import InMemoryHybridRetriever


def main(argv: list[str] | None = None) -> int:
    """Ingest the corpus once and evaluate both image retrieval modes."""

    parser = argparse.ArgumentParser(
        description="Run image retrieval evaluation.",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("demo/eval/image_retrieval.json"),
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path("demo/sample_data/local_docs"),
    )
    parser.add_argument(
        "--mock-s3-dir",
        type=Path,
        default=Path("demo/sample_data/mock_s3"),
    )
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args(argv)

    provider = configured_image_embedding_provider()
    ingestion = ingest_demo_sources(
        args.local_dir,
        args.mock_s3_dir,
        image_embedding_provider=provider,
    )
    report = evaluate_image_retrieval(
        cases_path=args.cases,
        retriever=InMemoryHybridRetriever(ingestion.kb_chunks),
        image_provider=provider,
        top_k=args.top_k,
        assets_root=args.cases.parent.parent,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
