"""Run the versioned Min-Max chunking configuration comparison."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from agent_workshop_demo.chunking_experiment import (
    run_chunking_experiment,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare versioned Min-Max chunking configs.",
    )
    parser.add_argument(
        "--configs",
        type=Path,
        default=Path("demo/eval/chunking_configs.json"),
    )
    parser.add_argument(
        "--anchors",
        type=Path,
        default=Path("demo/eval/chunking_anchors.json"),
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
    parser.add_argument(
        "--recommendation",
        type=Path,
        default=None,
        help="Optional reviewed recommendation artifact.",
    )
    args = parser.parse_args(argv)
    report = run_chunking_experiment(
        configs_path=args.configs,
        anchors_path=args.anchors,
        local_dir=args.local_dir,
        mock_s3_dir=args.mock_s3_dir,
        recommendation_path=args.recommendation,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
