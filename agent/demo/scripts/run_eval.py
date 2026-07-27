"""Run the deterministic golden-question evaluation."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from agent_workshop_demo.eval_runner import evaluate_questions


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
    args = parser.parse_args(argv)
    report = evaluate_questions(
        questions_path=args.questions,
        golden_answers_path=args.golden_answers,
        top_k=args.top_k,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
