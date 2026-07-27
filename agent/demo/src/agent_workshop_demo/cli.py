"""Command-line entrypoint for the local Agent Chat demo."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from agent_workshop_demo.langgraph_workflow import build_default_workflow


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Run the Agentic RAG workshop demo."
    )
    parser.add_argument(
        "question",
        nargs="?",
        default="我们 S3 文档同步流程是怎么设计的？",
        help="Question to ask the local knowledge base.",
    )
    parser.add_argument(
        "--department",
        help="Optional department filter, e.g. engineering.",
    )
    parser.add_argument(
        "--source-type",
        choices=["local", "s3", "mfs"],
        help="Optional source filter.",
    )
    parser.add_argument(
        "--session-id",
        default="session_cli",
        help="Stable session identifier for this request.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full response as JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one CLI query."""

    args = build_parser().parse_args(argv)
    filters: dict[str, str | list[str]] = {
        "source_type": ["local", "s3"],
        "doc_type": ["markdown", "pdf", "text", "image"],
    }
    if args.department:
        filters["department"] = args.department
    if args.source_type:
        filters["source_type"] = [args.source_type]

    try:
        response = build_default_workflow().run(
            args.question,
            filters=filters,
            session_id=args.session_id,
        )
    except ValueError as exc:
        raise SystemExit(f"Invalid request: {exc}") from exc

    if args.json:
        print(json.dumps(response, ensure_ascii=False, indent=2))
        return 0

    print(response["answer"])
    if response["citations"]:
        print("\nCitations:")
        for citation in response["citations"]:
            page = (
                f", page {citation['page_no']}"
                if citation["page_no"] is not None
                else ""
            )
            print(
                f"- [{citation['citation_id']}] {citation['title']} "
                f"({citation['chunk_id']}, "
                f"version {citation['doc_version']}{page})"
            )
    print("\nTrace:")
    print(json.dumps(response["trace"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
