"""Create and verify indexes on the demo Milvus collections."""

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

from agent_workshop_demo.schema.collections import (
    CONVERSATION_MEMORY_INDEXES,
    DOC_DEDUP_SIGNATURES_INDEXES,
    KB_CHUNKS_INDEXES,
)
from agent_workshop_demo.schema.pymilvus_adapter import create_indexes


def main(argv: list[str] | None = None) -> int:
    """Create indexes, or print their definitions in dry-run mode."""

    parser = argparse.ArgumentParser(
        description="Create and verify demo Milvus indexes."
    )
    parser.add_argument(
        "--uri",
        default=os.getenv("MILVUS_URI", "http://localhost:19530"),
        help="Milvus URI (default: MILVUS_URI or http://localhost:19530).",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("MILVUS_TOKEN"),
        help="Milvus token (default: MILVUS_TOKEN).",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and rebuild existing named indexes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print index definitions without connecting to Milvus.",
    )
    args = parser.parse_args(argv)
    definitions = {
        "kb_chunks": KB_CHUNKS_INDEXES,
        "conversation_memory": CONVERSATION_MEMORY_INDEXES,
        "doc_dedup_signatures": DOC_DEDUP_SIGNATURES_INDEXES,
    }
    if not args.dry_run:
        print(
            json.dumps(
                create_indexes(
                    args.uri,
                    args.token,
                    recreate=args.recreate,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    print(
        json.dumps(
            {"dry_run": True, "indexes": definitions},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
