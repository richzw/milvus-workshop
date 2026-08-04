"""Create and verify the demo Milvus collections."""

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
    CONVERSATION_MEMORY_COLLECTION,
    DOC_DEDUP_SIGNATURES_COLLECTION,
    GROUNDED_RESPONSE_CACHE_COLLECTION,
    KB_CHUNKS_COLLECTION,
    MEMORY_EVENTS_COLLECTION,
    MEMORY_FACTS_COLLECTION,
    MEMORY_CONSOLIDATION_JOURNAL_COLLECTION,
)
from agent_workshop_demo.schema.pymilvus_adapter import create_collections


def main(argv: list[str] | None = None) -> int:
    """Create collections, or print their schemas in dry-run mode."""

    parser = argparse.ArgumentParser(
        description="Create or inspect demo Milvus collections."
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
        "--drop-existing",
        action="store_true",
        help="Drop existing demo collections first.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print schemas without connecting to Milvus.",
    )
    args = parser.parse_args(argv)
    schemas = [
        KB_CHUNKS_COLLECTION,
        CONVERSATION_MEMORY_COLLECTION,
        MEMORY_EVENTS_COLLECTION,
        MEMORY_FACTS_COLLECTION,
        MEMORY_CONSOLIDATION_JOURNAL_COLLECTION,
        GROUNDED_RESPONSE_CACHE_COLLECTION,
        DOC_DEDUP_SIGNATURES_COLLECTION,
    ]
    if args.dry_run:
        print(json.dumps({"dry_run": True, "schemas": schemas}, indent=2))
        return 0
    report = create_collections(
        args.uri,
        args.token,
        drop_existing=args.drop_existing,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
