"""Drop the fixed set of demo Milvus collections before a schema rebuild."""

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

from agent_workshop_demo.schema.pymilvus_adapter import (
    DEMO_COLLECTION_NAMES,
    drop_demo_collections,
)


def main(argv: list[str] | None = None) -> int:
    """Preview or perform deletion of repository-owned demo collections."""

    parser = argparse.ArgumentParser(
        description=(
            "Drop demo Milvus collections and all records they contain. "
            "Other collections are never targeted."
        )
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
        "--confirm-drop-demo-data",
        action="store_true",
        help=(
            "Actually drop the three demo collections. Without this flag, "
            "the command is a connection-free preview."
        ),
    )
    args = parser.parse_args(argv)

    if not args.confirm_drop_demo_data:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "targeted": list(DEMO_COLLECTION_NAMES),
                    "next_step": (
                        "Re-run with --confirm-drop-demo-data to delete "
                        "these collections and all contained records."
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    report = drop_demo_collections(args.uri, args.token)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
