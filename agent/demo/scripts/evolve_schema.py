"""Plan or apply allow-listed Milvus 3.0 schema evolution."""

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

from agent_workshop_demo.schema.evolution import MilvusSchemaEvolution
from agent_workshop_demo.schema.pymilvus_adapter import _connect_milvus_client


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Dry-run-first additive kb_chunks schema evolution."
    )
    parser.add_argument("operation", choices=("add-retrieval-text", "backfill-retrieval-text", "add-sparse", "add-embedding", "add-bm25", "backfill-embedding"))
    parser.add_argument("--field-name")
    parser.add_argument("--dim", type=int)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--uri", default=os.getenv("MILVUS_URI", "http://localhost:19530"))
    parser.add_argument("--token", default=os.getenv("MILVUS_TOKEN"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    if args.operation not in {"add-bm25", "add-retrieval-text", "backfill-retrieval-text"} and not args.field_name:
        parser.error("--field-name is required for this operation")
    if args.operation in {"add-embedding", "backfill-embedding"} and args.dim is None:
        parser.error("--dim is required for embedding operations")
    if args.operation in {"backfill-embedding", "backfill-retrieval-text"} and args.records is None:
        parser.error("--records is required for backfill operations")
    try:
        report = _execute(args)
    except (ValueError, json.JSONDecodeError):
        report = {
            "status": "failed",
            "operation": args.operation,
            "error_code": "invalid_migration_input",
        }
        print(json.dumps(report, ensure_ascii=False), file=sys.stderr)
        return 1
    except OSError:
        report = {
            "status": "failed",
            "operation": args.operation,
            "error_code": "migration_input_unavailable",
        }
        print(json.dumps(report, ensure_ascii=False), file=sys.stderr)
        return 1
    except Exception:
        report = {
            "status": "failed",
            "operation": args.operation,
            "error_code": "milvus_migration_failed",
        }
        print(json.dumps(report, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _execute(args: argparse.Namespace) -> dict[str, object]:
    """Execute one validated operation; the CLI boundary sanitizes failures."""

    client = _connect_milvus_client(args.uri, args.token)
    evolution = MilvusSchemaEvolution(client)
    if args.operation == "add-retrieval-text":
        report = evolution.add_retrieval_text(apply=args.apply)
    elif args.operation == "backfill-retrieval-text":
        rows = [json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines() if line.strip()]
        report = evolution.backfill_retrieval_text(
            rows,
            batch_size=args.batch_size,
            apply=args.apply,
        )
    elif args.operation == "add-sparse":
        report = evolution.add_vector_field(args.field_name, kind="sparse", apply=args.apply)
    elif args.operation == "add-embedding":
        report = evolution.add_vector_field(args.field_name, kind="embedding", dim=args.dim, apply=args.apply)
    elif args.operation == "add-bm25":
        report = evolution.add_bm25_function(
            output_field_name=args.field_name or "sparse_vector_v2",
            apply=args.apply,
        )
    else:
        rows = [json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines() if line.strip()]
        report = evolution.backfill_embedding(
            args.field_name,
            rows,
            dim=args.dim,
            batch_size=args.batch_size,
            apply=args.apply,
        )
    return report


if __name__ == "__main__":
    raise SystemExit(main())
