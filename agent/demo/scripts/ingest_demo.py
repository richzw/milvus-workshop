"""Ingest the workshop fixtures into Milvus and verify the records."""

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

from agent_workshop_demo.embedding import text_embedding_fingerprint
from agent_workshop_demo.ingestion import (
    ingest_demo_sources,
    write_ingestion_result,
)
from agent_workshop_demo.schema.pymilvus_adapter import MilvusHybridRetriever


def main(argv: list[str] | None = None) -> int:
    """Build fixture chunks, insert them into Milvus, and read them back."""

    parser = argparse.ArgumentParser(description="Run demo ingestion.")
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
        "--version-manifest",
        type=Path,
        default=Path("demo/sample_data/document_versions.json"),
        help="Document-version manifest for stable doc families and current editions.",
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
        "--collection-name",
        default="kb_chunks",
        help="Target collection (default: kb_chunks).",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optionally also write the generated JSONL records here.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build records without connecting to Milvus.",
    )
    args = parser.parse_args(argv)
    result = ingest_demo_sources(
        args.local_dir,
        args.mock_s3_dir,
        version_manifest_path=args.version_manifest,
    )
    paths = (
        write_ingestion_result(result, args.output_dir)
        if args.output_dir
        else {}
    )
    report: dict[str, object] = {
        "source_documents": len(
            {chunk.doc_id for chunk in result.kb_chunks}
        ),
        "document_editions": len(
            {
                (chunk.doc_id, chunk.doc_version)
                for chunk in result.kb_chunks
            }
        ),
        "generated_chunks": len(result.kb_chunks),
        "embedded_chunks": len(result.kb_chunks),
        "text_embedding_fingerprint": text_embedding_fingerprint(),
        "jsonl": {key: str(value) for key, value in paths.items()},
    }
    if args.dry_run:
        report["dry_run"] = True
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    adapter = MilvusHybridRetriever.connect(
        args.uri,
        args.token,
        collection_name=args.collection_name,
        batch_size=args.batch_size,
    )
    insert_report = adapter.insert(result.kb_chunks)
    verified = adapter.verify_inserted(
        chunk.chunk_id for chunk in result.kb_chunks
    )
    report.update(
        {
            "collection": args.collection_name,
            **insert_report,
            "verified_count": verified,
        }
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
