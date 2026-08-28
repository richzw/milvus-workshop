"""Run the isolated offline StructArray profile comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent_workshop_demo.ingestion import ingest_demo_sources
from agent_workshop_demo.struct_array import (
    MilvusStructArrayRetriever,
    StructArrayProfile,
    build_struct_array_projection,
    load_projection_manifest,
    runtime_config_from_mapping,
)
from agent_workshop_demo.schema.pymilvus_adapter import MilvusHybridRetriever
from agent_workshop_demo.struct_array_eval import evaluate_struct_array_profiles


def main(argv: list[str] | None = None) -> int:
    """Build the local projection and print one strict comparison artifact."""

    parser = argparse.ArgumentParser(description="Run StructArray profile evaluation.")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("demo/eval/struct_array_cases.json"),
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
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--uri")
    parser.add_argument("--token")
    parser.add_argument("--projection-fingerprint")
    parser.add_argument(
        "--hardware-note",
        default="local deterministic workshop runtime",
    )
    args = parser.parse_args(argv)
    ingestion = ingest_demo_sources(args.local_dir, args.mock_s3_dir)
    projection = build_struct_array_projection(
        ingestion.kb_chunks,
        load_projection_manifest(),
    )
    native_args = (args.uri, args.projection_fingerprint)
    if any(native_args) and not all(native_args):
        parser.error("--uri and --projection-fingerprint must be provided together")
    profile_retrievers: dict[str, Any] | None = None
    execution_mode = "local_emulation"
    if all(native_args):
        if args.projection_fingerprint != projection.projection_fingerprint:
            parser.error("--projection-fingerprint differs from the fixed eval corpus")
        flat = MilvusHybridRetriever.connect(args.uri, args.token)
        profile_retrievers = {"flat_hybrid": flat}
        for profile in (
            StructArrayProfile.ELEMENT,
            StructArrayProfile.TWO_STAGE,
            StructArrayProfile.FUSED,
        ):
            config = runtime_config_from_mapping(
                {
                    "STRUCT_ARRAY_RETRIEVAL": profile.value,
                    "STRUCT_ARRAY_PROJECTION_FINGERPRINT": args.projection_fingerprint,
                }
            )
            retriever = MilvusStructArrayRetriever(flat.client, flat, config)
            retriever.ensure_ready()
            profile_retrievers[profile.value] = retriever
        execution_mode = "native_read_only"
    report = evaluate_struct_array_profiles(
        cases_path=args.cases,
        chunks=ingestion.kb_chunks,
        projection=projection,
        top_k=args.top_k,
        hardware_note=args.hardware_note,
        profile_retrievers=profile_retrievers,
        execution_mode=execution_mode,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
