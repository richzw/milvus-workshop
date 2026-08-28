from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_workshop_demo.ingestion import IngestionResult, ingest_demo_sources
from agent_workshop_demo.struct_array import (
    ProjectionBuild,
    build_struct_array_projection,
    load_projection_manifest,
)
from agent_workshop_demo.struct_array_eval import (
    evaluate_struct_array_profiles,
    validate_struct_array_eval_report,
)


class StructArrayEvalTests(unittest.TestCase):
    ingestion: IngestionResult
    projection: ProjectionBuild

    @classmethod
    def setUpClass(cls) -> None:
        cls.ingestion = ingest_demo_sources(
            Path("demo/sample_data/local_docs"),
            Path("demo/sample_data/mock_s3"),
        )
        cls.projection = build_struct_array_projection(
            cls.ingestion.kb_chunks,
            load_projection_manifest(),
        )

    def test_report_compares_all_profiles_without_parent_evidence(self) -> None:
        report = evaluate_struct_array_profiles(
            cases_path=Path("demo/eval/struct_array_cases.json"),
            chunks=self.ingestion.kb_chunks,
            projection=self.projection,
            top_k=20,
        )
        self.assertEqual(report["schema_version"], "struct-array-eval-v1")
        self.assertEqual(
            report["configured_profiles"],
            ["flat_hybrid", "struct_element", "struct_two_stage", "struct_fused"],
        )
        self.assertTrue(
            all(item["parent_only_evidence_count"] == 0 for item in report["profiles"])
        )
        self.assertEqual(
            report["build_storage_observations"]["native_observation_status"],
            "evaluation_incomplete",
        )
        self.assertEqual(
            report["end_to_end_quality"]["status"], "evaluation_incomplete"
        )
        self.assertEqual(report["profiles"][-1]["fusion_recipe"], "struct-rrf-v1")
        self.assertTrue(
            all(
                "selected_context_recall_at_5" in case
                and "same_element_predicate_failures" in case
                for profile in report["profiles"]
                for case in profile["cases"]
            )
        )

        broken = dict(report)
        broken.pop("end_to_end_quality")
        with self.assertRaisesRegex(ValueError, "shape"):
            validate_struct_array_eval_report(broken)

    def test_fixture_rejects_duplicate_ids_and_unknown_fields(self) -> None:
        fixture = {
            "schema_version": "struct-array-eval-cases-v1",
            "cases": [
                {
                    "case_id": "duplicate",
                    "queries": ["q"],
                    "filters": {},
                    "expected_chunk_ids": ["c"],
                    "unknown": True,
                },
                {
                    "case_id": "duplicate",
                    "queries": ["q"],
                    "filters": {},
                    "expected_chunk_ids": ["c"],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(json.dumps(fixture), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown fields"):
                evaluate_struct_array_profiles(
                    cases_path=path,
                    chunks=self.ingestion.kb_chunks,
                    projection=self.projection,
                )


if __name__ == "__main__":
    unittest.main()
