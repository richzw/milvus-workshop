"""Independent tests for executable selective-Memory eval metrics."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_workshop_demo.selective_memory_eval import evaluate_selective_memory


class SelectiveMemoryEvalTests(unittest.TestCase):
    def test_runner_executes_service_and_reports_bounded_provenance(self) -> None:
        with patch.dict("os.environ", {"EMBEDDING_PROVIDER": "deterministic"}):
            report = evaluate_selective_memory(
                Path("demo/eval/selective_memory_cases.json")
            )
        self.assertEqual(report["case_count"], 6)
        self.assertEqual(report["selection_precision"], 1.0)
        self.assertEqual(report["selection_recall"], 1.0)
        self.assertEqual(report["provenance"]["runner"], "LocalSelectiveMemoryStore")
        self.assertEqual(report["correction_accuracy"], 0.0)
        self.assertEqual(report["conflict_detection_accuracy"], 0.8333)
        self.assertEqual(report["stale_memory_intrusion_rate"], 0.0)
        self.assertEqual(report["memory_pack_size_violation_rate"], 0.0)
        self.assertEqual(report["consolidation_exact_once_accuracy"], 1.0)
        self.assertIn("user_preference", report["selection_by_event_class"])
        self.assertNotIn("query", str(report))

    def test_unbounded_cases_and_freeform_case_ids_fail_closed(self) -> None:
        fixture = {
            "schema_version": "selective-memory-eval-v2",
            "cases": [
                {
                    "case_id": "user secret must not leak",
                    "scenario": "ordinary_turn",
                    "expected_retain": False,
                    "expected_active_fact": False,
                    "expected_correction": False,
                    "expected_conflict": False,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(fixture))
            with self.assertRaisesRegex(ValueError, "case_id"):
                evaluate_selective_memory(path)
            fixture["cases"] = fixture["cases"] * 101
            path.write_text(json.dumps(fixture))
            with self.assertRaisesRegex(ValueError, "fixture"):
                evaluate_selective_memory(path)


if __name__ == "__main__":
    unittest.main()
