"""Strict metric-registry and error-analysis governance contracts."""

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

from agent_workshop_demo.eval_governance import (
    DEFAULT_METRIC_REGISTRY_PATH,
    load_error_analysis,
    load_metric_registry,
    main,
)


class EvalGovernanceTests(unittest.TestCase):
    def test_default_registry_has_all_roles_and_semantic_checksum(self) -> None:
        first = load_metric_registry()
        payload = json.loads(DEFAULT_METRIC_REGISTRY_PATH.read_text(encoding="utf-8"))
        payload["metrics"] = list(reversed(payload["metrics"]))

        with tempfile.TemporaryDirectory() as tmpdir:
            reordered_path = Path(tmpdir) / "registry.json"
            reordered_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=4),
                encoding="utf-8",
            )
            second = load_metric_registry(reordered_path)

        self.assertEqual(first.checksum, second.checksum)
        self.assertEqual(
            {metric.role for metric in first.active_metrics},
            {"goal", "guardrail", "operational"},
        )
        self.assertEqual(len(first.active_metrics), 10)

    def test_registry_rejects_unknown_measurement(self) -> None:
        payload = json.loads(DEFAULT_METRIC_REGISTRY_PATH.read_text(encoding="utf-8"))
        payload["metrics"][0]["measurement"] = "report.any_arbitrary_field"

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "registry.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "measurement is invalid"):
                load_metric_registry(path)

    def test_registry_rejects_retired_metric_without_reason(self) -> None:
        payload = json.loads(DEFAULT_METRIC_REGISTRY_PATH.read_text(encoding="utf-8"))
        payload["metrics"][0]["status"] = "retired"

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "registry.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires retirement_reason"):
                load_metric_registry(path)

    def test_registry_requires_explicit_retirement_condition(self) -> None:
        payload = json.loads(DEFAULT_METRIC_REGISTRY_PATH.read_text(encoding="utf-8"))
        del payload["metrics"][0]["retirement_condition"]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "registry.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fields are invalid"):
                load_metric_registry(path)

    def test_registry_rejects_unregistered_dataset_segment(self) -> None:
        payload = json.loads(DEFAULT_METRIC_REGISTRY_PATH.read_text(encoding="utf-8"))
        payload["metrics"][0]["dataset_segment"] = "typo_segment"

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "registry.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dataset_segment is invalid"):
                load_metric_registry(path)

    def test_bootstrap_error_analysis_validates_clusters_without_exposing_notes(
        self,
    ) -> None:
        payload = self._analysis_payload()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "analysis.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            artifact = load_error_analysis(path)

        self.assertEqual(artifact.case_count, 30)
        self.assertEqual(artifact.failed_case_count, 2)
        self.assertEqual(artifact.cluster_count, 1)
        self.assertNotIn("failure note", str(artifact))

    def test_bootstrap_error_analysis_requires_thirty_cases(self) -> None:
        payload = self._analysis_payload()
        payload["cases"] = payload["cases"][:20]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "analysis.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires 30 to 50 cases"):
                load_error_analysis(path)

    def test_error_clusters_must_cover_every_failed_trace(self) -> None:
        payload = self._analysis_payload()
        payload["clusters"][0]["trace_ids"] = ["trace_00"]
        payload["clusters"][0]["count"] = 1

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "analysis.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cover every and only"):
                load_error_analysis(path)

    def test_metric_candidate_must_be_a_generalization_failure(self) -> None:
        payload = self._analysis_payload()
        payload["clusters"][0]["generalization_failure"] = False

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "analysis.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be generalization failures"):
                load_error_analysis(path)

    def test_metric_candidate_must_link_to_registry(self) -> None:
        payload = self._analysis_payload()
        payload["clusters"][0]["metric_id"] = "goal.misspelled_metric"

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "analysis.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must link"):
                load_error_analysis(path)

    def test_module_cli_prints_only_registry_summary(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            result = main([])
        report = json.loads(output.getvalue())

        self.assertEqual(result, 0)
        self.assertEqual(
            report["metric_registry"]["active_counts"],
            {"goal": 3, "guardrail": 4, "operational": 3},
        )
        self.assertNotIn("metrics", report["metric_registry"])

    @staticmethod
    def _analysis_payload() -> dict[str, Any]:
        cases = [
            {
                "trace_id": f"trace_{index:02d}",
                "case_id": f"case_{index:02d}",
                "overall_pass": index >= 2,
                "review_note": (
                    "failure note" if index < 2 else "No observed failure."
                ),
            }
            for index in range(30)
        ]
        return {
            "artifact_version": "eval-error-analysis-v1",
            "analysis_id": "bootstrap_2026_08_21",
            "analysis_kind": "bootstrap",
            "change_reference": "metric-portfolio-v3",
            "sampled_at": "2026-08-21",
            "sampling_strata": ["retrieval", "abstention"],
            "reviewer": "workshop-author",
            "cases": cases,
            "clusters": [
                {
                    "category_id": "missing_required_fact",
                    "name": "Required fact missing",
                    "trace_ids": ["trace_00", "trace_01"],
                    "count": 2,
                    "severity": "P1",
                    "generalization_failure": True,
                    "disposition": "metric_candidate",
                    "metric_id": "goal.required_fact_coverage",
                }
            ],
        }


if __name__ == "__main__":
    unittest.main()
