"""Independent tests for selective-Memory distributions and lineage UI."""

from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

from agent_workshop_demo.selective_memory import (
    LocalSelectiveMemoryStore,
    MemorySelection,
    SelectiveMemoryService,
)
from agent_workshop_demo.streamlit_app import build_selective_memory_dashboard


class MemoryDashboardTests(unittest.TestCase):
    def test_dashboard_counts_distributions_and_resolves_opaque_lineage(self) -> None:
        records = [
            {
                "kind": "fact",
                "id": "fact_2",
                "status": "active",
                "revision": 2,
                "source_event_ids": ["event_1", "event_missing"],
                "supersedes_memory_id": "fact_1",
                "decay_profile": "durable_gentle",
                "sensitive_value": "never render me",
            },
            {
                "kind": "fact",
                "id": "fact_1",
                "status": "superseded",
                "revision": 1,
                "source_event_ids": ["event_1"],
                "supersedes_memory_id": None,
                "decay_profile": "durable_gentle",
            },
            {
                "kind": "episode",
                "id": "event_1",
                "status": "protected",
                "selection_reasons": ["explicit_remember", "explicit_remember"],
                "decay_profile": "no_time_decay",
                "parent_event_id": None,
                "branch_id": "main",
                "selector_name": "rule_based",
            },
        ]

        dashboard = build_selective_memory_dashboard(records)

        self.assertEqual(
            dashboard["distributions"]["retention_class"],
            {"protected": 1},
        )
        self.assertEqual(
            dashboard["distributions"]["selection_reason"],
            {"explicit_remember": 2},
        )
        self.assertEqual(
            dashboard["distributions"]["fact_status"],
            {"active": 1, "superseded": 1},
        )
        self.assertEqual(len(dashboard["lineage"]), 5)
        missing = next(
            edge
            for edge in dashboard["lineage"]
            if edge["to_id"] == "event_missing"
        )
        self.assertFalse(missing["resolved"])
        event_node = next(
            edge for edge in dashboard["lineage"] if edge["relation"] == "event_node"
        )
        self.assertEqual(event_node["branch_id"], "main")
        self.assertEqual(event_node["selector_name"], "rule_based")
        self.assertNotIn("never render me", str(dashboard))

    def test_dashboard_caps_edges_and_selection_reasons_are_registered(self) -> None:
        dashboard = build_selective_memory_dashboard(
            [
                {
                    "kind": "fact",
                    "id": "fact_many",
                    "status": "active",
                    "revision": 1,
                    "source_event_ids": [f"event_{index}" for index in range(700)],
                    "decay_profile": "durable_gentle",
                }
            ]
        )
        self.assertEqual(len(dashboard["lineage"]), 500)
        self.assertEqual(dashboard["lineage_truncated_count"], 200)

        with self.assertRaisesRegex(ValueError, "registered reason"):
            MemorySelection(
                event_type="user_statement",
                salience_score=0.4,
                selection_reason=("user secret rationale",),
                retention_class="ephemeral",
                decay_profile="episode_fast",
            )

    def test_service_rows_expose_lineage_metadata_without_payload(self) -> None:
        with patch.dict("os.environ", {"EMBEDDING_PROVIDER": "deterministic"}):
            service = SelectiveMemoryService()
            service.persist_turn(
                session_id="session_a",
                query_id="query_1",
                query="请记住以后用中文",
                answer="sensitive answer",
                terminal_status="memory_saved",
                remembered_statement="以后用中文",
                now_ms=1_000,
            )

        rows = service.list_session("session_a", now_ms=1_000)
        fact = next(row for row in rows if row["kind"] == "fact")
        episode = next(row for row in rows if row["kind"] == "episode")

        self.assertEqual(fact["revision"], 1)
        self.assertTrue(fact["source_event_ids"])
        self.assertIn("selection_reasons", episode)
        self.assertIn("selector_name", episode)
        self.assertNotIn("中文", str(rows))
        self.assertNotIn("sensitive answer", str(rows))

    def test_tombstoned_facts_have_zero_ui_visibility(self) -> None:
        with patch.dict("os.environ", {"EMBEDDING_PROVIDER": "deterministic"}):
            store = LocalSelectiveMemoryStore()
            service = SelectiveMemoryService(store)
            service.persist_turn(
                session_id="session_a",
                query_id="query_1",
                query="请记住以后用中文",
                answer="saved",
                terminal_status="memory_saved",
                remembered_statement="以后用中文",
                now_ms=1_000,
            )
            active = next(iter(store.facts.values()))
            store.upsert_facts((replace(active, status="tombstoned"),))

        rows = service.list_session("session_a", now_ms=1_000)
        self.assertFalse(any(row["kind"] == "fact" for row in rows))


if __name__ == "__main__":
    unittest.main()
