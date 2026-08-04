"""Independent recovery tests for the selective-Memory consolidation outbox."""

from __future__ import annotations

import threading
import struct
import unittest
from dataclasses import replace
from typing import Any
from unittest.mock import patch

from agent_workshop_demo.selective_memory import (
    LocalSelectiveMemoryStore,
    MilvusSelectiveMemoryStore,
    SelectiveMemoryError,
    SelectiveMemoryService,
)


class FailingConsolidationStore(LocalSelectiveMemoryStore):
    """Inject one fact or lifecycle failure after a journal enqueue."""

    def __init__(
        self,
        *,
        fail_fact: bool = False,
        fact_failures: int = 0,
        fail_event: bool = False,
    ) -> None:
        super().__init__()
        self.fact_failures = fact_failures + int(fail_fact)
        self.fail_event = fail_event

    def upsert_facts(self, facts: Any) -> int:
        if self.fact_failures:
            self.fact_failures -= 1
            raise SelectiveMemoryError("injected fact failure")
        return super().upsert_facts(facts)

    def append_events(self, events: Any) -> int:
        if self.fail_event and any(
            event.event_type in {"memory_promoted", "memory_reconfirmed"}
            for event in events
        ):
            self.fail_event = False
            raise SelectiveMemoryError("injected event failure")
        return super().append_events(events)


class BlockingLifecycleStore(LocalSelectiveMemoryStore):
    """Pause one projection write to coordinate erasure concurrency."""

    def __init__(self) -> None:
        super().__init__()
        self.lifecycle_entered = threading.Event()
        self.lifecycle_release = threading.Event()

    def append_events(self, events: Any) -> int:
        if any(
            event.event_type in {"memory_promoted", "memory_reconfirmed"}
            for event in events
        ):
            self.lifecycle_entered.set()
            if not self.lifecycle_release.wait(timeout=5):
                raise AssertionError("test did not release lifecycle write")
        return super().append_events(events)


def _persist_preference(
    service: SelectiveMemoryService,
    *,
    query_id: str = "query_1",
    now_ms: int = 1_000,
) -> Any:
    return service.persist_turn(
        session_id="session_a",
        query_id=query_id,
        query="请记住以后用中文",
        answer="saved",
        terminal_status="memory_saved",
        remembered_statement="以后用中文",
        now_ms=now_ms,
    )


class JournalMilvusClient:
    """Minimal scalar client for journal adapter parity."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.query_calls: list[dict[str, Any]] = []
        self.delete_calls: list[str] = []
        self.fail_journal_delete = False

    def query(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.query_calls.append(dict(kwargs))
        expression = str(kwargs["filter"])
        if 'status == "pending"' in expression:
            rows = [
                dict(row)
                for row in self.rows.values()
                if row["session_id"] in expression and row["status"] == "pending"
            ]
            rows.sort(key=lambda row: (row["created_at"], row["operation_id"]))
            return rows[: int(kwargs["limit"])]
        return [
            dict(row) for identity, row in self.rows.items() if identity in expression
        ]

    def insert(self, **kwargs: Any) -> dict[str, int]:
        row = dict(kwargs["data"][0])
        self.rows[str(row["operation_id"])] = row
        return {"insert_count": 1}

    def upsert(self, **kwargs: Any) -> dict[str, int]:
        row = dict(kwargs["data"][0])
        self.rows[str(row["operation_id"])] = row
        return {"upsert_count": 1}

    def flush(self, **kwargs: Any) -> None:
        del kwargs

    def delete(self, **kwargs: Any) -> dict[str, int]:
        collection = str(kwargs["collection_name"])
        self.delete_calls.append(collection)
        if collection == "memory_consolidation_journal" and self.fail_journal_delete:
            raise RuntimeError("injected journal delete failure")
        return {"delete_count": 0}

    def query_iterator(self, **kwargs: Any) -> Any:
        expression = str(kwargs["filter"])
        rows = [
            dict(row)
            for row in self.rows.values()
            if row["session_id"] in expression and row["status"] == "pending"
        ]
        batches = [rows, []]

        class Iterator:
            def next(self) -> list[dict[str, Any]]:
                return batches.pop(0)

            def close(self) -> None:
                return None

        return Iterator()


class PostWriteEventClient:
    """Persist an event, expose float32 readback, then fail the first flush."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.fail_flush_once = True

    def query(self, **kwargs: Any) -> list[dict[str, Any]]:
        expression = str(kwargs["filter"])
        return [
            dict(row)
            for event_id, row in self.rows.items()
            if event_id in expression
        ]

    def insert(self, **kwargs: Any) -> dict[str, int]:
        for raw in kwargs["data"]:
            row = dict(raw)
            vector = row["content_vector"]
            packed = struct.pack(f"<{len(vector)}f", *vector)
            row["content_vector"] = list(
                struct.unpack(f"<{len(vector)}f", packed)
            )
            self.rows[str(row["event_id"])] = row
        return {"insert_count": len(kwargs["data"])}

    def flush(self, **kwargs: Any) -> None:
        del kwargs
        if self.fail_flush_once:
            self.fail_flush_once = False
            raise RuntimeError("injected post-write flush failure")


class ConsolidationOutboxTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = patch.dict(
            "os.environ",
            {"EMBEDDING_PROVIDER": "deterministic"},
        )
        self.environment.start()

    def tearDown(self) -> None:
        self.environment.stop()

    def test_fact_failure_stays_pending_and_exact_replay_applies_once(self) -> None:
        store = FailingConsolidationStore(fail_fact=True)
        service = SelectiveMemoryService(store)

        result = _persist_preference(service)
        pending = store.list_pending_consolidations("session_a", limit=20)

        self.assertEqual(result.consolidation_status, "pending_retry")
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].attempts, 1)
        self.assertEqual(pending[0].last_error_code, "fact_write_failed")
        self.assertEqual(store.facts, {})

        self.assertEqual(
            service.drain_consolidation_outbox(
                "session_a",
                now_ms=2_000,
            ),
            1,
        )
        self.assertEqual(store.list_pending_consolidations("session_a", limit=20), [])
        self.assertEqual(len(store.facts), 1)
        self.assertEqual(
            len(
                [
                    event
                    for event in store.events.values()
                    if event.event_type == "memory_promoted"
                ]
            ),
            1,
        )
        self.assertEqual(
            service.drain_consolidation_outbox("session_a", now_ms=3_000),
            0,
        )

    def test_event_failure_replays_precomputed_revision_without_duplicates(
        self,
    ) -> None:
        store = FailingConsolidationStore(fail_event=True)
        service = SelectiveMemoryService(store)

        result = _persist_preference(service)
        fact_before = next(iter(store.facts.values()))

        self.assertEqual(result.consolidation_status, "pending_retry")
        self.assertEqual(fact_before.revision, 1)
        pending = store.list_pending_consolidations("session_a", limit=20)
        self.assertEqual(pending[0].last_error_code, "event_write_failed")

        service.drain_consolidation_outbox("session_a", now_ms=2_000)

        self.assertEqual(list(store.facts.values()), [fact_before])
        lifecycle = [
            event
            for event in store.events.values()
            if event.event_type == "memory_promoted"
        ]
        self.assertEqual(len(lifecycle), 1)
        self.assertEqual(lifecycle[0], pending[0].plan.lifecycle_event)

    def test_persistent_failure_fences_new_plan_then_rejoins_lineage(self) -> None:
        store = FailingConsolidationStore(fact_failures=2)
        service = SelectiveMemoryService(store)

        first = _persist_preference(service, query_id="query_1", now_ms=1_000)
        second = _persist_preference(service, query_id="query_2", now_ms=2_000)

        self.assertEqual(first.consolidation_status, "pending_retry")
        self.assertEqual(second.consolidation_status, "deferred_pending")
        self.assertEqual(
            len(store.list_pending_consolidations("session_a", limit=20)),
            1,
        )

        self.assertEqual(
            service.drain_consolidation_outbox("session_a", now_ms=3_000),
            1,
        )
        third = _persist_preference(service, query_id="query_3", now_ms=4_000)

        self.assertEqual(third.consolidation_status, "completed")
        active = store.list_facts("session_a", now_ms=5_000)
        self.assertEqual(active[0].revision, 2)
        self.assertEqual(len(active[0].source_event_ids), 3)
        self.assertEqual(
            store.list_pending_consolidations("session_a", limit=20),
            [],
        )

    def test_operation_collision_and_cross_session_listing_fail_closed(self) -> None:
        store = FailingConsolidationStore(fail_fact=True)
        service = SelectiveMemoryService(store)
        _persist_preference(service)
        entry = store.list_pending_consolidations("session_a", limit=20)[0]

        self.assertEqual(store.enqueue_consolidation(entry.plan), 0)
        changed_plan = replace(entry.plan, created_at=entry.plan.created_at + 1)
        with self.assertRaisesRegex(SelectiveMemoryError, "identity collision"):
            store.enqueue_consolidation(changed_plan)
        self.assertEqual(store.list_pending_consolidations("session_b", limit=20), [])
        store.delete_session("session_a")
        self.assertEqual(store.consolidation_journal, {})

    def test_milvus_journal_round_trips_exact_payload_and_attempt_state(self) -> None:
        local = FailingConsolidationStore(fail_fact=True)
        _persist_preference(SelectiveMemoryService(local))
        plan = local.list_pending_consolidations("session_a", limit=20)[0].plan
        client = JournalMilvusClient()
        store = MilvusSelectiveMemoryStore(client)

        self.assertEqual(store.enqueue_consolidation(plan), 1)
        self.assertEqual(store.enqueue_consolidation(plan), 0)
        encoded_payload = client.rows[plan.operation_id]["fact_update_0"]
        self.assertEqual(encoded_payload["codec"], "zlib-json-v1")
        self.assertLessEqual(len(encoded_payload["data"]), 40_000)
        pending = store.list_pending_consolidations("session_a", limit=20)
        self.assertEqual(pending[0].plan, plan)
        self.assertEqual(client.query_calls[-1]["limit"], 20)
        self.assertEqual(
            client.query_calls[-1]["order_by"],
            ["created_at:asc", "operation_id:asc"],
        )
        self.assertEqual(
            client.rows[plan.operation_id]["journal_anchor_vector"],
            [1.0, 0.0],
        )

        store.record_consolidation_attempt(
            plan.operation_id,
            session_id="session_a",
            now_ms=2_000,
            applied=False,
            error_code="event_write_failed",
        )
        pending = store.list_pending_consolidations("session_a", limit=20)
        self.assertEqual(pending[0].attempts, 1)
        self.assertEqual(pending[0].last_error_code, "event_write_failed")

        store.record_consolidation_attempt(
            plan.operation_id,
            session_id="session_a",
            now_ms=3_000,
            applied=True,
            error_code=None,
        )
        self.assertEqual(store.list_pending_consolidations("session_a", limit=20), [])

        client.rows[plan.operation_id]["fact_update_0"] = {
            "codec": "unknown",
            "data": "",
        }
        with self.assertRaisesRegex(
            SelectiveMemoryError,
            "Invalid stored Memory consolidation",
        ):
            store._journal_entry(plan.operation_id)

    def test_milvus_event_replay_accepts_float32_post_write_readback(self) -> None:
        local = FailingConsolidationStore(fail_fact=True)
        _persist_preference(SelectiveMemoryService(local))
        plan = local.list_pending_consolidations("session_a", limit=20)[0].plan
        client = PostWriteEventClient()
        store = MilvusSelectiveMemoryStore(client)

        with self.assertRaisesRegex(
            SelectiveMemoryError,
            "Unable to append selective-Memory events",
        ):
            store.append_events((plan.lifecycle_event,))

        self.assertEqual(store.append_events((plan.lifecycle_event,)), 0)
        self.assertEqual(len(client.rows), 1)
        self.assertEqual(
            tuple(client.rows[plan.lifecycle_event.event_id]["content_vector"]),
            plan.lifecycle_event.content_vector,
        )

    def test_milvus_erase_stops_before_projection_if_journal_delete_fails(
        self,
    ) -> None:
        client = JournalMilvusClient()
        client.fail_journal_delete = True
        store = MilvusSelectiveMemoryStore(client)

        with self.assertRaisesRegex(
            SelectiveMemoryError,
            "Unable to delete selective session Memory",
        ):
            store.delete_session("session_a")

        self.assertEqual(
            client.delete_calls,
            ["memory_consolidation_journal"],
        )

    def test_session_lock_serializes_inflight_projection_before_erasure(
        self,
    ) -> None:
        store = BlockingLifecycleStore()
        service = SelectiveMemoryService(store)
        write_finished = threading.Event()
        erase_finished = threading.Event()

        def persist() -> None:
            _persist_preference(service)
            write_finished.set()

        def erase() -> None:
            service.delete_session("session_a")
            erase_finished.set()

        writer = threading.Thread(target=persist)
        writer.start()
        self.assertTrue(store.lifecycle_entered.wait(timeout=2))
        eraser = threading.Thread(target=erase)
        eraser.start()
        self.assertFalse(erase_finished.wait(timeout=0.05))

        store.lifecycle_release.set()
        writer.join(timeout=2)
        eraser.join(timeout=2)

        self.assertTrue(write_finished.is_set())
        self.assertTrue(erase_finished.is_set())
        self.assertEqual(store.events, {})
        self.assertEqual(store.facts, {})
        self.assertEqual(store.consolidation_journal, {})

    def test_milvus_pending_backlog_pushes_oldest_order_and_limit_down(self) -> None:
        client = JournalMilvusClient()
        store = MilvusSelectiveMemoryStore(client)
        plans = []
        for index in range(5):
            local = FailingConsolidationStore(fail_fact=True)
            _persist_preference(
                SelectiveMemoryService(local),
                query_id=f"query_{index}",
                now_ms=(index + 1) * 1_000,
            )
            plan = local.list_pending_consolidations("session_a", limit=20)[0].plan
            plans.append(plan)
            store.enqueue_consolidation(plan)

        pending = store.list_pending_consolidations("session_a", limit=2)

        self.assertEqual(
            [entry.plan.operation_id for entry in pending],
            [plan.operation_id for plan in plans[:2]],
        )
        self.assertEqual(client.query_calls[-1]["limit"], 2)
        self.assertEqual(
            client.query_calls[-1]["order_by"],
            ["created_at:asc", "operation_id:asc"],
        )


if __name__ == "__main__":
    unittest.main()
