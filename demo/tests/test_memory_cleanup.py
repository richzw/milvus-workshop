"""Independent tests for bounded selective-Memory physical cleanup."""

from __future__ import annotations

import base64
import json
import re
import unittest
from typing import Any
from unittest.mock import patch

from agent_workshop_demo.selective_memory import (
    LocalSelectiveMemoryStore,
    MemoryEvent,
    MemoryFact,
    MilvusSelectiveMemoryStore,
    SESSION_PRIVATE_SCOPE_HASH,
    SelectiveMemoryError,
    SelectiveMemoryService,
)


def _event(
    event_id: str,
    *,
    session_id: str = "session_a",
    expires_at: int | None = 1_000,
) -> MemoryEvent:
    return MemoryEvent.create(
        event_id=event_id,
        session_id=session_id,
        query_id=f"query_{event_id}",
        turn_id=f"query_{event_id}",
        event_type="user_statement",
        content=f"payload {event_id}",
        summary="answered",
        outcome="answered",
        event_time=100,
        expires_at=expires_at,
        salience_score=0.5,
        selection_reason=("ordinary_turn",),
        retention_class="candidate",
        decay_profile="episode_fast",
        permission_scope_hash=SESSION_PRIVATE_SCOPE_HASH,
    )


def _fact(
    memory_id: str,
    source_event_id: str,
    *,
    status: str = "active",
    expires_at: int | None = None,
) -> MemoryFact:
    return MemoryFact.create(
        memory_id=memory_id,
        session_id="session_a",
        memory_type="user_fact",
        subject="user",
        predicate=memory_id,
        value=f"value {memory_id}",
        revision=1,
        source_event_ids=(source_event_id,),
        valid_from=100,
        last_confirmed_at=100,
        confidence=0.8,
        salience_score=0.8,
        permission_scope_hash=SESSION_PRIVATE_SCOPE_HASH,
        status=status,  # type: ignore[arg-type]
        expires_at=expires_at,
    )


class PendingFactFailureStore(LocalSelectiveMemoryStore):
    """Leave one exact consolidation plan pending."""

    def upsert_facts(self, facts: Any) -> int:
        del facts
        raise SelectiveMemoryError("injected projection failure")


class CleanupMilvusClient:
    """Validate the exact Milvus cleanup request shape."""

    def __init__(self) -> None:
        self.query_calls: list[dict[str, Any]] = []
        self.delete_calls: list[dict[str, Any]] = []

    def query(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.query_calls.append(dict(kwargs))
        collection = kwargs["collection_name"]
        expression = str(kwargs["filter"])
        if collection == "memory_consolidation_journal":
            return []
        if collection == "memory_facts" and "json_contains" in expression:
            return (
                [{"memory_id": "fact_retained"}]
                if '"event_expired_protected"' in expression
                else []
            )
        if collection == "memory_facts":
            return [{"memory_id": "fact_tombstoned"}]
        if collection == "memory_events":
            return [
                {"event_id": "event_expired_delete"},
                {"event_id": "event_expired_protected"},
            ]
        raise AssertionError(f"unexpected collection: {collection}")

    def delete(self, **kwargs: Any) -> dict[str, int]:
        self.delete_calls.append(dict(kwargs))
        match = re.search(r"\bin (\[[^\]]*\])", str(kwargs["filter"]))
        if match is None:
            raise AssertionError("cleanup delete must contain an exact id list")
        return {"delete_count": len(json.loads(match.group(1)))}


class MemoryCleanupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = patch.dict(
            "os.environ",
            {"EMBEDDING_PROVIDER": "deterministic"},
        )
        self.environment.start()

    def tearDown(self) -> None:
        self.environment.stop()

    def test_local_pages_delete_only_eligible_and_preserve_lineage(self) -> None:
        store = LocalSelectiveMemoryStore()
        store.append_events(
            (
                _event("event_expired_protected"),
                _event("event_expired_delete"),
                _event("event_live", expires_at=3_000),
            )
        )
        store.append_events((_event("event_other", session_id="session_b"),))
        store.upsert_facts(
            (
                _fact("fact_active", "event_expired_protected"),
                _fact(
                    "fact_tombstoned",
                    "event_live",
                    status="tombstoned",
                ),
                _fact(
                    "fact_expired",
                    "event_expired_delete",
                    expires_at=1_000,
                ),
            )
        )
        service = SelectiveMemoryService(store)

        first = service.cleanup_page(
            "session_a",
            now_ms=2_000,
            page_size=2,
        )
        self.assertEqual(first.status, "has_more")
        self.assertEqual(first.fact_deleted_count, 2)
        self.assertEqual(first.scanned_count, 2)
        self.assertIsNotNone(first.next_cursor)

        second = service.cleanup_page(
            "session_a",
            now_ms=2_000,
            page_size=2,
            cursor=first.next_cursor,
        )
        self.assertEqual(second.status, "completed")
        self.assertEqual(second.event_deleted_count, 1)
        self.assertEqual(second.protected_event_count, 1)
        self.assertEqual(second.scanned_count, 2)
        self.assertEqual(set(store.facts), {"fact_active"})
        self.assertEqual(
            set(store.events),
            {"event_expired_protected", "event_live", "event_other"},
        )

        replay = service.cleanup_page(
            "session_a",
            now_ms=2_000,
            page_size=2,
            cursor=first.next_cursor,
        )
        self.assertEqual(replay.event_deleted_count, 0)
        self.assertEqual(replay.protected_event_count, 1)

    def test_cursor_is_bound_to_session_and_time_before_mutation(self) -> None:
        store = LocalSelectiveMemoryStore()
        store.append_events((_event("event_1"), _event("event_2")))
        service = SelectiveMemoryService(store)
        first = service.cleanup_page("session_a", now_ms=2_000, page_size=1)
        remaining_before = dict(store.events)

        with self.assertRaisesRegex(ValueError, "does not match"):
            service.cleanup_page(
                "session_b",
                now_ms=2_000,
                page_size=1,
                cursor=first.next_cursor,
            )
        with self.assertRaisesRegex(ValueError, "does not match"):
            service.cleanup_page(
                "session_a",
                now_ms=2_001,
                page_size=1,
                cursor=first.next_cursor,
            )
        with self.assertRaisesRegex(ValueError, "cursor"):
            service.cleanup_page(
                "session_a",
                now_ms=2_000,
                page_size=1,
                cursor="not-a-valid-cursor!",
            )
        encoded, signature = str(first.next_cursor).split(".", 1)
        padding = "=" * (-len(encoded) % 4)
        payload = json.loads(
            base64.urlsafe_b64decode(encoded + padding).decode("utf-8")
        )
        payload["a"] = "zzzz"
        tampered_body = (
            base64.urlsafe_b64encode(
                json.dumps(
                    payload,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            )
            .decode("ascii")
            .rstrip("=")
        )
        with self.assertRaisesRegex(ValueError, "cursor"):
            service.cleanup_page(
                "session_a",
                now_ms=2_000,
                page_size=1,
                cursor=f"{tampered_body}.{signature}",
            )
        self.assertEqual(store.events, remaining_before)

    def test_pending_consolidation_blocks_cleanup_without_mutation(self) -> None:
        store = PendingFactFailureStore()
        service = SelectiveMemoryService(store)
        result = service.persist_turn(
            session_id="session_a",
            query_id="query_1",
            query="请记住以后用中文",
            answer="saved",
            terminal_status="memory_saved",
            remembered_statement="以后用中文",
            now_ms=1_000,
        )
        before = (dict(store.events), dict(store.facts))

        page = service.cleanup_page("session_a", now_ms=2_000, page_size=10)

        self.assertEqual(result.consolidation_status, "pending_retry")
        self.assertEqual(page.status, "blocked_pending")
        self.assertEqual(page.scanned_count, 0)
        self.assertEqual((store.events, store.facts), before)

    def test_milvus_pushes_keyset_bounds_and_exact_id_deletes(self) -> None:
        client = CleanupMilvusClient()
        service = SelectiveMemoryService(MilvusSelectiveMemoryStore(client))

        page = service.cleanup_page("session_a", now_ms=2_000, page_size=3)

        self.assertEqual(page.status, "completed")
        self.assertEqual(page.fact_deleted_count, 1)
        self.assertEqual(page.event_deleted_count, 1)
        self.assertEqual(page.protected_event_count, 1)
        self.assertEqual(page.scanned_count, 3)
        fact_page, event_page = client.query_calls[1], client.query_calls[2]
        self.assertEqual(fact_page["limit"], 4)
        self.assertEqual(fact_page["order_by"], ["memory_id:asc"])
        self.assertEqual(event_page["limit"], 3)
        self.assertEqual(event_page["order_by"], ["event_id:asc"])
        self.assertIn('session_id == "session_a"', fact_page["filter"])
        self.assertIn('status == "tombstoned"', fact_page["filter"])
        self.assertIn(
            'expires_at <= "1970-01-01T00:00:02.000Z"',
            event_page["filter"],
        )
        self.assertTrue(
            any("json_contains(source_event_ids" in call["filter"] for call in client.query_calls)
        )
        self.assertEqual(len(client.delete_calls), 2)
        for call in client.delete_calls:
            self.assertIn('session_id == "session_a"', call["filter"])
            self.assertIn(" in [", call["filter"])

    def test_milvus_rejects_unordered_page_before_delete(self) -> None:
        client = CleanupMilvusClient()

        def unordered(**kwargs: Any) -> list[dict[str, Any]]:
            if kwargs["collection_name"] == "memory_consolidation_journal":
                return []
            return [{"memory_id": "z"}, {"memory_id": "a"}]

        client.query = unordered  # type: ignore[method-assign]
        service = SelectiveMemoryService(MilvusSelectiveMemoryStore(client))

        with self.assertRaisesRegex(SelectiveMemoryError, "keyset order"):
            service.cleanup_page("session_a", now_ms=2_000, page_size=2)
        self.assertEqual(client.delete_calls, [])


if __name__ == "__main__":
    unittest.main()
