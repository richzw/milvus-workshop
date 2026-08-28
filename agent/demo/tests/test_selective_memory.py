from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import patch

from agent_workshop_demo.selective_memory import (
    DAY_MS,
    DECAY_PROFILES,
    SESSION_PRIVATE_SCOPE_HASH,
    LocalSelectiveMemoryStore,
    MemoryEvent,
    MemoryFact,
    MilvusSelectiveMemoryStore,
    RetentionClass,
    RuleBasedMemorySelector,
    SelectiveMemoryError,
    SelectiveMemoryService,
    decay_score,
)
from agent_workshop_demo.workflow import AgenticRAGWorkflow


def event(
    *,
    event_id: str = "event_1",
    session_id: str = "session_a",
    content: str = "remember this",
    event_time: int = 1_000,
    expires_at: int | None = None,
    salience: float = 0.2,
    retention_class: RetentionClass = "ephemeral",
    decay_profile: str = "episode_fast",
) -> MemoryEvent:
    return MemoryEvent.create(
        event_id=event_id,
        session_id=session_id,
        query_id="query_1",
        turn_id="query_1",
        event_type="user_statement",
        content=content,
        event_time=event_time,
        expires_at=expires_at,
        salience_score=salience,
        selection_reason=("ordinary_turn",),
        retention_class=retention_class,
        decay_profile=decay_profile,
        permission_scope_hash=SESSION_PRIVATE_SCOPE_HASH,
    )


class RecordingSelectiveClient:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, dict[str, Any]]] = {
            "memory_events": {},
            "memory_facts": {},
            "memory_consolidation_journal": {},
        }
        self.loaded: list[str] = []
        self.search_rows: dict[str, list[dict[str, Any]]] = {}
        self.query_calls: list[dict[str, Any]] = []

    def has_collection(self, *, collection_name: str) -> bool:
        return collection_name in self.rows

    def load_collection(self, *, collection_name: str) -> None:
        self.loaded.append(collection_name)

    def query(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.query_calls.append(dict(kwargs))
        collection = str(kwargs["collection_name"])
        expression = str(kwargs["filter"])
        return [
            dict(row)
            for identity, row in self.rows[collection].items()
            if identity in expression
        ]

    def insert(self, **kwargs: Any) -> dict[str, int]:
        collection = str(kwargs["collection_name"])
        for row in kwargs["data"]:
            identity_field = (
                "event_id" if collection == "memory_events" else "operation_id"
            )
            identity = str(row[identity_field])
            self.rows[collection][identity] = dict(row)
        return {"insert_count": len(kwargs["data"])}

    def upsert(self, **kwargs: Any) -> dict[str, int]:
        collection = str(kwargs["collection_name"])
        for row in kwargs["data"]:
            identity_field = (
                "memory_id" if collection == "memory_facts" else "operation_id"
            )
            identity = str(row[identity_field])
            self.rows[collection][identity] = dict(row)
        return {"upsert_count": len(kwargs["data"])}

    def flush(self, *, collection_name: str) -> None:
        del collection_name

    def search(self, **kwargs: Any) -> list[list[dict[str, Any]]]:
        collection = str(kwargs["collection_name"])
        return [[{"entity": dict(row)} for row in self.search_rows.get(collection, [])]]

    def query_iterator(self, **kwargs: Any) -> Any:
        self.query_calls.append(dict(kwargs))
        collection = str(kwargs["collection_name"])
        rows = [dict(row) for row in self.rows[collection].values()]
        batches = [rows, []]

        class Iterator:
            def next(self) -> list[dict[str, Any]]:
                return batches.pop(0)

            def close(self) -> None:
                return None

        return Iterator()

    def delete(self, **kwargs: Any) -> dict[str, int]:
        collection = str(kwargs["collection_name"])
        count = len(self.rows[collection])
        self.rows[collection] = {}
        return {"delete_count": count}


class SelectiveMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = patch.dict(
            "os.environ",
            {"EMBEDDING_PROVIDER": "deterministic"},
        )
        self.environment.start()

    def tearDown(self) -> None:
        self.environment.stop()

    def test_milvus_fact_filter_uses_timestamptz_literal(self) -> None:
        client = RecordingSelectiveClient()
        store = MilvusSelectiveMemoryStore(client)

        self.assertEqual(
            store.list_facts(
                "session_a",
                now_ms=2_000,
                permission_scope_hashes=frozenset(
                    {SESSION_PRIVATE_SCOPE_HASH}
                ),
            ),
            [],
        )

        expression = client.query_calls[-1]["filter"]
        self.assertIn(
            "expires_at > ISO '1970-01-01T00:00:02.000Z'",
            expression,
        )
        self.assertNotIn('expires_at > "1970-', expression)

    def test_decay_profiles_hit_offset_and_scale_points(self) -> None:
        now = 100 * DAY_MS
        for name in ("episode_fast", "experience_balanced", "task_deadline"):
            profile = DECAY_PROFILES[name]
            at_offset = now - profile.offset_ms
            at_scale = now - profile.offset_ms - profile.scale_ms
            with self.subTest(name=name):
                self.assertEqual(
                    decay_score(
                        profile,
                        timestamp_ms=at_offset,
                        now_ms=now,
                    ),
                    1.0,
                )
                self.assertAlmostEqual(
                    decay_score(
                        profile,
                        timestamp_ms=at_scale,
                        now_ms=now,
                    ),
                    profile.decay,
                )

    def test_rule_selector_protects_explicit_memory_and_correction(self) -> None:
        selector = RuleBasedMemorySelector()
        remembered = selector.select(
            query="请记住以后用中文",
            terminal_status="memory_saved",
            remembered_statement="以后用中文",
        )
        corrected = selector.select(
            query="不是 2.6，是 3.0",
            terminal_status="answered",
            remembered_statement=None,
        )
        ordinary = selector.select(
            query="你好",
            terminal_status="answered_without_retrieval",
            remembered_statement=None,
        )

        self.assertEqual(remembered.retention_class, "protected")
        self.assertEqual(remembered.event_type, "user_preference")
        self.assertEqual(corrected.event_type, "user_correction")
        self.assertEqual(corrected.salience_score, 1.0)
        self.assertEqual(ordinary.retention_class, "ephemeral")

    def test_event_append_is_idempotent_and_collision_fails_closed(self) -> None:
        store = LocalSelectiveMemoryStore()
        original = event()

        self.assertEqual(store.append_events((original,)), 1)
        self.assertEqual(store.append_events((original,)), 0)
        with self.assertRaisesRegex(
            SelectiveMemoryError,
            "identity collision",
        ):
            store.append_events((event(content="different"),))

    def test_preference_consolidates_and_recall_does_not_refresh_it(self) -> None:
        service = SelectiveMemoryService()
        result = service.persist_turn(
            session_id="session_a",
            query_id="query_1",
            query="请记住以后用中文",
            answer="saved",
            terminal_status="memory_saved",
            remembered_statement="以后用中文",
            now_ms=1_000,
        )
        before = service.store.list_facts(
            "session_a",
            now_ms=1_000,
            statuses=("active",),
        )
        pack = service.recall(
            "你还记得我的偏好吗",
            session_id="session_a",
            now_ms=2_000,
            include_episodes=True,
        )
        after = service.store.list_facts(
            "session_a",
            now_ms=2_000,
            statuses=("active",),
        )

        self.assertEqual(result.consolidation_status, "completed")
        self.assertEqual(pack.working_state[0].value, "zh-CN")
        self.assertEqual(before, after)
        self.assertEqual(pack.decay_mode, "application")

    def test_correction_creates_superseding_fact_revision(self) -> None:
        service = SelectiveMemoryService()
        for index, query in enumerate(
            ("不是 2.6，是 3.0", "不是 3.0，是 3.1"),
            start=1,
        ):
            service.persist_turn(
                session_id="session_a",
                query_id=f"query_{index}",
                query=query,
                answer="ok",
                terminal_status="answered",
                remembered_statement=None,
                now_ms=index * 1_000,
            )

        active = service.store.list_facts(
            "session_a",
            now_ms=3_000,
            statuses=("active",),
        )
        superseded = service.store.list_facts(
            "session_a",
            now_ms=3_000,
            statuses=("superseded",),
        )

        self.assertEqual(active[0].value, "3.1")
        self.assertEqual(active[0].revision, 2)
        self.assertEqual(active[0].supersedes_memory_id, superseded[0].memory_id)
        self.assertEqual(len(active[0].source_event_ids), 2)

    def test_explicit_reconfirmation_appends_lineage_and_revision(self) -> None:
        service = SelectiveMemoryService()
        for index in (1, 2):
            service.persist_turn(
                session_id="session_a",
                query_id=f"query_{index}",
                query="请记住以后用中文",
                answer="saved",
                terminal_status="memory_saved",
                remembered_statement="以后用中文",
                now_ms=index * 1_000,
            )

        active = service.store.list_facts(
            "session_a",
            now_ms=3_000,
            statuses=("active",),
        )
        events = service.store.list_events(
            "session_a",
            now_ms=3_000,
        )

        self.assertEqual(active[0].revision, 2)
        self.assertEqual(len(active[0].source_event_ids), 2)
        self.assertTrue(any(item.event_type == "memory_reconfirmed" for item in events))

    def test_unmatched_high_confidence_correction_becomes_disputed(self) -> None:
        service = SelectiveMemoryService()
        for index, query in enumerate(
            (
                "不是 2.6，是 3.0",
                "不是 4.0，是 5.0",
                "不是 6.0，是 7.0",
            ),
            start=1,
        ):
            service.persist_turn(
                session_id="session_a",
                query_id=f"query_{index}",
                query=query,
                answer="ok",
                terminal_status="answered",
                remembered_statement=None,
                now_ms=index * 1_000,
            )

        active = service.store.list_facts(
            "session_a",
            now_ms=3_000,
            statuses=("active",),
        )
        pack = service.recall(
            "版本更正",
            session_id="session_a",
            now_ms=3_000,
            include_episodes=False,
        )

        self.assertEqual(active, [])
        self.assertEqual(len(pack.conflicts), 3)
        self.assertEqual(pack.working_state, ())
        self.assertEqual(pack.durable_facts, ())

    def test_repeated_failure_consolidates_but_stays_permission_scoped(self) -> None:
        service = SelectiveMemoryService()
        for index in (1, 2):
            service.persist_turn(
                session_id="session_a",
                query_id=f"query_{index}",
                query="missing workshop document",
                answer="insufficient evidence",
                terminal_status="abstained",
                remembered_statement=None,
                now_ms=index * 1_000,
                permission_scope_hash_value="a" * 64,
            )

        facts = service.store.list_facts(
            "session_a",
            now_ms=3_000,
            statuses=("active",),
        )
        pack = service.recall(
            "missing workshop document",
            session_id="session_a",
            now_ms=3_000,
            include_episodes=True,
        )
        authorized = service.recall(
            "missing workshop document",
            session_id="session_a",
            now_ms=3_000,
            include_episodes=True,
            permission_scope_hash_value="a" * 64,
        )

        self.assertEqual(facts[0].memory_type, "failure_pattern")
        self.assertEqual(len(facts[0].source_event_ids), 2)
        self.assertNotIn(facts[0], pack.working_state)
        self.assertNotIn(facts[0], pack.durable_facts)
        self.assertIn(facts[0], authorized.durable_facts)

    def test_milvus_adapter_matches_application_decay_and_scope(self) -> None:
        client = RecordingSelectiveClient()
        store = MilvusSelectiveMemoryStore(client)
        record = event(
            event_time=1_000,
            expires_at=10 * DAY_MS,
        )
        store.ensure_collections_ready()
        self.assertEqual(store.append_events((record,)), 1)
        self.assertEqual(store.append_events((record,)), 0)
        client.search_rows["memory_events"] = [record.to_dict()]

        matches = store.search_events(
            "remember this",
            session_id="session_a",
            now_ms=DAY_MS + 1_000,
            decay_profile="episode_fast",
            top_k=3,
            permission_scope_hashes=frozenset({SESSION_PRIVATE_SCOPE_HASH}),
        )

        self.assertEqual(
            client.loaded,
            [
                "memory_events",
                "memory_facts",
                "memory_consolidation_journal",
            ],
        )
        self.assertEqual(matches[0].event.event_id, record.event_id)
        self.assertEqual(matches[0].event.selector_name, "rule_based")
        self.assertIn(
            "selector_fallback_reason",
            client.rows["memory_events"][record.event_id],
        )
        self.assertEqual(matches[0].decay_mode, "application")
        self.assertEqual(store.delete_session("session_a"), 1)

    def test_milvus_fact_upsert_preserves_immutable_lineage(self) -> None:
        client = RecordingSelectiveClient()
        store = MilvusSelectiveMemoryStore(client)
        store.append_events((event(),))
        original = MemoryFact.create(
            memory_id="memory_1",
            session_id="session_a",
            memory_type="user_fact",
            subject="user",
            predicate="name",
            value="张三",
            revision=1,
            source_event_ids=("event_1",),
            valid_from=1_000,
            last_confirmed_at=1_000,
            confidence=0.9,
            salience_score=0.8,
            permission_scope_hash=SESSION_PRIVATE_SCOPE_HASH,
        )
        rewritten = MemoryFact.create(
            memory_id="memory_1",
            session_id="session_a",
            memory_type="user_fact",
            subject="user",
            predicate="name",
            value="李四",
            revision=2,
            source_event_ids=("event_2",),
            valid_from=2_000,
            last_confirmed_at=2_000,
            confidence=0.9,
            salience_score=0.8,
            permission_scope_hash=SESSION_PRIVATE_SCOPE_HASH,
        )

        self.assertEqual(store.upsert_facts((original,)), 1)
        self.assertEqual(store.upsert_facts((original,)), 0)
        with self.assertRaisesRegex(
            SelectiveMemoryError,
            "immutable lineage",
        ):
            store.upsert_facts((rewritten,))

    def test_fact_store_rejects_missing_and_cross_session_sources(self) -> None:
        store = LocalSelectiveMemoryStore()
        fact = MemoryFact.create(
            memory_id="memory_1",
            session_id="session_a",
            memory_type="user_fact",
            subject="user",
            predicate="name",
            value="张三",
            revision=1,
            source_event_ids=("event_1",),
            valid_from=1_000,
            last_confirmed_at=1_000,
            confidence=0.9,
            salience_score=0.8,
            permission_scope_hash=SESSION_PRIVATE_SCOPE_HASH,
        )

        with self.assertRaisesRegex(SelectiveMemoryError, "source event"):
            store.upsert_facts((fact,))
        store.append_events((event(event_id="event_1", session_id="session_b"),))
        with self.assertRaisesRegex(SelectiveMemoryError, "source event"):
            store.upsert_facts((fact,))

    def test_selective_ui_rows_do_not_expose_memory_payload(self) -> None:
        service = SelectiveMemoryService()
        service.persist_turn(
            session_id="session_a",
            query_id="query_1",
            query="请记住我叫张三",
            answer="sensitive assistant answer",
            terminal_status="memory_saved",
            remembered_statement="我叫张三",
            now_ms=1_000,
        )

        rows = service.list_session("session_a", now_ms=1_000)

        self.assertTrue(rows)
        self.assertTrue(all("preview" not in row for row in rows))
        self.assertNotIn("张三", str(rows))
        self.assertNotIn("sensitive assistant answer", str(rows))

    def test_memory_only_answer_never_replays_assistant_answer(self) -> None:
        workflow = AgenticRAGWorkflow(wall_clock_ms=lambda: 1_000)
        workflow.run(
            "Milvus 3.0 有哪些新功能",
            session_id="session_a",
            query_id="query_1",
        )
        recalled = workflow.run(
            "你还记得我之前问了什么吗？",
            session_id="session_a",
            query_id="query_2",
        )

        self.assertEqual(recalled["terminal_status"], "answered_from_memory")
        self.assertIn("Milvus 3.0", recalled["answer"])
        self.assertNotIn("Storage Format V3", recalled["answer"])
        self.assertEqual(recalled["citations"], [])

    def test_workflow_uses_selective_memory_and_keeps_trace_content_free(
        self,
    ) -> None:
        workflow = AgenticRAGWorkflow(wall_clock_ms=lambda: 1_000)
        first = workflow.run(
            "请记住以后用中文",
            session_id="session_a",
            query_id="query_1",
        )
        second = workflow.run(
            "你还记得我的偏好吗？",
            session_id="session_a",
            query_id="query_2",
        )

        self.assertEqual(first["trace"]["memory"]["selective"]["status"], "saved")
        self.assertEqual(second["terminal_status"], "answered_from_memory")
        self.assertIn("zh-CN", second["answer"])
        self.assertNotIn(
            "zh-CN",
            str(second["trace"]["memory"]["recalled"]),
        )
        self.assertEqual(second["citations"], [])
        self.assertGreater(workflow.clear_memory("session_a"), 0)
        self.assertEqual(workflow.list_selective_memories("session_a"), [])


if __name__ == "__main__":
    unittest.main()
