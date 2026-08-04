"""Independent contract tests for probe-gated native Milvus decay."""

from __future__ import annotations

import math
import unittest
from typing import Any
from unittest.mock import patch

import agent_workshop_demo.selective_memory as selective_memory
from agent_workshop_demo.selective_memory import (
    DAY_MS,
    DECAY_PROFILES,
    SESSION_PRIVATE_SCOPE_HASH,
    LocalSelectiveMemoryStore,
    MemoryEvent,
    MemoryFact,
    MilvusSelectiveMemoryStore,
    SelectiveMemoryError,
    SelectiveMemoryService,
)


def _event(
    *,
    event_id: str = "event_1",
    session_id: str = "session_a",
    decay_profile: str = "episode_fast",
    event_time: int = 1_000,
    expires_at: int | None = None,
    salience: float = 0.2,
    permission_scope_hash: str = SESSION_PRIVATE_SCOPE_HASH,
) -> MemoryEvent:
    return MemoryEvent.create(
        event_id=event_id,
        session_id=session_id,
        query_id="query_1",
        turn_id="query_1",
        event_type="user_statement",
        content="remember this",
        event_time=event_time,
        expires_at=expires_at,
        salience_score=salience,
        selection_reason=("ordinary_turn",),
        retention_class="ephemeral",
        decay_profile=decay_profile,
        permission_scope_hash=permission_scope_hash,
    )


def _fact() -> MemoryFact:
    return MemoryFact.create(
        memory_id="memory_1",
        session_id="session_a",
        memory_type="user_fact",
        subject="user",
        predicate="language",
        value="zh-CN",
        revision=1,
        source_event_ids=("event_1",),
        valid_from=1_000,
        last_confirmed_at=1_000,
        confidence=0.8,
        salience_score=0.6,
        permission_scope_hash=SESSION_PRIVATE_SCOPE_HASH,
    )


class NativeDecayClient:
    """Record native calls while returning a PyMilvus-shaped envelope."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.rows: dict[str, list[tuple[dict[str, Any], float]]] = {
            "memory_events": [],
            "memory_facts": [],
            "memory_consolidation_journal": [],
        }
        self.fail_probe = False

    def has_collection(self, *, collection_name: str) -> bool:
        return collection_name in self.rows

    def load_collection(self, *, collection_name: str) -> None:
        del collection_name

    def search(self, **kwargs: Any) -> list[list[dict[str, Any]]]:
        self.calls.append(dict(kwargs))
        if "native_decay_probe_" in str(kwargs["filter"]):
            if self.fail_probe:
                raise RuntimeError("unsupported ranker")
            return [[]]
        collection = str(kwargs["collection_name"])
        return [
            [
                {"entity": dict(row), "distance": score}
                for row, score in self.rows[collection]
            ]
        ]

    def query_iterator(self, **kwargs: Any) -> Any:
        del kwargs

        class EmptyIterator:
            def next(self) -> list[dict[str, Any]]:
                return []

            def close(self) -> None:
                return None

        return EmptyIterator()


def _ranker(
    profile: selective_memory.DecayProfile,
    field_name: str,
    origin_ms: int,
) -> dict[str, object]:
    return {
        "function": profile.function,
        "field": field_name,
        "origin": origin_ms,
        "offset": profile.offset_ms,
        "scale": profile.scale_ms,
        "decay": profile.decay,
    }


class NativeMilvusDecayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = patch.dict(
            "os.environ",
            {"EMBEDDING_PROVIDER": "deterministic"},
        )
        self.environment.start()

    def tearDown(self) -> None:
        self.environment.stop()

    def test_public_pymilvus_function_contract_uses_numeric_field_and_ms(self) -> None:
        captured: dict[str, Any] = {}

        class FunctionType:
            RERANK = object()

        def function(**kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            return kwargs

        module = type(
            "FakePyMilvus",
            (),
            {"Function": staticmethod(function), "FunctionType": FunctionType},
        )
        with patch.object(selective_memory, "import_module", return_value=module):
            result = selective_memory._milvus_decay_ranker(
                DECAY_PROFILES["experience_balanced"],
                "last_confirmed_at",
                100 * DAY_MS,
            )

        self.assertEqual(result, captured)
        self.assertEqual(captured["input_field_names"], ["last_confirmed_at"])
        self.assertIs(captured["function_type"], FunctionType.RERANK)
        self.assertEqual(
            captured["params"],
            {
                "reranker": "decay",
                "function": "gauss",
                "origin": 100 * DAY_MS,
                "offset": 3 * DAY_MS,
                "scale": 30 * DAY_MS,
                "decay": 0.5,
            },
        )

    def test_startup_probe_is_empty_read_only_and_covers_all_functions(self) -> None:
        client = NativeDecayClient()
        store = MilvusSelectiveMemoryStore(
            client,
            decay_mode="milvus",
            ranker_factory=_ranker,
        )

        store.ensure_collections_ready()

        self.assertTrue(store.native_decay_verified)
        self.assertEqual(
            [call["ranker"]["function"] for call in client.calls],
            ["exp", "gauss", "linear"],
        )
        self.assertTrue(all(len(call["data"][0]) == 1_024 for call in client.calls))
        self.assertTrue(
            all("native_decay_probe_" in call["filter"] for call in client.calls)
        )
        self.assertTrue(all(call["limit"] == 1 for call in client.calls))

    def test_probe_failure_blocks_native_search_without_silent_fallback(self) -> None:
        client = NativeDecayClient()
        client.fail_probe = True
        store = MilvusSelectiveMemoryStore(
            client,
            decay_mode="milvus",
            ranker_factory=_ranker,
        )

        with self.assertRaisesRegex(SelectiveMemoryError, "probe failed"):
            store.ensure_collections_ready()
        self.assertFalse(store.native_decay_verified)
        with self.assertRaisesRegex(SelectiveMemoryError, "successful probe"):
            store.search_events(
                "remember",
                session_id="session_a",
                now_ms=2_000,
                decay_profile="episode_fast",
                top_k=1,
                permission_scope_hashes=frozenset({SESSION_PRIVATE_SCOPE_HASH}),
            )

    def test_event_and_fact_search_bind_native_fields_and_merge_factors(self) -> None:
        client = NativeDecayClient()
        event = _event(salience=0.2)
        fact = _fact()
        client.rows["memory_events"] = [(event.to_dict(), 0.4)]
        client.rows["memory_facts"] = [(fact.to_dict(), 0.5)]
        store = MilvusSelectiveMemoryStore(
            client,
            decay_mode="milvus",
            ranker_factory=_ranker,
        )
        store.ensure_collections_ready()

        event_match = store.search_events(
            "remember this",
            session_id="session_a",
            now_ms=2_000,
            decay_profile="episode_fast",
            top_k=1,
            permission_scope_hashes=frozenset({SESSION_PRIVATE_SCOPE_HASH}),
        )[0]
        fact_match = store.search_facts(
            "zh-CN",
            session_id="session_a",
            now_ms=2_000,
            decay_profile="durable_gentle",
            top_k=1,
            permission_scope_hashes=frozenset({SESSION_PRIVATE_SCOPE_HASH}),
        )[0]

        event_call, fact_call = client.calls[-2:]
        self.assertEqual(event_call["ranker"]["field"], "event_time")
        self.assertEqual(fact_call["ranker"]["field"], "last_confirmed_at")
        self.assertEqual(event_call["ranker"]["origin"], 2_000)
        self.assertIn('session_id == "session_a"', event_call["filter"])
        self.assertIn("expires_at", event_call["filter"])
        self.assertIn('status == "active"', fact_call["filter"])
        self.assertEqual(event_match.decay_mode, "milvus")
        self.assertTrue(math.isclose(event_match.final_score, 0.24))
        self.assertTrue(math.isclose(fact_match.final_score, 0.32))

    def test_no_time_decay_bypasses_ranker_but_preserves_native_mode(self) -> None:
        client = NativeDecayClient()
        client.rows["memory_events"] = [
            (_event(decay_profile="no_time_decay").to_dict(), 0.5)
        ]
        store = MilvusSelectiveMemoryStore(
            client,
            decay_mode="milvus",
            ranker_factory=_ranker,
        )
        store.ensure_collections_ready()

        match = store.search_events(
            "remember this",
            session_id="session_a",
            now_ms=2_000,
            decay_profile="no_time_decay",
            top_k=1,
            permission_scope_hashes=frozenset({SESSION_PRIVATE_SCOPE_HASH}),
        )[0]

        self.assertNotIn("ranker", client.calls[-1])
        self.assertEqual(match.decay_score, 1.0)
        self.assertEqual(match.decay_mode, "milvus")

    def test_native_and_application_decay_have_ranking_parity(self) -> None:
        records = (
            _event(event_id="event_recent", event_time=9 * DAY_MS),
            _event(event_id="event_old", event_time=1_000),
        )
        local = LocalSelectiveMemoryStore()
        local.append_events(records)
        expected = local.search_events(
            "remember this",
            session_id="session_a",
            now_ms=10 * DAY_MS,
            decay_profile="episode_fast",
            top_k=2,
            permission_scope_hashes=frozenset({SESSION_PRIVATE_SCOPE_HASH}),
        )

        client = NativeDecayClient()
        client.rows["memory_events"] = [
            (
                match.event.to_dict(),
                match.semantic_score * match.decay_score,
            )
            for match in expected
        ]
        native = MilvusSelectiveMemoryStore(
            client,
            decay_mode="milvus",
            ranker_factory=_ranker,
        )
        native.ensure_collections_ready()
        actual = native.search_events(
            "remember this",
            session_id="session_a",
            now_ms=10 * DAY_MS,
            decay_profile="episode_fast",
            top_k=2,
            permission_scope_hashes=frozenset({SESSION_PRIVATE_SCOPE_HASH}),
        )

        self.assertEqual(
            [match.event.event_id for match in actual],
            [match.event.event_id for match in expected],
        )
        for native_match, local_match in zip(actual, expected, strict=True):
            self.assertTrue(
                math.isclose(
                    native_match.final_score,
                    local_match.final_score,
                )
            )

    def test_permission_scope_is_filtered_before_bounded_native_ranking(self) -> None:
        class ScopeFilteringClient(NativeDecayClient):
            def search(self, **kwargs: Any) -> list[list[dict[str, Any]]]:
                if "native_decay_probe_" in str(kwargs["filter"]):
                    return super().search(**kwargs)
                self.calls.append(dict(kwargs))
                rows = [
                    (row, score)
                    for row, score in self.rows[str(kwargs["collection_name"])]
                    if row["permission_scope_hash"] in str(kwargs["filter"])
                ]
                rows.sort(key=lambda item: item[1], reverse=True)
                return [
                    [
                        {"entity": dict(row), "distance": score}
                        for row, score in rows[: int(kwargs["limit"])]
                    ]
                ]

        client = ScopeFilteringClient()
        denied_scope = "a" * 64
        client.rows["memory_events"] = [
            (
                _event(
                    event_id=f"denied_{index}",
                    permission_scope_hash=denied_scope,
                ).to_dict(),
                1.0 - index / 100,
            )
            for index in range(4)
        ]
        client.rows["memory_events"].append((_event().to_dict(), 0.1))
        store = MilvusSelectiveMemoryStore(
            client,
            decay_mode="milvus",
            ranker_factory=_ranker,
        )
        store.ensure_collections_ready()

        matches = store.search_events(
            "remember this",
            session_id="session_a",
            now_ms=2_000,
            decay_profile="episode_fast",
            top_k=1,
            permission_scope_hashes=frozenset({SESSION_PRIVATE_SCOPE_HASH}),
        )

        self.assertEqual([match.event.event_id for match in matches], ["event_1"])
        self.assertIn("permission_scope_hash in", client.calls[-1]["filter"])
        self.assertIn(SESSION_PRIVATE_SCOPE_HASH, client.calls[-1]["filter"])
        self.assertNotIn(denied_scope, client.calls[-1]["filter"])

    def test_returned_scope_is_revalidated_and_empty_pack_reports_mode(self) -> None:
        client = NativeDecayClient()
        store = MilvusSelectiveMemoryStore(
            client,
            decay_mode="milvus",
            ranker_factory=_ranker,
        )
        store.ensure_collections_ready()
        pack = SelectiveMemoryService(store).recall(
            "nothing",
            session_id="session_a",
            now_ms=2_000,
            include_episodes=False,
        )
        self.assertEqual(pack.decay_mode, "milvus")

        client.rows["memory_events"] = [(_event(session_id="session_b").to_dict(), 0.5)]
        with self.assertRaisesRegex(SelectiveMemoryError, "outside requested scope"):
            store.search_events(
                "remember",
                session_id="session_a",
                now_ms=2_000,
                decay_profile="episode_fast",
                top_k=1,
                permission_scope_hashes=frozenset({SESSION_PRIVATE_SCOPE_HASH}),
            )


if __name__ == "__main__":
    unittest.main()
