from __future__ import annotations

import unittest
from unittest.mock import patch

from agent_workshop_demo.embedding import (
    EMBEDDING_FINGERPRINT_KEY,
    text_embedding_fingerprint,
)
from agent_workshop_demo.langgraph_workflow import build_milvus_workflow
from agent_workshop_demo.schema.pymilvus_adapter import MilvusHybridRetriever
from agent_workshop_demo.workflow import AgenticRAGWorkflow


class CollectionClient:
    def __init__(self, *, exists: bool) -> None:
        self.exists = exists
        self.loaded: list[str] = []

    def has_collection(self, *, collection_name: str) -> bool:
        return self.exists

    def load_collection(self, *, collection_name: str) -> None:
        self.loaded.append(collection_name)


class FingerprintClient:
    def __init__(self, fingerprints: list[str | None], *, fail: bool = False) -> None:
        self.fingerprints = fingerprints
        self.fail = fail
        self.queries: list[dict[str, object]] = []

    def query(self, **kwargs: object) -> list[dict[str, object]]:
        self.queries.append(kwargs)
        if self.fail:
            raise TimeoutError("milvus unavailable")
        return [
            {
                "chunk_id": f"chunk_{index}",
                "metadata": (
                    {} if value is None else {EMBEDDING_FINGERPRINT_KEY: value}
                ),
            }
            for index, value in enumerate(self.fingerprints)
        ]


class EmbeddingSpaceGateTests(unittest.TestCase):
    def test_matching_fingerprint_passes_the_startup_gate(self) -> None:
        expected = text_embedding_fingerprint()
        client = FingerprintClient([expected, expected])

        observed = MilvusHybridRetriever(
            client,
            collection_name="workshop_chunks",
        ).ensure_embedding_space_ready()

        self.assertEqual(observed, expected)
        self.assertEqual(client.queries[0]["collection_name"], "workshop_chunks")
        self.assertEqual(client.queries[0]["limit"], 8)

    def test_stale_vector_space_fails_startup(self) -> None:
        client = FingerprintClient(["openai:text-embedding-3-small:1024"])

        with self.assertRaisesRegex(RuntimeError, "different vector space"):
            MilvusHybridRetriever(client).ensure_embedding_space_ready()

    def test_mixed_vector_spaces_fail_startup(self) -> None:
        client = FingerprintClient([text_embedding_fingerprint(), "legacy:512"])

        with self.assertRaisesRegex(RuntimeError, "different vector space"):
            MilvusHybridRetriever(client).ensure_embedding_space_ready()

    def test_missing_fingerprint_metadata_fails_startup(self) -> None:
        client = FingerprintClient([None])

        with self.assertRaisesRegex(RuntimeError, "different vector space"):
            MilvusHybridRetriever(client).ensure_embedding_space_ready()

    def test_empty_collection_is_not_a_mismatch(self) -> None:
        client = FingerprintClient([])

        observed = MilvusHybridRetriever(client).ensure_embedding_space_ready()

        self.assertEqual(observed, text_embedding_fingerprint())

    def test_query_failure_is_wrapped(self) -> None:
        client = FingerprintClient([], fail=True)

        with self.assertRaisesRegex(RuntimeError, "Unable to read the stored"):
            MilvusHybridRetriever(client).ensure_embedding_space_ready()

    def test_sample_size_is_bounded(self) -> None:
        client = FingerprintClient([text_embedding_fingerprint()])

        for size in (0, 65):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    MilvusHybridRetriever(client).ensure_embedding_space_ready(
                        sample_size=size,
                    )


class ServerWorkflowTests(unittest.TestCase):
    def test_milvus_builder_connects_loads_and_injects_retriever(self) -> None:
        with (
            patch(
                "agent_workshop_demo.langgraph_workflow.MilvusHybridRetriever.connect"
            ) as connect,
            patch(
                "agent_workshop_demo.langgraph_workflow.MilvusConversationMemoryStore"
            ) as memory_store_class,
            patch(
                "agent_workshop_demo.langgraph_workflow."
                "MilvusGroundedResponseCacheStore"
            ) as response_cache_class,
            patch(
                "agent_workshop_demo.langgraph_workflow.MilvusSelectiveMemoryStore"
            ) as selective_store_class,
        ):
            retriever = connect.return_value

            workflow = build_milvus_workflow(
                {
                    "MILVUS_URI": "http://milvus.test:19530",
                    "MILVUS_TOKEN": "test-token",
                    "MILVUS_COLLECTION_NAME": "workshop_chunks",
                    "MILVUS_SPARSE_FIELD": "sparse_vector_v2",
                    "MILVUS_MEMORY_COLLECTION_NAME": "workshop_memory",
                    "MILVUS_MEMORY_EVENTS_COLLECTION_NAME": ("workshop_memory_events"),
                    "MILVUS_MEMORY_FACTS_COLLECTION_NAME": ("workshop_memory_facts"),
                    "MILVUS_MEMORY_CONSOLIDATION_JOURNAL_COLLECTION_NAME": (
                        "workshop_memory_journal"
                    ),
                    "MILVUS_RESPONSE_CACHE_COLLECTION_NAME": (
                        "workshop_response_cache"
                    ),
                    "MEMORY_TOP_K": "4",
                    "MEMORY_TTL_SECONDS": "3600",
                    "RESPONSE_CACHE_TOP_K": "5",
                    "RESPONSE_CACHE_TTL_SECONDS": "7200",
                    "RESPONSE_CACHE_SIMILARITY_THRESHOLD": "0.95",
                    "KB_REVISION": "workshop-revision-2",
                }
            )

        connect.assert_called_once_with(
            "http://milvus.test:19530",
            "test-token",
            collection_name="workshop_chunks",
            sparse_field="sparse_vector_v2",
        )
        retriever.ensure_collection_ready.assert_called_once_with()
        retriever.ensure_embedding_space_ready.assert_called_once_with()
        memory_store_class.assert_called_once_with(
            retriever.client,
            collection_name="workshop_memory",
        )
        memory_store = memory_store_class.return_value
        memory_store.ensure_collection_ready.assert_called_once_with()
        selective_store_class.assert_called_once_with(
            retriever.client,
            events_collection_name="workshop_memory_events",
            facts_collection_name="workshop_memory_facts",
            journal_collection_name="workshop_memory_journal",
            decay_mode="application",
        )
        selective_store = selective_store_class.return_value
        selective_store.ensure_collections_ready.assert_called_once_with()
        response_cache_class.assert_called_once_with(
            retriever.client,
            collection_name="workshop_response_cache",
        )
        response_cache = response_cache_class.return_value
        response_cache.ensure_collection_ready.assert_called_once_with()
        configured_workflow = (
            workflow if isinstance(workflow, AgenticRAGWorkflow) else workflow.workflow
        )
        self.assertIs(configured_workflow.retriever, retriever)
        self.assertIs(
            configured_workflow.memory_store,
            memory_store,
        )
        self.assertEqual(configured_workflow.memory_top_k, 4)
        self.assertEqual(configured_workflow.memory_ttl_seconds, 3600)
        self.assertIs(configured_workflow.response_cache, response_cache)
        self.assertIs(
            configured_workflow.selective_memory.store,
            selective_store,
        )
        self.assertEqual(configured_workflow.response_cache_top_k, 5)
        self.assertEqual(
            configured_workflow.response_cache_ttl_seconds,
            7200,
        )
        self.assertEqual(
            configured_workflow.response_cache_similarity_threshold,
            0.95,
        )
        self.assertEqual(
            configured_workflow.kb_revision,
            "workshop-revision-2",
        )

    def test_collection_readiness_loads_existing_collection(self) -> None:
        client = CollectionClient(exists=True)

        MilvusHybridRetriever(
            client,
            collection_name="workshop_chunks",
        ).ensure_collection_ready()

        self.assertEqual(client.loaded, ["workshop_chunks"])

    def test_collection_readiness_rejects_missing_collection(self) -> None:
        client = CollectionClient(exists=False)

        with self.assertRaisesRegex(RuntimeError, "does not exist"):
            MilvusHybridRetriever(client).ensure_collection_ready()

    def test_milvus_builder_wraps_connection_failure(self) -> None:
        with patch(
            "agent_workshop_demo.langgraph_workflow.MilvusHybridRetriever.connect",
            side_effect=OSError("connection refused"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "Unable to initialize the Milvus retriever",
            ) as captured:
                build_milvus_workflow()

        self.assertIsInstance(captured.exception.__cause__, OSError)

    def test_milvus_builder_activates_fingerprint_gated_struct_array(self) -> None:
        fingerprint = "a" * 64
        with (
            patch(
                "agent_workshop_demo.langgraph_workflow.MilvusHybridRetriever.connect"
            ) as connect,
            patch(
                "agent_workshop_demo.langgraph_workflow.MilvusStructArrayRetriever"
            ) as struct_class,
            patch(
                "agent_workshop_demo.langgraph_workflow.MilvusConversationMemoryStore"
            ),
            patch(
                "agent_workshop_demo.langgraph_workflow."
                "MilvusGroundedResponseCacheStore"
            ),
            patch("agent_workshop_demo.langgraph_workflow.MilvusSelectiveMemoryStore"),
        ):
            flat = connect.return_value
            built = build_milvus_workflow(
                {
                    "STRUCT_ARRAY_RETRIEVAL": "struct_element",
                    "STRUCT_ARRAY_PROJECTION_FINGERPRINT": fingerprint,
                    "MILVUS_STRUCT_ARRAY_COLLECTION_NAME": "workshop_documents",
                    "STRUCT_ARRAY_PARENT_TOP_K": "7",
                }
            )

        config = struct_class.call_args.args[2]
        self.assertEqual(config.collection_name, "workshop_documents")
        self.assertEqual(config.projection_fingerprint, fingerprint)
        self.assertEqual(config.parent_top_k, 7)
        struct_class.assert_called_once_with(flat.client, flat, config)
        struct_class.return_value.ensure_ready.assert_called_once_with()
        workflow = built if isinstance(built, AgenticRAGWorkflow) else built.workflow
        self.assertIs(workflow.retriever, struct_class.return_value)

    def test_milvus_builder_enables_probe_gated_native_decay(self) -> None:
        with (
            patch(
                "agent_workshop_demo.langgraph_workflow.MilvusHybridRetriever.connect"
            ),
            patch(
                "agent_workshop_demo.langgraph_workflow.MilvusConversationMemoryStore"
            ),
            patch(
                "agent_workshop_demo.langgraph_workflow."
                "MilvusGroundedResponseCacheStore"
            ),
            patch(
                "agent_workshop_demo.langgraph_workflow.MilvusSelectiveMemoryStore"
            ) as selective_store_class,
        ):
            build_milvus_workflow({"MEMORY_DECAY_MODE": "milvus"})

        self.assertEqual(
            selective_store_class.call_args.kwargs["decay_mode"],
            "milvus",
        )
        selective_store_class.return_value.ensure_collections_ready.assert_called_once_with()

    def test_milvus_builder_rejects_invalid_memory_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "MEMORY_TOP_K"):
            build_milvus_workflow({"MEMORY_TOP_K": "0"})
        with self.assertRaisesRegex(ValueError, "MEMORY_TTL_SECONDS"):
            build_milvus_workflow({"MEMORY_TTL_SECONDS": "forever"})
        with self.assertRaisesRegex(ValueError, "RESPONSE_CACHE_ENABLED"):
            build_milvus_workflow({"RESPONSE_CACHE_ENABLED": "perhaps"})
        with self.assertRaisesRegex(ValueError, "SELECTIVE_MEMORY_ENABLED"):
            build_milvus_workflow({"SELECTIVE_MEMORY_ENABLED": "perhaps"})
        with self.assertRaisesRegex(ValueError, "MEMORY_CONTEXT_MAX_CHARS"):
            build_milvus_workflow({"MEMORY_CONTEXT_MAX_CHARS": "511"})
        with self.assertRaisesRegex(ValueError, "MEMORY_DECAY_MODE"):
            build_milvus_workflow({"MEMORY_DECAY_MODE": "native"})
        with self.assertRaisesRegex(
            ValueError,
            "MILVUS_MEMORY_CONSOLIDATION_JOURNAL_COLLECTION_NAME",
        ):
            build_milvus_workflow(
                {"MILVUS_MEMORY_CONSOLIDATION_JOURNAL_COLLECTION_NAME": " "}
            )
        with self.assertRaisesRegex(ValueError, "MEMORY_SELECTOR"):
            build_milvus_workflow({"MEMORY_SELECTOR": "maybe"})
        with self.assertRaisesRegex(ValueError, "ambiguity band"):
            build_milvus_workflow({"MEMORY_SELECTOR_AMBIGUITY_MAX": "0.61"})
        with self.assertRaisesRegex(
            ValueError,
            "RESPONSE_CACHE_SIMILARITY_THRESHOLD",
        ):
            build_milvus_workflow({"RESPONSE_CACHE_SIMILARITY_THRESHOLD": "1.1"})


if __name__ == "__main__":
    unittest.main()
