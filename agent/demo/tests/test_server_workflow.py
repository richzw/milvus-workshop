from __future__ import annotations

import unittest
from unittest.mock import patch

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


class ServerWorkflowTests(unittest.TestCase):
    def test_milvus_builder_connects_loads_and_injects_retriever(self) -> None:
        with (
            patch(
                "agent_workshop_demo.langgraph_workflow."
                "MilvusHybridRetriever.connect"
            ) as connect,
            patch(
                "agent_workshop_demo.langgraph_workflow."
                "MilvusConversationMemoryStore"
            ) as memory_store_class,
        ):
            retriever = connect.return_value

            workflow = build_milvus_workflow(
                {
                    "MILVUS_URI": "http://milvus.test:19530",
                    "MILVUS_TOKEN": "test-token",
                    "MILVUS_COLLECTION_NAME": "workshop_chunks",
                    "MILVUS_MEMORY_COLLECTION_NAME": "workshop_memory",
                    "MEMORY_TOP_K": "4",
                    "MEMORY_TTL_SECONDS": "3600",
                }
            )

        connect.assert_called_once_with(
            "http://milvus.test:19530",
            "test-token",
            collection_name="workshop_chunks",
        )
        retriever.ensure_collection_ready.assert_called_once_with()
        memory_store_class.assert_called_once_with(
            retriever.client,
            collection_name="workshop_memory",
        )
        memory_store = memory_store_class.return_value
        memory_store.ensure_collection_ready.assert_called_once_with()
        configured_workflow = (
            workflow
            if isinstance(workflow, AgenticRAGWorkflow)
            else workflow.workflow
        )
        self.assertIs(configured_workflow.retriever, retriever)
        self.assertIs(
            configured_workflow.memory_store,
            memory_store,
        )
        self.assertEqual(configured_workflow.memory_top_k, 4)
        self.assertEqual(configured_workflow.memory_ttl_seconds, 3600)

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
            "agent_workshop_demo.langgraph_workflow."
            "MilvusHybridRetriever.connect",
            side_effect=OSError("connection refused"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "Unable to initialize the Milvus retriever",
            ) as captured:
                build_milvus_workflow()

        self.assertIsInstance(captured.exception.__cause__, OSError)

    def test_milvus_builder_rejects_invalid_memory_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "MEMORY_TOP_K"):
            build_milvus_workflow({"MEMORY_TOP_K": "0"})
        with self.assertRaisesRegex(ValueError, "MEMORY_TTL_SECONDS"):
            build_milvus_workflow({"MEMORY_TTL_SECONDS": "forever"})


if __name__ == "__main__":
    unittest.main()
