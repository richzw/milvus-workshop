from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

from agent_workshop_demo.ingestion import ingest_demo_sources
from agent_workshop_demo.sample_data import load_kb_chunks
from agent_workshop_demo.schema.collections import (
    CONVERSATION_MEMORY_COLLECTION,
    DOC_DEDUP_SIGNATURES_COLLECTION,
    KB_CHUNKS_COLLECTION,
)
from agent_workshop_demo.schema.pymilvus_adapter import MilvusHybridRetriever
from agent_workshop_demo.schema.pymilvus_adapter import (
    create_collections,
    create_indexes,
    drop_demo_collections,
)


class RecordingIndexParams:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def add_index(self, **kwargs: Any) -> None:
        self.requests.append(kwargs)


class RecordingMilvusClient:
    def __init__(self) -> None:
        self.collections = {
            "kb_chunks",
            "conversation_memory",
            "doc_dedup_signatures",
        }
        self.deleted: list[dict[str, Any]] = []
        self.inserted: list[dict[str, Any]] = []
        self.flushed: list[str] = []
        self.loaded: list[str] = []
        self.search_calls: list[dict[str, Any]] = []
        self.search_results: dict[str, list[list[dict[str, Any]]]] = {}
        self.query_results: list[dict[str, Any]] | None = None

    def has_collection(self, *, collection_name: str) -> bool:
        return collection_name in self.collections

    def delete(self, **kwargs: Any) -> dict[str, int]:
        self.deleted.append(kwargs)
        return {"delete_count": 0}

    def insert(self, **kwargs: Any) -> dict[str, Any]:
        self.inserted.append(kwargs)
        return {"insert_count": len(kwargs["data"])}

    def search(self, **kwargs: Any) -> list[list[dict[str, Any]]]:
        self.search_calls.append(kwargs)
        return self.search_results[kwargs["anns_field"]]

    def flush(self, *, collection_name: str) -> None:
        self.flushed.append(collection_name)

    def load_collection(self, *, collection_name: str) -> None:
        self.loaded.append(collection_name)

    def query(self, **kwargs: Any) -> list[dict[str, Any]]:
        if self.query_results is not None:
            return self.query_results
        expected = {
            record["chunk_id"]
            for call in self.inserted
            for record in call["data"]
        }
        return [{"chunk_id": chunk_id} for chunk_id in sorted(expected)]


class RecordingTextEmbeddingProvider:
    name = "recording"

    def __init__(self) -> None:
        self.inputs: list[str] = []

    def embed(self, text: str, *, dimensions: int) -> list[float]:
        self.inputs.append(text)
        return [float(len(self.inputs))] + [0.0] * (dimensions - 1)

    def fingerprint(self, *, dimensions: int) -> str:
        return f"recording:test-model:{dimensions}"


class ProvisioningMilvusClient:
    def __init__(self) -> None:
        self.collections: set[str] = set()
        self.indexes: dict[str, set[str]] = {}

    def has_collection(self, *, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_collection(self, *, collection_name: str, schema: Any) -> None:
        self.collections.add(collection_name)
        self.indexes.setdefault(collection_name, set())

    def drop_collection(self, *, collection_name: str) -> None:
        self.collections.discard(collection_name)
        self.indexes.pop(collection_name, None)

    def prepare_index_params(self) -> RecordingIndexParams:
        return RecordingIndexParams()

    def create_index(
        self,
        *,
        collection_name: str,
        index_params: RecordingIndexParams,
        sync: bool,
    ) -> None:
        self.assert_sync(sync)
        self.indexes[collection_name].update(
            str(item["index_name"]) for item in index_params.requests
        )

    def list_indexes(self, *, collection_name: str) -> list[str]:
        return sorted(self.indexes[collection_name])

    def drop_index(
        self, *, collection_name: str, index_name: str
    ) -> None:
        self.indexes[collection_name].discard(index_name)

    @staticmethod
    def assert_sync(sync: bool) -> None:
        if not sync:
            raise AssertionError("index creation must be synchronous")


class SchemaTests(unittest.TestCase):
    def test_ingestion_embeds_chunks_before_milvus_insert(self) -> None:
        provider = RecordingTextEmbeddingProvider()
        client = RecordingMilvusClient()

        with patch(
            "agent_workshop_demo.embedding."
            "_default_text_embedding_provider",
            return_value=provider,
        ):
            result = ingest_demo_sources(
                Path("demo/sample_data/local_docs"),
                Path("demo/sample_data/mock_s3"),
            )
            chunks = result.kb_chunks[:2]
            MilvusHybridRetriever(client).insert(chunks)

        self.assertEqual(len(provider.inputs), len(result.kb_chunks))
        inserted_records = client.inserted[0]["data"]
        self.assertEqual(
            [record["text_vector"] for record in inserted_records],
            [chunk.text_vector for chunk in chunks],
        )
        self.assertTrue(
            all(
                record["metadata"]["text_embedding_fingerprint"]
                == "recording:test-model:1024"
                for record in inserted_records
            )
        )

    def test_milvus_adapter_inserts_replaceable_kb_chunk_batches(self) -> None:
        client = RecordingMilvusClient()
        adapter = MilvusHybridRetriever(client, batch_size=2)
        chunks = load_kb_chunks()[:3]

        result = adapter.insert(chunks)

        self.assertEqual(result, {"insert_count": 3, "batch_count": 2})
        self.assertEqual(len(client.deleted), 2)
        self.assertEqual(len(client.inserted), 2)
        self.assertEqual(client.flushed, ["kb_chunks"])
        self.assertEqual(client.loaded, ["kb_chunks"])
        first_record = client.inserted[0]["data"][0]
        self.assertNotIn("id", first_record)
        self.assertEqual(first_record["chunk_id"], chunks[0].chunk_id)
        self.assertTrue(first_record["sparse_vector"])
        self.assertTrue(
            all(
                isinstance(index, int)
                for index in first_record["sparse_vector"]
            )
        )
        self.assertEqual(
            first_record["metadata"]["text_embedding_fingerprint"],
            "deterministic:sha256-token-v1:1024",
        )
        records = [
            record
            for call in client.inserted
            for record in call["data"]
        ]
        self.assertTrue(
            any("image_vector" not in record for record in records)
        )

    def test_milvus_adapter_rejects_mismatched_embedding_space(self) -> None:
        client = RecordingMilvusClient()
        chunk = load_kb_chunks()[0]
        mismatched = replace(
            chunk,
            metadata={
                **(chunk.metadata or {}),
                "text_embedding_fingerprint": "openai:other-model:1024",
            },
        )

        with self.assertRaisesRegex(ValueError, "embedding fingerprint"):
            MilvusHybridRetriever(client).insert([mismatched])

        self.assertEqual(client.inserted, [])

    def test_milvus_adapter_rejects_unsafe_current_edition_replacement(
        self,
    ) -> None:
        client = RecordingMilvusClient()
        current = next(
            chunk
            for chunk in load_kb_chunks()
            if chunk.doc_id == "doc_go_button_guide" and chunk.is_current
        )
        client.query_results = [
            {
                "doc_id": current.doc_id,
                "doc_version": current.doc_version,
            }
        ]
        replacement = replace(
            current,
            chunk_id="doc_go_button_guide_v3_c001",
            doc_version="v3",
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "Incremental current-edition replacement is unsafe",
        ):
            MilvusHybridRetriever(client).insert([replacement])

        self.assertEqual(client.deleted, [])
        self.assertEqual(client.inserted, [])

    def test_milvus_adapter_returns_normalized_hybrid_results(self) -> None:
        client = RecordingMilvusClient()
        chunk = load_kb_chunks()[0]
        entity = chunk.to_dict()
        client.search_results = {
            "text_vector": [[{"id": 7, "distance": 0.91, "entity": entity}]],
            "sparse_vector": [
                [{"id": 7, "distance": 0.82, "entity": entity}]
            ],
        }
        adapter = MilvusHybridRetriever(client)

        results = adapter.search(
            "S3 文档同步",
            top_k=5,
            filters={
                "source_type": ["local", "s3"],
                "doc_version": "v1",
                "is_current": True,
            },
            order_by=["updated_at desc"],
        )

        self.assertEqual(len(client.search_calls), 2)
        self.assertEqual(
            {call["anns_field"] for call in client.search_calls},
            {"text_vector", "sparse_vector"},
        )
        self.assertTrue(
            all(
                call["filter"]
                == (
                    'source_type in ["local", "s3"] '
                    'and doc_version == "v1" and is_current == true'
                )
                for call in client.search_calls
            )
        )
        sparse_call = next(
            call
            for call in client.search_calls
            if call["anns_field"] == "sparse_vector"
        )
        self.assertTrue(
            all(isinstance(index, int) for index in sparse_call["data"][0])
        )
        self.assertEqual(
            sparse_call["search_params"]["metric_type"],
            "IP",
        )
        self.assertNotIn("text_vector", sparse_call["output_fields"])
        self.assertNotIn("sparse_vector", sparse_call["output_fields"])
        self.assertNotIn("image_vector", sparse_call["output_fields"])
        self.assertEqual(results[0].chunk.chunk_id, chunk.chunk_id)
        self.assertEqual(results[0].rank, 1)
        self.assertGreater(results[0].hybrid_score, 0.9)
        self.assertEqual(
            adapter.aggregations(results, ["source_type"]),
            {"source_type": {chunk.source_type: 1}},
        )

    def test_milvus_search_rejects_stored_embedding_mismatch(self) -> None:
        client = RecordingMilvusClient()
        chunk = load_kb_chunks()[0]
        entity = chunk.to_dict()
        entity["metadata"] = {
            **(chunk.metadata or {}),
            "text_embedding_fingerprint": "openai:other-model:1024",
        }
        hit = {"id": 7, "distance": 0.91, "entity": entity}
        client.search_results = {
            "text_vector": [[hit]],
            "sparse_vector": [[hit]],
        }

        with self.assertRaisesRegex(ValueError, "embedding fingerprint"):
            MilvusHybridRetriever(client).search("query", top_k=1)

    def test_milvus_reader_rejects_legacy_rows_without_version_fields(
        self,
    ) -> None:
        client = RecordingMilvusClient()
        entity = load_kb_chunks()[0].to_dict()
        entity.pop("doc_version")
        entity.pop("is_current")
        hit = {"id": 7, "distance": 0.91, "entity": entity}
        client.search_results = {
            "text_vector": [[hit]],
            "sparse_vector": [[hit]],
        }

        with self.assertRaises(KeyError):
            MilvusHybridRetriever(client).search("query", top_k=1)

    def test_kb_chunks_contains_required_multimodal_fields(self) -> None:
        fields = {
            field["name"]: field
            for field in KB_CHUNKS_COLLECTION["fields"]
        }

        self.assertEqual(fields["text_vector"]["type"], "FloatVector")
        self.assertEqual(
            fields["sparse_vector"]["type"],
            "SparseFloatVector",
        )
        self.assertTrue(fields["image_vector"]["nullable"])
        self.assertIn("has_image_vector", fields)
        self.assertIn("updated_at", fields)
        self.assertIn("priority", fields)
        self.assertFalse(fields["doc_version"]["nullable"])
        self.assertFalse(fields["is_current"]["nullable"])

    def test_all_requested_collections_are_named(self) -> None:
        self.assertEqual(
            KB_CHUNKS_COLLECTION["collection_name"],
            "kb_chunks",
        )
        self.assertEqual(
            CONVERSATION_MEMORY_COLLECTION["collection_name"],
            "conversation_memory",
        )
        self.assertEqual(
            DOC_DEDUP_SIGNATURES_COLLECTION["collection_name"],
            "doc_dedup_signatures",
        )

    def test_sample_data_has_nullable_image_vectors(self) -> None:
        chunks = load_kb_chunks()

        self.assertTrue(
            any(item.has_image_vector and item.image_vector for item in chunks)
        )
        self.assertTrue(
            any(
                not item.has_image_vector and item.image_vector is None
                for item in chunks
            )
        )

    def test_inserted_chunks_are_read_back_from_milvus(self) -> None:
        client = RecordingMilvusClient()
        chunks = load_kb_chunks()[:3]
        adapter = MilvusHybridRetriever(client, batch_size=2)

        adapter.insert(chunks)
        verified = adapter.verify_inserted(
            chunk.chunk_id for chunk in chunks
        )

        self.assertEqual(verified, 3)
        self.assertEqual(client.loaded, ["kb_chunks", "kb_chunks"])

    def test_create_collections_creates_and_verifies_all_schemas(self) -> None:
        client = ProvisioningMilvusClient()

        report = create_collections("unused", client=client)

        self.assertEqual(
            set(report["created"]),
            {
                "kb_chunks",
                "conversation_memory",
                "doc_dedup_signatures",
            },
        )
        self.assertEqual(set(report["verified"]), client.collections)

    def test_cleanup_drops_only_demo_collections_and_is_idempotent(self) -> None:
        client = ProvisioningMilvusClient()
        client.collections.update(
            {
                "kb_chunks",
                "conversation_memory",
                "doc_dedup_signatures",
                "unrelated_application_data",
            }
        )

        first = drop_demo_collections("unused", client=client)
        second = drop_demo_collections("unused", client=client)

        self.assertEqual(
            set(first["dropped"]),
            {
                "kb_chunks",
                "conversation_memory",
                "doc_dedup_signatures",
            },
        )
        self.assertEqual(first["already_absent"], [])
        self.assertEqual(second["dropped"], [])
        self.assertEqual(
            set(second["already_absent"]),
            set(first["targeted"]),
        )
        self.assertEqual(client.collections, {"unrelated_application_data"})

    def test_create_indexes_creates_and_verifies_server_indexes(self) -> None:
        client = ProvisioningMilvusClient()
        create_collections("unused", client=client)

        report = create_indexes("unused", client=client)

        self.assertIn("text_vector_idx", client.indexes["kb_chunks"])
        self.assertIn("sparse_vector_idx", client.indexes["kb_chunks"])
        self.assertIn("chunk_id_idx", client.indexes["kb_chunks"])
        self.assertIn(
            "minhash_signature_idx",
            client.indexes["doc_dedup_signatures"],
        )
        self.assertEqual(
            set(report["kb_chunks"]["verified"]),
            client.indexes["kb_chunks"],
        )


if __name__ == "__main__":
    unittest.main()
