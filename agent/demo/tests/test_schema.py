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
    CONVERSATION_MEMORY_INDEXES,
    DOC_DEDUP_SIGNATURES_COLLECTION,
    GROUNDED_RESPONSE_CACHE_COLLECTION,
    KB_CHUNKS_COLLECTION,
    KB_DOCUMENTS_COLLECTION,
    MEMORY_EVENTS_COLLECTION,
    MEMORY_FACTS_COLLECTION,
    MEMORY_CONSOLIDATION_JOURNAL_COLLECTION,
)
from agent_workshop_demo.schema.pymilvus_adapter import MilvusHybridRetriever
from agent_workshop_demo.schema.pymilvus_adapter import (
    _index_requests,
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
        self.query_calls: list[dict[str, Any]] = []
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
        self.query_calls.append(kwargs)
        if self.query_results is not None:
            return self.query_results
        expected = {
            record["chunk_id"] for call in self.inserted for record in call["data"]
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
        self.index_descriptions: dict[str, dict[str, dict[str, Any]]] = {}
        self.collection_calls: list[dict[str, Any]] = []

    def has_collection(self, *, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_collection(self, **kwargs: Any) -> None:
        collection_name = str(kwargs["collection_name"])
        self.collection_calls.append(kwargs)
        self.collections.add(collection_name)
        self.indexes.setdefault(collection_name, set())
        self.index_descriptions.setdefault(collection_name, {})

    def drop_collection(self, *, collection_name: str) -> None:
        self.collections.discard(collection_name)
        self.indexes.pop(collection_name, None)
        self.index_descriptions.pop(collection_name, None)

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
        for item in index_params.requests:
            self.index_descriptions[collection_name][str(item["index_name"])] = {
                **item,
                "state": "Finished",
            }

    def list_indexes(self, *, collection_name: str) -> list[str]:
        return sorted(self.indexes[collection_name])

    def drop_index(self, *, collection_name: str, index_name: str) -> None:
        self.indexes[collection_name].discard(index_name)
        self.index_descriptions[collection_name].pop(index_name, None)

    def describe_index(
        self,
        *,
        collection_name: str,
        index_name: str,
    ) -> dict[str, Any]:
        return dict(self.index_descriptions[collection_name][index_name])

    @staticmethod
    def assert_sync(sync: bool) -> None:
        if not sync:
            raise AssertionError("index creation must be synchronous")


class SchemaTests(unittest.TestCase):
    def test_ingestion_embeds_chunks_before_milvus_insert(self) -> None:
        provider = RecordingTextEmbeddingProvider()
        client = RecordingMilvusClient()

        with patch(
            "agent_workshop_demo.embedding._default_text_embedding_provider",
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
        self.assertNotIn("sparse_vector", first_record)
        self.assertIn(chunks[0].title, first_record["retrieval_text"])
        self.assertEqual(
            first_record["metadata"]["text_embedding_fingerprint"],
            "deterministic:sha256-token-v1:1024",
        )
        records = [record for call in client.inserted for record in call["data"]]
        self.assertTrue(any("image_vector" not in record for record in records))

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

    def test_milvus_rejects_existing_image_vector_space_before_mutation(
        self,
    ) -> None:
        client = RecordingMilvusClient()
        image_chunk = next(
            chunk for chunk in load_kb_chunks() if chunk.has_image_vector
        )
        client.query_results = [
            {
                "metadata": {
                    "image_embedding_fingerprint": (
                        "dinov3:other-model:pooler_output:l2:768"
                    )
                }
            }
        ]

        with self.assertRaisesRegex(
            RuntimeError,
            "image embedding-space replacement is unsafe",
        ):
            MilvusHybridRetriever(client).insert([image_chunk])

        self.assertEqual(client.deleted, [])
        self.assertEqual(client.inserted, [])
        self.assertEqual(
            client.query_calls[0]["filter"],
            "has_image_vector == true",
        )

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
            "sparse_vector": [[{"id": 7, "distance": 0.82, "entity": entity}]],
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
        self.assertEqual(sparse_call["data"], ["S3 文档同步"])
        self.assertEqual(
            sparse_call["search_params"]["metric_type"],
            "BM25",
        )
        self.assertNotIn("text_vector", sparse_call["output_fields"])
        self.assertNotIn("sparse_vector", sparse_call["output_fields"])
        self.assertIn("image_vector", sparse_call["output_fields"])
        self.assertEqual(results[0].chunk.chunk_id, chunk.chunk_id)
        self.assertEqual(results[0].rank, 1)
        self.assertGreater(results[0].hybrid_score, 0.9)
        client.query_results = [{"source_type": chunk.source_type, "count(*)": 1}]
        self.assertEqual(
            adapter.aggregations(results, ["source_type"]),
            {"source_type": {chunk.source_type: 1}},
        )

    def test_milvus_text_search_materializes_image_hits(self) -> None:
        client = RecordingMilvusClient()
        image_chunk = next(
            chunk for chunk in load_kb_chunks() if chunk.has_image_vector
        )
        hit = {
            "id": 12,
            "distance": 0.88,
            "entity": image_chunk.to_dict(),
        }
        client.search_results = {
            "text_vector": [[hit]],
            "sparse_vector": [[hit]],
        }

        results = MilvusHybridRetriever(client).search(
            "同步架构图",
            top_k=1,
        )

        self.assertEqual(results[0].chunk.chunk_id, image_chunk.chunk_id)
        self.assertEqual(
            results[0].chunk.image_vector,
            image_chunk.image_vector,
        )
        self.assertTrue(
            all("image_vector" in call["output_fields"] for call in client.search_calls)
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

    def test_milvus_document_expansion_uses_authorized_exact_family_query(
        self,
    ) -> None:
        client = RecordingMilvusClient()
        chunk = load_kb_chunks()[0]
        client.query_results = [chunk.to_dict()]

        chunks = MilvusHybridRetriever(client).fetch_document_chunks(
            doc_id=chunk.doc_id,
            doc_version=chunk.doc_version,
            filters={
                "department": chunk.department,
                "is_current": chunk.is_current,
            },
            limit=20,
        )

        self.assertEqual([item.chunk_id for item in chunks], [chunk.chunk_id])
        call = client.query_calls[-1]
        self.assertEqual(call["collection_name"], "kb_chunks")
        self.assertEqual(call["limit"], 20)
        self.assertIn(f'doc_id == "{chunk.doc_id}"', call["filter"])
        self.assertIn(
            f'doc_version == "{chunk.doc_version}"',
            call["filter"],
        )
        self.assertIn(
            f'department == "{chunk.department}"',
            call["filter"],
        )
        self.assertIn(
            f"is_current == {str(chunk.is_current).lower()}",
            call["filter"],
        )

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
        fields = {field["name"]: field for field in KB_CHUNKS_COLLECTION["fields"]}

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

    def test_grounded_response_cache_has_stable_scope_and_evidence(self) -> None:
        fields = {
            field["name"]: field
            for field in GROUNDED_RESPONSE_CACHE_COLLECTION["fields"]
        }

        self.assertTrue(fields["cache_id"]["primary_key"])
        self.assertFalse(fields["cache_id"]["auto_id"])
        self.assertEqual(fields["query_vector"]["dim"], 1024)
        self.assertEqual(fields["answer"]["max_length"], 12000)
        for name in (
            "version_scope",
            "entity_ids",
            "query_constraints",
            "citations",
            "evidence",
        ):
            self.assertEqual(fields[name]["type"], "JSON")
            self.assertFalse(fields[name]["nullable"])

    def test_all_requested_collections_are_named(self) -> None:
        self.assertEqual(
            KB_CHUNKS_COLLECTION["collection_name"],
            "kb_chunks",
        )
        self.assertEqual(KB_DOCUMENTS_COLLECTION["collection_name"], "kb_documents")
        struct_fields = KB_DOCUMENTS_COLLECTION["fields"][-1]
        self.assertEqual(struct_fields["element_type"], "Struct")
        self.assertEqual(struct_fields["max_capacity"], 1024)
        self.assertEqual(
            CONVERSATION_MEMORY_COLLECTION["collection_name"],
            "conversation_memory",
        )
        self.assertEqual(
            DOC_DEDUP_SIGNATURES_COLLECTION["collection_name"],
            "doc_dedup_signatures",
        )
        self.assertEqual(
            MEMORY_EVENTS_COLLECTION["collection_name"],
            "memory_events",
        )
        self.assertEqual(
            MEMORY_FACTS_COLLECTION["collection_name"],
            "memory_facts",
        )
        self.assertEqual(
            MEMORY_CONSOLIDATION_JOURNAL_COLLECTION["collection_name"],
            "memory_consolidation_journal",
        )
        journal_fields = {
            field["name"]: field
            for field in MEMORY_CONSOLIDATION_JOURNAL_COLLECTION["fields"]
        }
        self.assertEqual(journal_fields["journal_anchor_vector"]["type"], "FloatVector")
        self.assertEqual(journal_fields["journal_anchor_vector"]["dim"], 2)

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
        verified = adapter.verify_inserted(chunk.chunk_id for chunk in chunks)

        self.assertEqual(verified, 3)
        self.assertEqual(client.loaded, ["kb_chunks", "kb_chunks"])

    def test_create_collections_creates_and_verifies_all_schemas(self) -> None:
        client = ProvisioningMilvusClient()

        report = create_collections("unused", client=client)

        self.assertEqual(
            set(report["created"]),
            {
                "kb_chunks",
                "kb_documents",
                "conversation_memory",
                "grounded_response_cache",
                "doc_dedup_signatures",
                "memory_events",
                "memory_facts",
                "memory_consolidation_journal",
            },
        )
        self.assertEqual(set(report["verified"]), client.collections)
        ttl_calls = [
            call
            for call in client.collection_calls
            if call.get("properties") == {"ttl_field": "expires_at"}
        ]
        self.assertEqual(len(ttl_calls), 4)

    def test_cleanup_drops_only_demo_collections_and_is_idempotent(self) -> None:
        client = ProvisioningMilvusClient()
        client.collections.update(
            {
                "kb_chunks",
                "kb_documents",
                "conversation_memory",
                "grounded_response_cache",
                "doc_dedup_signatures",
                "memory_events",
                "memory_facts",
                "memory_consolidation_journal",
                "unrelated_application_data",
            }
        )

        first = drop_demo_collections("unused", client=client)
        second = drop_demo_collections("unused", client=client)

        self.assertEqual(
            set(first["dropped"]),
            {
                "kb_chunks",
                "kb_documents",
                "conversation_memory",
                "grounded_response_cache",
                "doc_dedup_signatures",
                "memory_events",
                "memory_facts",
                "memory_consolidation_journal",
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
            "passages_element_cosine_idx",
            client.indexes["kb_documents"],
        )
        self.assertIn(
            "passages_embedding_list_maxsim_idx",
            client.indexes["kb_documents"],
        )
        self.assertIn(
            "minhash_signature_idx",
            client.indexes["doc_dedup_signatures"],
        )
        self.assertIn(
            "content_vector_idx",
            client.indexes["memory_events"],
        )
        self.assertIn(
            "content_vector_idx",
            client.indexes["memory_facts"],
        )
        self.assertIn(
            "journal_anchor_vector_idx",
            client.indexes["memory_consolidation_journal"],
        )
        self.assertEqual(
            set(report["kb_chunks"]["verified"]),
            client.indexes["kb_chunks"],
        )

    def test_timestamptz_uses_stl_sort_not_inverted(self) -> None:
        requests = {
            request["field_name"]: request
            for request in _index_requests(CONVERSATION_MEMORY_INDEXES)
        }

        self.assertEqual(requests["expires_at"]["index_type"], "STL_SORT")
        self.assertEqual(requests["session_id"]["index_type"], "INVERTED")
        with self.assertRaisesRegex(ValueError, "Scalar index definition"):
            _index_requests({"scalar_indexes": [{"field_name": "expires_at"}]})

    def test_existing_index_type_mismatch_fails_closed(self) -> None:
        client = ProvisioningMilvusClient()
        create_collections("unused", client=client)
        create_indexes("unused", client=client)
        client.index_descriptions["conversation_memory"]["expires_at_idx"][
            "index_type"
        ] = "INVERTED"

        with self.assertRaisesRegex(RuntimeError, "rerun with --recreate"):
            create_indexes("unused", client=client)


if __name__ == "__main__":
    unittest.main()
