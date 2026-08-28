from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any

from agent_workshop_demo.ingestion import IngestionResult, ingest_demo_sources
from agent_workshop_demo.models import KBChunk, SearchResult
from agent_workshop_demo.retrieval import InMemoryHybridRetriever
from agent_workshop_demo.sample_data import load_kb_chunks
from agent_workshop_demo.schema.collections import KB_DOCUMENTS_COLLECTION
from agent_workshop_demo.struct_array import (
    ELEMENT_INDEX_NAME,
    EMBEDDING_LIST_INDEX_NAME,
    ElementPredicate,
    InMemoryStructArrayRetriever,
    MilvusStructArrayRetriever,
    MilvusStructArrayStore,
    ProjectionBuild,
    ProjectionManifest,
    StructArrayProfile,
    build_struct_array_projection,
    document_key,
    element_filter_expression,
    fuse_struct_and_bm25,
    load_projection_manifest,
    match_any_expression,
    runtime_config_from_mapping,
)
from agent_workshop_demo.workflow import AgenticRAGWorkflow


class NativeSearchClient:
    def __init__(self, hit: dict[str, Any]) -> None:
        self.hit = hit
        self.calls: list[dict[str, Any]] = []

    def search(self, **kwargs: Any) -> list[list[dict[str, Any]]]:
        self.calls.append(kwargs)
        return [[self.hit]]


class EmptyNativeSearchClient(NativeSearchClient):
    def __init__(self) -> None:
        super().__init__({})

    def search(self, **kwargs: Any) -> list[list[dict[str, Any]]]:
        self.calls.append(kwargs)
        return [[]]


class MatchAnyClient(NativeSearchClient):
    def __init__(self, row: dict[str, Any]) -> None:
        super().__init__({})
        self.row = row

    def query(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(kwargs)
        return [self.row]


class ReadyStoreClient:
    def __init__(self, build: ProjectionBuild) -> None:
        self.build = build
        self.inserted: list[dict[str, Any]] = []
        self.deleted: list[dict[str, Any]] = []

    def has_collection(self, *, collection_name: str) -> bool:
        return collection_name == "kb_documents"

    def list_indexes(self, *, collection_name: str) -> list[str]:
        return [ELEMENT_INDEX_NAME, EMBEDDING_LIST_INDEX_NAME]

    def describe_collection(self, *, collection_name: str) -> dict[str, Any]:
        def field_type(value: object) -> str:
            return {
                "FLOATVECTOR": "FLOAT_VECTOR",
                "VARCHAR": "VARCHAR",
            }.get(str(value).upper(), str(value).upper())

        fields: list[dict[str, Any]] = []
        for definition in KB_DOCUMENTS_COLLECTION["fields"]:
            field: dict[str, Any] = {
                "name": definition["name"],
                "type": field_type(definition["type"]),
                "params": {
                    key: definition[key]
                    for key in ("max_length", "dim", "max_capacity")
                    if key in definition
                },
                "is_primary": definition.get("primary_key", False),
            }
            if definition["type"] == "Array":
                field["element_type"] = "STRUCT"
                field["struct_fields"] = [
                    {
                        "name": item["name"],
                        "type": field_type(item["type"]),
                        "params": {
                            key: item[key]
                            for key in ("max_length", "dim")
                            if key in item
                        },
                    }
                    for item in definition["struct_fields"]
                ]
            fields.append(field)
        return {"fields": fields}

    def describe_index(
        self, *, collection_name: str, index_name: str
    ) -> dict[str, Any]:
        if index_name == ELEMENT_INDEX_NAME:
            return {
                "field_name": "passages[element_vector]",
                "index_type": "HNSW",
                "metric_type": "COSINE",
            }
        return {
            "field_name": "passages[embedding_list_vector]",
            "index_type": "HNSW",
            "metric_type": "MAX_SIM_COSINE",
            "emb_list_strategy": "tokenann",
        }

    def load_collection(self, *, collection_name: str) -> None:
        return None

    def delete(self, **kwargs: Any) -> dict[str, int]:
        self.deleted.append(kwargs)
        return {"delete_count": 0}

    def insert(self, **kwargs: Any) -> dict[str, int]:
        self.inserted.extend(kwargs["data"])
        return {"insert_count": len(kwargs["data"])}

    def flush(self, *, collection_name: str) -> None:
        return None

    def query(self, **kwargs: Any) -> list[dict[str, Any]]:
        if "!=" in kwargs["filter"]:
            return []
        if kwargs["output_fields"] == ["document_key", "passages"]:
            return [
                {
                    "document_key": parent.document_key,
                    "passages": [passage.to_record() for passage in parent.passages],
                }
                for parent in self.build.parents
            ]
        return [
            {
                "projection_fingerprint": parent.projection_fingerprint,
                "projection_parent_count": parent.projection_parent_count,
                "projection_passage_count": parent.projection_passage_count,
                "passage_count": parent.passage_count,
                "text_embedding_fingerprint": parent.text_embedding_fingerprint,
            }
            for parent in self.build.parents
        ]


class StructArrayTests(unittest.TestCase):
    ingestion: IngestionResult
    manifest: ProjectionManifest
    build: ProjectionBuild
    flat: InMemoryHybridRetriever

    @classmethod
    def setUpClass(cls) -> None:
        cls.ingestion = ingest_demo_sources(
            Path("demo/sample_data/local_docs"),
            Path("demo/sample_data/mock_s3"),
        )
        cls.manifest = load_projection_manifest()
        cls.build = build_struct_array_projection(
            cls.ingestion.kb_chunks,
            cls.manifest,
        )
        cls.flat = InMemoryHybridRetriever(cls.ingestion.kb_chunks)

    def test_projection_is_complete_ordered_and_fingerprinted(self) -> None:
        self.assertEqual(self.build.parent_count, 13)
        self.assertEqual(self.build.passage_count, 77)
        self.assertEqual(len(self.build.projection_fingerprint), 64)
        projected_ids = [
            passage.chunk_id
            for parent in self.build.parents
            for passage in parent.passages
        ]
        selected_ids = [
            chunk.chunk_id
            for chunk in self.ingestion.kb_chunks
            if chunk.doc_id in self.manifest.selected_doc_ids
        ]
        self.assertEqual(set(projected_ids), set(selected_ids))
        for parent in self.build.parents:
            self.assertEqual(
                parent.document_key, document_key(parent.doc_id, parent.doc_version)
            )
            self.assertEqual(parent.projection_parent_count, self.build.parent_count)
            self.assertEqual(parent.projection_passage_count, self.build.passage_count)
            self.assertEqual(
                [item.chunk_index for item in parent.passages],
                sorted(item.chunk_index for item in parent.passages),
            )
            self.assertTrue(
                all(
                    item.embedding_list_vector == item.element_vector
                    for item in parent.passages
                )
            )

    def test_projection_rejects_missing_mixed_and_oversize_inputs(self) -> None:
        missing = ProjectionManifest(
            "struct-array-projection-v1",
            ("missing-doc",),
            "negative fixture",
        )
        with self.assertRaisesRegex(ValueError, "missing document ids"):
            build_struct_array_projection(self.ingestion.kb_chunks, missing)

        base = self.ingestion.kb_chunks[0]
        mixed = replace(base, chunk_id=base.chunk_id + "_mixed", department="security")
        one_doc = ProjectionManifest(
            "struct-array-projection-v1",
            (base.doc_id,),
            "mixed parent fixture",
        )
        family = [
            chunk for chunk in self.ingestion.kb_chunks if chunk.doc_id == base.doc_id
        ]
        with self.assertRaisesRegex(ValueError, "mixed parent metadata"):
            build_struct_array_projection([*family, mixed], one_doc)

        oversize = [
            replace(base, chunk_id=f"oversize_{index}", chunk_index=index)
            for index in range(1025)
        ]
        with self.assertRaisesRegex(ValueError, "max_capacity"):
            build_struct_array_projection(oversize, one_doc)

    def test_element_expression_is_allow_listed_and_same_element(self) -> None:
        predicates = (
            ElementPredicate("record_type", "eq", "text_chunk"),
            ElementPredicate("chunk_index", "gte", 2),
        )
        self.assertEqual(
            element_filter_expression(predicates),
            'element_filter(passages, $[record_type] == "text_chunk" && $[chunk_index] >= 2)',
        )
        self.assertIn("MATCH_ANY(passages", match_any_expression(predicates))
        with self.assertRaisesRegex(
            ValueError, "Unsupported StructArray element field"
        ):
            ElementPredicate("department", "eq", "engineering")
        with self.assertRaisesRegex(ValueError, "At most four"):
            element_filter_expression(predicates * 3)

        local = InMemoryStructArrayRetriever(
            self.flat,
            self.build,
            profile=StructArrayProfile.ELEMENT,
        ).match_any_parents(predicates, filters={"department": "engineering"})
        self.assertTrue(local)
        self.assertTrue(all(item.rank > 0 for item in local))

    def test_same_element_predicates_do_not_match_across_offsets(self) -> None:
        target = next(
            parent for parent in self.build.parents if len(parent.passages) >= 2
        )
        passages = list(target.passages)
        passages[0] = replace(passages[0], record_type="wanted", language="other")
        passages[1] = replace(passages[1], record_type="other", language="wanted")
        changed_parent = replace(target, passages=tuple(passages))
        changed = replace(
            self.build,
            parents=tuple(
                changed_parent if parent.document_key == target.document_key else parent
                for parent in self.build.parents
            ),
        )
        self.assertTrue(any(item.record_type == "wanted" for item in passages))
        self.assertTrue(any(item.language == "wanted" for item in passages))
        matches = InMemoryStructArrayRetriever(
            self.flat, changed, profile=StructArrayProfile.ELEMENT
        ).match_any_parents(
            (
                ElementPredicate("record_type", "eq", "wanted"),
                ElementPredicate("language", "eq", "wanted"),
            )
        )
        self.assertNotIn(target.document_key, {item.document_key for item in matches})

    def test_runtime_config_defaults_disabled_and_gates_activation(self) -> None:
        disabled = runtime_config_from_mapping({})
        self.assertEqual(disabled.profile, StructArrayProfile.DISABLED)
        with self.assertRaisesRegex(ValueError, "PROJECTION_FINGERPRINT"):
            runtime_config_from_mapping({"STRUCT_ARRAY_RETRIEVAL": "struct_element"})
        enabled = runtime_config_from_mapping(
            {
                "STRUCT_ARRAY_RETRIEVAL": "struct_fused",
                "STRUCT_ARRAY_PROJECTION_FINGERPRINT": self.build.projection_fingerprint,
                "STRUCT_ARRAY_PARENT_TOP_K": "4",
            }
        )
        self.assertEqual(enabled.parent_top_k, 4)

    def test_local_element_two_stage_and_fused_return_passages_only(self) -> None:
        element = InMemoryStructArrayRetriever(
            self.flat,
            self.build,
            profile=StructArrayProfile.ELEMENT,
        )
        run = element.search_profile(
            ["S3 同步架构"],
            top_k=5,
            filters={"department": "engineering", "is_current": True},
        )
        self.assertTrue(run.results_by_query[0])
        self.assertTrue(
            all(item.element_offset is not None for item in run.results_by_query[0])
        )
        self.assertTrue(
            all(
                item.chunk.department == "engineering"
                for item in run.results_by_query[0]
            )
        )

        two_stage = InMemoryStructArrayRetriever(
            self.flat,
            self.build,
            profile=StructArrayProfile.TWO_STAGE,
            parent_top_k=3,
        ).search_profile(
            ["客户关心什么", "路线图计划"],
            top_k=5,
            filters={"department": "product", "is_current": True},
        )
        self.assertTrue(two_stage.document_candidates)
        self.assertTrue(
            all(
                item.element_offset is not None
                for group in two_stage.results_by_query
                for item in group
            )
        )

        struct_results = list(run.results_by_query[0])
        bm25 = self.flat.search_sparse(
            "S3 同步架构",
            top_k=5,
            filters={"department": "engineering", "is_current": True},
        )
        fused = fuse_struct_and_bm25(struct_results, bm25, top_k=5)
        self.assertTrue(fused)
        self.assertEqual(fused[0].retrieval_profile, "struct_fused")
        self.assertEqual(fused[0].fusion_recipe, "struct-rrf-v1")
        self.assertTrue(
            set(fused[0].retrieval_paths).issubset({"struct_element", "flat_bm25"})
        )

    def test_fused_lanes_reject_passage_identity_disagreement(self) -> None:
        """One chunk_id reached by two lanes must agree on edition and checksum."""

        chunks = load_kb_chunks()[:2]

        def result(chunk: KBChunk, rank: int, path: str) -> SearchResult:
            return SearchResult(
                chunk=chunk,
                rank=rank,
                dense_score=0.5,
                keyword_score=0.5,
                recency_score=0.0,
                priority_score=0.0,
                hybrid_score=0.5,
                retrieval_paths=(path,),
            )

        struct = [
            result(chunk, index, "struct_element")
            for index, chunk in enumerate(chunks, start=1)
        ]
        agreeing = [
            result(chunk, index, "flat_bm25")
            for index, chunk in enumerate(chunks, start=1)
        ]
        self.assertEqual(len(fuse_struct_and_bm25(struct, agreeing, top_k=5)), 2)

        for label, mutated in (
            ("edition", replace(chunks[0], doc_version="v9.9")),
            ("checksum", replace(chunks[0], checksum="0" * 64)),
        ):
            with self.subTest(disagreement=label):
                lexical = [
                    result(mutated, 1, "flat_bm25"),
                    result(chunks[1], 2, "flat_bm25"),
                ]
                with self.assertRaises(RuntimeError):
                    fuse_struct_and_bm25(struct, lexical, top_k=5)

    def test_native_element_filter_is_pushed_down_and_offset_revalidated(self) -> None:
        parent = next(
            item for item in self.build.parents if item.department == "engineering"
        )
        passage = parent.passages[0]
        hit = {
            "distance": 0.91,
            "offset": 0,
            "entity": {
                "document_key": parent.document_key,
                "doc_id": parent.doc_id,
                "doc_version": parent.doc_version,
                "department": parent.department,
                "is_current": parent.is_current,
                "text_embedding_fingerprint": parent.text_embedding_fingerprint,
                "projection_fingerprint": parent.projection_fingerprint,
                "passages": [passage.to_record()],
            },
        }
        client = NativeSearchClient(hit)
        config = runtime_config_from_mapping(
            {
                "STRUCT_ARRAY_RETRIEVAL": "struct_element",
                "STRUCT_ARRAY_PROJECTION_FINGERPRINT": self.build.projection_fingerprint,
            }
        )
        retriever = MilvusStructArrayRetriever(client, self.flat, config)
        results = retriever.search(
            "Milvus",
            top_k=1,
            filters={"department": "engineering", "is_current": True},
        )
        self.assertEqual(results[0].chunk.chunk_id, passage.chunk_id)
        expression = client.calls[0]["filter"]
        self.assertIn('department == "engineering"', expression)
        self.assertIn("is_current == true", expression)
        self.assertIn(self.build.projection_fingerprint, expression)

        missing_offset = dict(hit)
        missing_offset.pop("offset")
        with self.assertRaisesRegex(RuntimeError, "citeable offset"):
            MilvusStructArrayRetriever(
                NativeSearchClient(missing_offset), self.flat, config
            ).search("Milvus", top_k=1)

        drifted = json.loads(json.dumps(hit))
        drifted["entity"]["projection_fingerprint"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "identity failed"):
            MilvusStructArrayRetriever(
                NativeSearchClient(drifted), self.flat, config
            ).search("Milvus", top_k=1)

        match_client = MatchAnyClient(
            {
                "document_key": parent.document_key,
                "doc_id": parent.doc_id,
                "doc_version": parent.doc_version,
                "title": parent.title,
            }
        )
        matched = MilvusStructArrayRetriever(
            match_client, self.flat, config
        ).match_any_parents(
            [ElementPredicate("chunk_index", "gte", 0)],
            filters={"department": "engineering"},
        )
        self.assertEqual(matched[0].document_key, parent.document_key)
        self.assertIn("MATCH_ANY(passages", match_client.calls[0]["filter"])
        self.assertIn('department == "engineering"', match_client.calls[0]["filter"])

    def test_projection_store_requires_disabled_mode_and_exact_counts(self) -> None:
        client = ReadyStoreClient(self.build)
        store = MilvusStructArrayStore(client, batch_size=4)
        with self.assertRaisesRegex(RuntimeError, "requires.*disabled"):
            store.replace_projection(self.build, retrieval_mode="struct_element")
        report = store.replace_projection(self.build, retrieval_mode="disabled")
        self.assertEqual(report["parent_count"], self.build.parent_count)
        self.assertEqual(len(client.inserted), self.build.parent_count)
        self.assertEqual(client.deleted[0]["collection_name"], "kb_documents")

    def test_projection_preflight_rejects_invalid_bytes_before_delete(self) -> None:
        parent = self.build.parents[0]
        invalid = replace(
            self.build,
            parents=(replace(parent, title="界" * 513), *self.build.parents[1:]),
        )
        client = ReadyStoreClient(invalid)
        with self.assertRaisesRegex(ValueError, "UTF-8 bytes"):
            MilvusStructArrayStore(client).replace_projection(
                invalid, retrieval_mode="disabled"
            )
        self.assertEqual(client.deleted, [])
        self.assertEqual(client.inserted, [])

    def test_native_two_stage_empty_shortlist_does_not_search_elements(self) -> None:
        config = runtime_config_from_mapping(
            {
                "STRUCT_ARRAY_RETRIEVAL": "struct_two_stage",
                "STRUCT_ARRAY_PROJECTION_FINGERPRINT": self.build.projection_fingerprint,
            }
        )
        client = EmptyNativeSearchClient()
        run = MilvusStructArrayRetriever(client, self.flat, config).search_profile(
            ["aspect one", "aspect two"], top_k=3
        )
        self.assertEqual(run.results_by_query, ((), ()))
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(
            client.calls[0]["anns_field"], "passages[embedding_list_vector]"
        )

    def test_workflow_exposes_profile_without_treating_parents_as_evidence(
        self,
    ) -> None:
        retriever = InMemoryStructArrayRetriever(
            self.flat,
            self.build,
            profile=StructArrayProfile.TWO_STAGE,
        )
        response = AgenticRAGWorkflow(retriever=retriever).run(
            "Milvus 3.0 StructArray 有哪些检索方式和限制？"
        )

        self.assertEqual(response["retrieval_profile"], "struct_element")
        self.assertEqual(response["structarray_status"], "fallback_ineligible_group")
        self.assertEqual(response["document_candidates"], [])
        self.assertTrue(
            all(
                item["result_granularity"] == "passage"
                for item in response["milvus_recalled"]
            )
        )

        grouped = AgenticRAGWorkflow(retriever=retriever).run(
            "比较 Milvus StructArray 的 EmbeddingList 和 element search"
        )
        self.assertEqual(grouped["retrieval_profile"], "struct_two_stage")
        self.assertEqual(grouped["structarray_status"], "ready")
        self.assertGreater(len(grouped["document_candidates"]), 0)
        self.assertTrue(
            all(
                "resolved_element_count" in item and "resolved_to_evidence" in item
                for item in grouped["document_candidates"]
            )
        )
        candidate_keys = {
            item["document_key"] for item in grouped["document_candidates"]
        }
        self.assertTrue(
            all(
                item["document_key"] in candidate_keys
                and item["element_offset"] is not None
                for item in grouped["milvus_recalled"]
            )
        )

    def test_manifest_loader_rejects_unknown_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "struct-array-projection-v1",
                        "selected_doc_ids": ["x"],
                        "rationale": "test",
                        "extra": True,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "fields"):
                load_projection_manifest(path)


if __name__ == "__main__":
    unittest.main()
