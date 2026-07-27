from __future__ import annotations

import unittest

from agent_workshop_demo.retrieval import InMemoryHybridRetriever
from agent_workshop_demo.sample_data import load_kb_chunks


class RetrievalTests(unittest.TestCase):
    def test_hybrid_search_returns_citable_pipeline_chunks(self) -> None:
        retriever = InMemoryHybridRetriever(load_kb_chunks())
        results = retriever.search(
            "S3 文档同步流程 metadata chunking Milvus insertion",
            top_k=5,
            filters={
                "department": "engineering",
                "source_type": ["local", "s3"],
            },
            order_by=["updated_at desc", "priority desc"],
        )

        chunk_ids = [item.chunk.chunk_id for item in results]
        self.assertIn("doc_s3_sync_design_c003", chunk_ids)
        self.assertIn("doc_rag_arch_v1_p003_c002", chunk_ids)
        self.assertGreaterEqual(
            results[0].hybrid_score,
            results[-1].hybrid_score,
        )

    def test_aggregations_count_ui_fields(self) -> None:
        retriever = InMemoryHybridRetriever(load_kb_chunks())
        results = retriever.search(
            "Milvus image vector",
            top_k=20,
            filters={},
            order_by=[],
        )
        aggregations = retriever.aggregations(
            results,
            ["source_type", "doc_type", "department", "has_image_vector"],
        )

        self.assertIn("local", aggregations["source_type"])
        self.assertIn("image", aggregations["doc_type"])
        self.assertIn("engineering", aggregations["department"])
        self.assertIn("True", aggregations["has_image_vector"])

    def test_document_version_filters_isolate_current_and_exact_editions(
        self,
    ) -> None:
        retriever = InMemoryHybridRetriever(load_kb_chunks())

        current = retriever.search(
            "GO按钮",
            top_k=10,
            filters={"department": "product", "is_current": True},
        )
        historical = retriever.search(
            "GO按钮",
            top_k=10,
            filters={"department": "product", "doc_version": "v1"},
        )

        current_go = [
            item.chunk
            for item in current
            if item.chunk.doc_id == "doc_go_button_guide"
        ]
        historical_go = [
            item.chunk
            for item in historical
            if item.chunk.doc_id == "doc_go_button_guide"
        ]
        self.assertTrue(current_go)
        self.assertTrue(historical_go)
        self.assertEqual(
            {item.doc_version for item in current_go},
            {"v2"},
        )
        self.assertEqual(
            {item.doc_version for item in historical_go},
            {"v1"},
        )

    def test_search_rejects_unknown_or_invalid_filters(self) -> None:
        retriever = InMemoryHybridRetriever(load_kb_chunks())
        with self.assertRaisesRegex(ValueError, "Unsupported search filter"):
            retriever.search("Milvus", top_k=5, filters={"secret": "value"})
        with self.assertRaisesRegex(ValueError, "source_type"):
            retriever.search(
                "Milvus",
                top_k=5,
                filters={"source_type": ["unknown"]},
            )

    def test_search_rejects_invalid_top_k(self) -> None:
        retriever = InMemoryHybridRetriever(load_kb_chunks())
        with self.assertRaisesRegex(ValueError, "top_k"):
            retriever.search("Milvus", top_k=0)


if __name__ == "__main__":
    unittest.main()
