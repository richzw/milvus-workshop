from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

from agent_workshop_demo.embedding import (
    DeterministicTextEmbeddingProvider,
)
from agent_workshop_demo.image_embedding import (
    IMAGE_EMBEDDING_FINGERPRINT_KEY,
    DeterministicImageEmbeddingProvider,
)
from agent_workshop_demo.image_eval import evaluate_image_retrieval
from agent_workshop_demo.image_retrieval import (
    image_only_filters,
    search_similar_images,
)
from agent_workshop_demo.ingestion import IngestionResult, ingest_demo_sources
from agent_workshop_demo.models import (
    ImageSearchResult,
    KBChunk,
    SearchResult,
)
from agent_workshop_demo.retrieval import InMemoryHybridRetriever
from agent_workshop_demo.schema.pymilvus_adapter import (
    MilvusHybridRetriever,
)

IMAGE_ROOT = Path("demo/sample_data/local_docs/images")
IMAGE_CASES = Path("demo/eval/image_retrieval.json")


class _ImageSearchClient:
    def __init__(self, entity: dict[str, Any]) -> None:
        self.entity = entity
        self.calls: list[dict[str, Any]] = []

    def search(self, **kwargs: Any) -> list[list[dict[str, Any]]]:
        self.calls.append(kwargs)
        return [
            [
                {
                    "id": 42,
                    "distance": 0.97,
                    "entity": self.entity,
                }
            ]
        ]


class _OverReturningRetriever:
    def __init__(self, text_results: list[SearchResult]) -> None:
        self.text_results = text_results

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
        order_by: list[str] | None = None,
    ) -> list[SearchResult]:
        return list(self.text_results)

    def search_image_vector(
        self,
        query_vector: list[float],
        *,
        image_fingerprint: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[ImageSearchResult]:
        return []


class ImageRetrievalTests(unittest.TestCase):
    provider: ClassVar[DeterministicImageEmbeddingProvider]
    ingestion: ClassVar[IngestionResult]
    image_chunks: ClassVar[list[KBChunk]]
    retriever: ClassVar[InMemoryHybridRetriever]

    @classmethod
    def setUpClass(cls) -> None:
        cls.provider = DeterministicImageEmbeddingProvider()
        with patch(
            "agent_workshop_demo.embedding."
            "_default_text_embedding_provider",
            return_value=DeterministicTextEmbeddingProvider(),
        ):
            cls.ingestion = ingest_demo_sources(
                Path("demo/sample_data/local_docs"),
                Path("demo/sample_data/mock_s3"),
                image_embedding_provider=cls.provider,
            )
        cls.image_chunks = [
            chunk
            for chunk in cls.ingestion.kb_chunks
            if chunk.has_image_vector
        ]
        cls.retriever = InMemoryHybridRetriever(
            cls.ingestion.kb_chunks
        )

    def test_local_image_search_returns_exact_image_without_vectors(self) -> None:
        results = search_similar_images(
            IMAGE_ROOT / "rag_architecture.png",
            retriever=self.retriever,
            provider=self.provider,
            top_k=3,
            filters={"department": "engineering"},
        )

        self.assertEqual(
            results[0].chunk.source_uri,
            "sample_data/local_docs/images/rag_architecture.png",
        )
        self.assertAlmostEqual(results[0].image_score, 1.0, places=6)
        self.assertTrue(all(item.chunk.has_image_vector for item in results))
        public = results[0].to_dict()
        self.assertEqual(public["retrieval_mode"], "image_vector")
        self.assertNotIn("image_vector", public)
        self.assertNotIn("text_vector", public)
        self.assertNotIn("sparse_vector", public)

    def test_local_image_search_preserves_negative_cosine_order(self) -> None:
        query = self.provider.embed(
            IMAGE_ROOT / "ingestion_pipeline.png",
            dimensions=768,
        )
        fingerprint = self.provider.fingerprint(dimensions=768)

        results = self.retriever.search_image_vector(
            query,
            image_fingerprint=fingerprint,
            top_k=5,
        )
        expected_scores = {
            chunk.source_uri: sum(
                left * right
                for left, right in zip(
                    query,
                    chunk.image_vector or [],
                    strict=True,
                )
            )
            for chunk in self.image_chunks
        }

        self.assertEqual(
            [item.image_score for item in results],
            sorted(expected_scores.values(), reverse=True),
        )
        negative_scores = [
            item.image_score
            for item in results
            if item.image_score < 0
        ]
        self.assertGreaterEqual(len(negative_scores), 2)
        self.assertGreater(len(set(negative_scores)), 1)

    def test_image_search_filters_and_vectors_fail_closed(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "has_image_vector=false",
        ):
            image_only_filters({"has_image_vector": False})
        with (
            patch.object(
                self.provider,
                "embed",
                wraps=self.provider.embed,
            ) as embed,
            self.assertRaisesRegex(
                ValueError,
                "has_image_vector=false",
            ),
        ):
            search_similar_images(
                IMAGE_ROOT / "s3_sync_flow.png",
                retriever=self.retriever,
                provider=self.provider,
                top_k=3,
                filters={"has_image_vector": False},
            )
        embed.assert_not_called()
        with self.assertRaisesRegex(
            ValueError,
            "finite L2-normalized",
        ):
            self.retriever.search_image_vector(
                [0.0] * 768,
                image_fingerprint=self.provider.fingerprint(
                    dimensions=768
                ),
                top_k=3,
            )
        with self.assertRaisesRegex(ValueError, "top_k"):
            self.retriever.search_image_vector(
                [1.0] + [0.0] * 767,
                image_fingerprint=self.provider.fingerprint(
                    dimensions=768
                ),
                top_k=101,
            )
        with self.assertRaisesRegex(ValueError, "top_k"):
            self.retriever.search_image_vector(
                [1.0] + [0.0] * 767,
                image_fingerprint=self.provider.fingerprint(
                    dimensions=768
                ),
                top_k=True,
            )

        image_chunk = self.image_chunks[0]
        metadata = dict(image_chunk.metadata or {})
        metadata[IMAGE_EMBEDDING_FINGERPRINT_KEY] = "other-space"
        corrupted = replace(image_chunk, metadata=metadata)
        with self.assertRaisesRegex(
            ValueError,
            "does not match the query vector space",
        ):
            InMemoryHybridRetriever([corrupted]).search_image_vector(
                list(corrupted.image_vector or []),
                image_fingerprint=self.provider.fingerprint(
                    dimensions=768
                ),
                top_k=1,
            )

    def test_local_text_to_image_search_forces_image_records(self) -> None:
        with patch(
            "agent_workshop_demo.embedding."
            "_default_text_embedding_provider",
            return_value=DeterministicTextEmbeddingProvider(),
        ):
            results = self.retriever.search(
                "dense BM25 metadata filter reranking diagram",
                top_k=3,
                filters=image_only_filters({}),
            )

        self.assertEqual(
            results[0].chunk.source_uri,
            "sample_data/local_docs/images/milvus_hybrid_search.png",
        )
        self.assertTrue(all(item.chunk.doc_type == "image" for item in results))

    def test_milvus_image_search_uses_cosine_and_preserves_filters(self) -> None:
        image_chunk = self.image_chunks[0]
        client = _ImageSearchClient(image_chunk.to_dict())
        adapter = MilvusHybridRetriever(client)
        fingerprint = self.provider.fingerprint(dimensions=768)

        with (
            patch(
                "agent_workshop_demo.schema.pymilvus_adapter."
                "text_embedding_fingerprint",
                return_value=(
                    "deterministic:sha256-token-v1:1024"
                ),
            ),
            patch(
                "agent_workshop_demo.schema.pymilvus_adapter."
                "image_embedding_fingerprint",
                return_value=fingerprint,
            ),
        ):
            results = adapter.search_image_vector(
                list(image_chunk.image_vector or []),
                image_fingerprint=fingerprint,
                top_k=2,
                filters={"source_type": "local"},
            )

        self.assertEqual(results[0].chunk.chunk_id, image_chunk.chunk_id)
        self.assertEqual(results[0].image_score, 0.97)
        call = client.calls[0]
        self.assertEqual(call["anns_field"], "image_vector")
        self.assertEqual(call["search_params"]["metric_type"], "COSINE")
        self.assertEqual(
            call["filter"],
            'source_type == "local" and has_image_vector == true',
        )
        self.assertIn("image_vector", call["output_fields"])
        self.assertEqual(call["limit"], 2)

        client.calls.clear()
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            adapter.search_image_vector(
                list(image_chunk.image_vector or []),
                image_fingerprint="",
                top_k=2,
            )
        self.assertEqual(client.calls, [])
        with (
            patch(
                "agent_workshop_demo.schema.pymilvus_adapter."
                "image_embedding_fingerprint",
                return_value=fingerprint,
            ),
            self.assertRaisesRegex(
                ValueError,
                "configured collection vector space",
            ),
        ):
            adapter.search_image_vector(
                list(image_chunk.image_vector or []),
                image_fingerprint="other-valid-space",
                top_k=2,
            )
        self.assertEqual(client.calls, [])

    def test_image_eval_reports_both_modes_without_vector_payloads(self) -> None:
        with patch(
            "agent_workshop_demo.embedding."
            "_default_text_embedding_provider",
            return_value=DeterministicTextEmbeddingProvider(),
        ):
            report = evaluate_image_retrieval(
                cases_path=IMAGE_CASES,
                retriever=self.retriever,
                image_provider=self.provider,
                top_k=3,
                assets_root=Path("demo"),
            )

        self.assertEqual(report["quality_mode"], "pipeline_only")
        self.assertEqual(report["num_cases"], 10)
        self.assertEqual(report["num_text_to_image_cases"], 5)
        self.assertEqual(report["num_image_to_image_cases"], 5)
        self.assertEqual(report["text_to_image_recall_at_k"], 1.0)
        self.assertEqual(report["image_to_image_recall_at_k"], 1.0)
        self.assertEqual(report["image_to_image_mrr"], 1.0)
        serialized = json.dumps(report)
        self.assertNotIn('"image_vector"', serialized)
        self.assertNotIn('"text_vector"', serialized)

    def test_image_eval_slices_over_returning_adapter_at_k(self) -> None:
        with patch(
            "agent_workshop_demo.embedding."
            "_default_text_embedding_provider",
            return_value=DeterministicTextEmbeddingProvider(),
        ):
            text_results = self.retriever.search(
                "diagram",
                top_k=5,
                filters=image_only_filters({}),
            )
        expected = text_results[3].chunk.source_uri
        fixture = {
            "schema_version": "image-retrieval-v1",
            "cases": [
                {
                    "case_id": "bounded",
                    "mode": "text",
                    "query": "ignored by fake",
                    "expected_sources": [expected],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "cases.json"
            path.write_text(json.dumps(fixture), encoding="utf-8")
            report = evaluate_image_retrieval(
                cases_path=path,
                retriever=_OverReturningRetriever(text_results),
                image_provider=self.provider,
                top_k=3,
                assets_root=root,
            )

        self.assertEqual(report["text_to_image_recall_at_k"], 0.0)
        self.assertEqual(
            len(report["cases"][0]["retrieved_sources"]),
            3,
        )

    def test_image_eval_rejects_invalid_fixture_and_unsafe_path(self) -> None:
        invalid_cases = [
            [
                {
                    "case_id": "bad",
                    "mode": "unknown",
                    "expected_sources": ["source"],
                }
            ],
            [
                {
                    "case_id": "bad",
                    "mode": "image",
                    "image_path": "../secret.png",
                    "expected_sources": ["source"],
                }
            ],
        ]
        for cases in invalid_cases:
            with self.subTest(cases=cases):
                with tempfile.TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    path = root / "cases.json"
                    path.write_text(
                        json.dumps(
                            {
                                "schema_version": "image-retrieval-v1",
                                "cases": cases,
                            }
                        ),
                        encoding="utf-8",
                    )
                    with self.assertRaises(ValueError):
                        evaluate_image_retrieval(
                            cases_path=path,
                            retriever=self.retriever,
                            image_provider=self.provider,
                            assets_root=root,
                        )


if __name__ == "__main__":
    unittest.main()
