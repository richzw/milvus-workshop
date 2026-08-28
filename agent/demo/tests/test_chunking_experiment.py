from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent_workshop_demo.chunking import (
    BOUNDARY_POLICY,
    ChunkingConfig,
    count_chunk_tokens,
    split_text,
)
from agent_workshop_demo.chunking_experiment import (
    load_chunking_configs,
    run_chunking_experiment,
)
from agent_workshop_demo.embedding import (
    DeterministicTextEmbeddingProvider,
)
from agent_workshop_demo.image_embedding import (
    DeterministicImageEmbeddingProvider,
)
from agent_workshop_demo.ingestion import ingest_demo_sources

CONFIGS = Path("demo/eval/chunking_configs.json")
ANCHORS = Path("demo/eval/chunking_anchors.json")
LOCAL_DIR = Path("demo/sample_data/local_docs")
MOCK_S3_DIR = Path("demo/sample_data/mock_s3")


class ChunkingExperimentTests(unittest.TestCase):
    def test_config_validation_and_fingerprint_are_strict(self) -> None:
        config = ChunkingConfig(
            name="focused",
            min_tokens=24,
            max_tokens=80,
            overlap_tokens=8,
        )
        self.assertEqual(config.boundary_policy, BOUNDARY_POLICY)
        self.assertEqual(config.fingerprint, config.fingerprint)
        self.assertTrue(config.fingerprint.startswith("minmax:"))

        invalid = [
            {
                "name": "Upper",
                "min_tokens": 24,
                "max_tokens": 80,
                "overlap_tokens": 8,
            },
            {
                "name": "bad",
                "min_tokens": 0,
                "max_tokens": 80,
                "overlap_tokens": 8,
            },
            {
                "name": "bad",
                "min_tokens": 81,
                "max_tokens": 80,
                "overlap_tokens": 8,
            },
            {
                "name": "bad",
                "min_tokens": 24,
                "max_tokens": 80,
                "overlap_tokens": 24,
            },
            {
                "name": "bad",
                "min_tokens": 24,
                "max_tokens": 80,
                "overlap_tokens": 8,
                "boundary_policy": "unknown",
            },
        ]
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    ChunkingConfig(**values)

    def test_splitter_is_bounded_overlapping_and_deterministic(self) -> None:
        text = "\n\n".join(
            f"Sentence {index} contains stable retrieval evidence."
            for index in range(1, 31)
        )
        config = ChunkingConfig(
            name="split",
            min_tokens=20,
            max_tokens=40,
            overlap_tokens=6,
        )

        first = split_text(text, config=config)
        second = split_text(text, config=config)

        self.assertEqual(first, second)
        self.assertGreater(len(first), 2)
        self.assertTrue(
            all(fragment.token_count <= 40 for fragment in first)
        )
        self.assertTrue(
            all(
                fragment.token_count >= 20
                for fragment in first
                if fragment is not first[-1]
            )
        )
        self.assertEqual(first[0].applied_overlap_tokens, 0)
        self.assertTrue(
            all(
                fragment.applied_overlap_tokens == 6
                for fragment in first[1:]
            )
        )
        self.assertTrue(
            all(
                left.end_token - right.start_token == 6
                for left, right in zip(first, first[1:])
            )
        )
        self.assertEqual(
            [fragment.start_token for fragment in first],
            sorted(fragment.start_token for fragment in first),
        )

    def test_splitter_plans_ahead_to_avoid_undersized_tail(self) -> None:
        no_overlap = ChunkingConfig(
            name="no_overlap",
            min_tokens=3,
            max_tokens=4,
            overlap_tokens=0,
        )
        overlapping = ChunkingConfig(
            name="overlapping",
            min_tokens=3,
            max_tokens=4,
            overlap_tokens=1,
        )

        nine_tokens = split_text(
            "one two three four five six seven eight nine",
            config=no_overlap,
        )
        five_tokens = split_text(
            "one two three four five",
            config=overlapping,
        )

        self.assertEqual(
            [fragment.token_count for fragment in nine_tokens],
            [3, 3, 3],
        )
        self.assertEqual(
            [fragment.token_count for fragment in five_tokens],
            [3, 3],
        )
        self.assertEqual(five_tokens[1].applied_overlap_tokens, 1)

    def test_configured_ingestion_preserves_hard_boundaries_and_metadata(
        self,
    ) -> None:
        config = ChunkingConfig(
            name="focused",
            min_tokens=24,
            max_tokens=80,
            overlap_tokens=8,
        )
        with patch(
            "agent_workshop_demo.embedding."
            "_default_text_embedding_provider",
            return_value=DeterministicTextEmbeddingProvider(),
        ):
            result = ingest_demo_sources(
                LOCAL_DIR,
                MOCK_S3_DIR,
                image_embedding_provider=(
                    DeterministicImageEmbeddingProvider()
                ),
                chunking_config=config,
            )

        configured = [
            chunk
            for chunk in result.kb_chunks
            if (chunk.metadata or {}).get("chunking")
        ]
        self.assertTrue(configured)
        for chunk in configured:
            metadata = (chunk.metadata or {})["chunking"]
            self.assertEqual(
                metadata["config_fingerprint"],
                config.fingerprint,
            )
            self.assertEqual(
                metadata["token_count"],
                count_chunk_tokens(chunk.text),
            )
            self.assertLessEqual(metadata["token_count"], 80)
        markdown = [
            chunk for chunk in configured if chunk.doc_type == "markdown"
        ]
        self.assertTrue(all(chunk.section for chunk in markdown))
        self.assertTrue(
            all(
                chunk.section
                == (chunk.metadata or {})["heading_path"][-1]
                for chunk in markdown
            )
        )
        pdf = [
            chunk for chunk in result.kb_chunks if chunk.doc_type == "pdf"
        ]
        self.assertTrue(pdf)
        self.assertTrue(all(chunk.page_no is not None for chunk in pdf))
        release_chunks = [
            chunk
            for chunk in configured
            if chunk.doc_id == "doc_milvus_release_notes"
        ]
        self.assertTrue(release_chunks)
        self.assertTrue(
            all(
                chunk.doc_version in {"v2.6", "v3.0"}
                for chunk in release_chunks
            )
        )

    def test_configured_pdf_parser_splits_only_within_each_page(self) -> None:
        config = ChunkingConfig(
            name="pdf_pages",
            min_tokens=12,
            max_tokens=24,
            overlap_tokens=4,
        )
        page_texts = [
            (
                " ".join(
                    f"page_one_token_{index}" for index in range(4)
                )
                + "\n\n"
                + " ".join(
                    f"page_one_token_{index}" for index in range(4, 70)
                )
            ),
            " ".join(f"page_two_token_{index}" for index in range(65)),
        ]
        fake_pypdf = SimpleNamespace(
            PdfReader=lambda _path: SimpleNamespace(
                pages=[
                    SimpleNamespace(
                        extract_text=lambda text=text: text,
                    )
                    for text in page_texts
                ]
            )
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            local_dir = root / "local_docs"
            mock_s3_dir = root / "mock_s3"
            pdf_path = local_dir / "engineering" / "architecture.pdf"
            pdf_path.parent.mkdir(parents=True)
            mock_s3_dir.mkdir()
            pdf_path.write_bytes(b"%PDF-fake")
            with (
                patch(
                    "agent_workshop_demo.embedding."
                    "_default_text_embedding_provider",
                    return_value=DeterministicTextEmbeddingProvider(),
                ),
                patch(
                    "agent_workshop_demo.ingestion.import_module",
                    return_value=fake_pypdf,
                ),
            ):
                result = ingest_demo_sources(
                    local_dir,
                    mock_s3_dir,
                    image_embedding_provider=(
                        DeterministicImageEmbeddingProvider()
                    ),
                    chunking_config=config,
                )

        pdf_chunks = [
            chunk
            for chunk in result.kb_chunks
            if chunk.doc_type == "pdf"
        ]
        self.assertGreater(len(pdf_chunks), 1)
        self.assertTrue(all(chunk.page_no is not None for chunk in pdf_chunks))
        self.assertEqual({chunk.page_no for chunk in pdf_chunks}, {1, 2})
        self.assertTrue(
            all(
                (chunk.metadata or {})["chunking"]["token_count"]
                >= config.min_tokens
                for chunk in pdf_chunks
            )
        )
        for chunk in pdf_chunks:
            metadata = (chunk.metadata or {})["chunking"]
            self.assertLessEqual(metadata["token_count"], config.max_tokens)
            self.assertEqual(
                metadata["token_count"],
                count_chunk_tokens(chunk.text),
            )
            self.assertIn(
                f"_p{chunk.page_no:03d}_",
                chunk.chunk_id,
            )
            expected_prefix = (
                "page_one_token_" if chunk.page_no == 1
                else "page_two_token_"
            )
            self.assertIn(expected_prefix, chunk.text)
            unexpected_prefix = (
                "page_two_token_" if chunk.page_no == 1
                else "page_one_token_"
            )
            self.assertNotIn(unexpected_prefix, chunk.text)

    def test_configured_plain_text_combines_short_paragraphs(self) -> None:
        config = ChunkingConfig(
            name="plain_document",
            min_tokens=8,
            max_tokens=20,
            overlap_tokens=2,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            local_dir = root / "local_docs"
            mock_s3_dir = root / "mock_s3"
            text_path = local_dir / "engineering" / "notes.txt"
            text_path.parent.mkdir(parents=True)
            mock_s3_dir.mkdir()
            text_path.write_text(
                "alpha beta gamma delta\n\nepsilon zeta eta theta",
                encoding="utf-8",
            )
            with patch(
                "agent_workshop_demo.embedding."
                "_default_text_embedding_provider",
                return_value=DeterministicTextEmbeddingProvider(),
            ):
                result = ingest_demo_sources(
                    local_dir,
                    mock_s3_dir,
                    image_embedding_provider=(
                        DeterministicImageEmbeddingProvider()
                    ),
                    chunking_config=config,
                )

        self.assertEqual(len(result.kb_chunks), 1)
        chunk = result.kb_chunks[0]
        self.assertEqual(count_chunk_tokens(chunk.text), 8)
        self.assertIn("delta\n\nepsilon", chunk.text)

    def test_versioned_config_loader_rejects_ambiguous_inputs(self) -> None:
        configs = load_chunking_configs(CONFIGS)
        self.assertEqual(
            [config.name for config in configs],
            ["small_32_128", "medium_64_256", "large_96_512"],
        )
        invalid_payloads = [
            {"schema_version": "wrong", "configs": []},
            {
                "schema_version": "chunking-experiment-v2",
                "configs": [
                    {
                        "name": "only",
                        "min_tokens": 1,
                        "max_tokens": 2,
                        "overlap_tokens": 0,
                        "boundary_policy": BOUNDARY_POLICY,
                    }
                ],
            },
            {
                "schema_version": "chunking-experiment-v2",
                "configs": [
                    {
                        "name": name,
                        "min_tokens": 10,
                        "max_tokens": 20,
                        "overlap_tokens": 3,
                        "boundary_policy": BOUNDARY_POLICY,
                    }
                    for name in ("same_a", "same_b", "same_c")
                ],
            },
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with tempfile.TemporaryDirectory() as temp_dir:
                    path = Path(temp_dir) / "configs.json"
                    path.write_text(
                        json.dumps(payload),
                        encoding="utf-8",
                    )
                    with self.assertRaises(ValueError):
                        load_chunking_configs(path)

    def test_runner_compares_same_corpus_and_recommends_by_metrics(self) -> None:
        times = iter([1.0, 1.1, 2.0, 2.2, 3.0, 3.3])
        with patch(
            "agent_workshop_demo.embedding."
            "_default_text_embedding_provider",
            return_value=DeterministicTextEmbeddingProvider(),
        ):
            report = run_chunking_experiment(
                configs_path=CONFIGS,
                anchors_path=ANCHORS,
                local_dir=LOCAL_DIR,
                mock_s3_dir=MOCK_S3_DIR,
                image_provider=DeterministicImageEmbeddingProvider(),
                clock=lambda: next(times),
            )

        self.assertEqual(report["schema_version"], "chunking-experiment-v2")
        self.assertEqual(report["num_configs"], 3)
        self.assertEqual(report["num_anchor_cases"], 4)
        self.assertIsNone(report["recommended_config"])
        self.assertEqual(report["evaluation_status"], "evaluation_incomplete")
        self.assertEqual(
            set(report["recommendation"]["rejected"]),
            {"small_32_128", "medium_64_256", "large_96_512"},
        )
        small, medium, large = report["results"]
        self.assertEqual(small["ingestion_time_ms"], 100.0)
        self.assertEqual(medium["ingestion_time_ms"], 200.0)
        self.assertEqual(large["ingestion_time_ms"], 300.0)
        self.assertGreater(
            small["corpus"]["textual_chunk_count"],
            large["corpus"]["textual_chunk_count"],
        )
        for result in report["results"]:
            self.assertEqual(result["corpus"]["over_max_chunk_count"], 0)
            self.assertIn("character_length", result["corpus"])
            self.assertEqual(result["index_size_bytes"], None)
            self.assertEqual(result["index_size_status"], "not_built")
            self.assertEqual(
                result["isolation"]["index_status"],
                "not_built_local_in_memory",
            )
            self.assertIn("reranked_recall_at_8", result)
            self.assertIn("citation_precision", result)
            self.assertIn("required_fact_coverage", result)
            self.assertEqual(result["abstention_correctness"], 1.0)
            self.assertIsNone(result["faithfulness"])
            self.assertIsNone(result["answer_relevancy"])
            self.assertEqual(result["usage"]["provider_call_count"], 0)
            self.assertIn("p95", result["latency_ms"]["end_to_end"])

    def test_runner_rejects_malformed_anchor_fixture(self) -> None:
        payload = {
            "schema_version": "chunking-anchors-v2",
            "cases": [
                {
                    "case_id": "bad",
                    "question": "query",
                    "filters": {},
                    "should_abstain": False,
                    "expected_anchors": [
                        {
                            "source_uri": "source",
                            "required_terms": [],
                        }
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "anchors.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "anchor"):
                run_chunking_experiment(
                    configs_path=CONFIGS,
                    anchors_path=path,
                    local_dir=LOCAL_DIR,
                    mock_s3_dir=MOCK_S3_DIR,
                )

    def test_calibrated_frontier_requires_matching_review_artifact(self) -> None:
        with patch(
            "agent_workshop_demo.embedding."
            "_default_text_embedding_provider",
            return_value=DeterministicTextEmbeddingProvider(),
        ), patch.dict(
            "agent_workshop_demo.chunking_experiment.COMMITTED_QUALITY_GATES",
            {
                "retrieval_recall_at_20": 0.0,
                "selected_context_recall_at_5": 0.0,
                "citation_precision": 0.0,
                "citation_coverage": 0.0,
                "required_fact_coverage": 0.0,
                "abstention_correctness": 0.0,
                "markdown_section_preservation_rate": 0.0,
                "release_boundary_preservation_rate": 0.0,
                "pdf_page_preservation_rate": 0.0,
            },
            clear=True,
        ):
            def grader(
                _question: str,
                _answer: str,
                _contexts: list[str],
            ) -> dict[str, float]:
                return {"faithfulness": 1.0, "answer_relevancy": 1.0}
            evaluated = run_chunking_experiment(
                configs_path=CONFIGS,
                anchors_path=ANCHORS,
                local_dir=LOCAL_DIR,
                mock_s3_dir=MOCK_S3_DIR,
                image_provider=DeterministicImageEmbeddingProvider(),
                semantic_grader=grader,
                grader_calibrated=True,
                semantic_grader_profile="test-grader-v1-calibrated",
            )
            chosen_name = evaluated["recommendation"]["pareto_frontier"][0]
            chosen = next(
                item["config"]
                for item in evaluated["results"]
                if item["config"]["name"] == chosen_name
            )
            with tempfile.TemporaryDirectory() as temp_dir:
                recommendation = Path(temp_dir) / "recommendation.json"
                recommendation.write_text(
                    json.dumps(
                        {
                            "schema_version": "chunking-recommendation-v1",
                            "experiment_fingerprint": evaluated[
                                "experiment_fingerprint"
                            ],
                            "evaluation_fingerprint": evaluated[
                                "evaluation_fingerprint"
                            ],
                            "config_name": chosen["name"],
                            "config_fingerprint": chosen[
                                "config_fingerprint"
                            ],
                            "reviewer": "workshop-owner",
                            "rationale": "Reviewed the independent dimensions.",
                        }
                    ),
                    encoding="utf-8",
                )
                reviewed = run_chunking_experiment(
                    configs_path=CONFIGS,
                    anchors_path=ANCHORS,
                    local_dir=LOCAL_DIR,
                    mock_s3_dir=MOCK_S3_DIR,
                    image_provider=DeterministicImageEmbeddingProvider(),
                    semantic_grader=grader,
                    grader_calibrated=True,
                    semantic_grader_profile="test-grader-v1-calibrated",
                    recommendation_path=recommendation,
                )

        self.assertEqual(reviewed["evaluation_status"], "complete")
        self.assertEqual(reviewed["recommended_config"], chosen["name"])


if __name__ == "__main__":
    unittest.main()
