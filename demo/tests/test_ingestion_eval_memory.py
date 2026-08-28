from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from agent_workshop_demo.eval_runner import (
    _cross_version_contamination_count,
    evaluate_questions,
)
from agent_workshop_demo.ingestion import (
    DocumentVersion,
    _versioned_identity_prefix,
    ingest_demo_sources,
    write_ingestion_result,
)
from agent_workshop_demo.knowledge_tools import ALL_DEPARTMENTS, PermissionDecision
from agent_workshop_demo.memory import ConversationMemoryStore
from agent_workshop_demo.reranker import build_reranker
from agent_workshop_demo.workflow import AgenticRAGWorkflow


class IngestionEvalMemoryTests(unittest.TestCase):
    def test_eval_detects_missing_and_unrequested_comparison_sides(
        self,
    ) -> None:
        response = {
            "version_scope": {
                "mode": "comparison",
                "doc_versions": ["v1", "v2"],
                "sides": [
                    {"mode": "exact", "doc_version": "v1"},
                    {"mode": "exact", "doc_version": "v2"},
                ],
            },
            "milvus_recalled": [
                {"doc_id": "doc_guide", "doc_version": "v1"},
                {"doc_id": "doc_guide", "doc_version": "v3"},
            ],
            "reranked": [
                {
                    "doc_id": "doc_guide",
                    "doc_version": "v1",
                    "selected": True,
                }
            ],
            "citations": [
                {"doc_id": "doc_guide", "doc_version": "v1"}
            ],
        }

        self.assertEqual(
            _cross_version_contamination_count(response),
            2,
        )

    def test_ingestion_reads_sources_and_builds_dedup_records(self) -> None:
        result = ingest_demo_sources(
            Path("demo/sample_data/local_docs"),
            Path("demo/sample_data/mock_s3"),
        )

        self.assertEqual(
            list(Path("demo/sample_data").rglob("*.caption.md")),
            [],
        )
        self.assertGreaterEqual(len(result.kb_chunks), 12)
        self.assertEqual(
            len(result.dedup_signatures),
            len(result.kb_chunks),
        )
        self.assertTrue(
            any(item.source_type == "local" for item in result.kb_chunks)
        )
        self.assertTrue(
            any(item.source_type == "s3" for item in result.kb_chunks)
        )
        self.assertTrue(
            any(item.department == "hr" for item in result.kb_chunks)
        )
        self.assertTrue(
            any(item.doc_type == "pdf" for item in result.kb_chunks)
        )
        image_count = sum(
            1 for item in result.kb_chunks if item.has_image_vector
        )
        self.assertGreaterEqual(image_count, 5)
        self.assertTrue(
            all("minhash_signature" not in item for item in result.dedup_signatures)
        )
        self.assertTrue(
            all(item["normalized_text"] for item in result.dedup_signatures)
        )
        go_versions = {
            (item.doc_version, item.is_current)
            for item in result.kb_chunks
            if item.doc_id == "doc_go_button_guide"
        }
        self.assertEqual(go_versions, {("v1", False), ("v2", True)})
        release_chunks = [
            item
            for item in result.kb_chunks
            if item.doc_id == "doc_milvus_release_notes"
        ]
        self.assertEqual(
            {
                (item.doc_version, item.is_current)
                for item in release_chunks
            },
            {("v2.6", False), ("v3.0", True)},
        )
        self.assertTrue(
            all(item.section and item.doc_version for item in release_chunks)
        )
        self.assertTrue(
            all(
                item.metadata
                and item.metadata.get("heading_path")
                and item.section == item.metadata["heading_path"][-1]
                for item in release_chunks
            )
        )
        self.assertTrue(
            all(
                item.metadata
                and item.metadata.get("retrieval_text_version")
                == "title-heading-text-v1"
                and "milvus" in item.sparse_vector
                for item in release_chunks
            )
        )
        self.assertEqual(
            sum(
                item.section == "Storage Format V2"
                for item in release_chunks
            ),
            1,
        )
        self.assertEqual(
            sum(
                item.section == "Storage Format V3"
                for item in release_chunks
            ),
            1,
        )
        self.assertTrue(all(item.doc_version for item in result.kb_chunks))

    def test_markdown_heading_keeps_paragraphs_in_one_feature_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            local = root / "local"
            mock_s3 = root / "mock_s3"
            local.mkdir()
            mock_s3.mkdir()
            (local / "release.md").write_text(
                "# Release\n\n"
                "Edition overview.\n\n"
                "## Feature A\n\n"
                "First feature paragraph.\n\n"
                "Second feature paragraph.\n\n"
                "## Feature B\n\n"
                "Another feature.",
                encoding="utf-8",
            )

            result = ingest_demo_sources(local, mock_s3)

        self.assertEqual(
            [item.section for item in result.kb_chunks],
            ["Release", "Feature A", "Feature B"],
        )
        feature_a = result.kb_chunks[1]
        self.assertIn("First feature paragraph.", feature_a.text)
        self.assertIn("Second feature paragraph.", feature_a.text)
        self.assertEqual(
            feature_a.metadata["heading_path"],
            ["Release", "Feature A"],
        )

    def test_sample_assets_are_openable_file_formats(self) -> None:
        pdf_path = Path(
            "demo/sample_data/local_docs/engineering/"
            "rag_architecture_v1.pdf"
        )
        image_paths = sorted(
            Path("demo/sample_data/local_docs/images").glob("*.png")
        )

        self.assertEqual(pdf_path.read_bytes()[:5], b"%PDF-")
        self.assertGreater(pdf_path.stat().st_size, 1_000)
        pdf_bytes = pdf_path.read_bytes()
        required = [
            b"bucket scanning",
            b"change detection",
            b"document parsing",
            b"chunking",
            b"Milvus insertion",
        ]
        for term in required:
            self.assertIn(term, pdf_bytes)
        self.assertEqual(len(image_paths), 5)
        for image_path in image_paths:
            self.assertEqual(
                image_path.read_bytes()[:8],
                b"\x89PNG\r\n\x1a\n",
            )
            self.assertGreater(image_path.stat().st_size, 500)

    def test_sample_text_supports_golden_answers(self) -> None:
        paths = [
            Path("demo/sample_data/mock_s3/engineering/s3_sync_design.md"),
            Path(
                "demo/sample_data/local_docs/engineering/"
                "milvus_feature_map.md"
            ),
            Path("demo/sample_data/asset_manifest.json"),
        ]
        combined = "\n".join(
            path.read_text(encoding="utf-8") for path in paths
        )
        required = [
            "bucket scanning",
            "change detection",
            "document parsing",
            "chunking",
            "embedding generation",
            "Milvus insertion",
            "hybrid retrieval",
            "metadata filter",
            "order_by",
            "aggregation",
        ]
        for term in required:
            self.assertIn(term, combined)

    def test_ingestion_writes_jsonl_files(self) -> None:
        result = ingest_demo_sources(
            Path("demo/sample_data/local_docs"),
            Path("demo/sample_data/mock_s3"),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_ingestion_result(result, Path(tmpdir))
            self.assertTrue(paths["kb_chunks"].exists())
            self.assertTrue(paths["doc_dedup_signatures"].exists())
            self.assertGreater(paths["kb_chunks"].stat().st_size, 0)

    def test_ingestion_ids_are_stable_across_checkout_paths(self) -> None:
        with tempfile.TemporaryDirectory() as first:
            with tempfile.TemporaryDirectory() as second:
                first_root = Path(first) / "sample_data"
                second_root = Path(second) / "sample_data"
                shutil.copytree(Path("demo/sample_data"), first_root)
                shutil.copytree(Path("demo/sample_data"), second_root)
                first_result = ingest_demo_sources(
                    first_root / "local_docs",
                    first_root / "mock_s3",
                )
                second_result = ingest_demo_sources(
                    second_root / "local_docs",
                    second_root / "mock_s3",
                )

        first_ids = [
            (item.doc_id, item.doc_version, item.chunk_id)
            for item in first_result.kb_chunks
        ]
        second_ids = [
            (item.doc_id, item.doc_version, item.chunk_id)
            for item in second_result.kb_chunks
        ]
        self.assertEqual(first_ids, second_ids)

    def test_version_identity_prefix_does_not_collapse_punctuation(self) -> None:
        prefixes = {
            _versioned_identity_prefix(
                DocumentVersion("doc_guide", version, version == "v1_0")
            )
            for version in ("v1.0", "v1_0", "v1/0")
        }

        self.assertEqual(len(prefixes), 3)

    def test_ingestion_rejects_multiple_current_editions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            local = root / "local_docs"
            mock_s3 = root / "mock_s3"
            local.mkdir()
            (mock_s3 / "product").mkdir(parents=True)
            first = mock_s3 / "product" / "guide_v1.md"
            second = mock_s3 / "product" / "guide_v2.md"
            first.write_text("# Guide v1\n\nOld.", encoding="utf-8")
            second.write_text("# Guide v2\n\nNew.", encoding="utf-8")
            manifest = root / "document_versions.json"
            manifest.write_text(
                json.dumps(
                    {
                        "s3://internal-agent-chat-demo/product/guide_v1.md": {
                            "doc_id": "doc_guide",
                            "doc_version": "v1",
                            "is_current": True,
                        },
                        "s3://internal-agent-chat-demo/product/guide_v2.md": {
                            "doc_id": "doc_guide",
                            "doc_version": "v2",
                            "is_current": True,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "exactly one current edition",
            ):
                ingest_demo_sources(
                    local,
                    mock_s3,
                    version_manifest_path=manifest,
                )

    def test_ingestion_rejects_missing_source_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "missing"
            with self.assertRaisesRegex(FileNotFoundError, "source directory"):
                ingest_demo_sources(
                    missing,
                    Path("demo/sample_data/mock_s3"),
                )

    def test_ingestion_rejects_unhandled_document_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local = root / "local"
            mock_s3 = root / "mock_s3"
            local.mkdir()
            mock_s3.mkdir()
            (local / "unsupported.docx").write_bytes(b"not a demo document")

            with self.assertRaisesRegex(
                ValueError,
                "Unsupported document type.*docx",
            ):
                ingest_demo_sources(local, mock_s3)

    def test_ingestion_extracts_unmanifested_pdf_pages(self) -> None:
        class FakePage:
            def __init__(self, text: str) -> None:
                self.text = text

            def extract_text(self) -> str:
                return self.text

        class FakeReader:
            def __init__(self, path: Path) -> None:
                self.pages = [
                    FakePage("# Page One\n\nFirst PDF evidence."),
                    FakePage("Second PDF evidence."),
                ]
                self.metadata = SimpleNamespace(title="Parsed PDF")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local = root / "local"
            mock_s3 = root / "mock_s3"
            (local / "engineering").mkdir(parents=True)
            mock_s3.mkdir()
            (local / "engineering" / "new.pdf").write_bytes(b"%PDF-1.7")

            with patch(
                "agent_workshop_demo.ingestion.import_module",
                return_value=SimpleNamespace(PdfReader=FakeReader),
            ):
                result = ingest_demo_sources(local, mock_s3)

        self.assertEqual(len(result.kb_chunks), 3)
        self.assertEqual(
            [item.page_no for item in result.kb_chunks],
            [1, 1, 2],
        )
        self.assertTrue(
            all(item.record_type == "pdf_page" for item in result.kb_chunks)
        )
        self.assertTrue(
            all(
                item.metadata["parser"] == "pypdf"
                for item in result.kb_chunks
                if item.metadata is not None
            )
        )

    def test_eval_runner_reports_recall_and_citation_metrics(self) -> None:
        class ScenarioPermissionChecker:
            def __init__(self, *, allowed: bool) -> None:
                self.allowed = allowed

            def check(
                self,
                *,
                session_id: str,
                intent: str,
                query_type: str,
            ) -> PermissionDecision:
                del session_id, intent, query_type
                return PermissionDecision(
                    allowed=self.allowed,
                    allowed_departments=ALL_DEPARTMENTS if self.allowed else (),
                    reason="Test fixture permission decision.",
                    checker_name="test-scenario-permission",
                )

        questions_path = Path("demo/eval/questions.json")
        fixture_case_count = len(
            json.loads(questions_path.read_text(encoding="utf-8"))
        )

        report = evaluate_questions(
            questions_path=questions_path,
            # A scenario factory must honour every declared key; ignoring one
            # and carrying on is what spec 70 § 3 forbids.
            scenario_workflow_factory=lambda scenario: AgenticRAGWorkflow(
                permission_checker=ScenarioPermissionChecker(
                    allowed=scenario["permission"] == "allow"
                ),
                reranker=(
                    build_reranker({"RERANKER": "auto"})
                    if scenario.get("reranker") == "fallback"
                    else None
                ),
            ),
        )

        self.assertEqual(report["report_version"], "rag-eval-v3")
        self.assertEqual(report["evaluation"]["mode"], "deterministic")
        # Assert coverage of the committed fixture, not a frozen count.
        self.assertEqual(report["num_questions"], fixture_case_count)
        self.assertIn("recall_at_k", report)
        self.assertIn("reranked_recall_at_8", report)
        self.assertIn("selected_context_recall_at_5", report)
        self.assertIn("citation_coverage", report)
        self.assertIn("citation_precision", report)
        self.assertIn("required_fact_coverage", report)
        self.assertEqual(report["abstention_accuracy"], 1.0)
        self.assertGreaterEqual(report["recall_at_k"], 0.5)
        self.assertEqual(report["tool_selection_accuracy"], 1.0)
        self.assertEqual(report["entity_resolution_accuracy"], 1.0)
        self.assertEqual(report["cross_version_contamination_count"], 0)
        self.assertEqual(report["permission_denial_case_count"], 1)
        self.assertEqual(report["permission_bypass_count"], 0)
        self.assertEqual(
            report["dimensions"]["trajectory"]["contract_pass_rate"]["value"],
            1.0,
        )
        self.assertIn("goal", report["metric_portfolio"])
        self.assertIn("guardrail", report["metric_portfolio"])
        self.assertIn("operational", report["metric_portfolio"])
        self.assertEqual(
            report["transcript_review"]["failed_case_ids"],
            ["q001"],
        )
        self.assertEqual(
            report["cases"][0]["first_failure_layer"],
            "outcome",
        )

    def test_eval_metrics_use_per_question_denominators(self) -> None:
        class PartialWorkflow:
            def run(
                self,
                question: str,
                filters: dict[str, Any] | None = None,
                *,
                session_id: str | None = None,
                query_id: str | None = None,
            ) -> dict[str, Any]:
                del question, filters, session_id, query_id
                return {
                    "milvus_recalled": [{"chunk_id": "expected_1"}],
                    "citations": [
                        {"chunk_id": "expected_1"},
                        {"chunk_id": "unrelated"},
                    ],
                    "enough_evidence": True,
                }

        questions = [
            {
                "question_id": "partial",
                "question": "partial recall",
                "expected_sources": ["expected_1", "expected_2"],
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            questions_path = Path(tmpdir) / "questions.json"
            questions_path.write_text(
                json.dumps(questions),
                encoding="utf-8",
            )
            report = evaluate_questions(
                questions_path=questions_path,
                workflow_factory=PartialWorkflow,
            )

        self.assertEqual(report["recall_at_k"], 0.5)
        self.assertEqual(report["citation_coverage"], 0.5)
        self.assertEqual(report["citation_precision"], 0.5)
        self.assertEqual(report["cases"][0]["recall_at_k"], 0.5)

    def test_conversation_memory_filters_expired_records(self) -> None:
        store = ConversationMemoryStore(now_ms=1_000)
        store.add_turn(
            session_id="session_demo",
            turn_id="turn_old",
            role="user",
            content="expired S3 question",
            memory_type="short_term",
            expires_at=900,
        )
        store.add_turn(
            session_id="session_demo",
            turn_id="turn_live",
            role="summary",
            content="S3 document sync pipeline summary",
            memory_type="session_summary",
            expires_at=2_000,
        )

        results = store.search(
            "S3 sync",
            session_id="session_demo",
            top_k=5,
            now_ms=1_000,
        )
        self.assertEqual([item.turn_id for item in results], ["turn_live"])

    def test_workflow_stream_yields_chunks_then_final_response(self) -> None:
        events = list(
            AgenticRAGWorkflow().stream(
                "我们 S3 文档同步流程是怎么设计的？"
            )
        )

        deltas = [event for event in events if event["type"] == "answer_delta"]
        self.assertGreaterEqual(len(deltas), 2)
        self.assertEqual(events[-1]["type"], "final")
        grade = events[-1]["response"]["trace"]["evidence_grading"]
        self.assertTrue(grade["enough_evidence"])

    def test_workshop_notebook_sequence_exists_and_is_valid_json(self) -> None:
        notebooks = sorted(Path("demo/notebooks").glob("*.ipynb"))
        expected = [
            "01_ingestion_local_s3.ipynb",
            "02_text_image_embedding.ipynb",
            "03_milvus_schema_and_insert.ipynb",
            "04_milvus_hybrid_search.ipynb",
            "05_langgraph_agentic_rag.ipynb",
            "06_streamlit_ui_demo.ipynb",
        ]
        self.assertEqual([path.name for path in notebooks], expected)
        for path in notebooks:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["nbformat"], 4)
            self.assertGreaterEqual(len(data["cells"]), 2)


if __name__ == "__main__":
    unittest.main()
