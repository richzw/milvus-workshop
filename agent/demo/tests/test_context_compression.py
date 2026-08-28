from __future__ import annotations

import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from typing import cast

from agent_workshop_demo.context_compression import (
    AtomicFallbackContextCompressor,
    CompressionRun,
    EffectiveCompressionMode,
    OpenAIContextCompressor,
    build_context_compressor,
    validate_compression_run,
)
from agent_workshop_demo.generation import GenerationContext


class RecordingResponses:
    def __init__(self, output: object) -> None:
        self.output = output
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=json.dumps(self.output, ensure_ascii=False))


def context(chunk_id: str, text: str) -> GenerationContext:
    return GenerationContext(
        citation_id=f"C{chunk_id[-1]}",
        chunk_id=chunk_id,
        doc_id="doc",
        doc_version="v3.0",
        title="Title",
        page_no=None,
        section="Section",
        prompt_text=text,
    )


class ContextCompressionTests(unittest.TestCase):
    def test_selective_mode_accepts_only_exact_ordered_spans(self) -> None:
        source = "前置信息。Force Merge 会合并 sealed segments。无关结尾。"
        quote = "Force Merge 会合并 sealed segments。"
        start = source.index(quote)
        responses = RecordingResponses(
            {
                "contexts": [
                    {
                        "chunk_id": "chunk_1",
                        "prompt_text": quote,
                        "support_spans": [
                            {"start": start, "end": start + len(quote), "quote": quote}
                        ],
                    }
                ]
            }
        )
        compressor = OpenAIContextCompressor(
            client=SimpleNamespace(responses=responses),
            model="test-model",
            mode="selective",
        )

        run = compressor.compress("Force Merge 是什么？", [context("chunk_1", source)])

        self.assertEqual(run.contexts[0].prompt_text, quote)
        self.assertEqual(run.contexts[0].compression_mode, "selective")
        self.assertEqual(len(responses.calls), 1)
        self.assertTrue(run.contexts[0].source_text_checksum)

    def test_summary_and_extraction_keep_exact_support(self) -> None:
        source = "Force Merge 合并 sealed segments，并回收已删除实体占用的空间。"
        for mode, derived in (
            ("summary", "Force Merge 用于整理存储。"),
            ("extraction", "事实：可回收删除实体占用的空间。"),
        ):
            with self.subTest(mode=mode):
                responses = RecordingResponses(
                    {
                        "contexts": [
                            {
                                "chunk_id": "chunk_1",
                                "prompt_text": derived,
                                "support_spans": [
                                    {"start": 0, "end": len(source), "quote": source}
                                ],
                            }
                        ]
                    }
                )
                run = OpenAIContextCompressor(
                    client=SimpleNamespace(responses=responses),
                    model="test-model",
                    mode=cast(EffectiveCompressionMode, mode),
                ).compress("Force Merge", [context("chunk_1", source)])
                self.assertEqual(run.contexts[0].prompt_text, source)
                self.assertEqual(run.contexts[0].compression_mode, mode)

    def test_one_invalid_item_restores_the_whole_original_batch(self) -> None:
        first = context("chunk_1", "第一段可靠证据。")
        second = context("chunk_2", "第二段可靠证据。")
        responses = RecordingResponses(
            {
                "contexts": [
                    {
                        "chunk_id": "chunk_1",
                        "prompt_text": first.prompt_text,
                        "support_spans": [
                            {
                                "start": 0,
                                "end": len(first.prompt_text),
                                "quote": first.prompt_text,
                            }
                        ],
                    },
                    {
                        "chunk_id": "chunk_2",
                        "prompt_text": "伪造内容",
                        "support_spans": [
                            {"start": 0, "end": 4, "quote": "并不存在"}
                        ],
                    },
                ]
            }
        )
        primary = OpenAIContextCompressor(
            client=SimpleNamespace(responses=responses),
            model="test-model",
            mode="selective",
        )
        compressor = AtomicFallbackContextCompressor(
            primary=primary,
            configured_mode="selective",
        )

        run = compressor.compress("问题", [first, second])

        self.assertEqual(run.fallback_reason, "invalid_model_output")
        self.assertEqual(run.effective_mode, "disabled")
        self.assertEqual(
            [item.prompt_text for item in run.contexts],
            [first.prompt_text, second.prompt_text],
        )
        self.assertTrue(all(item.compression_mode == "disabled" for item in run.contexts))

    def test_auto_below_trigger_and_configuration_are_network_free(self) -> None:
        responses = RecordingResponses({"contexts": []})
        compressor = build_context_compressor(
            {
                "CONTEXT_COMPRESSION_MODE": "auto",
                "CONTEXT_COMPRESSION_TRIGGER_CHARS": "1000",
                "OPENAI_API_KEY": "test-key",
                "OPENAI_MODEL": "test-model",
            },
            client_factory=lambda _key: SimpleNamespace(responses=responses),
        )

        run = compressor.compress("问题", [context("chunk_1", "短证据")])

        self.assertEqual(run.fallback_reason, "below_trigger")
        self.assertEqual(responses.calls, [])
        self.assertEqual(build_context_compressor({}).name, "identity")
        with self.assertRaises(ValueError):
            build_context_compressor({"CONTEXT_COMPRESSION_MODE": "summary"})

    def test_injected_same_id_fabrication_is_atomically_restored(self) -> None:
        original = context("chunk_1", "可信原文。")
        fabricated = GenerationContext(
            citation_id=original.citation_id,
            chunk_id=original.chunk_id,
            doc_id=original.doc_id,
            doc_version=original.doc_version,
            title=original.title,
            page_no=original.page_no,
            section=original.section,
            prompt_text="伪造摘要。",
            compression_mode="summary",
            support_spans=(
                {"start": 0, "end": 5, "quote": "不存在文本"},
            ),
            source_text_checksum="forged",
        )
        run = validate_compression_run(
            CompressionRun(
                contexts=(fabricated,),
                configured_mode="summary",
                effective_mode="summary",
                compressor_name="injected",
            ),
            [original],
        )

        self.assertEqual(run.effective_mode, "disabled")
        self.assertEqual(run.fallback_reason, "invalid_model_output")
        self.assertEqual(run.contexts[0].prompt_text, original.prompt_text)

    def test_projection_dropping_query_relevant_evidence_falls_back(self) -> None:
        original = context(
            "chunk_1",
            "无关前言。Force Merge 会合并 sealed segments。",
        )
        projected = GenerationContext(
            citation_id=original.citation_id,
            chunk_id=original.chunk_id,
            doc_id=original.doc_id,
            doc_version=original.doc_version,
            title=original.title,
            page_no=original.page_no,
            section=original.section,
            prompt_text="无关前言。",
            compression_mode="selective",
            support_spans=(
                {"start": 0, "end": 5, "quote": "无关前言。"},
            ),
        )

        run = validate_compression_run(
            CompressionRun(
                contexts=(projected,),
                configured_mode="selective",
                effective_mode="selective",
                compressor_name="injected",
            ),
            [original],
            query="Milvus Force Merge 为什么这样工作？",
        )

        self.assertEqual(run.effective_mode, "disabled")
        self.assertEqual(run.fallback_reason, "invalid_model_output")
        self.assertEqual(run.contexts[0].prompt_text, original.prompt_text)

    def test_comparison_side_cannot_borrow_retention_from_another_version(self) -> None:
        old = replace(
            context("chunk_old", "前言。v2.6 Force Merge 旧版行为。"),
            doc_version="v2.6",
        )
        new = replace(
            context("chunk_new", "v3.0 Force Merge 新版行为。"),
            doc_version="v3.0",
        )
        projected_old = replace(
            old,
            prompt_text="前言。",
            compression_mode="selective",
            support_spans=(
                {"start": 0, "end": 3, "quote": "前言。"},
            ),
        )
        projected_new = replace(
            new,
            compression_mode="selective",
            support_spans=(
                {
                    "start": 0,
                    "end": len(new.prompt_text),
                    "quote": new.prompt_text,
                },
            ),
        )

        run = validate_compression_run(
            CompressionRun(
                contexts=(projected_old, projected_new),
                configured_mode="selective",
                effective_mode="selective",
                compressor_name="injected",
            ),
            [old, new],
            query="比较 v2.6 和 v3.0 的 Force Merge",
        )

        self.assertEqual(run.effective_mode, "disabled")
        self.assertEqual(run.fallback_reason, "invalid_model_output")
        self.assertEqual(
            [item.prompt_text for item in run.contexts],
            [old.prompt_text, new.prompt_text],
        )


if __name__ == "__main__":
    unittest.main()
