"""Deterministic sample records for the workshop query fallback."""

from __future__ import annotations

import hashlib
from typing import Any

from agent_workshop_demo.embedding import (
    dense_vector,
    embedding_metadata,
    image_vector,
    sparse_vector,
)
from agent_workshop_demo.models import KBChunk

MAY_2026 = 1782604800000


def checksum(text: str) -> str:
    """Return a compact deterministic checksum for sample records."""

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
    return f"sha256:{digest}"


def chunk(
    *,
    doc_id: str,
    chunk_id: str,
    record_type: str,
    source_type: str,
    source_uri: str,
    doc_type: str,
    title: str,
    section: str | None,
    text: str,
    department: str,
    priority: int,
    chunk_index: int,
    page_no: int | None = None,
    parent_id: str | None = None,
    bucket: str | None = None,
    object_key: str | None = None,
    has_image_vector: bool = False,
    metadata: dict[str, Any] | None = None,
    doc_version: str = "unversioned",
    is_current: bool = True,
) -> KBChunk:
    """Build one validated sample chunk."""

    return KBChunk(
        doc_id=doc_id,
        chunk_id=chunk_id,
        parent_id=parent_id,
        record_type=record_type,
        source_type=source_type,
        source_uri=source_uri,
        bucket=bucket,
        object_key=object_key,
        doc_type=doc_type,
        title=title,
        section=section,
        page_no=page_no,
        chunk_index=chunk_index,
        text=text,
        text_summary=text[:180],
        language="mixed",
        department=department,
        updated_at=MAY_2026 - chunk_index * 86_400_000,
        created_at=MAY_2026 - (chunk_index + 10) * 86_400_000,
        priority=priority,
        doc_version=doc_version,
        is_current=is_current,
        checksum=checksum(text),
        metadata=embedding_metadata(metadata),
        has_image_vector=has_image_vector,
        text_vector=dense_vector(text),
        sparse_vector=sparse_vector(text),
        image_vector=image_vector(text) if has_image_vector else None,
    )


def load_kb_chunks() -> list[KBChunk]:
    """Load the curated local retrieval corpus."""

    return [
        chunk(
            doc_id="doc_s3_sync_design",
            chunk_id="doc_s3_sync_design_c003",
            record_type="text_chunk",
            source_type="s3",
            source_uri=(
                "s3://internal-agent-chat-demo/engineering/"
                "s3_sync_design.md"
            ),
            bucket="internal-agent-chat-demo",
            object_key="engineering/s3_sync_design.md",
            doc_type="markdown",
            title="S3 Sync Design",
            section="Ingestion Pipeline",
            chunk_index=3,
            department="engineering",
            priority=10,
            text=(
                "S3 文档同步流程会扫描 bucket，检测 updated objects，抽取 metadata，"
                "解析 Markdown/PDF/image，执行 chunking，生成 text embeddings "
                "和可选 image embeddings，最后写入 Milvus kb_chunks。"
                "The scheduler records object_key, checksum, updated_at, "
                "department, and priority for metadata filter and order_by."
            ),
            metadata={
                "parser": "markdown",
                "heading_path": ["Architecture", "Ingestion Pipeline"],
            },
        ),
        chunk(
            doc_id="doc_rag_arch",
            chunk_id="doc_rag_arch_v1_p003_c002",
            record_type="pdf_page",
            source_type="local",
            source_uri=(
                "sample_data/local_docs/engineering/"
                "rag_architecture_v1.pdf"
            ),
            doc_type="pdf",
            title="RAG Architecture v1",
            section="S3 Sync Flow",
            page_no=3,
            chunk_index=2,
            department="engineering",
            priority=8,
            doc_version="v1",
            is_current=True,
            text=(
                "Page 3 describes the Agentic RAG architecture. Local docs "
                "and S3 docs enter ingestion, then Milvus handles hybrid "
                "retrieval with dense vectors, sparse BM25, metadata filter, "
                "order_by, aggregation, and chunk/page citation."
            ),
            metadata={
                "parser": "pymupdf",
                "page_count": 12,
                "mime_type": "application/pdf",
            },
        ),
        chunk(
            doc_id="doc_rag_arch",
            chunk_id="img_s3_sync_flow",
            parent_id="doc_rag_arch_v1_p003_c002",
            record_type="image",
            source_type="local",
            source_uri="sample_data/local_docs/images/s3_sync_flow.png",
            doc_type="image",
            title="S3 Sync Flow Diagram",
            section="S3 Sync Flow",
            page_no=3,
            chunk_index=0,
            department="engineering",
            priority=8,
            doc_version="v1",
            is_current=True,
            has_image_vector=True,
            text=(
                "Diagram showing MinIO bucket scanning, document parsing, "
                "chunking, embedding generation, metadata enrichment, and "
                "Milvus insertion."
            ),
            metadata={
                "width": 1280,
                "height": 720,
                "caption_source": "manual",
                "image_model": "deterministic-placeholder",
            },
        ),
        chunk(
            doc_id="doc_milvus_feature_map",
            chunk_id="doc_milvus_feature_map_c001",
            record_type="text_chunk",
            source_type="local",
            source_uri=(
                "sample_data/local_docs/engineering/"
                "milvus_feature_map.md"
            ),
            doc_type="markdown",
            title="Milvus 3.0 Feature Map",
            section="Retrieval Layer",
            chunk_index=1,
            department="engineering",
            priority=9,
            text=(
                "Milvus 在 RAG 架构里负责向量与混合检索层。kb_chunks stores "
                "text_vector, sparse_vector, nullable image_vector, source "
                "metadata, updated_at, priority, and aggregation fields."
            ),
            metadata={"parser": "markdown"},
        ),
        chunk(
            doc_id="doc_ttl_memory_policy",
            chunk_id="doc_ttl_memory_policy_c001",
            record_type="text_chunk",
            source_type="s3",
            source_uri=(
                "s3://internal-agent-chat-demo/security/"
                "ttl_memory_policy.md"
            ),
            bucket="internal-agent-chat-demo",
            object_key="security/ttl_memory_policy.md",
            doc_type="markdown",
            title="TTL Memory Policy",
            section="Temporary Uploads",
            chunk_index=4,
            department="security",
            priority=7,
            text=(
                "Temporary documents and short-term memory use expires_at "
                "for TTL demos. The UI does not implement production ACL; "
                "filters apply by department, source_type, and doc_type."
            ),
            metadata={"parser": "markdown", "policy": "ttl"},
        ),
        chunk(
            doc_id="doc_agent_ui_mvp",
            chunk_id="doc_agent_ui_mvp_c001",
            record_type="text_chunk",
            source_type="local",
            source_uri=(
                "sample_data/local_docs/product/agent_ui_mvp.md"
            ),
            doc_type="markdown",
            title="Agent UI MVP",
            section="Demo UI",
            chunk_index=5,
            department="product",
            priority=6,
            text=(
                "The Streamlit Chat UI is query-only. It shows streaming "
                "answers, citations, recall/rerank evidence, Agent trace, "
                "aggregations, and nullable image_vector examples."
            ),
            metadata={"parser": "markdown"},
        ),
        chunk(
            doc_id="doc_customer_meeting_notes",
            chunk_id="doc_customer_meeting_notes_c001",
            record_type="text_chunk",
            source_type="s3",
            source_uri=(
                "s3://internal-agent-chat-demo/product/"
                "customer_meeting_notes.md"
            ),
            bucket="internal-agent-chat-demo",
            object_key="product/customer_meeting_notes.md",
            doc_type="markdown",
            title="Q2 Customer Meeting Notes",
            section="Top Customer Concerns",
            chunk_index=6,
            department="product",
            priority=9,
            text=(
                "本季度客户会议纪要显示，客户最关心 S3 data freshness、"
                "可核查的 citation 与 Agent trace，以及细粒度 permission "
                "和 production ACL。需要与产品路线图逐项对齐。"
            ),
            metadata={"parser": "markdown", "document_kind": "meeting_notes"},
        ),
        chunk(
            doc_id="doc_product_roadmap",
            chunk_id="doc_product_roadmap_c001",
            record_type="text_chunk",
            source_type="s3",
            source_uri=(
                "s3://internal-agent-chat-demo/product/roadmap_notes.md"
            ),
            bucket="internal-agent-chat-demo",
            object_key="product/roadmap_notes.md",
            doc_type="markdown",
            title="Product Roadmap Notes",
            section="Customer Concern Coverage",
            chunk_index=7,
            department="product",
            priority=9,
            text=(
                "产品路线图已经覆盖 S3 data freshness、citation source "
                "cards 和 Agent trace。细粒度 permission 与 production "
                "ACL 尚未覆盖，属于后续生产化阶段。"
            ),
            metadata={"parser": "markdown", "document_kind": "roadmap"},
        ),
        chunk(
            doc_id="doc_go_button_guide",
            chunk_id="doc_go_button_guide_v1_c001",
            record_type="text_chunk",
            source_type="s3",
            source_uri=(
                "s3://internal-agent-chat-demo/product/"
                "go_button_guide_v1.md"
            ),
            bucket="internal-agent-chat-demo",
            object_key="product/go_button_guide_v1.md",
            doc_type="markdown",
            title="GO Button Guide v1",
            section="Legacy Behavior",
            chunk_index=1,
            department="product",
            priority=6,
            doc_version="v1",
            is_current=False,
            text=(
                "GO按钮 v1 只表示页面跳转按钮。点击后打开活动详情页，"
                "不会直接领取奖励。"
            ),
            metadata={"parser": "markdown", "document_kind": "ui_guide"},
        ),
        chunk(
            doc_id="doc_go_button_guide",
            chunk_id="doc_go_button_guide_v1_c002",
            record_type="text_chunk",
            source_type="s3",
            source_uri=(
                "s3://internal-agent-chat-demo/product/"
                "go_button_guide_v1.md"
            ),
            bucket="internal-agent-chat-demo",
            object_key="product/go_button_guide_v1.md",
            doc_type="markdown",
            title="GO Button Guide v1",
            section="Legacy Copy",
            chunk_index=2,
            department="product",
            priority=6,
            doc_version="v1",
            is_current=False,
            text=(
                "v1 文案把跳转按钮统一标记为 GO，领取动作需要在详情页"
                "再次确认。"
            ),
            metadata={"parser": "markdown", "document_kind": "ui_guide"},
        ),
        chunk(
            doc_id="doc_go_button_guide",
            chunk_id="doc_go_button_guide_v2_c001",
            record_type="text_chunk",
            source_type="s3",
            source_uri=(
                "s3://internal-agent-chat-demo/product/"
                "go_button_guide_v2.md"
            ),
            bucket="internal-agent-chat-demo",
            object_key="product/go_button_guide_v2.md",
            doc_type="markdown",
            title="GO Button Guide v2",
            section="Current Behavior",
            chunk_index=1,
            department="product",
            priority=10,
            doc_version="v2",
            is_current=True,
            text=(
                "GO按钮 v2 表示跳转或领取按钮。满足领取条件时点击后直接"
                "领取奖励，否则跳转到活动详情页。"
            ),
            metadata={"parser": "markdown", "document_kind": "ui_guide"},
        ),
        chunk(
            doc_id="doc_go_button_guide",
            chunk_id="doc_go_button_guide_v2_c002",
            record_type="text_chunk",
            source_type="s3",
            source_uri=(
                "s3://internal-agent-chat-demo/product/"
                "go_button_guide_v2.md"
            ),
            bucket="internal-agent-chat-demo",
            object_key="product/go_button_guide_v2.md",
            doc_type="markdown",
            title="GO Button Guide v2",
            section="Current Copy",
            chunk_index=2,
            department="product",
            priority=10,
            doc_version="v2",
            is_current=True,
            text=(
                "v2 将领取按钮与跳转按钮复用为 GO按钮，并根据用户状态"
                "动态决定直接领取还是页面跳转。"
            ),
            metadata={"parser": "markdown", "document_kind": "ui_guide"},
        ),
    ]
