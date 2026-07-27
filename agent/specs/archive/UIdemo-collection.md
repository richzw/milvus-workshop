下面给出一版可以直接进入代码实现的 **Milvus collections 结构定稿**。按照你确认的方向，最终包含 3 个 Milvus collections：

```text
1. kb_chunks
   主知识库 collection，负责 Agent Chat 查询、hybrid search、citation、metadata filter、order by、aggregation、nullable image_vector。

2. conversation_memory
   Chat memory collection，用于多轮会话记忆、历史上下文检索、TTL 演示。

3. doc_dedup_signatures
   文档 / chunk 去重 collection，用于 MinHash / ingestion 去重演示。
```

`query_traces` 不建 Milvus collection，使用 `Streamlit session_state`。
`eval_questions` 不建 Milvus collection，使用：

```text
eval/questions.json
eval/golden_answers.yaml
```

---

# 1. Collection 1：`kb_chunks`

## 1.1 用途

`kb_chunks` 是主 collection，负责存储企业知识库中的：

```text
- 本地 Markdown / TXT 文档 chunk
- MinIO/mock S3 文档 chunk
- PDF page chunk
- 少量 image record
- 可选 OCR / caption 文本
```

它需要支撑：

```text
- Milvus hybrid search
- dense vector search
- sparse/BM25 search
- metadata filter
- order by updated_at / priority
- aggregation by source_type / doc_type / department
- chunk/page 级 citation
- reranker candidate recall
- nullable image_vector
```

---

## 1.2 字段总表

| 字段名                | 类型                           | 是否必需 | 是否 nullable |   是否建议建索引 | 用途                                            |
| ------------------ | ---------------------------- | ---: | ----------: | --------: | --------------------------------------------- |
| `id`               | `Int64`                      |    是 |           否 |        主键 | Milvus primary key，建议 `auto_id=True`          |
| `doc_id`           | `VarChar(128)`               |    是 |           否 |         是 | 文档级 ID                                        |
| `chunk_id`         | `VarChar(128)`               |    是 |           否 |         是 | chunk/page/image 级 ID，用于 citation             |
| `parent_id`        | `VarChar(128)`               |    否 |           是 |        可选 | image/table record 指向父 chunk/page             |
| `record_type`      | `VarChar(32)`                |    是 |           否 |         是 | `text_chunk` / `pdf_page` / `image` / `table` |
| `source_type`      | `VarChar(32)`                |    是 |           否 |         是 | `local` / `s3` / `mfs`                        |
| `source_uri`       | `VarChar(1024)`              |    是 |           否 |        可选 | 文件路径或 `s3://...`                              |
| `bucket`           | `VarChar(128)`               |    否 |           是 |        可选 | MinIO bucket                                  |
| `object_key`       | `VarChar(1024)`              |    否 |           是 |        可选 | MinIO object key                              |
| `doc_type`         | `VarChar(32)`                |    是 |           否 |         是 | `markdown` / `pdf` / `text` / `image`         |
| `title`            | `VarChar(512)`               |    是 |           否 |        可选 | 文档标题或文件名                                      |
| `section`          | `VarChar(512)`               |    否 |           是 |        可选 | Markdown heading / PDF section                |
| `page_no`          | `Int32`                      |    否 |           是 |        可选 | PDF 页码，用于 page citation                       |
| `chunk_index`      | `Int32`                      |    是 |           否 |        可选 | chunk 在文档中的顺序                                 |
| `text`             | `VarChar(16384)`             |    是 |           否 | 是，全文/BM25 | chunk 正文、OCR 文本或 image caption                |
| `text_summary`     | `VarChar(2048)`              |    否 |           是 |         否 | UI source card/snippet                        |
| `language`         | `VarChar(16)`                |    是 |           否 |        可选 | `zh` / `en` / `mixed`                         |
| `department`       | `VarChar(64)`                |    是 |           否 |         是 | filter / aggregation                          |
| `updated_at`       | `Int64`                      |    是 |           否 |         是 | epoch milliseconds，用于 order by                |
| `created_at`       | `Int64`                      |    否 |           是 |        可选 | epoch milliseconds                            |
| `priority`         | `Int32`                      |    是 |           否 |         是 | 文档优先级，用于 order by                             |
| `version`          | `VarChar(64)`                |    否 |           是 |        可选 | 文档版本                                          |
| `checksum`         | `VarChar(128)`               |    否 |           是 |        可选 | 文件或 chunk hash                                |
| `metadata`         | `JSON`                       |    否 |           是 |        可选 | parser、bbox、mime_type 等扩展信息                   |
| `has_image_vector` | `Bool`                       |    是 |           否 |         是 | UI 展示 nullable image vector                   |
| `text_vector`      | `FloatVector(dim=TEXT_DIM)`  |    是 |           否 |         是 | 文本 dense embedding                            |
| `sparse_vector`    | `SparseFloatVector`          |    是 |           否 |         是 | BM25 / sparse retrieval                       |
| `image_vector`     | `FloatVector(dim=IMAGE_DIM)` |    否 |           是 |      是，进阶 | 图片 embedding，大多数文本 chunk 为 null               |

---

## 1.3 字段详细定义

```python
KB_CHUNKS_COLLECTION = {
    "collection_name": "kb_chunks",
    "description": "Enterprise knowledge chunks for Agentic RAG demo",

    "fields": [
        # -------------------------
        # Primary key
        # -------------------------
        {
            "name": "id",
            "type": "Int64",
            "primary_key": True,
            "auto_id": True,
            "description": "Milvus primary key."
        },

        # -------------------------
        # Identity fields
        # -------------------------
        {
            "name": "doc_id",
            "type": "VarChar",
            "max_length": 128,
            "nullable": False,
            "description": "Stable document-level ID, usually hash(source_uri)."
        },
        {
            "name": "chunk_id",
            "type": "VarChar",
            "max_length": 128,
            "nullable": False,
            "description": "Stable chunk/page/image-level ID, used for citation."
        },
        {
            "name": "parent_id",
            "type": "VarChar",
            "max_length": 128,
            "nullable": True,
            "description": "Parent chunk/page ID. Useful for image/table records."
        },

        # -------------------------
        # Source fields
        # -------------------------
        {
            "name": "record_type",
            "type": "VarChar",
            "max_length": 32,
            "nullable": False,
            "description": "Record type: text_chunk, pdf_page, image, table."
        },
        {
            "name": "source_type",
            "type": "VarChar",
            "max_length": 32,
            "nullable": False,
            "description": "Source type: local, s3, mfs."
        },
        {
            "name": "source_uri",
            "type": "VarChar",
            "max_length": 1024,
            "nullable": False,
            "description": "Original file URI or object URI."
        },
        {
            "name": "bucket",
            "type": "VarChar",
            "max_length": 128,
            "nullable": True,
            "description": "MinIO/S3 bucket name."
        },
        {
            "name": "object_key",
            "type": "VarChar",
            "max_length": 1024,
            "nullable": True,
            "description": "MinIO/S3 object key."
        },
        {
            "name": "doc_type",
            "type": "VarChar",
            "max_length": 32,
            "nullable": False,
            "description": "Document type: markdown, pdf, text, image."
        },
        {
            "name": "title",
            "type": "VarChar",
            "max_length": 512,
            "nullable": False,
            "description": "Document title or filename."
        },

        # -------------------------
        # Citation fields
        # -------------------------
        {
            "name": "section",
            "type": "VarChar",
            "max_length": 512,
            "nullable": True,
            "description": "Heading, section title, or logical document section."
        },
        {
            "name": "page_no",
            "type": "Int32",
            "nullable": True,
            "description": "PDF page number. Null for non-page-based documents."
        },
        {
            "name": "chunk_index",
            "type": "Int32",
            "nullable": False,
            "description": "Chunk sequence number inside a document."
        },

        # -------------------------
        # Content fields
        # -------------------------
        {
            "name": "text",
            "type": "VarChar",
            "max_length": 16384,
            "nullable": False,
            "description": "Chunk text, OCR text, table text, or image caption."
        },
        {
            "name": "text_summary",
            "type": "VarChar",
            "max_length": 2048,
            "nullable": True,
            "description": "Short snippet or summary for UI rendering."
        },
        {
            "name": "language",
            "type": "VarChar",
            "max_length": 16,
            "nullable": False,
            "description": "Language tag: zh, en, mixed."
        },

        # -------------------------
        # Filter / aggregation / order fields
        # -------------------------
        {
            "name": "department",
            "type": "VarChar",
            "max_length": 64,
            "nullable": False,
            "description": "Business department: engineering, product, hr, security, general."
        },
        {
            "name": "updated_at",
            "type": "Int64",
            "nullable": False,
            "description": "Last updated timestamp in epoch milliseconds."
        },
        {
            "name": "created_at",
            "type": "Int64",
            "nullable": True,
            "description": "Created timestamp in epoch milliseconds."
        },
        {
            "name": "priority",
            "type": "Int32",
            "nullable": False,
            "description": "Document priority for order_by. Higher means more important."
        },
        {
            "name": "version",
            "type": "VarChar",
            "max_length": 64,
            "nullable": True,
            "description": "Document version, such as v1, v2, 2026-Q2."
        },
        {
            "name": "checksum",
            "type": "VarChar",
            "max_length": 128,
            "nullable": True,
            "description": "File or chunk checksum, such as sha256."
        },

        # -------------------------
        # Flexible metadata
        # -------------------------
        {
            "name": "metadata",
            "type": "JSON",
            "nullable": True,
            "description": "Extra metadata: parser, bbox, mime_type, image size, heading path, etc."
        },

        # -------------------------
        # Multimodal flag
        # -------------------------
        {
            "name": "has_image_vector",
            "type": "Bool",
            "nullable": False,
            "description": "Whether image_vector is present."
        },

        # -------------------------
        # Vector fields
        # -------------------------
        {
            "name": "text_vector",
            "type": "FloatVector",
            "dim": "TEXT_DIM",
            "nullable": False,
            "description": "Dense text embedding vector."
        },
        {
            "name": "sparse_vector",
            "type": "SparseFloatVector",
            "nullable": False,
            "description": "Sparse vector for BM25 / full-text retrieval."
        },
        {
            "name": "image_vector",
            "type": "FloatVector",
            "dim": "IMAGE_DIM",
            "nullable": True,
            "description": "Image embedding vector. Most text chunks have null image_vector."
        }
    ]
}
```

---

## 1.4 推荐枚举值

```python
KB_ENUMS = {
    "record_type": [
        "text_chunk",
        "pdf_page",
        "image",
        "table"
    ],
    "source_type": [
        "local",
        "s3",
        "mfs"
    ],
    "doc_type": [
        "markdown",
        "pdf",
        "text",
        "image",
        "table"
    ],
    "department": [
        "engineering",
        "product",
        "hr",
        "security",
        "general"
    ],
    "language": [
        "zh",
        "en",
        "mixed"
    ]
}
```

---

## 1.5 推荐索引

```python
KB_CHUNKS_INDEXES = {
    "text_vector": {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {
            "M": 16,
            "efConstruction": 200
        }
    },

    "sparse_vector": {
        "index_type": "SPARSE_INVERTED_INDEX",
        "metric_type": "BM25",
        "params": {}
    },

    "image_vector": {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {
            "M": 16,
            "efConstruction": 200
        },
        "note": "Advanced image retrieval. MVP only has a few non-null image_vector records."
    },

    "scalar_indexes": [
        "doc_id",
        "chunk_id",
        "record_type",
        "source_type",
        "doc_type",
        "department",
        "updated_at",
        "priority",
        "has_image_vector"
    ]
}
```

---

## 1.6 默认搜索参数

```python
KB_CHUNKS_SEARCH_DEFAULTS = {
    "search_mode": "hybrid",

    "dense_search": {
        "field": "text_vector",
        "top_k": 20,
        "metric_type": "COSINE",
        "params": {
            "ef": 64
        }
    },

    "sparse_search": {
        "field": "sparse_vector",
        "top_k": 20,
        "metric_type": "BM25"
    },

    "hybrid": {
        "milvus_top_k": 20,
        "reranker_top_k": 8,
        "answer_context_top_k": 5
    },

    "filters": {
        "source_type": ["local", "s3"],
        "doc_type": ["markdown", "pdf", "text", "image"]
    },

    "order_by": [
        "updated_at desc",
        "priority desc"
    ]
}
```

---

## 1.7 示例记录：Markdown chunk

```json
{
  "doc_id": "doc_s3_sync_design",
  "chunk_id": "doc_s3_sync_design_c003",
  "parent_id": null,

  "record_type": "text_chunk",
  "source_type": "s3",
  "source_uri": "s3://internal-agent-chat-demo/engineering/s3_sync_design.md",
  "bucket": "internal-agent-chat-demo",
  "object_key": "engineering/s3_sync_design.md",
  "doc_type": "markdown",
  "title": "S3 Sync Design",

  "section": "Ingestion Pipeline",
  "page_no": null,
  "chunk_index": 3,

  "text": "The S3 sync pipeline scans the bucket, detects updated objects, extracts metadata, chunks documents, generates embeddings, and writes records into Milvus...",
  "text_summary": "S3 sync pipeline overview.",
  "language": "en",

  "department": "engineering",
  "updated_at": 1782604800000,
  "created_at": 1782518400000,
  "priority": 10,
  "version": "v1",
  "checksum": "sha256:xxx",

  "metadata": {
    "parser": "markdown",
    "heading_path": ["Architecture", "Ingestion Pipeline"],
    "mime_type": "text/markdown"
  },

  "has_image_vector": false,
  "text_vector": [0.01, 0.02],
  "sparse_vector": {},
  "image_vector": null
}
```

---

## 1.8 示例记录：PDF page chunk

```json
{
  "doc_id": "doc_rag_arch_v1",
  "chunk_id": "doc_rag_arch_v1_p003_c002",
  "parent_id": null,

  "record_type": "pdf_page",
  "source_type": "local",
  "source_uri": "sample_data/local_docs/engineering/rag_architecture_v1.pdf",
  "bucket": null,
  "object_key": null,
  "doc_type": "pdf",
  "title": "RAG Architecture v1",

  "section": "S3 Sync Flow",
  "page_no": 3,
  "chunk_index": 2,

  "text": "Page 3 describes how local and S3 documents are processed by the ingestion pipeline...",
  "text_summary": "PDF page describing the S3 sync flow.",
  "language": "en",

  "department": "engineering",
  "updated_at": 1782518400000,
  "created_at": 1782432000000,
  "priority": 8,
  "version": "v1",
  "checksum": "sha256:yyy",

  "metadata": {
    "parser": "pymupdf",
    "page_count": 12,
    "mime_type": "application/pdf"
  },

  "has_image_vector": false,
  "text_vector": [0.01, 0.02],
  "sparse_vector": {},
  "image_vector": null
}
```

---

## 1.9 示例记录：Image record

```json
{
  "doc_id": "doc_rag_arch_v1",
  "chunk_id": "img_s3_sync_flow",
  "parent_id": "doc_rag_arch_v1_p003_c002",

  "record_type": "image",
  "source_type": "local",
  "source_uri": "sample_data/local_docs/images/s3_sync_flow.png",
  "bucket": null,
  "object_key": null,
  "doc_type": "image",
  "title": "S3 Sync Flow Diagram",

  "section": "S3 Sync Flow",
  "page_no": 3,
  "chunk_index": 0,

  "text": "Diagram showing the S3 sync flow: MinIO bucket scanning, document parsing, chunking, embedding, and Milvus insertion.",
  "text_summary": "S3 sync flow diagram.",
  "language": "en",

  "department": "engineering",
  "updated_at": 1782518400000,
  "created_at": 1782432000000,
  "priority": 8,
  "version": "v1",
  "checksum": "sha256:zzz",

  "metadata": {
    "width": 1280,
    "height": 720,
    "caption_source": "manual",
    "image_model": "dinov3",
    "mime_type": "image/png"
  },

  "has_image_vector": true,
  "text_vector": [0.01, 0.02],
  "sparse_vector": {},
  "image_vector": [0.03, 0.04]
}
```

---

# 2. Collection 2：`conversation_memory`

## 2.1 用途

`conversation_memory` 用于 chat memory 演示，支持：

```text
- 多轮对话历史检索
- 会话摘要存储
- 临时上下文记忆
- TTL / bounded memory 演示
- “刚才我们讨论了什么？”这类问题
```

这个 collection 不参与主知识库检索，而是作为 Agent workflow 的一个可选 memory retriever。

---

## 2.2 字段总表

| 字段名              | 类型                          | 是否必需 | 是否 nullable | 是否建议建索引 | 用途                                              |
| ---------------- | --------------------------- | ---: | ----------: | ------: | ----------------------------------------------- |
| `id`             | `Int64`                     |    是 |           否 |      主键 | Milvus primary key，建议 `auto_id=True`            |
| `session_id`     | `VarChar(128)`              |    是 |           否 |       是 | 会话 ID                                           |
| `turn_id`        | `VarChar(128)`              |    是 |           否 |       是 | 单轮对话 ID                                         |
| `role`           | `VarChar(32)`               |    是 |           否 |       是 | `user` / `assistant` / `system` / `summary`     |
| `content`        | `VarChar(8192)`             |    是 |           否 |    可选全文 | 原始对话内容或摘要内容                                     |
| `summary`        | `VarChar(2048)`             |    否 |           是 |       否 | 对话摘要                                            |
| `memory_type`    | `VarChar(32)`               |    是 |           否 |       是 | `short_term` / `session_summary` / `task_state` |
| `created_at`     | `Int64`                     |    是 |           否 |       是 | epoch milliseconds                              |
| `expires_at`     | `Int64`                     |    否 |           是 |       是 | 过期时间，用于 TTL/filter                              |
| `metadata`       | `JSON`                      |    否 |           是 |      可选 | 扩展信息                                            |
| `content_vector` | `FloatVector(dim=TEXT_DIM)` |    是 |           否 |       是 | memory embedding                                |

---

## 2.3 字段详细定义

```python
CONVERSATION_MEMORY_COLLECTION = {
    "collection_name": "conversation_memory",
    "description": "Conversation memory for Agent Chat demo",

    "fields": [
        # -------------------------
        # Primary key
        # -------------------------
        {
            "name": "id",
            "type": "Int64",
            "primary_key": True,
            "auto_id": True,
            "description": "Milvus primary key."
        },

        # -------------------------
        # Conversation identity
        # -------------------------
        {
            "name": "session_id",
            "type": "VarChar",
            "max_length": 128,
            "nullable": False,
            "description": "Chat session ID."
        },
        {
            "name": "turn_id",
            "type": "VarChar",
            "max_length": 128,
            "nullable": False,
            "description": "Conversation turn ID."
        },
        {
            "name": "role",
            "type": "VarChar",
            "max_length": 32,
            "nullable": False,
            "description": "Message role: user, assistant, system, summary."
        },

        # -------------------------
        # Memory content
        # -------------------------
        {
            "name": "content",
            "type": "VarChar",
            "max_length": 8192,
            "nullable": False,
            "description": "Raw message content or memory text."
        },
        {
            "name": "summary",
            "type": "VarChar",
            "max_length": 2048,
            "nullable": True,
            "description": "Optional summarized content."
        },
        {
            "name": "memory_type",
            "type": "VarChar",
            "max_length": 32,
            "nullable": False,
            "description": "Memory type: short_term, session_summary, task_state."
        },

        # -------------------------
        # Time / TTL
        # -------------------------
        {
            "name": "created_at",
            "type": "Int64",
            "nullable": False,
            "description": "Created timestamp in epoch milliseconds."
        },
        {
            "name": "expires_at",
            "type": "Int64",
            "nullable": True,
            "description": "Expiration timestamp in epoch milliseconds."
        },

        # -------------------------
        # Metadata
        # -------------------------
        {
            "name": "metadata",
            "type": "JSON",
            "nullable": True,
            "description": "Extra metadata: query_id, related_doc_ids, topic, etc."
        },

        # -------------------------
        # Vector
        # -------------------------
        {
            "name": "content_vector",
            "type": "FloatVector",
            "dim": "TEXT_DIM",
            "nullable": False,
            "description": "Embedding of memory content."
        }
    ]
}
```

---

## 2.4 推荐枚举值

```python
CONVERSATION_MEMORY_ENUMS = {
    "role": [
        "user",
        "assistant",
        "system",
        "summary"
    ],
    "memory_type": [
        "short_term",
        "session_summary",
        "task_state"
    ]
}
```

---

## 2.5 推荐索引

```python
CONVERSATION_MEMORY_INDEXES = {
    "content_vector": {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {
            "M": 16,
            "efConstruction": 200
        }
    },

    "scalar_indexes": [
        "session_id",
        "turn_id",
        "role",
        "memory_type",
        "created_at",
        "expires_at"
    ]
}
```

---

## 2.6 默认查询参数

```python
CONVERSATION_MEMORY_SEARCH_DEFAULTS = {
    "top_k": 5,
    "metric_type": "COSINE",
    "params": {
        "ef": 64
    },

    "filters": {
        "session_id": "$current_session_id",
        "expires_at": "expires_at is null or expires_at > now()"
    },

    "order_by": [
        "created_at desc"
    ]
}
```

---

## 2.7 示例记录：user message

```json
{
  "session_id": "session_demo_001",
  "turn_id": "turn_001",
  "role": "user",

  "content": "我们 S3 文档同步流程是怎么设计的？",
  "summary": null,
  "memory_type": "short_term",

  "created_at": 1782604800000,
  "expires_at": 1782691200000,

  "metadata": {
    "query_id": "query_001",
    "topic": "s3_sync",
    "ui_source": "streamlit"
  },

  "content_vector": [0.01, 0.02]
}
```

---

## 2.8 示例记录：assistant summary

```json
{
  "session_id": "session_demo_001",
  "turn_id": "turn_001_summary",
  "role": "summary",

  "content": "User asked about the S3 document sync pipeline. Assistant explained bucket scanning, metadata extraction, chunking, embedding, and Milvus insertion.",
  "summary": "Discussion about S3 document sync pipeline.",
  "memory_type": "session_summary",

  "created_at": 1782604860000,
  "expires_at": 1785196800000,

  "metadata": {
    "related_doc_ids": [
      "doc_s3_sync_design",
      "doc_rag_arch_v1"
    ],
    "citations": [
      "doc_s3_sync_design_c003",
      "doc_rag_arch_v1_p003_c002"
    ]
  },

  "content_vector": [0.01, 0.02]
}
```

---

# 3. Collection 3：`doc_dedup_signatures`

## 3.1 用途

`doc_dedup_signatures` 用于 ingestion 阶段的去重演示，支持：

```text
- 精确重复检测：checksum
- 近重复检测：MinHash signature
- doc-level dedup
- chunk-level dedup
- 多数据源重复文档识别
```

它不参与在线问答主链路，主要用于离线 ingestion notebook / CLI demo。

---

## 3.2 字段总表

| 字段名                 | 类型                              | 是否必需 | 是否 nullable | 是否建议建索引 | 用途                                   |
| ------------------- | ------------------------------- | ---: | ----------: | ------: | ------------------------------------ |
| `id`                | `Int64`                         |    是 |           否 |      主键 | Milvus primary key，建议 `auto_id=True` |
| `doc_id`            | `VarChar(128)`                  |    是 |           否 |       是 | 文档 ID                                |
| `chunk_id`          | `VarChar(128)`                  |    否 |           是 |       是 | chunk ID，doc-level 记录可为空             |
| `source_uri`        | `VarChar(1024)`                 |    是 |           否 |      可选 | 来源 URI                               |
| `source_type`       | `VarChar(32)`                   |    是 |           否 |       是 | `local` / `s3` / `mfs`               |
| `record_level`      | `VarChar(32)`                   |    是 |           否 |       是 | `doc` / `chunk`                      |
| `normalized_text`   | `VarChar(16384)`                |    是 |           否 |    可选全文 | 归一化后的文本                              |
| `checksum`          | `VarChar(128)`                  |    是 |           否 |       是 | 精确 hash                              |
| `minhash_signature` | `BinaryVector(dim=MINHASH_DIM)` |    是 |           否 |       是 | MinHash 签名                           |
| `created_at`        | `Int64`                         |    是 |           否 |       是 | epoch milliseconds                   |
| `metadata`          | `JSON`                          |    否 |           是 |      可选 | 扩展信息                                 |

---

## 3.3 字段详细定义

```python
DOC_DEDUP_SIGNATURES_COLLECTION = {
    "collection_name": "doc_dedup_signatures",
    "description": "Document and chunk dedup signatures for ingestion demo",

    "fields": [
        # -------------------------
        # Primary key
        # -------------------------
        {
            "name": "id",
            "type": "Int64",
            "primary_key": True,
            "auto_id": True,
            "description": "Milvus primary key."
        },

        # -------------------------
        # Identity
        # -------------------------
        {
            "name": "doc_id",
            "type": "VarChar",
            "max_length": 128,
            "nullable": False,
            "description": "Document-level ID."
        },
        {
            "name": "chunk_id",
            "type": "VarChar",
            "max_length": 128,
            "nullable": True,
            "description": "Chunk-level ID. Null for doc-level dedup record."
        },

        # -------------------------
        # Source
        # -------------------------
        {
            "name": "source_uri",
            "type": "VarChar",
            "max_length": 1024,
            "nullable": False,
            "description": "Original source URI."
        },
        {
            "name": "source_type",
            "type": "VarChar",
            "max_length": 32,
            "nullable": False,
            "description": "Source type: local, s3, mfs."
        },
        {
            "name": "record_level",
            "type": "VarChar",
            "max_length": 32,
            "nullable": False,
            "description": "Dedup level: doc or chunk."
        },

        # -------------------------
        # Dedup content
        # -------------------------
        {
            "name": "normalized_text",
            "type": "VarChar",
            "max_length": 16384,
            "nullable": False,
            "description": "Normalized text used for MinHash generation."
        },
        {
            "name": "checksum",
            "type": "VarChar",
            "max_length": 128,
            "nullable": False,
            "description": "Exact checksum, such as sha256."
        },

        # -------------------------
        # MinHash vector
        # -------------------------
        {
            "name": "minhash_signature",
            "type": "BinaryVector",
            "dim": "MINHASH_DIM",
            "nullable": False,
            "description": "Binary MinHash signature for near-duplicate search."
        },

        # -------------------------
        # Time / metadata
        # -------------------------
        {
            "name": "created_at",
            "type": "Int64",
            "nullable": False,
            "description": "Created timestamp in epoch milliseconds."
        },
        {
            "name": "metadata",
            "type": "JSON",
            "nullable": True,
            "description": "Extra metadata: shingle_size, num_perm, parser version, etc."
        }
    ]
}
```

---

## 3.4 推荐枚举值

```python
DOC_DEDUP_ENUMS = {
    "source_type": [
        "local",
        "s3",
        "mfs"
    ],
    "record_level": [
        "doc",
        "chunk"
    ]
}
```

---

## 3.5 推荐索引

```python
DOC_DEDUP_SIGNATURES_INDEXES = {
    "minhash_signature": {
        "index_type": "BIN_IVF_FLAT",
        "metric_type": "HAMMING",
        "params": {
            "nlist": 128
        }
    },

    "scalar_indexes": [
        "doc_id",
        "chunk_id",
        "source_type",
        "record_level",
        "checksum",
        "created_at"
    ]
}
```

---

## 3.6 默认查询参数

```python
DOC_DEDUP_SEARCH_DEFAULTS = {
    "search_mode": "minhash",

    "top_k": 5,

    "metric_type": "HAMMING",

    "params": {
        "nprobe": 8
    },

    "near_duplicate_threshold": {
        "hamming_distance_max": 32
    },

    "filters": {
        "record_level": ["doc", "chunk"]
    }
}
```

`hamming_distance_max` 只是 demo 起点，后续需要根据 `MINHASH_DIM` 和样本调。

---

## 3.7 示例记录：doc-level dedup

```json
{
  "doc_id": "doc_s3_sync_design",
  "chunk_id": null,

  "source_uri": "s3://internal-agent-chat-demo/engineering/s3_sync_design.md",
  "source_type": "s3",
  "record_level": "doc",

  "normalized_text": "s3 sync pipeline scans bucket detects updated objects extracts metadata chunks documents generates embeddings writes records into milvus",
  "checksum": "sha256:doc-level-hash",

  "minhash_signature": "binary-vector-placeholder",

  "created_at": 1782604800000,

  "metadata": {
    "normalizer": "lowercase_remove_punctuation",
    "shingle_size": 5,
    "num_perm": 128,
    "source_doc_type": "markdown"
  }
}
```

---

## 3.8 示例记录：chunk-level dedup

```json
{
  "doc_id": "doc_s3_sync_design",
  "chunk_id": "doc_s3_sync_design_c003",

  "source_uri": "s3://internal-agent-chat-demo/engineering/s3_sync_design.md",
  "source_type": "s3",
  "record_level": "chunk",

  "normalized_text": "s3 sync pipeline scans bucket detects updated objects extracts metadata chunks documents generates embeddings writes records into milvus",
  "checksum": "sha256:chunk-level-hash",

  "minhash_signature": "binary-vector-placeholder",

  "created_at": 1782604800000,

  "metadata": {
    "normalizer": "lowercase_remove_punctuation",
    "shingle_size": 5,
    "num_perm": 128,
    "source_doc_type": "markdown",
    "chunk_index": 3
  }
}
```

---

# 4. 非 Milvus 存储结构

## 4.1 `query_traces`：Streamlit session_state

不建 Milvus collection。建议结构：

```python
st.session_state["query_traces"] = {
    "query_001": {
        "query_id": "query_001",
        "session_id": "session_demo_001",
        "user_query": "我们 S3 文档同步流程是怎么设计的？",

        "answer": "...",

        "citations": [
            {
                "citation_id": "C1",
                "title": "S3 Sync Design",
                "source_uri": "s3://internal-agent-chat-demo/engineering/s3_sync_design.md",
                "page_no": None,
                "chunk_id": "doc_s3_sync_design_c003",
                "section": "Ingestion Pipeline"
            }
        ],

        "milvus_recalled": [
            {
                "rank": 1,
                "chunk_id": "doc_s3_sync_design_c003",
                "score": 0.82,
                "source_uri": "s3://internal-agent-chat-demo/engineering/s3_sync_design.md"
            }
        ],

        "reranked": [
            {
                "rerank": 1,
                "old_rank": 3,
                "chunk_id": "doc_rag_arch_v1_p003_c002",
                "rerank_score": 0.91,
                "selected": True
            }
        ],

        "trace": {
            "classify_query": {
                "query_type": "architecture",
                "need_retrieval": True
            },
            "rewrite_query": {
                "rounds": [
                    {
                        "round": 0,
                        "queries": [
                            "S3 文档同步流程",
                            "S3 document sync pipeline",
                            "object storage ingestion architecture"
                        ]
                    }
                ]
            },
            "milvus_search": {
                "mode": "hybrid",
                "top_k": 20,
                "filters": {
                    "source_type": ["local", "s3"]
                },
                "order_by": ["updated_at desc", "priority desc"]
            },
            "reranker": {
                "name": "bge-reranker",
                "input_candidates": 20,
                "output_top_k": 8
            },
            "evidence_grading": {
                "enough_evidence": True,
                "retry_count": 1,
                "max_retry": 3,
                "relevant_chunks": 5
            }
        },

        "aggregations": {
            "source_type": {
                "local": 4,
                "s3": 6
            },
            "doc_type": {
                "markdown": 5,
                "pdf": 3,
                "image": 1
            },
            "department": {
                "engineering": 8,
                "product": 1
            },
            "has_image_vector": {
                "true": 1,
                "false": 8
            }
        },

        "metrics": {
            "latency_ms": 1320,
            "retrieval_latency_ms": 180,
            "rerank_latency_ms": 210,
            "generation_latency_ms": 930,
            "num_retrieved": 20,
            "num_reranked": 8,
            "num_context_chunks": 5
        }
    }
}
```

---

## 4.2 `eval/questions.json`

建议结构：

```json
[
  {
    "question_id": "q001",
    "question": "我们 S3 文档同步流程是怎么设计的？",
    "category": "architecture",
    "expected_sources": [
      "doc_s3_sync_design_c003",
      "doc_rag_arch_v1_p003_c002"
    ],
    "metadata_filters": {
      "department": "engineering"
    }
  },
  {
    "question_id": "q002",
    "question": "RAG 架构里 Milvus 负责哪一层？",
    "category": "architecture",
    "expected_sources": [
      "doc_rag_arch_v1_p001_c001"
    ],
    "metadata_filters": {
      "department": "engineering"
    }
  }
]
```

---

## 4.3 `eval/golden_answers.yaml`

建议结构：

```yaml
q001:
  question: "我们 S3 文档同步流程是怎么设计的？"
  golden_answer: >
    S3 文档同步流程包括 bucket 扫描、对象变更检测、文档解析、
    chunking、embedding 生成，以及写入 Milvus collection。
  required_facts:
    - "bucket scanning"
    - "change detection"
    - "document parsing"
    - "chunking"
    - "embedding generation"
    - "Milvus insertion"
  required_citations:
    - "doc_s3_sync_design_c003"
    - "doc_rag_arch_v1_p003_c002"

q002:
  question: "RAG 架构里 Milvus 负责哪一层？"
  golden_answer: >
    Milvus 负责向量与混合检索层，存储文本向量、稀疏向量和可选图片向量，
    并支持 metadata filter、order by 和 aggregation。
  required_facts:
    - "hybrid retrieval"
    - "text vector"
    - "sparse vector"
    - "metadata filter"
    - "order by"
  required_citations:
    - "doc_rag_arch_v1_p001_c001"
```

---

# 5. 全局常量建议

为了代码实现方便，建议先定义统一常量。

```python
COLLECTION_NAMES = {
    "kb_chunks": "kb_chunks",
    "conversation_memory": "conversation_memory",
    "doc_dedup_signatures": "doc_dedup_signatures"
}

VECTOR_DIMS = {
    # 按实际 embedding 模型调整
    "TEXT_DIM": 1024,
    "IMAGE_DIM": 768,
    "MINHASH_DIM": 256
}

TIME_UNIT = "epoch_milliseconds"

DEFAULT_SEARCH_PARAMS = {
    "max_retry": 3,
    "milvus_top_k": 20,
    "reranker_top_k": 8,
    "answer_context_top_k": 5
}
```

---

# 6. 最终落地建议

下一步代码实现时，可以按这个顺序做：

```text
1. 定义 schema constants
   src/schema/collections.py

2. 写 create_collection 脚本
   scripts/create_collections.py

3. 写 index 创建脚本
   scripts/create_indexes.py

4. 写 ingestion 数据模型
   src/ingestion/models.py

5. 写 kb_chunks 插入逻辑
   src/milvus/insert_kb_chunks.py

6. 写 doc_dedup_signatures 插入和查询逻辑
   src/milvus/dedup.py

7. 写 conversation_memory 写入和检索逻辑
   src/milvus/memory.py

8. 写 hybrid search + reranker pipeline
   src/retrieval/hybrid_search.py
```

最终 collections 可以总结为：

```text
Milvus:
  kb_chunks
    - 主知识库
    - hybrid search
    - citation
    - metadata filter/order/aggregation
    - nullable image_vector

  conversation_memory
    - chat memory
    - session-level semantic memory
    - TTL/bounded memory demo

  doc_dedup_signatures
    - ingestion dedup
    - checksum exact match
    - MinHash near-duplicate search

Non-Milvus:
  query_traces
    - Streamlit session_state

  eval_questions
    - eval/questions.json
    - eval/golden_answers.yaml
```
