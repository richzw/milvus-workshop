<div align="center">
  <img src="ch4/images/milvus-logo.svg" alt="Milvus Logo" width="300">
</div>

**Milvus Workshop：从入门到应用**

[![en](https://img.shields.io/badge/lang-en-red.svg)](./README.md)

----

**Workshop 目标:**
*   理解向量数据库的概念及其在 AI 领域的价值。
*   掌握 Milvus 的安装、基本架构和核心组件功能。
*   能够使用 Milvus Python SDK 进行 Collection 管理、数据插入、向量搜索和数据查询等基础操作。
*   实践 Milvus 在 图片搜索、 RAG (检索增强生成)、AI Agent 等实际应用中的作用和实现方式。

**环境准备 (参与者需提前完成或Workshop开始时提供指引):**
*   安装 Docker 和 Docker Compose。
*   安装 Python 3.9+。
*   安装所需的 Python 库：`pymilvus`, `torch` (或 `transformers` 等用于生成向量), `langchain`, `langgraph`等。
*   安装 Jupyter Notebook 或 JupyterLab。

---

**第一部分：Milvus 初探 - 安装、概念与核心组件**

*   [**1.1 向量数据库Milvus概览**](./ch1/ch1_1.ipynb)
    *   什么是向量嵌入 (Vector Embedding)
    *   为什么需要向量数据库？(对比传统数据库，ANN 搜索需求)
    *   Milvus 是什么？(定位、核心特性、优势、社区概览)
    *   Milvus 核心概念解析
        *   Collection (集合)：类比关系型数据库的表。
        *   Schema (模式)：定义 Collection 的结构 (Field: Primary Key, Vector Field, Scalar Fields)。
        *   Entity (实体)：数据行，包含 ID、向量和标量属性。
        *   Index (索引)：加速向量搜索的关键，不同索引类型介绍 (如 FLAT, IVF_FLAT, HNSW)。
        *   Search/Query (搜索/查询)：ANN 搜索 vs. 基于标量字段的过滤查询。
        *   Consistency Level (一致性级别)：不同级别对数据可见性的影响。
        *   Partition (分区)：Collection 内的数据逻辑分组 (提高查询效率), 优化数据管理和查询性能。
    *   Milvus部署：Lite vs Standalone vs Cluster 模式对比 (Workshop 使用 Standalone)

*   [**1.2 Milvus 安装实战**](./ch1/ch1_2.ipynb)
    *   介绍不同的安装方式 ( Docker Compose, Kubernetes, Cloud Services )。
    *   **实操：Zilliz Cloud试用版** 
        *   注册 Zilliz Cloud 账号。
        *   创建 Milvus 实例。
        *   连接 Milvus 实例 (使用 Python SDK 或 Milvus CLI)。
    *   **实操：使用 Docker Compose 本地快速安装 Milvus Standalone （环境允许的前提）**
        *   下载配置文件 (`milvus-standalone.yaml` 或 `milvus-cluster.yaml`)。
        *   使用 `docker compose up -d` 命令启动。
        *   解释 Docker Compose 文件中的关键服务和端口。
    *   验证安装：
        *   检查 Docker 容器状态。
        *   Milvus SDKs 概览 (Python, Go, Java, Node.js, C# - 本次聚焦 Python)
        *   使用 `milvus_cli` 或 Python SDK 连接测试。
        *   介绍 WebUI：图形化管理工具。
        *   介绍 Attu：Milvus 的可视化管理工具（安装与基本使用）。

*   [**1.3 Milvus 核心架构与组件解析**](./ch1/ch1_3.ipynb) 
    *   Milvus 的分布式架构概览 (解耦设计)。
    *   核心组件功能详解：
        *   **Proxy:** 请求入口，负载均衡。
        *   **Root Coord:** 集群大脑，管理拓扑和任务。
        *   **Data Coord:** 数据写入协调者，管理数据段。
        *   **Query Coord:** 查询协调者，管理查询节点和索引加载。
        *   **Index Coord:** 索引协调者，管理索引构建任务。
        *   **Data Node:** 写入节点，处理数据写入和持久化。
        *   **Query Node:** 查询节点，加载数据和索引，执行搜索/查询。
        *   **Index Node:** 索引构建节点，执行索引构建任务。
    *   Milvus 2.6 在架构中的优化 
        *   **Streaming Node:** 实时数据流处理。

*   [**1.4 Milvus 访问控制 RBAC**](./ch1/ch1_4.ipynb)
    *   为什么使用 Milvus 时访问控制很重要？
    *   Milvus 中的 RBAC 是什么？
    *   RBAC 在 Milvus 中的工作原理
    *   如何在 Milvus 中通过 RBAC 配置访问控制
    *   示例：Milvus 驱动的 RAG 系统的访问控制设计
    *   快速提示：如何在生产中安全操作访问控制
    *   **Q&A & 小结**

---

**第二部分：Milvus 基础操作 - 使用 Python SDK**

*   [**2.1 连接 Milvus 并管理 Collections**](./ch2/ch2_1.ipynb)
    *   **概念：Collection (集合)** 
    *   **实操：定义 Collection Schema (结构)。**
        *   Field 的概念 (主键 Field, Vector Field, Scalar Field)。
        *   数据类型 (Int, Float, String, Array, JSON, FloatVector)。
        *   如何定义主键 (Primary Key)。
        *   如何定义 Vector Field (维度 Dimension)。
        *   如何定义自动ID。
    *   **实操：创建、删除、查看 Collection。**
    *   **实操：加载 (Load) 和释放 (Release) Collection。** 
    *   **Hands-on Exercise 1:** 创建一个简单的 Collection，定义几个 Scalar Field 和一个 Vector Field。

*   [**2.2 数据插入与管理**](./ch2/ch2_2.ipynb)
    *   **概念：Entity (实体)**。
    *   **概念：Partition (分区)** 。
    *   **实操：准备数据进行插入。** 。
    *   **实操：将数据插入 Collection (或指定 Partition)。**
        *   解释数据格式要求。
        *   批量插入的技巧。
    *   **实操：删除数据 (按 ID 或过滤条件)。** 
    *   **Hands-on Exercise 2:** 插入一批模拟数据到之前创建的 Collection 中，并尝试删除其中几条。

*   [**2.3 索引的构建与管理**](./ch2/ch2_3.ipynb)
    *   **概念：索引 (Index)** 
    *   **核心：近似最近邻搜索 ( Annoyingly Near Neighbor Search, ANNS )。** 
    *   介绍常见的向量索引类型及其适用场景 (简要)：
        *   **FLAT:** 精确但慢。
        *   **IVF_Flat:** 聚类 + 精确。
        *   **IVF_SQ8:** 聚类 + 量化 (压缩)。
        *   **IVF_PQ:** 聚类 + 乘积量化。
        *   **HNSW:** 基于图的索引 (高性能，常用)。
    *   介绍距离度量 ( Distance Metrics ): Euclidean distance (L2), Inner Product (IP), Cosine Similarity。
    *   **实操：为 Vector Field 创建索引。**
        *   选择索引类型和参数 (如 `nlist` for IVF, `M`, `efConstruction` for HNSW)。
    *   **实操：查看索引状态。**
    *   **Hands-on Exercise 3:** 为你的 Collection 中的 Vector Field 创建一个 HNSW 索引。

*   [**2.4 向量相似度搜索 (Search) 与混合查询 (Query/Hybrid Search)**](./ch2/ch2_4.ipynb)
    *   **概念：向量搜索 (Search)。**
    *   **实操：执行向量搜索。**
        *   准备搜索向量。
        *   设置搜索参数 (如 `anns_field`, `param`, `limit`, `output_fields`)。
        *   解释搜索结果 (`id`, `distance`, `fields`)。
    *   **概念：数据查询 (Query)。** 
    *   **实操：执行数据查询。** 
    *   **概念：混合查询 (Hybrid Search)。** - 结合向量相似度和 Sparse-BM25 过滤条件进行搜索。
    *   **实操：执行混合查询。** 
    *   **Hands-on Exercise 4:** 执行一次简单的向量搜索，一次基于 Scalar Field 的查询，一次混合查询。
    *   **Q&A & 小结**

---

**第三部分：Milvus 应用实践 - 图片搜索，RAG 与 Agent 案例**

*   [**3.1 Milvus 在图片搜索中的应用**](./ch3/ch3_1.ipynb)
    *   **流程：** 图片加载 -> 图片向量化 (Embedding) -> 存储 -> 检索。
    *   **Milvus 在图片搜索中的角色：** 存储图片特征向量，实现以图搜图 (Image-to-Image Search) 或以文搜图 (Text-to-Image Search)。
    *   **案例演示/代码讲解：** 
        *   使用预训练模型 (如 CLIP, ResNet等) 对图片数据集进行向量化。
        *   将图片路径/ID 和向量插入 Milvus。
        *   输入一张图片或一段描述文字，通过同一模型生成向量。
        *   在 Milvus 中进行向量搜索，返回相似图片的 ID 或路径。
    *   **Hands-on Exercise 1:** 实操对少量图片通过跨模态搜索 (CLIP)进行文搜图搜索。

*   [**3.2 Milvus 在 RAG (检索增强生成) 中的应用**](./ch3/ch3_2.ipynb)
    *   **RAG 流程回顾：** 文档加载 -> 分块 (Chunking) -> 向量化 (Embedding) -> 存储 -> 检索 -> 生成。
    *   **Milvus 在 RAG 中的角色：** 作为外部知识库，存储文本块及其向量，快速检索相关文本块。
    *   **案例演示/代码讲解：** 
        *   使用 LangChain 进行文档处理和向量化 (使用 Sentence Transformers, OpenAI embeddings 等)。
        *   将文本块和元数据向量化之后 插入 Milvus。
        *   接收用户问题，向量化。
        *   使用 Milvus 进行向量搜索，检索相关的文本块。
        *   将检索到的文本块作为上下文喂给 LLM 生成回答
    *   讨论：如何优化 RAG 中的检索效果 (Chunk 大小、embedding 模型选择、混合搜索的使用)。
    *   **Hands-on Exercise 2:** 尝试修改搜索参数，观察检索结果的变化。
    * 【TODO】: 添加更多的 RAG案例 或者 RAG best practice

*   [**3.3 Milvus 在 AI Agent 中的应用**](./ch3/ch3_3.ipynb)
    *   **AI Agent 架构概览：** Planning, Memory, Tools。
    *   **Milvus 在 Agent 中的角色：**
        *   **External Knowledge Base:** 存储 Agent 需要查询的外部信息 (类似于 RAG)。
        *   **Memory:** 存储 Agent 的对话历史、学习到的经验、规划步骤等 (以向量形式)。
    *   **案例演示/代码讲解：** 
        *   使用 LangGraph 实现一个 Agent
        *   Agent 识别需要外部信息 -> 将查询转化为向量 -> 在 Milvus 中搜索相关知识 -> 获取信息 -> 继续规划。
        *   Agent 存储对话片段向量 -> 在新对话开始时搜索相似历史 -> 召回相关记忆。
    *   讨论：Milvus 如何赋能 Agent 更智能地执行任务。
    *   **Hands-on Exercise 3:** 实操AI Agent Demo。

*   [**3.4 Milvus周边功能一览**](./ch3/ch3_4.ipynb)
    *   Milvus 周边工具简述 (数据同步VTS， Milvus CDC， Milvus Backup，VectorDBBench, DeepSearcher，MCP)。
    *   Milvus 在更多领域的应用前景 (推荐系统、异常检测等)。
    *   学习资源推荐 (官方文档、社区、Github)。

----------


**第四部分：Milvus 进阶实战**
*   [**4.1 Milvus 可观测性运维实战**](./ch4/ch4_1.ipynb)
    *   **可观测性部署**：基于 Prometheus + Loki + Jaeger + Grafana 完整可观测性解决方案
    *   **核心监控指标**：深入理解 Milvus 关键性能指标和健康状态
*   [**4.2 VectorDBBench 基准测试实战**](./ch4/ch4_2.ipynb)
    *   **VectorDBBench 简介**
    *   **部署与安装**
    *   **Web 界面与功能**
    *   **默认数据集基准测试**
    *   **Custom 数据集基准测试**
    *   **结果解析**
*   [**4.3 Milvus调优**](./ch4/ch4_3.ipynb)
    *   **Milvus社区问题汇总**
        *   内存：省一点，再省一点
        *   插入：丝滑入库是影响开发体验的第一步
        *   配置：半数使用问题是配置问题
    *   **优化 Milvus 性能**
        *   合理的预计数据量，表数目大小，QPS 参数等指标
        *   选择合适的索引类型和参数
        *   合理选择流式插入和批量导入
        *   谨慎使用标量过滤，删除特性等特性
        *   部署监控并观察集群情况
        *   一些常见的参数调整
    *   **加速Milvus实践: 从资源配置到参数调优**
        *   物理资源配置
        *   与向量索引相关的参数调整
    *   **调试Milvus慢查询**
        *   慢查询的定义
        *   定位工具
        *   常见慢查询场景分析
*   [**4.4 Milvus升级**](./ch4/ch4_4.ipynb)
    *   **Milvus 2.5 → 2.6 升级**

----

本workshop涵盖了 Milvus 从安装、基础概念、核心操作到实际应用的完整流程，结合理论讲解和大量实操，能有效帮助新手快速掌握 Milvus 并应用于自己的项目中。
