<div align="center">
  <img src="ch4/images/milvus-logo.svg" alt="Milvus Logo" width="300">
</div>

# Milvus Workshop: From Getting Started to Application

[![zh](https://img.shields.io/badge/lang-zh-yellow.svg)](./README-zh.md)

**Workshop Objectives:**
*   Understand the concept of a vector database and its value in the AI field.
*   Master the installation, basic architecture, and core component functions of Milvus.
*   Be able to use the Milvus Python SDK for basic operations such as Collection management, data insertion, vector search, and data query.
*   Practice the role and implementation of Milvus in practical applications like image search, RAG (Retrieval-Augmented Generation), and AI Agents.

**Environment Preparation (Participants need to complete this in advance or guidance will be provided at the beginning of the workshop):**
*   Install Docker and Docker Compose.
*   Install Python 3.9+.
*   Install the required Python libraries: `pymilvus`, `torch` (or `transformers` for generating vectors), `langchain`, `langgraph`, etc.
*   Install Jupyter Notebook or JupyterLab.

---

**Part 1: Exploring Milvus - Installation, Concepts, and Core Components**

*   [**1.1 Overview of Milvus Vector Database**](./ch1/ch1_1_en.ipynb)
    *   What is a Vector Embedding?
    *   Why do we need a vector database? (Comparison with traditional databases, the need for ANN search)
    *   What is Milvus? (Positioning, core features, advantages, community overview)
    *   Analysis of Milvus Core Concepts
        *   Collection: Analogous to a table in a relational database.
        *   Schema: Defines the structure of a Collection (Fields: Primary Key, Vector Field, Scalar Fields).
        *   Entity: A row of data, containing an ID, a vector, and scalar attributes.
        *   Index: Key to accelerating vector search, introduction to different index types (e.g., FLAT, IVF_FLAT, HNSW).
        *   Search/Query: ANN search vs. filtered queries based on scalar fields.
        *   Consistency Level: The impact of different levels on data visibility.
        *   Partition: Logical grouping of data within a Collection (to improve query efficiency), optimizing data management and query performance.
    *   Milvus Deployment: Comparison of Lite vs. Standalone vs. Cluster modes (The workshop will use Standalone).

*   [**1.2 Hands-on Milvus Installation**](./ch1/ch1_2_en.ipynb)
    *   Introduction to different installation methods (Docker Compose, Kubernetes, Cloud Services).
    *   **Hands-on: Zilliz Cloud Free Tier** 
        *   Register for a Zilliz Cloud account.
        *   Create a Milvus instance.
        *   Connect to the Milvus instance (using the Python SDK or Milvus CLI).
    *   **Hands-on: Quick Local Installation of Milvus Standalone using Docker Compose (if the environment permits)**
        *   Download the configuration file (`milvus-standalone.yaml` or `milvus-cluster.yaml`).
        *   Start with the `docker compose up -d` command.
        *   Explain the key services and ports in the Docker Compose file.
    *   Verify Installation:
        *   Check the Docker container status.
        *   Overview of Milvus SDKs (Python, Go, Java, Node.js, C# - this workshop focuses on Python).
        *   Test the connection using `milvus_cli` or the Python SDK.
        *   Introduction to the WebUI: A graphical management tool.
        *   Introduction to Attu: Milvus's visualization management tool (installation and basic usage).

*   [**1.3 Analysis of Milvus Core Architecture and Components**](./ch1/ch1_3_en.ipynb) 
    *   Overview of Milvus's distributed architecture (decoupled design).
    *   Detailed explanation of core component functions:
        *   **Proxy:** Request entry point, load balancing.
        *   **Root Coord:** Cluster brain, manages topology and tasks.
        *   **Data Coord:** Data ingestion coordinator, manages data segments.
        *   **Query Coord:** Query coordinator, manages query nodes and index loading.
        *   **Index Coord:** Index coordinator, manages index-building tasks.
        *   **Data Node:** Ingestion node, handles data writing and persistence.
        *   **Query Node:** Query node, loads data and indexes, executes searches/queries.
        *   **Index Node:** Index-building node, executes index-building tasks.
    *   Architectural Optimizations in Milvus 2.6
        *   **Streaming Node:** Real-time data stream processing.
    *   **Q&A & Summary**

---

**Part 2: Basic Milvus Operations - Using the Python SDK**

*   [**2.1 Connecting to Milvus and Managing Collections**](./ch2/ch2_1_en.ipynb)
    *   **Concept: Collection** 
    *   **Hands-on: Defining a Collection Schema.**
        *   Concept of Fields (Primary Key Field, Vector Field, Scalar Field).
        *   Data types (Int, Float, String, Array, JSON, FloatVector).
        *   How to define a Primary Key.
        *   How to define a Vector Field (Dimension).
        *   How to define an auto-ID.
    *   **Hands-on: Creating, deleting, and viewing a Collection.**
    *   **Hands-on: Loading and releasing a Collection.** 
    *   **Hands-on Exercise 1:** Create a simple Collection with several Scalar Fields and one Vector Field.

*   [**2.2 Data Insertion and Management**](./ch2/ch2_2_en.ipynb)
    *   **Concept: Entity**.
    *   **Concept: Partition**.
    *   **Hands-on: Preparing data for insertion.**
    *   **Hands-on: Inserting data into a Collection (or a specific Partition).**
        *   Explain data format requirements.
        *   Techniques for batch insertion.
    *   **Hands-on: Deleting data (by ID or filter conditions).** 
    *   **Hands-on Exercise 2:** Insert a batch of mock data into the previously created Collection and try deleting some of it.

*   [**2.3 Building and Managing Indexes**](./ch2/ch2_3_en.ipynb)
    *   **Concept: Index** 
    *   **Core Concept: Approximate Nearest Neighbor Search (ANNS).** 
    *   Brief introduction to common vector index types and their use cases:
        *   **FLAT:** Accurate but slow.
        *   **IVF_FLAT:** Clustering + exact search.
        *   **IVF_SQ8:** Clustering + quantization (compression).
        *   **IVF_PQ:** Clustering + product quantization.
        *   **HNSW:** Graph-based index (high performance, commonly used).
    *   Introduction to Distance Metrics: Euclidean distance (L2), Inner Product (IP), Cosine Similarity.
    *   **Hands-on: Creating an index for a Vector Field.**
        *   Selecting the index type and parameters (e.g., `nlist` for IVF, `M`, `efConstruction` for HNSW).
    *   **Hands-on: Checking the index status.**
    *   **Hands-on Exercise 3:** Create an HNSW index for the Vector Field in your Collection.

*   [**2.4 Vector Similarity Search, Query, and Hybrid Search**](./ch2/ch2_4_en.ipynb)
    *   **Concept: Vector Search.**
    *   **Hands-on: Executing a vector search.**
        *   Prepare search vectors.
        *   Set search parameters (e.g., `anns_field`, `param`, `limit`, `output_fields`).
        *   Explain the search results (`id`, `distance`, `fields`).
    *   **Concept: Data Query.** 
    *   **Hands-on: Executing a data query.** 
    *   **Concept: Hybrid Search.** - Combining vector similarity and Sparse-BM25 filtering conditions for searching.
    *   **Hands-on: Executing a hybrid search.** 
    *   **Hands-on Exercise 4:** Perform a simple vector search, a query based on a Scalar Field, and a hybrid search.
    *   **Q&A & Summary**

---

**Part 3: Milvus Application Practice - Image Search, RAG, and Agent Use Cases**

*   [**3.1 Applying Milvus in Image Search**](./ch3/ch3_1_en.ipynb)
    *   **Process:** Image Loading -> Image Vectorization (Embedding) -> Storage -> Retrieval.
    *   **Role of Milvus in Image Search:** Store image feature vectors to enable Image-to-Image Search or Text-to-Image Search.
    *   **Case Study / Code Walkthrough:** 
        *   Use a pre-trained model (like CLIP, ResNet, etc.) to vectorize an image dataset.
        *   Insert image paths/IDs and vectors into Milvus.
        *   Input an image or a text description, and generate a vector using the same model.
        *   Perform a vector search in Milvus to return the IDs or paths of similar images.
    *   **Hands-on Exercise 1:** Perform a text-to-image search on a small number of images using cross-modal search (CLIP).

*   [**3.2 Applying Milvus in RAG (Retrieval-Augmented Generation)**](./ch3/ch3_2_en.ipynb)
    *   **RAG Process Review:** Document Loading -> Chunking -> Vectorization (Embedding) -> Storage -> Retrieval -> Generation.
    *   **Role of Milvus in RAG:** Acts as an external knowledge base to store text chunks and their vectors for fast retrieval of relevant text.
    *   **Case Study / Code Walkthrough:** 
        *   Use LangChain for document processing and vectorization (using Sentence Transformers, OpenAI embeddings, etc.).
        *   Insert vectorized text chunks and metadata into Milvus.
        *   Receive a user question and vectorize it.
        *   Use Milvus to perform a vector search and retrieve relevant text chunks.
        *   Feed the retrieved text chunks as context to an LLM to generate an answer.
    *   Discussion: How to optimize retrieval effectiveness in RAG (Chunk size, choice of embedding model, use of hybrid search).
    *   **Hands-on Exercise 2:** Try modifying the search parameters and observe the changes in the retrieval results.
    *   [TODO]: Add more RAG use cases or RAG best practices.

*   [**3.3 Applying Milvus in AI Agents**](./ch3/ch3_3_en.ipynb)
    *   **AI Agent Architecture Overview:** Planning, Memory, Tools.
    *   **Role of Milvus in Agents:**
        *   **External Knowledge Base:** Stores external information that the Agent needs to query (similar to RAG).
        *   **Memory:** Stores the Agent's dialogue history, learned experiences, planning steps, etc. (in vector form).
    *   **Case Study / Code Walkthrough:** 
        *   Implement an Agent using LangGraph.
        *   The Agent identifies the need for external information -> converts the query into a vector -> searches for relevant knowledge in Milvus -> obtains information -> continues planning.
        *   The Agent stores conversation fragment vectors -> searches for similar history at the start of a new conversation -> recalls relevant memories.
    *   Discussion: How Milvus empowers Agents to perform tasks more intelligently.
    *   **Hands-on Exercise 3:** Hands-on with an AI Agent Demo.

*   [**3.4 Overview of Milvus Ecosystem Features**](./ch3/ch3_4_en.ipynb)
    *   Brief introduction to Milvus ecosystem tools (data synchronization VTS, Milvus CDC, Milvus Backup, VectorDBBench, DeepSearcher, MCP).
    *   Future applications of Milvus in other fields (recommendation systems, anomaly detection, etc.).
    *   Recommended learning resources (official documentation, community, GitHub).

----------


**Part 4: Advanced Milvus Practice**
*   [**4.1 Hands-on Milvus Observability and Operations**](./ch4/ch4_1_en.ipynb)
    *   **Observability Deployment**: A complete observability solution based on Prometheus + Loki + Jaeger + Grafana.
    *   **Core Monitoring Metrics**: In-depth understanding of Milvus's key performance indicators and health status.
*   [**4.2 Hands-on Benchmarking with VectorDBBench**](./ch4/ch4_2_en.ipynb)
    *   **Introduction to VectorDBBench**
    *   **Deployment and Installation**
    *   **Web Interface and Features**
    *   **Benchmarking with Default Datasets**
    *   **Benchmarking with Custom Datasets**
    *   **Result Analysis**
*   [**4.3 Milvus Performance Tuning**](./ch4/ch4_3_en.ipynb)
    *   **Summary of Common Issues from the Milvus Community**
        *   Memory: Saving as much as possible.
        *   Insertion: Smooth data ingestion is the first step to a good developer experience.
        *   Configuration: Half of all usage problems are configuration problems.
    *   **Optimizing Milvus Performance**
        *   Reasonably estimate metrics like data volume, number of tables, and QPS.
        *   Choose appropriate index types and parameters.
        *   Choose wisely between streaming insertion and bulk import.
        *   Use features like scalar filtering and deletion with caution.
        *   Deploy monitoring and observe the cluster status.
        *   Some common parameter adjustments.
    *   **Accelerating Milvus in Practice: From Resource Allocation to Parameter Tuning**
        *   Physical resource configuration.
        *   Parameter tuning related to vector indexes.
*   [**4.4 Milvus Upgrade**](./ch4/ch4_4_en.ipynb)
    *   **Milvus 2.5 → 2.6 upgrade**

