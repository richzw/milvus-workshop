主线用 LangGraph，Notebook 可以这样分：

01_ingestion_local_s3.ipynb
02_text_image_embedding.ipynb
03_milvus_schema_and_insert.ipynb
04_milvus_hybrid_search.ipynb
05_langgraph_agentic_rag.ipynb
06_streamlit_ui_demo.ipynb

其中第 5 个 notebook 是核心：

Step 1: 定义 AgentState
Step 2: 写 classify_query node
Step 3: 写 rewrite_query node
Step 4: 写 retriever node
Step 5: 写 evidence_grader node
Step 6: 写 answer node
Step 7: 加 conditional edge
Step 8: 打印完整 trace
