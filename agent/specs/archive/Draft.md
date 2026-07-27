我想以此为题，搞一个workshop，题目大体是 基于 Milvus 3.0 开发一个公司内部使用的 Agent Chat（RAG升级）。

大体规划的内容如下：
- 公司内部资料可能分布于 很多地方，云doc，S3等，异构数据源，适合Milvus 3.0，数据源集成以及处理，可以考虑使用Milvus社区的MFS  https://github.com/zilliztech/mfs/tree/main
- Embedding 方面可以试试 DINO https://huggingface.co/docs/transformers/en/model_doc/dinov3，因为公司内部资料基本参杂着文本和图片
- 应用一些Agent开发技巧，比如agentic rag等

数据构造：
- 本地文档数据； 存在于S3的文档 等
                  
Workshop 具体形式：
- 有个UI demo，直观演示，用户输入公司内部资料内容之后，输出相关答案  ：面向开箱即用的用户
- 可以运行的demo 代码，以及 jupyter notebook 详解              ：面向有开发经验的同学
- 给出一个 vibe coding的 实操步骤                             ：面向Vibe Coding的同学

首先，先搜索下Milvus 3.0 新功能有哪些https://milvus.io/docs/release_notes.md，然后看看哪些特性可以用于这个workshop项目里；然后搜索下 Agent开发技巧，用于该workshop里。先整理一个大纲
