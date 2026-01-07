## Milvus Community Q&A Summary

----

### About Milvus Version Upgrade

- Upgrade path, recommended 2.3.x——>2.3 latest——>2.4 latest——>2.5 latest --> 2.6 ...
- The most stable minor version for 2.4 is currently 2.4.23. Remember a general principle, the larger the third minor version number, the more stable.


----

### Insert: Smooth Insertion is the First Step Affecting Development Experience

"Insert" should be one of the most used database operations, and all subsequent work is built on the foundation of successfully inserting data. A smooth insertion experience is the first step affecting development experience, and it's a crucial step.

The Milvus community's discussion about "insert" mainly focuses on practical experience with data insertion:

- "How to optimize insertion speed?"
- "Inserting data in batches, should collection.flush() be called for each batch, or should collection.flush() be called at the end?"
- "After setting this as a primary key, why can I still continue to insert the same value repeatedly?"

Regarding this keyword, I'd like to share 3 points about data insertion:

- Batch insertion is faster than single insertion, file import (bulk_insert) is faster than batch insertion (insert). The ordinary insert interface requires data to go through a long process: Proxy—>MQ—>DataNode—Object Storage. But the file import bulk_insert insertion process is only: Proxy—>DataNode—Object Storage, which can reduce the time-consuming MQ stage. For large batch (tens of millions and above) data import, strongly recommend using the bulk_insert interface.
- When inserting data, don't call flush() interface for each batch. Milvus internally calls flush() interface periodically. Just call flush once after all data insertion is complete. Frequent flush calls after inserting data will generate many fragmented segment files, creating significant compaction pressure for the system.
- When Milvus uses the insert interface for data insertion, it won't deduplicate primary keys. If you want primary key deduplication, you can use the upsert interface. However, since upsert internally performs an additional query operation, insertion performance will be worse than insert.

-----

### Configuration: Half of Usage Issues are Configuration Issues

- "How to enable username and password verification for Milvus configuration?"
- "This service is already up via k8s, how should I modify the configuration file without wanting to shut it down?"
- "If etcd is deployed independently, are there recommended reference values for this configuration?"

As a distributed vector database, Milvus has many functional modules and also depends on third-party components such as object storage, message queue, and etcd. To ensure the Milvus cluster can deliver optimal performance in different application scenarios, Milvus exposes a large number of configuration parameters. However, facing these configuration parameters, "which to tune", "how to tune", and "how much to tune" have become difficult problems encountered by many users. It can be said that if you understand Milvus configuration parameters well, half of Milvus usage issues are resolved.

Regarding the "which to tune" issue, this is the most difficult of the three configuration questions. In different usage scenarios, the parameters to tune are different. Take performance optimization for example, parameters for search performance optimization and parameters for insertion performance optimization are definitely different. This relies heavily on actual usage experience. Zilliz (Milvus original factory) official public account has many technical articles written by Milvus senior R&D colleagues and Milvus deep users, which mention a lot of configuration tuning experience. Everyone can fully utilize these resources.

[**Milvus Configuration Explained**](https://mp.weixin.qq.com/s?__biz=MzUzMDI5OTA5NQ==&mid=2247496778&idx=1&sn=018256ae415356e3ed357ee473dc1627&chksm=fa5155f2cd26dce4095b57fa4eb5c7e67e9eb49ed2762ce618e1134bb33022049e75c25e49ab&scene=21#wechat_redirect)







## Optimizing Milvus Performance

------

### Experience 1: Reasonably Estimate Data Volume, Number of Tables, QPS and Other Indicators

Before deploying Milvus, you first need to decide on machine resources, specifications, and some dependent resources. Here are factors you need to consider:

- How many tables?
- How much data in each table?
- What are the QPS requirements for each table?
- Do you need to store scalar fields? If there are strings, what's the average string length?
- Are there deletions and streaming inserts? Approximately what proportion of data needs to be updated daily?
  
Based on the above factors, you can follow these empirical conclusions:

- Node resource usage can be calculated through [sizing tool](https://milvus.io/tools/sizing/). Typically 8G memory can support more than 5m 128dim vector data and 1m 768dim data.
- By default, Milvus creates 256 message queue topics. If the number of tables is relatively small, you can adjust `rootCoord.dmlChannelNum` to reduce the number of topics and lower message queue load.
- By default, each collection uses 2 message queue topics (shards). If writes are very large or data volume is extremely large, you need to adjust the collection's shard count. It's recommended that each shard doesn't exceed 10M/s for write/delete, and single shard data volume doesn't exceed 1B vectors. Too many shards will also affect write performance, so it's not recommended to exceed 8 shards per table.
- Calculate needed CPU resources based on [benchmark](https://milvus.io/docs/benchmark.md) results. For small data volume scenarios (less than 5m), using multiple replicas can expand query performance, but it's recommended not to exceed 10 replicas. For medium to large data volume scenarios, usually expanding querynode can automatically load balance, no need to use multiple replicas to improve QPS.
- All scalar fields are currently also loaded into memory and will consume memory. Please reserve more than twice the original data type memory in capacity planning.
- Milvus has a lot of redundant data in the process of storing data (https://github.com/milvus-io/milvus/issues/20453). Considering Minio's 2,4 erasure code has two replica redundancy, we recommend Minio contains at least 6 times the data disk storage. At the same time, Pulsar/Kafka needs to contain three times the storage of recent five days' write volume. Reasonably adjusting data retention time and GC time can greatly reduce disk usage. By default, data will be retained for 5 days. Personally recommend appropriately shortening data expiration time, but try to retain more than 1 day to avoid data loss or accidental deletion.
- Etcd as Milvus's metadata storage and service discovery node, please try to use SSD disk and deploy independently. Typically Etcd memory usage won't exceed 4GB. By adjusting parameters, you can quickly clean up historical versions in etcd to reduce memory usage.
- Pulsar/Kafka as Milvus's log storage, the zookeeper cluster it depends on also has relatively high performance requirements. It's recommended to use SSD and deploy independently.

----

### Experience 2: Choose Appropriate Index Type and Parameters

Index selection is crucial for vector recall performance. Milvus supports multiple different indexes such as Annoy, Faiss, HNSW, DiskANN, and users can choose based on latency, memory usage, and recall rate requirements.

Index selection steps are generally as follows:

- 1) Do you need exact results?
  - Only Faiss's Flat index supports exact results, but note that Flat index retrieval speed is very slow. Query performance is usually more than two orders of magnitude lower than other index types supported by Milvus, so it's only suitable for tens of millions data volume small queries (Flat on GPU is on the way, stay tuned)

- 2) Can the data volume be loaded into memory?
  - For large data volumes with insufficient memory scenarios, Milvus provides two solutions:
    - DiskANN
      - DiskANN relies on high-performance disk indexes, using NVMe disks to cache full data, only storing quantized data in memory.
      - DiskANN is suitable for scenarios with high query Recall requirements but low QPS.
      - DiskANN key parameters:
        - **search_list**: Larger search_list means higher recall but worse performance. search_list size should not be smaller than K. For smaller K, it's recommended to set the ratio of search_list to K relatively larger. This ratio can gradually approach 1 as K increases.
    - IVF_PQ
      - For scenarios with low accuracy requirements or extremely high performance requirements.
      - IVF PQ core is two algorithms, IVF + PQ quantization, where quantization can greatly reduce vector memory usage.
        - IVF parameters
          - **nlist**: Generally recommend nlist = 4*sqrt(N). For Milvus, a Segment default is 512M data. For 128dim vectors, a segment contains 1m data, so optimal nlist is around 1000.
          - **nprobe**: nprobe can adjust search data volume during Search. Larger nprobe means higher recall but worse performance. Specific nprobe needs to be decided based on query accuracy requirements. Starting from nprobe = 16 would be a good try.
        - PQ parameters
          - **M**: Number of segments for vector PQ, generally recommend setting to 1/4 of vector dimension. Smaller M means less memory usage, faster query speed, and lower accuracy.
          - **N bits**: Number of bits occupied by each segment quantizer, default is 8, not recommended to adjust.
- 3) Are index building and memory resources sufficient?
  - **Performance priority, choose HNSW index**
    - HNSW index is currently the fastest performing index supported by Milvus. Our test reports are also based on HNSW as the test basis.
    - HNSW memory overhead is relatively high, typically requiring more than 1.5 - 2 times the original vector memory.
    - HNSW parameters
      - M: Represents the number of edges for each vector during table building. Larger M means higher memory consumption and better query performance on high-dimensional datasets. Usually recommended to set between 8-32.
      - ef_construction: Controls index time and index accuracy. Larger ef_construction means longer index building but higher query accuracy. Note that increasing ef_construction cannot infinitely increase index quality. Common ef_construction parameter is 128.
      - ef: Controls search accuracy and search performance. Note that ef must be greater than K.
  - **Resource priority, choose IVF_FLAT or IVF_SQ8 index**
    - IVF index after Milvus sharding can also achieve relatively good recall rates, with memory usage and index building speed much lower than HNSW
    - IVF_SQ8 compared to IVF converts vector data from float32 to int8, can reduce 4 times memory usage, but has a significant impact on recall rate. Not recommended if requiring more than 95% recall accuracy.
    - IVF class index parameters are similar to IVFPQ, so no detailed introduction here.
  - During retrieval, Milvus's **query consistency** will also have a significant impact on queries. Usually, for scenarios with high consistency requirements, it's recommended to use eventual consistency or bounded consistency. By default, Milvus chooses bounded consistency with a 3s window.

-----

### Experience 3: Reasonably Choose Streaming Insert and Bulk Import

Milvus natively supports stream-batch integration, supporting both streaming write and batch write (BulkInsert) modes. Most users, when first encountering Milvus, will choose streaming write, which has good real-time performance and also avoids Compaction pressure from small files in batch write.

If there are scenarios with large amounts of offline writing, it's recommended to use BulkInsert. The reason is that BulkInsert won't cause too much impact on query performance and also greatly reduces pressure on message queue from streaming writes. How to reasonably choose streaming or batch write?

- Single write exceeding 100MB or more, recommend batch write
- Want to minimize write impact on online queries, recommend batch write
- Want writes to be visible in real-time, recommend streaming write
- Single write less than 10MB, recommend streaming write

Based on choosing the write method, there are several more experiences to note:

- Try to write in batches as much as possible, overall throughput will be higher. Recommend each write size controlled at 10M
- Single Shard streaming write volume should not exceed 10M/s
- When DataNode is more than Shard, some DataNodes may not be able to get load
- Current import supported file size limit is 1GB, will support larger import file size limits next
- Don't recommend frequent import of small files, will put relatively large pressure on compaction

----

### Experience 4: Use Scalar Filtering, Deletion Features and Other Features Carefully

As a database, Milvus supports advanced features such as deletion, scalar filtering, TimeTravel, etc. If you don't understand the underlying principles, using these advanced features may have a relatively serious impact on stability and performance. Here are some usage considerations:

- Milvus uses pre-filtering, i.e., first do scalar filtering to generate Bitset, then remove entities that don't meet conditions based on Bitset during vector retrieval. For graph indexes like HNSW, scalar filtering won't speed up queries and may even cause worse performance. Especially for strong filtering conditions (like PK=1 which is globally unique), scalar filtering may even cause single query time longer than brute force search. For this situation, users can also choose to bypass through post-filtering, first query TopK data based on Milvus, then filter based on other databases.
- For scenarios where filtering conditions are relatively certain, using Partition to physically partition data and specifying Partition during query performs better.
- Milvus deletion is mark deletion, cleaned during compaction, so deleted data still occupies memory. A lot of deletions will also cause query performance degradation, and a lot of compaction may cause increased index building pressure and a series of impacts. In scenarios requiring frequent large-scale deletions, you may need to adjust some compaction parameters to ensure deleted data can be cleaned up in time.
- Milvus supports automatic data expiration function (TTL), which can regularly clean up expired data.
- If you need to fully update a Collection's data, recommend using new table + import data + Alias switch solution.
- When specifying Output field, if you need to get scalar fields, it will be obtained from object storage, and throughput and latency will be significantly affected.

Of course, Milvus subsequent versions will make targeted optimizations for the above capabilities, especially deletion and scalar filtering scenarios. Milvus's new generation scalar execution engine is also in development. Everyone is welcome to participate and give more constructive opinions.

----

### Experience 5: Deploy Monitoring and Observe Cluster Status

Observability is a very important part of user production environment deployment. Milvus 2.2 reorganized monitoring metrics and corrected metric accuracy. We strongly recommend your production cluster deploy monitoring and conduct performance testing before going live.

Here are some monitoring metrics we recommend you pay attention to:

- Proxy
  - Query latency: milvus_proxy_sq_latency/milvus_proxy_collection_sq_latency
  - Write/delete latency: milvus_proxy_mutation_latency
  - Write traffic: milvus_proxy_receive_bytes_count
  - Query return traffic: milvus_proxy_send_bytes_count
- QueryNode
  - Loaded data volume: milvus_querynode_entity_num
  - Query request queue time: milvus_querynode_sq_queue_latency
  - Single Segment query time: milvus_querynode_sq_segment_latency
- IndexNode
  - Index building time: milvus_indexnode_build_index_latency
- DataNode
  - Flush time: milvus_datanode_save_latency
  - Compaction time: milvus_datanode_compaction_latency

-----

### Experience 6: Some Common Parameter Adjustments

To make Milvus run faster and more stable, making some customized adjustments for your usage scenario and hardware resource situation is naturally unavoidable. You can start by understanding the following parameters:

- **Segment size**: Larger Segment size means better query performance, slower index building, and less easy load balancing. Milvus default choice of 512M Segment size mainly considers machines with less memory. For users with 8G-16G memory, it's recommended to adjust Segment size to 1024M, machines with 16G or more can adjust to 2G.

- **Segment seal portion**: When Growing Segment reaches Segment size * seal portion, streaming data will be converted to batch data. Usually it's recommended to control Growing segment size around 100-200M, adjusting this value smaller helps reduce query latency in streaming write scenarios.

- **DataNode Segment SyncPeriod**: Milvus regularly Syncs data to object storage. More frequent Sync means faster failure recovery, but too frequent sync will cause Milvus to generate a lot of small files, putting significant pressure on object storage.

- **Quota related parameters**: Currently supports limiting Milvus write, delete traffic, query QPS, and memory protection. When encountering performance issues, also observe whether corresponding throttling was triggered.





## Accelerating Milvus Practice: From Resource Configuration to Parameter Tuning

------

### Overview

With the rise of Retrieval Augmented Generation (RAG) technology, **vector databases are increasingly becoming a focus of industry attention**. **Milvus, as a widely popular open-source vector database, has been widely deployed in actual applications by many developers and enterprises**. The community's demand for Milvus performance tuning is also growing.

Based on the team's practical experience in engineering development and proof of concept (PoC) projects, several key directions for performance tuning have been summarized. Before delving into tuning strategies, we need to clarify where Milvus's resource consumption is mainly concentrated. The core difference between vector databases and traditional databases lies in **vector search**. In most scenarios, **vector search accounts for the vast majority of Milvus's computing resources**. Therefore, this discussion will focus on how to accelerate vector computation, mainly covering the following two major types of methods:
*   First, the most direct means is to **increase physical resources**. But in distributed systems, how to reasonably allocate new resources to achieve optimal results is a question worth exploring.
*   Second, vector search efficiency is highly dependent on **vector indexes**. We will deeply analyze the characteristics of different vector indexes and explore how to tune parameters. Specifically, which parameters are tunable? Which parameters have the most significant impact on performance?

It must be emphasized that **any tuning strategy comes with corresponding trade-offs**, please be sure to implement cautiously with a full understanding of business requirements and system impacts.

-------

### 1. Physical Resource Configuration
For a distributed system, adding physical resources doesn't take effect immediately. Milvus needs to **explicitly allocate resources to specific functional modules**.

The two modules with the greatest impact on performance are:
*   **`index_node`**: Affects **vector index building performance**, externally manifested as **building speed**.
*   **`query_node`**: Affects **vector index search performance**, externally manifested as **Search Latency and QPS** (Queries Per Second).

Milvus supports **dynamic scaling**. For example, if users need to build a read-only system where data is imported once and there will be no subsequent data updates, the best practice is: when inserting data, **increase the number of `index_node` as much as possible or increase the resources of a single `index_node`**. After index building is complete, you can remove extra `index_node` and allocate resources to `query_node`.

In production scenarios, Milvus provides more operations tool support (prometheus/grafana). Users can manually adjust an appropriate architecture ratio according to actual business volume changes to fully utilize resources.

Besides the number of nodes, **the underlying architecture of physical machines** is also one of the important influencing factors. Vector index operations are **compute-intensive tasks**. Milvus's performance bottleneck most often appears in **CPU** (except when mmap or diskann is enabled). Milvus will continuously track the most advanced SIMD technology. Newer and more complete CPU instruction sets have huge benefits for vector computation.

-------

### 2. Parameter Adjustments Related to Vector Indexes
Except brute force search (FLAT), vector indexes generally perform **approximate vector search**. From the algorithm level, they can be roughly divided into two categories: **IVF-based and Graph-based**. Among them, **Graph indexes have very obvious advantages in Search performance, and HNSW is the recommended choice for the vast majority of vector databases on the market**. The cost of excellent Search performance is mainly that more resources are needed when building indexes, and the built indexes will also be relatively larger, occupying relatively more memory.

Several optimization directions related to Graph indexes have been organized.

#### 2.1 Increase `segment_size`
Milvus divides data into multiple **`segments`** for management, and each `segment` will build a separate vector index. Graph indexes have a relatively counter-intuitive characteristic: the amount of computation per search is not very sensitive to `segment` size. Simply put, if a Graph expands to 10x data volume, the computation amount per search may only increase by less than 50%. This means that when Milvus uses **HNSW**, when importing the same amount of data, if **`segment_size` is increased, the number of `segments` will decrease, and performance will significantly improve**.

The current default `dataCoord.segment.maxSize` is **1024MB**, which can be modified through `milvus.yaml`.

**Cost**: If `segment` is too large, there will be certain **stability risks**. Especially when data is frequently updated and memory waterline is high, it may cause load failure.

**Adjustment suggestions**:
*   It's recommended to control `segment_size` at **10% of single `querynode` memory**.
*   For common 8c32g pods, it can be adjusted to **3-4GB**.
*   For scenarios with less data updates, or sufficient memory space, pursuing ultimate performance, it can also be adjusted larger, but still needs to be cautious.

#### 2.2 Use `quantization`
**Quantization** refers to **compressing float32 data** when building indexes. After compression, there are two advantages:
1.  **Index memory usage becomes less**;
2.  Compressed data supports **instruction set optimization, computation efficiency greatly improves**.

**Cost**: Using quantization alone **irreversibly loses precision**, not suitable for **high recall** requirements >=99%.

Milvus provides a large number of quantization indexes for users to choose from, such as **IVF / HNSW stacked with SQ / PQ / PRQ / RabitQ**, etc.

**Adjustment suggestions**: Based on long-term testing experience, it's quite recommended to set `index_type` to **`HNSW_SQ`**, quantization type set to **`SQ8`**. Compared to HNSW without quantization, in **recall 0.95 scenarios, memory is reduced by more than half, QPS increases by about 25%**.

#### 2.3 Index Parameters
Index parameters will directly affect vector approximate search algorithm execution. Can be simply understood as, **any index parameter adjustment is a trade-off between recall and QPS**.

Index parameters can be divided into two categories:
1.  **Index build parameters**: Specified when `create_index`, cannot be modified during `search`. For example **`M`**, **`efConstruction`**. Larger values mean higher recall and lower QPS.
2.  **Index query parameters**: Can be modified according to actual business conditions during `search`. For example **`ef`**. Larger values mean higher recall and lower QPS.

**Parameter suggestions**: For **HNSW**, common suggestion is to set **`M=16`**, **`efConstruction=256`**. **`ef`** set to **`topk` size**.

------

### 3. Other Tips
#### 3.1 `flush`
Milvus uses **`growing segment`** to manage newly inserted data. The `growing` design ensures real-time visibility of newly inserted data and dynamic data growth. But as a cost, the search performance of the index used by `growing` is poor and will slow down overall search. Calling **`flush()`** can explicitly require Milvus to convert current `growing segment` to **`sealed segment`** to build more efficient indexes for search.

**Cost**: After `flush()`, new `growing segment` will be created. **Frequent `flush()` will cause `segment` fragmentation**. As mentioned in 2.1, too many `segment` numbers is a disaster for performance.

**Adjustment suggestions**: It's recommended to **call after stage-based concentrated data insertion is complete**. For indefinite time and indefinite amount data update behaviors without clear behavior expectations, Milvus also has **auto-flush mechanism** to handle.

#### 3.2 `wait_index_building`
**Index building requires a lot of computing resources**. The completion of `insert request` doesn't mean the index has been built. Especially in large-scale write situations, if querying immediately after `insert` completion, Milvus cannot perform at its best.

You can monitor index building progress through `utility.index_building_progress`.
Related PyMilvus API reference: `https://milvus.io/api-reference/pymilvus/v2.5.x/ORM/utility/index_building_progress.md`.

#### 3.3 `compact`
Data deletion and modification will also have a certain degree of impact on indexes. Especially after `upsert` or `delete` operations increase, vector indexes will be destroyed to a certain extent, affecting search performance. Users can explicitly call **`compact()`** to require Milvus to reorganize the current collection's data. In this process, **fragmented `segments` will be merged, vector indexes will also be rebuilt and repaired, and performance will recover**.

**Cost**: `compact()` **involves data repartitioning and index rebuilding, requires a lot of resources**. Not recommended to call frequently. Milvus background also has **auto-compact mechanism** to merge fragmented `segments` and clean up data that needs to be `deleted`.

#### 3.4 Special Scenario - Large K
Common Search behaviors mostly have `topk<1000`, at which time **Graph indexes have obvious advantages**. But when users have higher requirements for `topk` (`>10,000`), **IVF will have more advantages**.

#### 3.5 Special Scenario - Ultimate Compression
HNSW can provide particularly high recall (`>90%`), but some users don't have such high recall requirements and want to trade recall for less memory usage and higher performance. You can consider setting the index to **`SCANN`** or **`IVF_RABITQ`**. Among them, `IVF_RABITQ` has ultra-high compression ratio. In public dataset Cohere 1M * 768dim testing, it achieved more than 75% recall with 1/32 compression ratio.

#### 3.6 Special Scenario - Scalar Filtering
**Scalar filtering is a very challenging task in the vector search field**. Milvus has made a lot of filtering optimizations for HNSW, leading competitors in most production scenarios. The optimized HNSW maintains very high recall under any filtering conditions. But it's also noted that its **QPS performance is poor at 85-95% filtering volume**. If users have a large number of 90% filtering volume requirements, you can consider using **IVF index**.

------

Finally, it must be emphasized that Milvus is a very complex system. Its flexibility can support various user needs, but **there is no set of configuration parameters that satisfies all users**. **Any tuning behavior is a trade-off for different needs, it's recommended to adjust dynamically according to actual business**.


## [Debugging Milvus Slow Queries](https://milvus.io/blog/how-to-debug-slow-requests-in-milvus.md)

------

### Overview

Milvus searches usually only take milliseconds, but under high concurrency, complex filtering, or limited runtime environments, latency can rise to seconds, affecting user experience. Locating slow queries needs to answer two questions:

- ①Occurrence frequency;
- ②At which stage is the time consumed

-------

###  Location Tools

#### Metrics (Prometheus + Grafana)
  - Service Quality → Slow Query:
    - Requests exceeding `proxy.slowQuerySpanInSeconds` (default 5 s) will be marked.
  - Service Quality → Search Latency:
    - Overall distribution;
    - If the panel is normal but the client is still slow, most likely it's a network or application layer issue.
  - Query Node → Search Latency by Phase:
    - Splits into queue / query / reduce;
    - Auxiliary panels Scalar Filter Latency, Vector Search Latency, Wait tSafe Latency can further locate.
#### Logs
  - Milvus adds "[Search slow]" marker for requests executing > 1 s and records traceID, collection, filter DSL, topk, metric_type, nprobe, nq and other detailed parameters.
  - Experience thresholds:
    - `< 30 ms` = healthy;
    - `> 100 ms` = needs attention;
    - `> 1 s` = slow query.
#### Metrics tell you where the time is going; logs tell you which queries are hit.

-------

###  Common Root Causes and Fix Solutions

#### Overload
  - Phenomenon: All request latency increases, queue latency is significant; a request NQ is very large in logs.
  - Solution: Control single NQ, or horizontally scale Query Node.
#### Low Filtering Efficiency
  - Phenomenon: Only queries with filters become slow, Scalar Filter Latency or Wait tSafe Latency increases.
  - Solution:
    - Use IN instead of long OR chains; use expression templates to reduce parsing overhead.
    - Build scalar index for filter fields; use path/flat index for JSON fields (provided from 2.6), JSON shredding will be supported in the future.
    - If consistency requirements are not high, change to Bounded / Eventually, reduce tSafe waiting.
#### Inappropriate Vector Index Selection
  - Phenomenon: High Vector Search Latency, or disk I/O saturation (DiskANN/MMAP), slow cold start after restart.
  - Suggestions:
    - Float vectors: HNSW (memory priority), IVF series (trade-off), DiskANN (billions level data, needs high bandwidth).
    - Binary vectors: 2.6 newly added MINHASH_LSH + MHJACCARD.
    - Enable MMAP to map indexes on demand; reasonably adjust index / search parameters; warm up hot segments after restart.
#### Runtime and Environment
  - Phenomenon: CPU, I/O peaks during background compaction / migration / build index; frequent upsert generates many unindexed small segments; version Bugs.
  - Countermeasures:
    - Schedule background tasks to off-peak hours; release unused collections; warm up cache.
    - Batch upsert to reduce small segments; upgrade versions timely; reserve resources for latency-sensitive loads


