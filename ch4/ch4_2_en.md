# 4.2 VectorDBBench Benchmark Testing Practice

This section will introduce and practice the deployment and use of VectorDBBench, a mainstream vector database benchmark testing tool.

## 4.2.1 VectorDBBench Introduction

VectorDBBench (VDBBench) is an open-source benchmark testing tool for mainstream vector databases and cloud services, supporting performance and cost-effectiveness comparison of multiple databases, providing a visual interface and rich test scenarios, making it convenient for users to reproduce results or test new systems.

- Supports multiple databases (such as Milvus, Zilliz Cloud, Qdrant, Weaviate, PgVector, Redis, Chroma, etc.)
- Provides multiple test scenarios such as insert, search, filter search, streaming search, etc.
- Built-in multiple public datasets (such as SIFT, GIST, Cohere, OpenAI C4, etc.)
- Supports visual interface, convenient for configuring tests, viewing and comparing test results

Main Features:
1. Easy-to-use Web UI, supports test configuration and visualization analysis of results
2. Standardized testing process and metrics collection, supports multi-scenario expansion (such as filtering, streaming)
3. Supports multiple mainstream and emerging vector databases, convenient for horizontal comparison

For more introduction, see [Official Documentation](https://github.com/zilliztech/VectorDBBench)

## 4.2.2 VectorDBBench Deployment (Multiple Database Clients)

### Environment Requirements
- Python >= 3.11

### Installation
Install only Milvus/Zilliz Cloud client:
```shell
pip install vectordb-bench
```

Install all supported database clients (if comparing multiple databases):
```shell
pip install vectordb-bench[all]
```

Install specified database client (such as Qdrant):
```shell
pip install vectordb-bench[qdrant]
```

Supported database clients and installation commands are as follows:

| Database Client | Installation Command |
|-------------|----------|
| pymilvus, zilliz_cloud | `pip install vectordb-bench` |
| all         | `pip install vectordb-bench[all]` |
| qdrant      | `pip install vectordb-bench[qdrant]` |
| pinecone    | `pip install vectordb-bench[pinecone]` |
| weaviate    | `pip install vectordb-bench[weaviate]` |
| elastic, aliyun_elasticsearch | `pip install vectordb-bench[elastic]` |
| pgvector, pgvectorscale, pgdiskann, alloydb | `pip install vectordb-bench[pgvector]` |
| redis       | `pip install vectordb-bench[redis]` |
| chromadb    | `pip install vectordb-bench[chromadb]` |
| awsopensearch | `pip install vectordb-bench[opensearch]` |
| oceanbase   | `pip install vectordb-bench[oceanbase]` |
| ...         | ... |

For more database support and installation methods, see the official README.

## 4.2.3 VectorDBBench Web Startup and Feature Introduction

### Start Web Interface

After installation, run directly:
```shell
init_bench
```
or
```shell
python -m vectordb_bench
```

By default, it will start a local Web service (such as http://localhost:8501), open a browser to access.

### Main Function Modules
- **Run Test**: Select database, fill in connection information, select test cases, initiate benchmark test. Supports multiple databases, multiple test cases, and multiple datasets combination testing.
- **Result**: View all test results, supports multi-round comparison and filtering, supports multi-dimensional display such as QPS, latency, cost-effectiveness, etc.
- **Custom Dataset**: Custom datasets and test cases, supports detailed parameter configuration (such as dimension, data volume, label distribution, etc.).
- **Queries Per Dollar**: Shows the number of queries that can be processed per dollar, convenient for cost-effectiveness analysis.
- **Tables**: Compares various indicators under different datasets in table form.
- **Concurrent Performance**: Shows QPS and latency trends under different concurrency levels.
- **Label Filter Performance**: Shows performance under different label filtering ratios.
- **Int Filter Performance**: Shows performance under different integer filtering ratios.
- **Streaming Performance**: Shows retrieval performance under continuous insertion pressure.

#### Typical Interface Examples

![VDBBench Main Interface](./images/VectorDBBench.png)

![VDBBench Results Page](https://github.com/zilliztech/VectorDBBench/assets/105927039/8a981327-c1c6-4796-8a85-c86154cb5472)

## 4.2.4 Benchmark Testing Using Default Datasets

### Test Process Description
VDBBench standard testing process is divided into three main stages:

1. **Load (Insert + Optimize)**: Single-process serial insertion of all data, record insertion time (insert_duration); some databases will also perform index optimization, record optimization time (optimize_duration). Total time (load_duration) reflects the overall loading capability of the database from zero to queryable.
2. **Serial Search Test (Serial Retrieval)**: Single-process serial retrieval, record recall and latency (latency_p99) for each query. p99 latency focuses on the slowest 1% of requests, suitable for high-requirement scenarios.
3. **Concurrent Search Test (Concurrent Retrieval)**: Multi-process concurrent retrieval, gradually increase concurrency (such as 1~80), run 30 seconds for each group, record QPS and latency under different concurrency levels, finally take the maximum QPS as max-qps.

Also supports:
- **Filter Search Test**: Add label or integer filter conditions during retrieval, examine performance changes under different filtering ratios.
- **Streaming Search Test**: Stage retrieval under continuous insertion pressure, examine the retrieval capability of the database in streaming write scenarios.

### Practice Steps
1. After starting the Web interface, enter the **Run Test** page
2. Select the database system to test (such as Milvus, Qdrant, PgVector, etc.), fill in connection information
3. Select test cases (such as Capacity, Performance, Filtering, Streaming, etc.), multiple selections allowed
4. Select default dataset (such as SIFT, GIST, Cohere, OpenAI C4, etc.)
5. Fill in Task Label, click submit, wait for test completion

### Notes
- Default datasets are built-in, no need to manually upload
- Can choose different test cases and data scales according to actual needs
- Recommended to deploy test client and database service in the same LAN to reduce network latency impact

After test completion, you can view detailed results and comparative analysis in the **Result** page

## 4.2.5 Benchmark Testing Using Custom Dataset

### 4.2.5.1 Data Preparation

#### Script to Generate Initial Data
```python
import pandas as pd
import numpy as np

def generate_csv(num_records: int, dim: int, filename: str):
    ids = range(num_records)
    vectors = np.random.rand(num_records, dim).round(6)  # Keep 6 decimal places
    emb_str = [str(list(vec)) for vec in vectors]
    df = pd.DataFrame({
        'id': ids,
        'emb': emb_str
    })
    df.to_csv(filename, index=False)
    print(f"Generated file {filename}, total {num_records} records, vector dimension {dim}")

if __name__ == "__main__":
    num_records = 3000  # Number of data to generate
    dim = 768           # Vector dimension

    generate_csv(num_records, dim, "train.csv")
    generate_csv(num_records, dim, "test.csv")

```

#### Prepare Your Own Initial Data
Data requirements:
##### **1. CSV Format**

- First column is **id** (unique identifier)
- Second column is **vector** (string form of float array, such as `[0.1, 0.2, 0.3, ...]`)
- Other columns are optional (metadata, labels)

**Example:**

```
id,emb,label
1,"[0.12,0.56,0.89,...]",A
2,"[0.33,0.48,0.90,...]",B
```

##### **2. NPY Format**

- A two-dimensional array, shape = `(num_vectors, dim)`
- Vector order is assigned id starting from 0 by default
- Labels can be provided separately in a CSV (id,label)

**Example:**

```python
import numpy as np
vectors = np.random.rand(10000, 768).astype('float32')
np.save("vectors.npy", vectors)
```
### 4.2.5.2 Use Script to Convert Data File Format
- Install dependencies:

```
pip install numpy pandas faiss-cpu
```

- Start command:

```shell
python convert_to_vdb_format.py \
  --train data/train.csv \
  --test data/test.csv \
  --out datasets/custom \
  --topk 10
```

- Parameter description:

| Parameter | Required | Type | Description | Default |
| ---------- | -------- | ------ | ------------------------------------------------------------ | ------ |
| `--train`  | Yes       | String | Training data path, supports CSV or NPY format. CSV needs `emb` column, auto-generates `id` column if not present | None |
| `--test`   | Yes       | String | Query data path, supports CSV or NPY format. Same format as training data | None |
| `--out`    | Yes       | String | Output directory path, saves converted parquet files and neighbor index files | None |
| `--labels` | No       | String | Label CSV path, must contain `labels` column (format is string list), used to save labels | None |
| `--topk`   | No       | Integer | Number of neighbors returned when computing nearest neighbors | 10 |
- Output directory structure

```
datasets/custom/
├── train.parquet          # Training vectors
├── test.parquet           # Query vectors
├── neighbors.parquet      # Ground Truth
└── scalar_labels.parquet  # Optional labels
```
- Script code
```python
import os
import argparse
import numpy as np
import pandas as pd
import faiss
from ast import literal_eval
from typing import Optional


def load_csv(path: str):
    df = pd.read_csv(path)
    if 'emb' not in df.columns:
        raise ValueError(f"CSV file missing 'emb' column: {path}")
    df['emb'] = df['emb'].apply(literal_eval)
    if 'id' not in df.columns:
        df.insert(0, 'id', range(len(df)))
    return df


def load_npy(path: str):
    arr = np.load(path)
    df = pd.DataFrame({
        'id': range(arr.shape[0]),
        'emb': arr.tolist()
    })
    return df


def load_vectors(path: str) -> pd.DataFrame:
    if path.endswith('.csv'):
        return load_csv(path)
    elif path.endswith('.npy'):
        return load_npy(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")


def compute_ground_truth(train_vectors: np.ndarray, test_vectors: np.ndarray, top_k: int = 10):
    dim = train_vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(train_vectors)
    _, indices = index.search(test_vectors, top_k)
    return indices


def save_ground_truth(df_path: str, indices: np.ndarray):
    df = pd.DataFrame({
        "id": np.arange(indices.shape[0]),
        "neighbors_id": indices.tolist()
    })
    df.to_parquet(df_path, index=False)
    print(f"✅ Ground truth saved successfully: {df_path}")


def main(train_path: str, test_path: str, output_dir: str,
         label_path: Optional[str] = None, top_k: int = 10):
    
    os.makedirs(output_dir, exist_ok=True)

    # Load training and query data
    print("📥 Loading training data...")
    train_df = load_vectors(train_path)
    print("📥 Loading query data...")
    test_df = load_vectors(test_path)

    # Extract vectors and convert to numpy
    train_vectors = np.array(train_df['emb'].to_list(), dtype='float32')
    test_vectors = np.array(test_df['emb'].to_list(), dtype='float32')

    # Save parquet files retaining all fields
    train_df.to_parquet(os.path.join(output_dir, 'train.parquet'), index=False)
    print(f"✅ train.parquet saved successfully, total {len(train_df)} records")

    test_df.to_parquet(os.path.join(output_dir, 'test.parquet'), index=False)
    print(f"✅ test.parquet saved successfully, total {len(test_df)} records")

    # Compute ground truth
    print("🔍 Computing Ground Truth (nearest neighbors)...")
    gt_indices = compute_ground_truth(train_vectors, test_vectors, top_k=top_k)
    save_ground_truth(os.path.join(output_dir, 'neighbors.parquet'), gt_indices)

    # Load and save label file (if exists)
    if label_path:
        print("📥 Loading label file...")
        label_df = pd.read_csv(label_path)
        if 'labels' not in label_df.columns:
            raise ValueError("Label file must contain 'labels' column")
        label_df['labels'] = label_df['labels'].apply(literal_eval)
        label_df.to_parquet(os.path.join(output_dir, 'scalar_labels.parquet'), index=False)
        print("✅ Label file saved as scalar_labels.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV/NPY vectors to VectorDBBench data format (retain all columns)")
    parser.add_argument("--train", required=True, help="Training data path (CSV or NPY)")
    parser.add_argument("--test", required=True, help="Query data path (CSV or NPY)")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--labels", help="Label CSV path (optional)")
    parser.add_argument("--topk", type=int, default=10, help="Ground truth")

    args = parser.parse_args()
    main(args.train, args.test, args.out, args.labels, args.topk)

```
### 4.2.5.3 Set Custom Data and Test
- Enter the Web UI homepage, select **Custom Dataset** on the main page:

![image-20250809160755529](./images/image-20250809160755529.png)

- After selection, we can see the explanation of **Custom Dataset** and the content that needs to be filled in:

![image-20250809160941449](./images/image-20250809160941449.png)

- Parameter Details

| Field Name | Meaning | Filling Suggestions |
| --------------------------- | ------------------------------------------- | ------------------------------------------------------------ |
| **Name**                    | Dataset name (unique identifier) | Arbitrary, e.g., `my_custom_dataset` |
| **Folder Path**             | Dataset folder path | e.g., `/data/datasets/custom` |
| **dim**                     | Vector dimension | Consistent with data file, e.g., `768` |
| **size**                    | Vector count (optional) | Can be left blank, system can auto-read |
| **metric type**             | Similarity metric method | Common `L2` (Euclidean distance) or `IP` (inner product) |
| **train file name**         | Training set file name (without `.parquet` suffix) | If it's `train.parquet`, fill in `train`. Multiple files separated by comma, e.g., `train1,train2` |
| **test file name**          | Query set file name (without `.parquet` suffix) | If it's `test.parquet`, fill in `test` |
| **ground truth file name**  | Ground Truth file name (without `.parquet` suffix) | If it's `neighbors.parquet`, fill in `neighbors` |
| **train id name**           | Training data ID column name | Generally `id` |
| **train emb name**          | Training data vector column name | If script generated column name is `emb`, fill in `emb` |
| **test emb name**           | Test data vector column name | Generally same as train emb name, e.g., `emb` |
| **ground truth emb name**   | Neighbor column name in Ground Truth | If column name is `neighbors_id`, fill in `neighbors_id` |
| **scalar labels file name** | (Optional) Label file name (without `.parquet` suffix) | If generated `scalar_labels.parquet`, fill in `scalar_labels`, otherwise leave blank |
| **label percentages**       | (Optional) Label filtering ratio | e.g., `0.001,0.02,0.5`, leave blank if no label filtering requirement |
| **description**             | Dataset description | Can write business background or generation method |

Click **Save** to save.


### 4.2.5.3 Configure Test Plan and Run Test

1. Enter **Run Test** page in Web UI:

   ![image-20250809170426143](./images/image-20250809170426143.png)

2. Check and fill in the vector database to test, this article uses milvus as an example:

   ![image-20250809170449053](./images/image-20250809170449053.png)

3. Select our created Custom dataset:

   ![image-20250809170511831](./images/image-20250809170511831.png)

4. Set task label

   ![image-20250809170553869](./images/image-20250809170553869.png)

5. Start testing

![image-20250809170644233](./images/image-20250809170644233.png)

## 4.2.6 Result Interpretation
#### Main Indicator Description

- QPS (Queries Per Second):
  - Number of queries processed per second. QPS is an indicator to measure system query processing capability, higher QPS means the system can process more queries in unit time.
- Recall:
  - An accuracy indicator of the retrieval system, used to measure the ratio of relevant items returned in query results to actual relevant items. Higher Recall indicates more correct matches returned in query results. Used to evaluate the effectiveness of the system in approximate queries.
- Load Duration:
  - Data loading time, representing the total time spent loading data into the database. This indicator measures the database's loading efficiency, generally more data volume leads to longer loading time.
- Serial Latency P99:
  - This is the upper limit of 99% query processing time, representing the longest time required for the system to process 99% of queries (99th percentile latency). This indicator is used to measure the consistency of system response time, lower values mean more stable system response. Higher P99 latency means the system occasionally has slow queries.

For detailed rules, refer to [Leaderboard Description](https://zilliz.com/benchmark)
