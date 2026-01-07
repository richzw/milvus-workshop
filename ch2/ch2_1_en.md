# Part 2: Basic Milvus Operations - Using the Python SDK [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/richzw/milvus-workshop/blob/main/ch2/ch2_1_en.ipynb)
Welcome to Part 2 of the Milvus Workshop! In this part, we will learn how to use the Python SDK (PyMilvus) to interact with Milvus, focusing on the management of collections.
**Before we begin, please ensure you have:**
1. Install and start the Milvus server. 
2. Install `pymilvus` Python SDK (`pip install pymilvus`).


```python
!pip install pymilvus==2.5.8
```

## 2.1 Connect to Milvus and Manage Collections

First, we need to connect to the running Milvus server.


```python
# Import the necessary libraries
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema 
```


```python
# Connect to Milvus server
# MilvusClient uses a URI or host/port for connection.
MILVUS_HOST = "localhost" # Or your Milvus server IP
MILVUS_PORT = "19530"
MILVUS_URI = f"http://{MILVUS_HOST}:{MILVUS_PORT}" # Recommended format

# Or, if you have the username and password (for Zilliz Cloud or Milvus with auth)
# MILVUS_USER = "username"
# MILVUS_PASSWORD = "password"
# client = MilvusClient(uri=MILVUS_URI, user=MILVUS_USER, password=MILVUS_PASSWORD)

try:
    # Create a MilvusClient instance
    client = MilvusClient(
        uri=MILVUS_URI
        # token="YOUR_API_KEY_OR_TOKEN" # For Zilliz Cloud serverless or other token-based auth
        # db_name="default" # Specify database if not default (Milvus 2.2.9+)
    )
    print(f"Successfully created MilvusClient and connected to the Milvus server: {MILVUS_URI}")
    print(f"Milvus server version (via client): {client.get_server_version()}")
except Exception as e:
    print(f"Failed to create MilvusClient or connect to Milvus server: {e}")
```

    Successfully created MilvusClient and connected to the Milvus server: http://localhost:19530
    Milvus server version (via client): 2.5.11


### Concept: Collection

In Milvus, a **Collection** is a set of entities, analogous to a “table” in a relational database (such as MySQL) or an “index” in Elasticsearch.

- A Collection contains a set of entities with the same schema.
- Each entity can contain multiple fields, including vector fields and scalar fields.
- A Collection is the fundamental unit for performing vector searches, inserting data, and managing data.

### Hands-on: Defining a Collection Schema
To create a Collection, you must first define its **Schema**. The Schema specifies the name, data type, and other properties for each field within the Collection.
#### Concept of Fields
- **Primary Key Field**:
  - Used to uniquely identify each entity within a Collection.
  - Each Collection must have exactly one primary key field.
  - Data type is typically `INT64` or `VARCHAR`.
  - Specified in `FieldSchema` via `is_primary=True`.
- **Vector Field**:
  - Used to store vector embeddings. This is the core of Milvus' similarity search.
  - Data type is typically `FLOAT_VECTOR` or `BINARY_VECTOR`.
  - The vector dimension `dim` must be specified during creation.
- **Scalar Field**:
  - Used to store non-vector attribute data such as IDs, names, categories, timestamps, etc.
  - Can be used for filtering query results (Attribute Filtering) or returned as metadata.
  - Supports multiple data types.

#### Data Types
Milvus supports the following primary data types:
- `DataType.BOOL`: Boolean (True/False)
- `DataType.INT8`, `DataType.INT16`, `DataType.INT32`, `DataType.INT64`: Integer types with different bit lengths 
- `DataType.FLOAT`, `DataType.DOUBLE`: Single-precision and double-precision floating-point types
- `DataType.STRING`: String type (typically used for longer text, UTF-8 encoded, theoretically unlimited length but practically constrained by gRPC transmission size)
- `DataType.VARCHAR`: Variable-length string type (UTF-8 encoded, requires specifying `max_length` during creation, maximum length 65535)
- `DataType.ARRAY`: Array type, can contain scalar data. Requires specifying `element_type` and `max_capacity`.
- `DataType.JSON`: JSON type, used for storing semi-structured data.
- `DataType.FLOAT_VECTOR`: Floating-point vector.
- `DataType.BINARY_VECTOR`: Binary vector.

#### How to define a Primary Key
The primary key field uniquely identifies each entity.
- `is_primary=True`
- Data type is typically `INT64` or `VARCHAR` (requires specifying `max_length`).
- If the primary key field is of type `INT64`, set `auto_id=True` to have Milvus automatically generate a unique ID. If it is `VARCHAR`, `auto_id` must be `False`, and the user must provide a unique value when inserting data.


```python
# Example: Define a primary key field of type INT64 and enable automatic ID generation.
pk_field_auto_id = FieldSchema(
  name="doc_id",          # Field name
  dtype=DataType.INT64,   # Data type
  is_primary=True,        # Set as primary key
  auto_id=True,           # Enable automatic ID generation
  description="Document ID (auto-generated)" # Field description
)
print(f"Primary Key Field (Auto-Generated ID): {pk_field_auto_id}")
```

    Primary Key Field (Auto-Generated ID): {'name': 'doc_id', 'description': 'Document ID (auto-generated)', 'type': <DataType.INT64: 5>, 'is_primary': True, 'auto_id': True}



```python
# Example: Define a VARCHAR primary key field (requires user-provided ID)
pk_field_user_id = FieldSchema(
  name="user_uuid",
  dtype=DataType.VARCHAR,
  max_length=36,          # VARCHAR must specify the maximum length
  is_primary=True,
  auto_id=False,          # VARCHAR primary key cannot automatically generate ID
  description="User unique identifier (provided by user)"
)
print(f"Primary Key Field (User-provided ID): {pk_field_user_id}")
```

    Primary Key Field (User-provided ID): {'name': 'user_uuid', 'description': 'User unique identifier (provided by user)', 'type': <DataType.VARCHAR: 21>, 'params': {'max_length': 36}, 'is_primary': True, 'auto_id': False}


#### How to define an auto-ID

As mentioned above, when the primary key field's data type is `INT64`, setting `auto_id=True` enables Milvus to automatically generate a unique ID for each inserted entity. This simplifies the data insertion process, as users no longer need to manage and provide primary key values themselves.

If `auto_id=False` (the default value, unless `is_primary=True` and `dtype=DataType.INT64`, in which case `auto_id` behavior may vary across versions; explicit setting is recommended), or if the primary key type is not `INT64`, users must provide a unique value for the primary key field when inserting data.

#### How to define a Vector Field (Dimension)
The vector field is the core of Milvus, used for storing vector embeddings.
- `dtype` must be `DataType.FLOAT_VECTOR` or `DataType.BINARY_VECTOR`.
- The `dim` parameter must be specified, representing the dimension of the vector. For example, a vector generated by a BERT model may have 768 dimensions.


```python
# Example: Define a 128-dimensional floating-point vector field
vector_field_128d = FieldSchema(
  name="embedding",
  dtype=DataType.FLOAT_VECTOR,
  dim=128,                 # Vector dimension
  description="128-dimensional float vector embedding"
)
print(f"Vector field: {vector_field_128d}")
```

    Vector field: {'name': 'embedding', 'description': '128-dimensional float vector embedding', 'type': <DataType.FLOAT_VECTOR: 101>, 'params': {'dim': 128}}


### Hands-on: Creating, deleting, and viewing a Collection

Now we will combine the above concepts to actually operate Collection.

**1. Define a Complete Collection Schema**


```python
COLLECTION_NAME_DEMO = "my_collection" 

# First attempt deletion to ensure a clean environment (if it doesn't exist, it will be ignored).
if client.has_collection(collection_name=COLLECTION_NAME_DEMO):
    print(f"Detect an existing Collection '{COLLECTION_NAME_DEMO}', deleting it.")
    client.drop_collection(collection_name=COLLECTION_NAME_DEMO)

# Define Field
field1 = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
field2 = FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512)
field3 = FieldSchema(name="views", dtype=DataType.INT32)
field4 = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=8) # Use a smaller dimension for demonstration purpose

# Define Schema
schema = CollectionSchema(
  fields=[field1, field2, field3, field4],
  description="A simple collection for demonstration purposes (using MilvusClient)",
  enable_dynamic_field=False # Enable Dynamic Schema?
  # True: Allows insertion of fields not defined in the schema; these fields will be stored as JSON.
  # False: (Default) Insert data strictly according to the schema definition.
)

print("Schema definition completed.")
```

    Schema definition completed.


**2. Creating a Collection**

Use the defined Schema and Collection name to create the Collection.

The `consistency_level` parameter specifies the Collection's consistency level, affecting data visibility and search accuracy. Common levels:
- `Strong`: Strong consistency. After a write operation completes, subsequent read/search operations immediately see the latest data.
- `Bounded`: Allows reading slightly stale data within a time window, offering better performance.
- `Session`: Ensures consistency within the same session.
- `Eventually`: Writes become visible eventually, with potentially significant delays but optimal performance.

For most applications, `Bounded` or `Strong` are common choices. The default is `Bounded`.


```python
# Create Collection
try:
    # 注意：consistency_level 是一个重要的参数
    client.create_collection(
        collection_name=COLLECTION_NAME_DEMO,
        schema=schema, # Pass the CollectionSchema object
        # alternatively, you can pass dim directly for simple cases:
        # collection_name=COLLECTION_NAME_DEMO,
        # dimension=8, # Only if you have one vector field and a primary key
        consistency_level="Strong" # 或者 "Bounded", "Session", "Eventually"
    )
    print(f"Collection '{COLLECTION_NAME_DEMO}' created successfully!")
except Exception as e:
    print(f"Create Collection '{COLLECTION_NAME_DEMO}' failed: {e}")
```

    Collection 'my_collection' created successfully!


**3. Viewing Collection Information**

We can view all Collections in the current Milvus instance or retrieve detailed information about a specific Collection.


```python
# List all Collections
all_collections = client.list_collections()
print(f"All Collections in Milvus: {all_collections}")

# Check if a specific collection exists
exists = client.has_collection(collection_name=COLLECTION_NAME_DEMO)
print(f"Does the Collection '{COLLECTION_NAME_DEMO}' exist: {exists}")

if exists:
    # Retrieve collection description information (including schema, num_entities, etc.)
    desc = client.describe_collection(collection_name=COLLECTION_NAME_DEMO)
    print(f"\nCollection '{desc['collection_name']}' Description Information:")
    print(f"  - Description: {desc['description']}")
    print(f"  - Auto ID: {desc['auto_id']}")
    print(f"  - Consistency Level: {desc['consistency_level']}")
    print(f"  - Number of Partitions: {desc['num_partitions']}") # MilvusClient returns num_entities directly in describe

    print(f"\n  - Schema Fields:")
    for field_info in desc['fields']:
        print(f"    - Name: {field_info['name']}, Type: {field_info['type']}, Is Primary: {field_info.get('is_primary', False)}"
              f", Dim: {field_info.get('params', {}).get('dim', 'N/A')}") # field_info['params']['dim'] if vector

    # Retrieve Collection Statistics (with a greater emphasis on row_count, sealed/growing segments, etc.)
    stats = client.get_collection_stats(collection_name=COLLECTION_NAME_DEMO)
    print(f"\nCollection '{COLLECTION_NAME_DEMO}' Statistics: {stats}")
    
```

    All Collections in Milvus: ['my_collection']
    Does the Collection 'my_collection' exist: True
    
    Collection 'my_collection' Description Information:
      - Description: A simple collection for demonstration purposes (using MilvusClient)
      - Auto ID: True
      - Consistency Level: 0
      - Number of Partitions: 1
    
      - Schema Fields:
        - Name: id, Type: 5, Is Primary: True, Dim: N/A
        - Name: title, Type: 21, Is Primary: False, Dim: N/A
        - Name: views, Type: 4, Is Primary: False, Dim: N/A
        - Name: embedding, Type: 101, Is Primary: False, Dim: 8
    
    Collection 'my_collection' Statistics: {'row_count': 0}


**4. Deleting a Collection**

If a Collection is no longer needed, it can be deleted. **Note: Deletion is irreversible and will permanently remove the Collection and all its data.**


```python
# Delete Collection
try:
    client.drop_collection(collection_name=COLLECTION_NAME_DEMO)
    print(f"Collection '{COLLECTION_NAME_DEMO}' successfully deleted.")
except Exception as e:
    print(f"Deleting Collection '{COLLECTION_NAME_DEMO}' failed: {e}")

# Check again whether the Collection exists
exists_after_drop = client.has_collection(collection_name=COLLECTION_NAME_DEMO)
print(f"Does the Collection '{COLLECTION_NAME_DEMO}' exist (after deletion): {exists_after_drop}")
```

    Collection 'my_collection' successfully deleted.
    Does the Collection 'my_collection' exist (after deletion): False


### Hands-on: Loading and releasing a Collection

In Milvus, data is stored on disk (or object storage) by default. For efficient searching and querying, it is necessary to load a Collection's data (or a portion of it, such as specific Segments or Partitions) into the memory of a Milvus QueryNode.

- **`client.load_collection()`**: Loads Collection data into memory, making it available for search/query operations.
- **`client.release_collection()`**: Releases Collection data from memory to conserve memory resources. The data itself is not deleted and remains stored.

**Importance of Loading**:

- **Performance**: Searching in-memory data is significantly faster than searching disk-based data.
- **Resource Management**: For extremely large datasets, loading all data simultaneously into memory may be impractical. Milvus allows loading specific partitions or enables automatic resource management by Milvus based on available resources.
- **Readiness**: Only fully loaded Collections can be effectively queried.

We recreate the previous `simple_collection` to demonstrate loading and unloading.


```python
# 1. Recreate the Collection (if it does not exist)
if not client.has_collection(collection_name=COLLECTION_NAME_DEMO):
    # The schema variable has already been defined earlier and can be used directly here.
    client.create_collection(
        collection_name=COLLECTION_NAME_DEMO,
        schema=schema,
        consistency_level="Strong"
    )
    print(f"Collection '{COLLECTION_NAME_DEMO}' has been recreated.")
else:
    print(f"Collection '{COLLECTION_NAME_DEMO}' already exists, use it directly.")
```

    Collection 'my_collection' has been recreated.



```python
# 2. Load the Collection
# Before loading, you must first create indexes for vector fields (indexing details will be covered later).
index_params = MilvusClient.prepare_index_params()

index_params.add_index(
    field_name="embedding",
    metric_type="COSINE",
    index_type="IVF_FLAT",
    index_name="vector_index",
    params={ "nlist": 128 }
)

client.create_index(
    collection_name=COLLECTION_NAME_DEMO,
    index_params=index_params,
    sync=False # Whether to wait for index creation to complete before returning. Defaults to True.
)

try:
    print(f"Loading Collection '{COLLECTION_NAME_DEMO}'...")
    client.load_collection(collection_name=COLLECTION_NAME_DEMO)
    print(f"Collection '{COLLECTION_NAME_DEMO}' loading command has been sent.")

    # Get loading status (MilvusClient 2.3+)
    # `get_load_state` returns a dictionary like {'state': <LoadState: Loaded: 1 NotLoad: 2 Loading: 3>}
    load_state_info = client.get_load_state(collection_name=COLLECTION_NAME_DEMO)
    print(f"Loading status information: {load_state_info}")
except Exception as e:
    print(f"Loading Collection '{COLLECTION_NAME_DEMO}' failed: {e}")
    raise

```

    Loading Collection 'my_collection'...
    Collection 'my_collection' loading command has been sent.
    Loading status information: {'state': <LoadState: Loaded>}


**Note**:
- For a newly created Collection with no data, the `load_collection()` operation completes quickly.
- When the Collection contains data, the loading process may take some time.
- `load_collection()` is an asynchronous operation that returns immediately. Use `client.get_load_state()` to check the loading status.
- Before performing any search operations on the Collection, ensure it has finished loading.


```python
# 3. Release the Collection
try:
    print(f"Begin releasing the Collection '{COLLECTION_NAME_DEMO}'...")
    client.release_collection(collection_name=COLLECTION_NAME_DEMO)
    print(f"Collection '{COLLECTION_NAME_DEMO}' has been released from memory.")
except Exception as e:
    print(f"Releasing Collection '{COLLECTION_NAME_DEMO}' failed: {e}")
```

    Begin releasing the Collection 'my_collection'...
    Collection 'my_collection' has been released from memory.



### Hands-on: Create a Collection with TTL (Time-To-Live)

Time-To-Live (TTL) is a property of Milvus Collection that allows you to set a “lifetime” for data. Once data exceeds this configured duration, Milvus will automatically delete it. TTL is measured in **seconds**.

TTL-related operations

- Create collection
  ```py
  client.create_collection(
        properties={ "collection.ttl.seconds": TTL_SECONDS}
    )
  ```
- Add TTL to an existing collection
  ```py
  client.alter_collection_properties(
        properties={"collection.ttl.seconds": TTL_SECONDS}
  )
  ```
- Delete TTL
  ```py
  client.drop_collection_properties(
    property_keys=["collection.ttl.seconds"]
  )
  ```


```python
COLLECTION_NAME_TTL = "my_ttl_collection"
TTL_SECONDS = 300 # Data survives for 300 seconds (5 minutes)

# First ensure the collection does not exist.
if client.has_collection(collection_name=COLLECTION_NAME_TTL):
    print(f"The existing Collection '{COLLECTION_NAME_TTL}'has been detected and will be deleted.")
    client.drop_collection(collection_name=COLLECTION_NAME_TTL)

# 1. Define fields
ttl_pk_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
ttl_data_field = FieldSchema(name="message", dtype=DataType.VARCHAR, max_length=1024)
ttl_vector_field = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=4) 

# 2. Define Schema
ttl_schema = CollectionSchema(
    fields=[ttl_pk_field, ttl_data_field, ttl_vector_field],
    description=f"Collection with TTL of {TTL_SECONDS} seconds",
    enable_dynamic_field=False
)

# 3. Create a Collection with TTL
try:
    client.create_collection(
        collection_name=COLLECTION_NAME_TTL,
        schema=ttl_schema,
        consistency_level="Strong",
        properties={
            "collection.ttl.seconds": TTL_SECONDS
        }
    )
    print(f"Collection '{COLLECTION_NAME_TTL}' with TTL has been successfully created!")
except Exception as e:
    print(f"Failed to create Collection '{COLLECTION_NAME_TTL}' with TTL: {e}")
    raise

# 4. Verify TTL Settings
# Milvus stores TTL information in the Collection's properties.
if client.has_collection(collection_name=COLLECTION_NAME_TTL):
    desc_ttl_collection = client.describe_collection(collection_name=COLLECTION_NAME_TTL)
    print(f"\nDescription of Collection '{COLLECTION_NAME_TTL}':")
    # print(desc_ttl_collection) # Print full description to view structure
    
    # TTL information is typically found under the 'properties' field, with the key 'collection.ttl.seconds'.
    collection_properties = desc_ttl_collection.get('properties', {})
    actual_ttl = collection_properties.get('collection.ttl.seconds', '未设置或获取失败')
    
    print(f"  - Description: {desc_ttl_collection.get('description')}")
    print(f"  - Configured TTL (seconds): {actual_ttl}") # Note: This returns a string.
    
    if str(actual_ttl) == str(TTL_SECONDS):
        print(f"  - TTL verification successful! Set to {TTL_SECONDS} seconds.")
    else:
        print(f"  - TTL validation failed or mismatched. Expected value: {TTL_SECONDS}, Actual value: {actual_ttl}")

    print(f"\nNote: Data inserted into '{COLLECTION_NAME_TTL}' will be automatically purged by Milvus after {TTL_SECONDS} seconds.")
    print("To verify the TTL effect, you can: ")
    print("  1. Insert some data into this Collection.")
    print(f"  2. Waiting for more than {TTL_SECONDS} seconds.")
    print("  3. When querying the Collection, the data should have been deleted (num_entities becomes 0 or decreases).")

# # Delete TTL
# client.drop_collection_properties(
#     collection_name=COLLECTION_NAME_TTL,
#     property_keys=["collection.ttl.seconds"]
# )
```

    Collection 'my_ttl_collection' with TTL has been successfully created!
    
    Description of Collection 'my_ttl_collection':
      - Description: Collection with TTL of 300 seconds
      - Configured TTL (seconds): 300
      - TTL verification successful! Set to 300 seconds.
    
    Note: Data inserted into 'my_ttl_collection' will be automatically purged by Milvus after 300 seconds.
    To verify the TTL effect, you can: 
      1. Insert some data into this Collection.
      2. Waiting for more than 300 seconds.
      3. When querying the Collection, the data should have been deleted (num_entities becomes 0 or decreases).


### Hands-on Exercise 1: Create a simple Collection

**Task**: Create a Collection named `book_search_mc` to store book information. This Collection should contain the following fields:

1.  `book_id`:
    *   Type: `INT64`
    *   Attributes: Primary Key, Auto ID
    *   Description: "Book's unique identifier"
2.  `book_title`:
    *   Type: `VARCHAR`
    *   Attributes: max_length 512
    *   Description: "Title of the book"
3.  `publication_year`:
    *   Type: `INT32`
    *   Description: "Year the book was published"
4.  `book_embedding`:
    *   Type: `FLOAT_VECTOR`
    *   Attributes: Dimension 768 (e.g., a common sentence embedding dimension)
    *   Description: "Vector embedding of the book's content or title"

**Steps**:
1. Define the `FieldSchema` for each field.
2. Define the `CollectionSchema` using these fields.
3. Create the Collection using `client.create_collection()`.
4. Verify the Collection was successfully created using `client.describe_collection()`, and print its Schema and entity count.
5. (Optional) Delete the created Collection using `client.drop_collection()` to clean up the environment.



```python

```


```python

```
