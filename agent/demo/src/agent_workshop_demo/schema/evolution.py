"""Allow-listed, dry-run-first Milvus 3.0 schema evolution helpers."""

from __future__ import annotations

import math
import re
from json import loads
from importlib import import_module
from typing import Any, Iterable, Literal

FieldKind = Literal["sparse", "embedding"]
_FIELD_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_RETRIEVAL_ANALYZER = {
    "tokenizer": "standard",
    "filter": [
        "lowercase",
        {
            "type": "synonym",
            "synonyms": [
                "object storage, s3, minio",
                "vector database, vector db",
                "full text, bm25",
            ],
        },
    ],
}


class MilvusSchemaEvolution:
    """Plan and apply additive vector-field/function migrations only."""

    def __init__(self, client: Any, *, collection_name: str = "kb_chunks") -> None:
        if collection_name != "kb_chunks":
            raise ValueError("Schema evolution is allow-listed to kb_chunks")
        self.client = client
        self.collection_name = collection_name

    def add_retrieval_text(self, *, apply: bool = False) -> dict[str, Any]:
        """Add the nullable analyzed BM25 input field to an existing collection."""

        plan = {
            "operation": "add_retrieval_text",
            "collection": self.collection_name,
            "field_name": "retrieval_text",
            "nullable": True,
            "applied": False,
        }
        fields = _described_fields(self.client, self.collection_name)
        if "retrieval_text" in fields:
            _require_retrieval_text_field(fields["retrieval_text"])
            plan["status"] = "already_exists"
            return plan
        if not apply:
            plan["status"] = "planned"
            return plan
        milvus = import_module("pymilvus")
        self.client.add_collection_field(
            collection_name=self.collection_name,
            field_name="retrieval_text",
            data_type=milvus.DataType.VARCHAR,
            desc="BM25 Function input added by Agent Workshop migration",
            nullable=True,
            max_length=32_768,
            enable_analyzer=True,
            analyzer_params=_RETRIEVAL_ANALYZER,
        )
        fields = _described_fields(self.client, self.collection_name)
        if "retrieval_text" not in fields:
            raise RuntimeError("Milvus schema revalidation did not find retrieval_text")
        _require_retrieval_text_field(fields["retrieval_text"])
        plan.update({"status": "applied", "applied": True})
        return plan

    def add_vector_field(
        self,
        field_name: str,
        *,
        kind: FieldKind,
        dim: int | None = None,
        apply: bool = False,
    ) -> dict[str, Any]:
        """Add one nullable sparse or dense vector field idempotently."""

        _validate_field_name(field_name)
        if kind == "embedding":
            if isinstance(dim, bool) or not isinstance(dim, int) or not 1 <= dim <= 65_536:
                raise ValueError("Embedding dim must be between 1 and 65536")
        elif kind == "sparse":
            if dim is not None:
                raise ValueError("Sparse fields do not accept dim")
        else:
            raise ValueError("Unsupported vector field kind")
        plan = {
            "operation": "add_vector_field",
            "collection": self.collection_name,
            "field_name": field_name,
            "kind": kind,
            "dim": dim,
            "nullable": True,
            "applied": False,
        }
        fields = _described_fields(self.client, self.collection_name)
        if field_name in fields:
            _require_matching_vector_field(
                fields[field_name],
                kind=kind,
                dim=dim,
            )
            plan["status"] = "already_exists"
            return plan
        if not apply:
            plan["status"] = "planned"
            return plan
        milvus = import_module("pymilvus")
        data_type = (
            milvus.DataType.SPARSE_FLOAT_VECTOR
            if kind == "sparse"
            else milvus.DataType.FLOAT_VECTOR
        )
        kwargs: dict[str, Any] = {"nullable": True}
        if dim is not None:
            kwargs["dim"] = dim
        self.client.add_collection_field(
            collection_name=self.collection_name,
            field_name=field_name,
            data_type=data_type,
            desc="Agent Workshop additive migration field",
            **kwargs,
        )
        fields = _described_fields(self.client, self.collection_name)
        if field_name not in fields:
            raise RuntimeError("Milvus schema revalidation did not find the new field")
        _require_matching_vector_field(fields[field_name], kind=kind, dim=dim)
        plan.update({"status": "applied", "applied": True})
        return plan

    def add_bm25_function(
        self,
        *,
        output_field_name: str = "sparse_vector_v2",
        apply: bool = False,
    ) -> dict[str, Any]:
        """Atomically add BM25 output, Function and default SINDI index."""

        _validate_field_name(output_field_name)
        function_name = f"bm25_{output_field_name}"
        if len(function_name) > 64:
            raise ValueError("BM25 function name would exceed 64 characters")

        plan = {
            "operation": "add_bm25_function",
            "collection": self.collection_name,
            "function": function_name,
            "output_field": output_field_name,
            "physical_backfill": True,
            "activation": {"MILVUS_SPARSE_FIELD": output_field_name},
            "applied": False,
            "status": "planned",
        }
        description = self.client.describe_collection(
            collection_name=self.collection_name
        )
        raw_functions = (
            description.get("functions", [])
            if isinstance(description, dict)
            else getattr(description, "functions", [])
        )
        functions = {
            str(_value(item, "name", "")): item for item in raw_functions
        }
        if function_name in functions:
            _require_bm25_function(
                functions[function_name],
                input_field="retrieval_text",
                output_field=output_field_name,
            )
            fields = _described_fields(self.client, self.collection_name)
            if output_field_name not in fields:
                raise RuntimeError("Existing BM25 function output field is missing")
            _require_matching_vector_field(
                fields[output_field_name], kind="sparse", dim=None
            )
            _require_bm25_index(self.client, self.collection_name, output_field_name)
            plan["status"] = "already_exists"
            return plan
        fields = _described_fields(self.client, self.collection_name)
        if "retrieval_text" not in fields:
            raise RuntimeError("BM25 migration requires retrieval_text")
        _require_retrieval_text_field(fields["retrieval_text"])
        if output_field_name in fields:
            raise RuntimeError(
                "BM25 atomic migration requires a new output field name"
            )
        if not apply:
            return plan
        milvus = import_module("pymilvus")
        function = milvus.Function(
            name=function_name,
            function_type=milvus.FunctionType.BM25,
            input_field_names=["retrieval_text"],
            output_field_names=[output_field_name],
            params={},
        )
        field_schema = milvus.FieldSchema(
            name=output_field_name,
            dtype=milvus.DataType.SPARSE_FLOAT_VECTOR,
            nullable=True,
        )
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name=output_field_name,
            index_name=f"{output_field_name}_idx",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={},
        )
        _add_function_field_with_physical_backfill(
            self.client,
            collection_name=self.collection_name,
            field_schema=field_schema,
            function=function,
            index_params=index_params,
        )
        description = self.client.describe_collection(
            collection_name=self.collection_name
        )
        fields = _fields_from_description(description)
        if output_field_name not in fields:
            raise RuntimeError("Milvus BM25 migration revalidation failed")
        _require_matching_vector_field(
            fields[output_field_name], kind="sparse", dim=None
        )
        functions = {
            str(_value(item, "name", "")): item
            for item in _value(description, "functions", [])
        }
        if function_name not in functions:
            raise RuntimeError("Milvus BM25 function revalidation failed")
        _require_bm25_function(
            functions[function_name],
            input_field="retrieval_text",
            output_field=output_field_name,
        )
        _require_bm25_index(self.client, self.collection_name, output_field_name)
        plan.update({"status": "applied", "applied": True})
        return plan

    def backfill_retrieval_text(
        self,
        records: Iterable[dict[str, Any]],
        *,
        batch_size: int = 100,
        apply: bool = False,
    ) -> dict[str, Any]:
        """Backfill analyzed BM25 input by primary key before Function backfill."""

        if not 1 <= batch_size <= 1_000:
            raise ValueError("batch_size must be between 1 and 1000")
        fields = _described_fields(self.client, self.collection_name)
        if "retrieval_text" not in fields:
            raise RuntimeError("Retrieval-text backfill target field does not exist")
        _require_retrieval_text_field(fields["retrieval_text"])
        pending: list[dict[str, Any]] = []
        seen: set[int] = set()
        for record in records:
            if set(record) != {"id", "retrieval_text"}:
                raise ValueError("Retrieval-text rows must contain only id and retrieval_text")
            primary_id = record["id"]
            text = record["retrieval_text"]
            if isinstance(primary_id, bool) or not isinstance(primary_id, int):
                raise ValueError("Backfill id must be an integer")
            if primary_id in seen:
                raise ValueError("Backfill ids must be unique")
            if not isinstance(text, str) or not text.strip() or len(text) > 32_768:
                raise ValueError("retrieval_text must contain 1..32768 characters")
            seen.add(primary_id)
            pending.append({"id": primary_id, "retrieval_text": text.strip()})
        report = {
            "operation": "backfill_retrieval_text",
            "collection": self.collection_name,
            "validated_count": len(pending),
            "applied_count": 0,
            "status": "planned" if not apply else "applied",
        }
        if not apply:
            return report
        applied = 0
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            result = self.client.upsert(
                collection_name=self.collection_name,
                data=batch,
                partial_update=True,
            )
            count = result.get("upsert_count") if isinstance(result, dict) else None
            if count is not None and int(count) != len(batch):
                raise RuntimeError("Milvus reported incomplete retrieval-text backfill")
            applied += len(batch)
        report["applied_count"] = applied
        return report

    def backfill_embedding(
        self,
        field_name: str,
        records: Iterable[dict[str, Any]],
        *,
        dim: int,
        batch_size: int = 100,
        apply: bool = False,
    ) -> dict[str, Any]:
        """Validate complete primary-key/vector rows and partial-update in batches."""

        _validate_field_name(field_name)
        if not 1 <= dim <= 65_536:
            raise ValueError("Embedding dim must be between 1 and 65536")
        if not 1 <= batch_size <= 1_000:
            raise ValueError("batch_size must be between 1 and 1000")
        fields = _described_fields(self.client, self.collection_name)
        if field_name not in fields:
            raise RuntimeError("Embedding backfill target field does not exist")
        _require_matching_vector_field(
            fields[field_name], kind="embedding", dim=dim
        )
        pending: list[dict[str, Any]] = []
        seen: set[int] = set()
        for record in records:
            if set(record) != {"id", field_name}:
                raise ValueError("Backfill rows must contain only id and target field")
            primary_id = record["id"]
            vector = record[field_name]
            if isinstance(primary_id, bool) or not isinstance(primary_id, int):
                raise ValueError("Backfill id must be an integer")
            if primary_id in seen:
                raise ValueError("Backfill ids must be unique")
            if not isinstance(vector, list) or len(vector) != dim or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in vector
            ):
                raise ValueError("Backfill vector has an invalid dimension or value")
            seen.add(primary_id)
            pending.append({"id": primary_id, field_name: [float(v) for v in vector]})
        report = {
            "operation": "backfill_embedding",
            "collection": self.collection_name,
            "field_name": field_name,
            "validated_count": len(pending),
            "applied_count": 0,
            "status": "planned" if not apply else "applied",
        }
        if not apply:
            return report
        applied = 0
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            result = self.client.upsert(
                collection_name=self.collection_name,
                data=batch,
                partial_update=True,
            )
            count = result.get("upsert_count") if isinstance(result, dict) else None
            if count is not None and int(count) != len(batch):
                raise RuntimeError("Milvus reported an incomplete embedding backfill")
            applied += len(batch)
        report["applied_count"] = applied
        return report


def _validate_field_name(field_name: str) -> None:
    if not _FIELD_NAME.fullmatch(field_name):
        raise ValueError("Invalid additive field name")


def _described_fields(client: Any, collection_name: str) -> dict[str, Any]:
    description = client.describe_collection(collection_name=collection_name)
    return _fields_from_description(description)


def _fields_from_description(description: Any) -> dict[str, Any]:
    if isinstance(description, dict):
        raw_fields = description.get("fields", [])
    else:
        raw_fields = getattr(description, "fields", [])
    output: dict[str, Any] = {}
    for field in raw_fields:
        name = field.get("name") if isinstance(field, dict) else getattr(field, "name", None)
        if isinstance(name, str):
            output[name] = field
    return output


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _normalized_type(value: Any) -> str:
    return str(getattr(value, "name", value)).upper().replace("_", "").replace(".", "")


def _field_param(field: Any, key: str) -> Any:
    direct = _value(field, key)
    if direct is not None:
        return direct
    params = _value(field, "params", {})
    return params.get(key) if isinstance(params, dict) else None


def _canonical_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value in {"True", "False", "true", "false"}:
        return value.lower() == "true"
    raise RuntimeError(f"Existing {field_name} boolean metadata is invalid")


def _require_retrieval_text_field(field: Any) -> None:
    raw_type = _value(field, "type", _value(field, "data_type"))
    if raw_type is None or _normalized_type(raw_type) not in {
        "VARCHAR",
        "DATATYPEVARCHAR",
    }:
        raise RuntimeError("Existing retrieval_text field must be VarChar")
    if int(_field_param(field, "max_length") or 0) != 32_768:
        raise RuntimeError("Existing retrieval_text max_length must be 32768")
    nullable = _field_param(field, "nullable")
    # PyMilvus omits nullable=False from canonical FieldSchema dictionaries.
    if nullable is None:
        nullable = False
    _canonical_bool(nullable, field_name="retrieval_text nullable")
    if not _canonical_bool(
        _field_param(field, "enable_analyzer"),
        field_name="retrieval_text enable_analyzer",
    ):
        raise RuntimeError("Existing retrieval_text analyzer must be enabled")
    analyzer = _field_param(field, "analyzer_params")
    if isinstance(analyzer, str):
        try:
            analyzer = loads(analyzer)
        except ValueError as exc:
            raise RuntimeError("Existing retrieval_text analyzer is invalid") from exc
    if analyzer != _RETRIEVAL_ANALYZER:
        raise RuntimeError("Existing retrieval_text analyzer configuration differs")


def _field_names(function: Any, singular: str, plural: str) -> list[str]:
    raw = _value(function, plural)
    if raw is None:
        raw = _value(function, singular, [])
    if isinstance(raw, str):
        return [raw]
    return [str(item) for item in raw]


def _require_bm25_function(
    function: Any,
    *,
    input_field: str,
    output_field: str,
) -> None:
    raw_type = _value(function, "type", _value(function, "function_type"))
    if raw_type is None or _normalized_type(raw_type) not in {
        "BM25",
        "FUNCTIONTYPEBM25",
    }:
        raise RuntimeError("Existing function is not BM25")
    if _field_names(function, "input_fields", "input_field_names") != [input_field]:
        raise RuntimeError("Existing BM25 function input differs")
    if _field_names(function, "output_fields", "output_field_names") != [output_field]:
        raise RuntimeError("Existing BM25 function output differs")


def _require_bm25_index(
    client: Any,
    collection_name: str,
    output_field: str,
) -> None:
    raw = client.describe_index(
        collection_name=collection_name,
        index_name=f"{output_field}_idx",
    )
    field_name = _value(raw, "field_name", _value(raw, "field"))
    if field_name != output_field:
        raise RuntimeError("BM25 index is not bound to the output field")
    index_type = _value(raw, "index_type")
    metric_type = _value(raw, "metric_type")
    if _normalized_type(index_type) != "SPARSEINVERTEDINDEX":
        raise RuntimeError("BM25 index type differs")
    if _normalized_type(metric_type) != "BM25":
        raise RuntimeError("BM25 index metric differs")
    params = _value(raw, "params", {})
    if isinstance(params, dict) and "inverted_index_algo" in params:
        raise RuntimeError("BM25 index must use the Milvus 3 SINDI default")


def _add_function_field_with_physical_backfill(
    client: Any,
    *,
    collection_name: str,
    field_schema: Any,
    function: Any,
    index_params: Any,
) -> None:
    """Set the 3.0 protobuf flag omitted by PyMilvus 3.0.1's wrapper."""

    get_connection = getattr(client, "_get_connection", None)
    generate_context = getattr(client, "_generate_call_context", None)
    if not callable(get_connection) or not callable(generate_context):
        client.add_function_field(
            collection_name=collection_name,
            field_schema=field_schema,
            func=function,
            index_params=index_params,
            do_physical_backfill=True,
        )
        return

    milvus_prepare = import_module("pymilvus.client.prepare")
    milvus_index = import_module("pymilvus.milvus_client.index")
    call_context = import_module("pymilvus.client.call_context")
    client_utils = import_module("pymilvus.client.utils")
    index_name, index_extra_params = milvus_index.extract_bound_index_param(
        field_schema.name, index_params
    )
    request = milvus_prepare.Prepare.alter_collection_schema_request(
        collection_name=collection_name,
        field_schema=field_schema,
        func=function,
        index_name=index_name,
        index_extra_params=index_extra_params,
    )
    add_request = request.action.add_request
    if not hasattr(add_request, "do_physical_backfill"):
        raise RuntimeError("PyMilvus does not expose physical Function backfill")
    add_request.do_physical_backfill = True
    if add_request.do_physical_backfill is not True:
        raise RuntimeError("Unable to enable physical Function backfill")
    connection = get_connection()
    context = generate_context()
    response = connection._stub.AlterCollectionSchema(  # noqa: SLF001
        request,
        timeout=None,
        metadata=call_context._api_level_md(context),
    )
    client_utils.check_status(response.alter_status)
    connection._invalidate_schema(  # noqa: SLF001
        collection_name,
        db_name=context.get_db_name(),
    )


def _require_matching_vector_field(
    field: Any,
    *,
    kind: FieldKind,
    dim: int | None,
) -> None:
    """Reject an idempotency shortcut when the existing field shape differs."""

    raw_type = _value(field, "type", _value(field, "data_type"))
    expected = "SPARSEFLOATVECTOR" if kind == "sparse" else "FLOATVECTOR"
    if raw_type is None or _normalized_type(raw_type) not in {
        expected,
        f"DATATYPE{expected}",
    }:
        raise RuntimeError("Existing vector field has a different type")
    if kind == "embedding":
        raw_dim = _field_param(field, "dim")
        if raw_dim is None or int(raw_dim) != dim:
            raise RuntimeError("Existing embedding field has a different dimension")
