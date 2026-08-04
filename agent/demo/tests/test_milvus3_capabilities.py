from __future__ import annotations

import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from importlib import import_module
from io import StringIO
from unittest.mock import patch
from types import SimpleNamespace
from typing import Any, cast

from agent_workshop_demo.eval_snapshot import MilvusEvalSnapshot
from agent_workshop_demo.dedup import near_duplicate_jaccard
from agent_workshop_demo.milvus_time import (
    encode_expiry,
    epoch_ms_from_milvus,
    milvus_timestamp,
)
from agent_workshop_demo.retrieval import InMemoryHybridRetriever
from agent_workshop_demo.sample_data import load_kb_chunks
from agent_workshop_demo.schema.collections import (
    DOC_DEDUP_SIGNATURES_COLLECTION,
    DOC_DEDUP_SIGNATURES_INDEXES,
    KB_CHUNKS_COLLECTION,
    KB_CHUNKS_INDEXES,
)
from agent_workshop_demo.schema.evolution import (
    MilvusSchemaEvolution,
    _add_function_field_with_physical_backfill,
    _require_retrieval_text_field,
)
from agent_workshop_demo.schema.pymilvus_adapter import (
    MilvusDedupStore,
    MilvusHybridRetriever,
    _index_requests,
)
from demo.scripts.run_eval import main as run_eval_main


class SearchClient:
    def __init__(self, hits: dict[str, Any]) -> None:
        self.hits = hits
        self.search_calls: list[dict[str, Any]] = []
        self.query_calls: list[dict[str, Any]] = []

    def search(self, **kwargs: Any) -> Any:
        self.search_calls.append(kwargs)
        return self.hits[kwargs["anns_field"]]

    def query(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.query_calls.append(kwargs)
        return [
            {
                "source_type": "local",
                "department": "engineering",
                "count(*)": 2,
            }
        ]


class DedupClient:
    def __init__(self) -> None:
        self.inserted: list[dict[str, Any]] = []

    def has_collection(self, **kwargs: Any) -> bool:
        return True

    def load_collection(self, **kwargs: Any) -> None:
        pass

    def delete(self, **kwargs: Any) -> dict[str, int]:
        return {"delete_count": 0}

    def insert(self, **kwargs: Any) -> dict[str, int]:
        self.inserted.extend(kwargs["data"])
        return {"insert_count": len(kwargs["data"])}

    def flush(self, **kwargs: Any) -> None:
        pass


class SnapshotClient:
    def __init__(self) -> None:
        self.collections = {"kb_chunks"}
        self.calls: list[str] = []

    def has_collection(self, *, collection_name: str) -> bool:
        return collection_name in self.collections

    def describe_snapshot(self, **kwargs: Any) -> Any:
        raise RuntimeError("missing")

    def list_snapshots(self, **kwargs: Any) -> list[Any]:
        return []

    def flush(self, **kwargs: Any) -> None:
        self.calls.append("flush")

    def create_snapshot(self, **kwargs: Any) -> None:
        self.calls.append("create")

    def restore_snapshot(self, **kwargs: Any) -> int:
        self.calls.append("restore")
        self.collections.add(str(kwargs["target_collection_name"]))
        return 17

    def get_restore_snapshot_state(self, **kwargs: Any) -> Any:
        return SimpleNamespace(state="RestoreSnapshotCompleted")

    def list_restore_snapshot_jobs(self, **kwargs: Any) -> list[Any]:
        return [
            SimpleNamespace(
                job_id=17,
                snapshot_name="eval_v1",
                collection_name="eval_kb_v1",
                state="RestoreSnapshotCompleted",
            )
        ]

    def load_collection(self, **kwargs: Any) -> None:
        self.calls.append("load")


class EvolutionClient:
    def __init__(self) -> None:
        self.fields: list[dict[str, Any]] = []
        self.functions: list[dict[str, Any]] = []
        self.added: list[dict[str, Any]] = []
        self.upserts: list[dict[str, Any]] = []
        self.function_calls: list[dict[str, Any]] = []

    def describe_collection(self, **kwargs: Any) -> dict[str, Any]:
        return {"fields": self.fields, "functions": self.functions}

    def add_collection_field(self, **kwargs: Any) -> None:
        self.added.append(kwargs)
        self.fields.append(
            {
                "name": kwargs["field_name"],
                "type": kwargs["data_type"],
                **{
                    key: value
                    for key, value in kwargs.items()
                    if key
                    in {
                        "nullable",
                        "dim",
                        "max_length",
                        "enable_analyzer",
                        "analyzer_params",
                    }
                },
            }
        )

    def upsert(self, **kwargs: Any) -> dict[str, int]:
        self.upserts.append(kwargs)
        return {"upsert_count": len(kwargs["data"])}

    def prepare_index_params(self) -> Any:
        class Params:
            def __init__(self) -> None:
                self.requests: list[dict[str, Any]] = []

            def add_index(self, **kwargs: Any) -> None:
                self.requests.append(kwargs)

        return Params()

    def add_function_field(self, **kwargs: Any) -> None:
        self.function_calls.append(kwargs)
        self.fields.append(
            {
                "name": kwargs["field_schema"].name,
                "type": kwargs["field_schema"].dtype,
            }
        )
        function = kwargs["func"]
        self.functions.append(
            {
                "name": function.name,
                "type": function.type,
                "input_field_names": function.input_field_names,
                "output_field_names": function.output_field_names,
            }
        )

    def describe_index(self, **kwargs: Any) -> dict[str, Any]:
        request = self.function_calls[-1]["index_params"].requests[0]
        return {
            "field_name": request["field_name"],
            "index_type": request["index_type"],
            "metric_type": request["metric_type"],
            "params": request["params"],
        }


class Milvus3CapabilityTests(unittest.TestCase):
    def test_server_scalar_order_and_local_parity_are_explicit(self) -> None:
        first, second = load_kb_chunks()[:2]
        first = replace(first, updated_at=1, priority=1)
        second = replace(second, updated_at=2, priority=2)
        hit1 = {"distance": 0.99, "entity": first.to_dict()}
        hit2 = {"distance": 0.50, "entity": second.to_dict()}
        client = SearchClient(
            {"text_vector": [[hit1, hit2]], "sparse_vector": [[hit1, hit2]]}
        )
        milvus_results = MilvusHybridRetriever(client).search(
            "Milvus",
            top_k=2,
            order_by=["updated_at desc"],
            order_mode="scalar",
        )
        local_results = InMemoryHybridRetriever([first, second]).search(
            "Milvus",
            top_k=2,
            order_by=["updated_at desc"],
            order_mode="scalar",
        )
        self.assertEqual(milvus_results[0].chunk.chunk_id, second.chunk_id)
        self.assertEqual(local_results[0].chunk.chunk_id, second.chunk_id)
        self.assertTrue(
            all(
                call["order_by_fields"]
                == [{"field": "updated_at", "order": "desc"}]
                for call in client.search_calls
            )
        )

    def test_sparse_field_cutover_is_explicit(self) -> None:
        chunk = load_kb_chunks()[0]
        hits = [[{"distance": 1.0, "entity": chunk.to_dict()}]]
        client = SearchClient({"text_vector": hits, "sparse_vector_v2": hits})
        MilvusHybridRetriever(client, sparse_field="sparse_vector_v2").search(
            "Milvus", top_k=1
        )
        self.assertEqual(client.search_calls[1]["anns_field"], "sparse_vector_v2")
        with self.assertRaisesRegex(ValueError, "valid Milvus field"):
            MilvusHybridRetriever(client, sparse_field="unsafe-field")

    def test_facets_use_one_bounded_query_aggregation(self) -> None:
        chunks = load_kb_chunks()[:2]
        hits = [[{"distance": 1.0, "entity": item.to_dict()} for item in chunks]]
        client = SearchClient({"text_vector": hits, "sparse_vector": hits})
        adapter = MilvusHybridRetriever(client)
        results = adapter.search("Milvus", top_k=2)
        facets = adapter.aggregations(results, ["source_type", "department"])
        self.assertEqual(facets["source_type"], {"local": 2})
        self.assertEqual(client.query_calls[0]["group_by_fields"], ["source_type", "department"])
        self.assertEqual(client.query_calls[0]["output_fields"][-1], "count(*)")
        with self.assertRaisesRegex(ValueError, "Unsupported aggregation"):
            InMemoryHybridRetriever(chunks).aggregations(results, ["text"])

    def test_bm25_sindi_and_minhash_dido_definitions(self) -> None:
        self.assertEqual(KB_CHUNKS_COLLECTION["functions"][0]["type"], "BM25")
        default_sparse = next(
            item for item in _index_requests(KB_CHUNKS_INDEXES) if item["field_name"] == "sparse_vector"
        )
        legacy_sparse = next(
            item
            for item in _index_requests(
                KB_CHUNKS_INDEXES,
                sparse_compatibility_daat_maxscore=True,
            )
            if item["field_name"] == "sparse_vector"
        )
        self.assertNotIn("inverted_index_algo", default_sparse["params"])
        self.assertEqual(legacy_sparse["params"]["inverted_index_algo"], "DAAT_MAXSCORE")
        dedup_functions = DOC_DEDUP_SIGNATURES_COLLECTION["functions"]
        self.assertIsInstance(dedup_functions, list)
        self.assertEqual(list(dedup_functions)[0]["type"], "MINHASH")
        dedup_index = cast(
            dict[str, Any],
            DOC_DEDUP_SIGNATURES_INDEXES["minhash_signature"],
        )
        self.assertIsInstance(dedup_index, dict)
        self.assertEqual(dedup_index["index_type"], "MINHASH_LSH")

    def test_dedup_store_never_writes_function_output(self) -> None:
        client = DedupClient()
        record = {
            "doc_id": "doc",
            "chunk_id": "chunk",
            "source_uri": "safe/path",
            "source_type": "local",
            "record_level": "chunk",
            "normalized_text": "some text",
            "checksum": "sha256:abc",
            "created_at": 1,
            "metadata": {},
        }
        MilvusDedupStore(client).insert([record])
        self.assertNotIn("minhash_signature", client.inserted[0])

    def test_local_near_duplicate_estimator_is_deterministic(self) -> None:
        exact = near_duplicate_jaccard("alpha beta gamma", "alpha beta gamma")
        near = near_duplicate_jaccard(
            "alpha beta gamma delta", "alpha beta gamma epsilon"
        )
        unrelated = near_duplicate_jaccard(
            "alpha beta gamma", "red green blue"
        )
        self.assertEqual(exact, 1.0)
        self.assertGreater(near, unrelated)
        self.assertEqual(unrelated, 0.0)

    def test_timestamptz_codec_is_strict_and_reversible(self) -> None:
        self.assertEqual(milvus_timestamp(2_000), "1970-01-01T00:00:02.000Z")
        self.assertEqual(epoch_ms_from_milvus("1970-01-01T00:00:02.000Z"), 2_000)
        self.assertEqual(encode_expiry({"expires_at": 2_000})["expires_at"], "1970-01-01T00:00:02.000Z")
        with self.assertRaises(ValueError):
            epoch_ms_from_milvus("1970-01-01T00:00:02")

    def test_snapshot_pins_and_loads_restored_target(self) -> None:
        client = SnapshotClient()
        provenance = MilvusEvalSnapshot(client, sleeper=lambda _: None).pin(
            source_collection="kb_chunks",
            snapshot_name="eval_v1",
            target_collection="eval_kb_v1",
        )
        self.assertEqual(provenance.restore_job_id, 17)
        self.assertEqual(client.calls, ["flush", "create", "restore", "load"])

    def test_snapshot_list_string_fallback_fails_closed(self) -> None:
        client = SnapshotClient()
        client.list_snapshots = lambda **_: ["eval_v1"]  # type: ignore[method-assign]
        with self.assertRaisesRegex(RuntimeError, "Unable to validate"):
            MilvusEvalSnapshot(client).pin(
                source_collection="kb_chunks",
                snapshot_name="eval_v1",
                target_collection="eval_kb_v1",
            )

    def test_schema_evolution_is_dry_run_and_partial_update(self) -> None:
        client = EvolutionClient()
        evolution = MilvusSchemaEvolution(client)
        planned = evolution.add_vector_field(
            "new_embedding", kind="embedding", dim=2
        )
        self.assertEqual(planned["status"], "planned")
        self.assertEqual(client.added, [])
        evolution.add_vector_field(
            "new_embedding", kind="embedding", dim=2, apply=True
        )
        report = evolution.backfill_embedding(
            "new_embedding",
            [{"id": 7, "new_embedding": [1.0, 0.0]}],
            dim=2,
            apply=True,
        )
        self.assertEqual(report["applied_count"], 1)
        self.assertTrue(client.upserts[0]["partial_update"])

    def test_schema_evolution_rejects_wrong_shape_and_uses_atomic_bm25(self) -> None:
        client = EvolutionClient()
        client.fields = [
            {
                "name": "wrong_embedding",
                "type": "VarChar",
            }
        ]
        evolution = MilvusSchemaEvolution(client)
        with self.assertRaisesRegex(RuntimeError, "different type"):
            evolution.add_vector_field(
                "wrong_embedding", kind="embedding", dim=2
            )
        client.fields = [
            {
                "name": "retrieval_text",
                "type": "VarChar",
                "nullable": True,
                "max_length": 32_768,
                "enable_analyzer": True,
                "analyzer_params": {
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
                },
            }
        ]
        report = evolution.add_bm25_function(
            output_field_name="sparse_vector_v2",
            apply=True,
        )
        self.assertTrue(report["applied"])
        call = client.function_calls[0]
        self.assertEqual(call["func"].output_field_names, ["sparse_vector_v2"])
        self.assertEqual(
            call["index_params"].requests[0]["index_type"],
            "SPARSE_INVERTED_INDEX",
        )
        self.assertTrue(call["do_physical_backfill"])
        self.assertEqual(
            report["activation"],
            {"MILVUS_SPARSE_FIELD": "sparse_vector_v2"},
        )

    def test_exact_sdk_request_enables_physical_backfill(self) -> None:
        try:
            milvus = import_module("pymilvus")
            common_pb = import_module("pymilvus.grpc_gen.common_pb2")
            call_context = import_module("pymilvus.client.call_context")
        except ImportError:
            self.skipTest("optional pymilvus is not installed")

        class Stub:
            request: Any = None

            def AlterCollectionSchema(self, request: Any, **kwargs: Any) -> Any:
                self.request = request
                return SimpleNamespace(alter_status=common_pb.Status(error_code=0))

        class Connection:
            def __init__(self) -> None:
                self._stub = Stub()
                self.invalidated: list[tuple[str, str]] = []

            def _invalidate_schema(self, collection: str, *, db_name: str) -> None:
                self.invalidated.append((collection, db_name))

        class Client:
            def __init__(self) -> None:
                self.connection = Connection()

            def _get_connection(self) -> Any:
                return self.connection

            def _generate_call_context(self) -> Any:
                return call_context.CallContext(db_name="default")

        client = Client()
        field = milvus.FieldSchema(
            name="sparse_vector_v2",
            dtype=milvus.DataType.SPARSE_FLOAT_VECTOR,
            nullable=True,
        )
        function = milvus.Function(
            name="bm25_sparse_vector_v2",
            function_type=milvus.FunctionType.BM25,
            input_field_names=["retrieval_text"],
            output_field_names=["sparse_vector_v2"],
        )
        indexes = milvus.MilvusClient.prepare_index_params()
        indexes.add_index(
            field_name="sparse_vector_v2",
            index_name="sparse_vector_v2_idx",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={},
        )
        _add_function_field_with_physical_backfill(
            client,
            collection_name="kb_chunks",
            field_schema=field,
            function=function,
            index_params=indexes,
        )
        request = client.connection._stub.request
        self.assertTrue(request.action.add_request.do_physical_backfill)
        self.assertEqual(
            client.connection.invalidated,
            [("kb_chunks", "default")],
        )

    def test_exact_sdk_wire_field_shape_revalidates(self) -> None:
        try:
            milvus = import_module("pymilvus")
            prepare = import_module("pymilvus.client.prepare")
            abstract = import_module("pymilvus.client.abstract")
        except ImportError:
            self.skipTest("optional pymilvus is not installed")
        field = milvus.FieldSchema(
            name="retrieval_text",
            dtype=milvus.DataType.VARCHAR,
            nullable=True,
            max_length=32_768,
            enable_analyzer=True,
            analyzer_params={
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
            },
        )
        proto, _, _ = prepare.Prepare.get_field_schema(field.to_dict())
        described = abstract.FieldSchema(proto).dict()
        self.assertEqual(described["params"]["enable_analyzer"], "True")
        _require_retrieval_text_field(described)

    def test_snapshot_eval_inherits_sparse_reader_cutover(self) -> None:
        with (
            patch.dict("os.environ", {"MILVUS_SPARSE_FIELD": "sparse_vector_v2"}),
            patch("demo.scripts.run_eval.MilvusHybridRetriever.connect") as connect,
            patch("demo.scripts.run_eval.MilvusEvalSnapshot") as snapshot,
            patch("demo.scripts.run_eval.evaluate_questions", return_value={}),
            redirect_stdout(StringIO()),
        ):
            snapshot.return_value.pin.return_value = SimpleNamespace(
                to_dict=lambda: {}
            )
            result = run_eval_main(
                [
                    "--milvus-uri",
                    "http://milvus.test:19530",
                    "--milvus-token",
                    "token",
                    "--snapshot-name",
                    "eval_v1",
                    "--source-collection",
                    "kb_chunks",
                    "--target-collection",
                    "eval_kb_v1",
                ]
            )
        self.assertEqual(result, 0)
        connect.assert_called_once_with(
            "http://milvus.test:19530",
            "token",
            collection_name="eval_kb_v1",
            sparse_field="sparse_vector_v2",
        )


if __name__ == "__main__":
    unittest.main()
