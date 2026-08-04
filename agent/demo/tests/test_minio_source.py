"""Independent contract tests for the real MinIO ingestion adapter."""

from __future__ import annotations

import io
import tempfile
import traceback
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from agent_workshop_demo.embedding import DeterministicTextEmbeddingProvider
from agent_workshop_demo.ingestion import (
    IngestionResult,
    MinIOIngestionResult,
    ingest_minio_sources,
)
from agent_workshop_demo import object_store
from agent_workshop_demo.object_store import (
    MinIOAdapterError,
    MinIOConfig,
    MinIOSourceAdapter,
    build_minio_source_adapter,
)
from demo.scripts import ingest_demo as ingest_script


@dataclass
class ListedObject:
    """Fake MinIO list entry."""

    object_name: str
    size: int | None
    is_dir: bool = False


class RecordingResponse:
    """Bounded streaming response with cleanup observations."""

    def __init__(
        self,
        content: bytes,
        *,
        fail_read: bool = False,
        fail_close: bool = False,
    ) -> None:
        self.content = content
        self.offset = 0
        self.fail_read = fail_read
        self.fail_close = fail_close
        self.closed = False
        self.released = False

    def read(self, amount: int | None = None) -> bytes:
        if self.fail_read:
            raise RuntimeError("secret response body")
        size = len(self.content) if amount is None else amount
        chunk = self.content[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk

    def close(self) -> None:
        self.closed = True
        if self.fail_close:
            raise RuntimeError("secret cleanup body")

    def release_conn(self) -> None:
        self.released = True


class RecordingMinIOClient:
    """Injectable MinIO fake implementing the production protocol."""

    def __init__(
        self,
        objects: list[ListedObject],
        contents: dict[str, bytes],
        *,
        bucket_exists: bool = True,
        fail_listing: bool = False,
        fail_read_key: str | None = None,
        fail_get_key: str | None = None,
        fail_close_key: str | None = None,
    ) -> None:
        self.objects = objects
        self.contents = contents
        self.exists = bucket_exists
        self.fail_listing = fail_listing
        self.fail_read_key = fail_read_key
        self.fail_get_key = fail_get_key
        self.fail_close_key = fail_close_key
        self.list_calls: list[tuple[str, str | None, bool]] = []
        self.get_calls: list[tuple[str, str]] = []
        self.responses: dict[str, RecordingResponse] = {}

    def bucket_exists(self, bucket_name: str) -> bool:
        return self.exists

    def list_objects(
        self,
        bucket_name: str,
        prefix: str | None = None,
        recursive: bool = False,
    ) -> list[ListedObject]:
        self.list_calls.append((bucket_name, prefix, recursive))
        if self.fail_listing:
            raise RuntimeError("endpoint?access_key=do-not-expose")
        return self.objects

    def get_object(self, bucket_name: str, object_name: str) -> RecordingResponse:
        self.get_calls.append((bucket_name, object_name))
        if object_name == self.fail_get_key:
            raise RuntimeError("access_key=TOPSECRET response=RAWBODY")
        response = RecordingResponse(
            self.contents[object_name],
            fail_read=object_name == self.fail_read_key,
            fail_close=object_name == self.fail_close_key,
        )
        self.responses[object_name] = response
        return response


def config(
    *,
    max_objects: int = 10,
    max_object_bytes: int = 1_024,
) -> MinIOConfig:
    """Build a small bounded test configuration."""

    return MinIOConfig(
        endpoint="localhost:9000",
        bucket="workshop-data",
        prefix="corpus",
        max_objects=max_objects,
        max_object_bytes=max_object_bytes,
    )


def deterministic_vector(text: str) -> list[float]:
    """Return a test-local vector independent of ``demo/.env``."""

    return DeterministicTextEmbeddingProvider().embed(text, dimensions=1_024)


def deterministic_metadata(
    metadata: dict[str, object] | None = None,
    *,
    dimensions: int = 1_024,
) -> dict[str, object]:
    """Attach a deterministic fingerprint without consulting process config."""

    output = dict(metadata or {})
    output["text_embedding_fingerprint"] = (
        f"deterministic:sha256-token-v1:{dimensions}"
    )
    return output


class MinIOSourceAdapterTests(unittest.TestCase):
    """Validate configuration, safety, cleanup and ingestion reuse."""

    def test_config_from_env_validates_credentials_bounds_and_safe_uri(self) -> None:
        result = MinIOConfig.from_env(
            {
                "MINIO_ENDPOINT": "minio.internal:9000",
                "MINIO_BUCKET": "workshop-data",
                "MINIO_PREFIX": "/team/corpus/",
                "MINIO_SECURE": "true",
                "MINIO_ACCESS_KEY": "access",
                "MINIO_SECRET_KEY": "secret",
                "MINIO_MAX_OBJECTS": "20",
                "MINIO_MAX_OBJECT_BYTES": "4096",
            }
        )
        self.assertEqual(result.endpoint, "minio.internal:9000")
        self.assertEqual(result.prefix, "team/corpus")
        self.assertTrue(result.secure)
        self.assertEqual(result.base_uri, "s3://workshop-data/team/corpus")

        invalid_values = (
            {"MINIO_ENDPOINT": "https://minio.example"},
            {"MINIO_BUCKET": "UPPERCASE"},
            {"MINIO_PREFIX": "../private"},
            {"MINIO_ACCESS_KEY": "access"},
            {"MINIO_MAX_OBJECTS": "0"},
            {"MINIO_SECURE": "perhaps"},
        )
        for values in invalid_values:
            with self.subTest(values=values), self.assertRaises(ValueError):
                MinIOConfig.from_env(values)

    def test_download_is_sorted_bounded_and_always_releases_responses(self) -> None:
        objects = [
            ListedObject("corpus/product/z.md", 4),
            ListedObject("corpus/engineering/a.md", 5),
            ListedObject("corpus/folder/", 0, is_dir=True),
        ]
        client = RecordingMinIOClient(
            objects,
            {
                "corpus/product/z.md": b"# Z\n",
                "corpus/engineering/a.md": b"# A\n\n",
            },
        )
        adapter = MinIOSourceAdapter(client, config())
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot = adapter.download_to(Path(temp_dir))
            self.assertEqual(snapshot.object_count, 2)
            self.assertEqual(snapshot.total_bytes, 9)
            self.assertEqual(snapshot.base_uri, "s3://workshop-data/corpus")
            self.assertEqual(
                (snapshot.root / "engineering/a.md").read_bytes(),
                b"# A\n\n",
            )
            self.assertEqual(
                client.get_calls,
                [
                    ("workshop-data", "corpus/engineering/a.md"),
                    ("workshop-data", "corpus/product/z.md"),
                ],
            )
        self.assertEqual(
            client.list_calls,
            [("workshop-data", "corpus/", True)],
        )
        self.assertTrue(all(item.closed for item in client.responses.values()))
        self.assertTrue(all(item.released for item in client.responses.values()))

    def test_missing_empty_unsafe_and_count_boundaries_fail_closed(self) -> None:
        cases = (
            (
                RecordingMinIOClient([], {}, bucket_exists=False),
                config(),
                "does not exist",
            ),
            (RecordingMinIOClient([], {}), config(), "contains no objects"),
            (
                RecordingMinIOClient(
                    [ListedObject("corpus/../private.md", 1)],
                    {"corpus/../private.md": b"x"},
                ),
                config(),
                "unsafe object key",
            ),
            (
                RecordingMinIOClient(
                    [
                        ListedObject("corpus/a.md", 1),
                        ListedObject("corpus/b.md", 1),
                    ],
                    {"corpus/a.md": b"a", "corpus/b.md": b"b"},
                ),
                config(max_objects=1),
                "exceeds 1 objects",
            ),
            (
                RecordingMinIOClient(
                    [ListedObject("corpus/a.md", 2_000)],
                    {"corpus/a.md": b""},
                ),
                config(max_object_bytes=100),
                "exceeds 100 bytes",
            ),
        )
        for client, test_config, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as root:
                with self.assertRaisesRegex(MinIOAdapterError, message):
                    MinIOSourceAdapter(client, test_config).download_to(Path(root))

    def test_raw_key_boundaries_and_destination_collisions_fail_closed(self) -> None:
        unsafe_keys = (
            "/corpus/a.md",
            r"corpus\private.md",
            "corpus/\x00private.md",
            "corpus/../private.md",
            "corpus/./a.md",
            "corpus//a.md",
            "corpus-old/a.md",
        )
        for key in unsafe_keys:
            client = RecordingMinIOClient(
                [ListedObject(key, 1)],
                {key: b"x"},
            )
            with self.subTest(key=key), tempfile.TemporaryDirectory() as root:
                with self.assertRaises(MinIOAdapterError):
                    MinIOSourceAdapter(client, config()).download_to(Path(root))

        duplicate = RecordingMinIOClient(
            [
                ListedObject("corpus/a.md", 1),
                ListedObject("corpus/a.md", 1),
            ],
            {"corpus/a.md": b"x"},
        )
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(
                MinIOAdapterError,
                "colliding object destinations",
            ):
                MinIOSourceAdapter(duplicate, config()).download_to(Path(root))

    def test_size_change_get_and_cleanup_failures_are_sanitized(self) -> None:
        key = "corpus/a.md"
        clients = (
            RecordingMinIOClient(
                [ListedObject(key, 2)],
                {key: b"x"},
            ),
            RecordingMinIOClient(
                [ListedObject(key, 1)],
                {key: b"x"},
                fail_get_key=key,
            ),
            RecordingMinIOClient(
                [ListedObject(key, 1)],
                {key: b"x"},
                fail_close_key=key,
            ),
        )
        for client in clients:
            with tempfile.TemporaryDirectory() as root:
                with self.assertRaises(MinIOAdapterError) as raised:
                    MinIOSourceAdapter(client, config()).download_to(Path(root))
                rendered = "".join(
                    traceback.TracebackException.from_exception(
                        raised.exception
                    ).format()
                )
                self.assertNotIn("TOPSECRET", rendered)
                self.assertNotIn("RAWBODY", rendered)
                self.assertNotIn("secret cleanup body", rendered)

    def test_streaming_oversize_and_read_failure_remove_partial_files(self) -> None:
        for fail_read, expected in ((False, "exceeds 4 bytes"), (True, "Unable to download")):
            key = "corpus/product/a.md"
            client = RecordingMinIOClient(
                [ListedObject(key, None)],
                {key: b"12345"},
                fail_read_key=key if fail_read else None,
            )
            with self.subTest(fail_read=fail_read), tempfile.TemporaryDirectory() as root:
                with self.assertRaisesRegex(MinIOAdapterError, expected) as raised:
                    MinIOSourceAdapter(
                        client,
                        config(max_object_bytes=4),
                    ).download_to(Path(root))
                self.assertNotIn("secret response body", str(raised.exception))
                self.assertFalse((Path(root) / "product/a.md").exists())
                response = client.responses[key]
                self.assertTrue(response.closed)
                self.assertTrue(response.released)

    def test_failed_snapshot_is_empty_and_retryable(self) -> None:
        first = "corpus/a.md"
        second = "corpus/b.md"
        objects = [ListedObject(first, 1), ListedObject(second, 1)]
        client = RecordingMinIOClient(
            objects,
            {first: b"a", second: b"b"},
            fail_read_key=second,
        )
        with tempfile.TemporaryDirectory() as root:
            target = Path(root)
            with self.assertRaises(MinIOAdapterError):
                MinIOSourceAdapter(client, config()).download_to(target)
            self.assertEqual(list(target.iterdir()), [])

            client.fail_read_key = None
            snapshot = MinIOSourceAdapter(client, config()).download_to(target)
            self.assertEqual((snapshot.root / "a.md").read_bytes(), b"a")
            self.assertEqual((snapshot.root / "b.md").read_bytes(), b"b")

    def test_listing_failure_is_sanitized(self) -> None:
        client = RecordingMinIOClient([], {}, fail_listing=True)
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(
                MinIOAdapterError,
                "Unable to list MinIO bucket 'workshop-data'",
            ) as raised:
                MinIOSourceAdapter(client, config()).download_to(Path(root))
        self.assertNotIn("access_key", str(raised.exception))

    def test_builder_is_lazy_and_passes_validated_sdk_arguments(self) -> None:
        sdk_calls: list[tuple[str, str | None, str | None, bool]] = []
        client = RecordingMinIOClient([], {})

        class FakeSDK:
            @staticmethod
            def Minio(
                endpoint: str,
                *,
                access_key: str | None,
                secret_key: str | None,
                secure: bool,
            ) -> RecordingMinIOClient:
                sdk_calls.append((endpoint, access_key, secret_key, secure))
                return client

        values = {
            "MINIO_ENDPOINT": "minio.internal:9000",
            "MINIO_BUCKET": "workshop-data",
            "MINIO_ACCESS_KEY": "access",
            "MINIO_SECRET_KEY": "secret",
            "MINIO_SECURE": "true",
        }
        with patch.object(object_store, "import_module", return_value=FakeSDK):
            adapter = build_minio_source_adapter(values)
        self.assertIs(adapter.client, client)
        self.assertEqual(
            sdk_calls,
            [("minio.internal:9000", "access", "secret", True)],
        )

    def test_cli_modes_are_explicit_and_minio_error_has_no_traceback(self) -> None:
        empty = IngestionResult([], [])
        stdout = io.StringIO()
        with (
            patch.object(
                ingest_script,
                "ingest_demo_sources",
                return_value=empty,
            ),
            patch.object(
                ingest_script,
                "build_minio_source_adapter",
            ) as builder,
            patch("sys.stdout", stdout),
        ):
            self.assertEqual(ingest_script.main(["--dry-run"]), 0)
        builder.assert_not_called()
        self.assertIn('"s3_source": "mock"', stdout.getvalue())

        adapter = MinIOSourceAdapter(RecordingMinIOClient([], {}), config())
        minio_output = MinIOIngestionResult(
            ingestion=empty,
            source_base_uri="s3://workshop-data/corpus",
            source_object_count=2,
            source_total_bytes=20,
        )
        stdout = io.StringIO()
        with (
            patch.object(
                ingest_script,
                "build_minio_source_adapter",
                return_value=adapter,
            ),
            patch.object(
                ingest_script,
                "ingest_minio_sources",
                return_value=minio_output,
            ),
            patch("sys.stdout", stdout),
        ):
            self.assertEqual(
                ingest_script.main(["--s3-source", "minio", "--dry-run"]),
                0,
            )
        self.assertIn('"s3_source": "minio"', stdout.getvalue())

        stderr = io.StringIO()
        unsafe_error = MinIOAdapterError("safe MinIO failure")
        unsafe_error.__cause__ = RuntimeError("access_key=TOPSECRET RAWBODY")
        with (
            patch.object(
                ingest_script,
                "build_minio_source_adapter",
                side_effect=unsafe_error,
            ),
            patch("sys.stderr", stderr),
            self.assertRaises(SystemExit),
        ):
            ingest_script.main(["--s3-source", "minio", "--dry-run"])
        self.assertIn("safe MinIO failure", stderr.getvalue())
        self.assertNotIn("TOPSECRET", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_snapshot_reuses_ingestion_and_preserves_object_identity(self) -> None:
        key = "corpus/product/guide.md"
        content = b"# Guide\n\nUse the MinIO adapter."
        client = RecordingMinIOClient(
            [ListedObject(key, len(content))],
            {key: content},
        )
        with tempfile.TemporaryDirectory() as root:
            workspace = Path(root)
            local = workspace / "local"
            local.mkdir()
            (local / "local.md").write_text(
                "# Local\n\nLocal golden path.",
                encoding="utf-8",
            )
            manifest = workspace / "versions.json"
            manifest.write_text(
                """
{
  "s3://workshop-data/corpus/product/guide.md": {
    "doc_id": "doc_minio_guide",
    "doc_version": "v1",
    "is_current": true
  }
}
""".strip(),
                encoding="utf-8",
            )
            with (
                patch(
                    "agent_workshop_demo.ingestion.dense_vector",
                    side_effect=deterministic_vector,
                ),
                patch(
                    "agent_workshop_demo.ingestion.embedding_metadata",
                    side_effect=deterministic_metadata,
                ),
            ):
                result = ingest_minio_sources(
                    local,
                    MinIOSourceAdapter(client, config()),
                    version_manifest_path=manifest,
                )

        minio_chunks = [
            chunk
            for chunk in result.ingestion.kb_chunks
            if chunk.source_type == "s3"
        ]
        self.assertEqual(result.source_object_count, 1)
        self.assertEqual(result.source_total_bytes, len(content))
        self.assertEqual(len(minio_chunks), 1)
        self.assertEqual(minio_chunks[0].doc_id, "doc_minio_guide")
        self.assertEqual(minio_chunks[0].bucket, "workshop-data")
        self.assertEqual(minio_chunks[0].object_key, key)
        self.assertEqual(
            minio_chunks[0].source_uri,
            "s3://workshop-data/corpus/product/guide.md",
        )


if __name__ == "__main__":
    unittest.main()
