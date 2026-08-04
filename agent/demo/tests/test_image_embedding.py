from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from agent_workshop_demo.image_embedding import (
    DEFAULT_DINOV3_MODEL,
    IMAGE_EMBEDDING_FINGERPRINT_KEY,
    DINOv3ImageEmbeddingProvider,
    DeterministicImageEmbeddingProvider,
    ImageEmbeddingError,
    _TransformersDINOv3Runtime,
    _build_transformers_runtime,
    build_image_embedding_provider,
    configured_image_embedding_provider,
)
from agent_workshop_demo.ingestion import (
    _safe_manifest_asset_path,
    ingest_demo_sources,
    ingest_minio_sources,
)
from agent_workshop_demo.object_store import (
    MinIOSnapshot,
    MinIOSourceAdapter,
)
from agent_workshop_demo.sample_data import load_kb_chunks
from agent_workshop_demo.schema.pymilvus_adapter import (
    MilvusHybridRetriever,
)

IMAGE_DIR = Path("demo/sample_data/local_docs/images")
S3_IMAGE = IMAGE_DIR / "s3_sync_flow.png"


class _FakeRuntime:
    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.calls: list[Path] = []

    def embed(self, image_path: Path) -> list[float]:
        self.calls.append(image_path)
        return list(self.vector)


class _RecordingProvider:
    name = "recording"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.paths: list[Path] = []

    def embed(self, image_path: Path, *, dimensions: int) -> list[float]:
        self.paths.append(image_path)
        if self.fail:
            raise ImageEmbeddingError("inference_error")
        return [1.0] + [0.0] * (dimensions - 1)

    def fingerprint(self, *, dimensions: int) -> str:
        return f"recording:real-file-test:l2:{dimensions}"


class _FakeBatch(dict[str, object]):
    def __init__(self) -> None:
        super().__init__(pixel_values="pixels")
        self.moved_to: list[str] = []

    def to(self, device: str) -> _FakeBatch:
        self.moved_to.append(device)
        return self


class _FakeTensor:
    def __getitem__(self, index: int) -> _FakeTensor:
        if index != 0:
            raise IndexError(index)
        return self

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def tolist(self) -> list[float]:
        return [2.0] * 768


class _FixtureMinIOAdapter(MinIOSourceAdapter):
    def __init__(self, image_path: Path) -> None:
        self.image_path = image_path

    def download_to(self, target_dir: Path) -> MinIOSnapshot:
        root = target_dir / "snapshot"
        image_target = root / "images" / "minio-source.png"
        image_target.parent.mkdir(parents=True)
        image_target.write_bytes(self.image_path.read_bytes())
        manifest = [
            {
                "asset_path": "images/minio-source.png",
                "text": "MinIO 中的图片同步拓扑。",
                "doc_type": "image",
                "record_type": "image_asset",
                "title": "MinIO Image",
                "department": "engineering",
            }
        ]
        manifest_path = root / "asset_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False),
            encoding="utf-8",
        )
        return MinIOSnapshot(
            root=root,
            base_uri="s3://test-bucket/exercises",
            object_count=2,
            total_bytes=(
                image_target.stat().st_size + manifest_path.stat().st_size
            ),
        )


class ImageEmbeddingTests(unittest.TestCase):
    def test_deterministic_provider_hashes_validated_image_bytes(self) -> None:
        provider = DeterministicImageEmbeddingProvider()

        first = provider.embed(S3_IMAGE, dimensions=768)
        repeated = provider.embed(S3_IMAGE, dimensions=768)
        other = provider.embed(
            IMAGE_DIR / "rag_architecture.png",
            dimensions=768,
        )

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, other)
        self.assertEqual(len(first), 768)
        self.assertAlmostEqual(
            math.sqrt(sum(value * value for value in first)),
            1.0,
            places=6,
        )
        self.assertEqual(
            provider.fingerprint(dimensions=768),
            "deterministic:sha256-image-bytes-v1:l2:768",
        )

    def test_source_format_and_size_fail_closed(self) -> None:
        provider = DeterministicImageEmbeddingProvider(max_image_bytes=32)
        with self.assertRaisesRegex(ImageEmbeddingError, "oversize"):
            provider.embed(S3_IMAGE, dimensions=768)
        with self.assertRaisesRegex(
            ImageEmbeddingError,
            "missing_source",
        ):
            provider.embed(IMAGE_DIR / "missing.png", dimensions=768)
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid = Path(temp_dir) / "not-image.png"
            invalid.write_bytes(b"not a png")
            with self.assertRaisesRegex(
                ImageEmbeddingError,
                "unsupported_format",
            ):
                DeterministicImageEmbeddingProvider().embed(
                    invalid,
                    dimensions=768,
                )

    def test_dinov3_runtime_is_lazy_reused_and_l2_normalized(self) -> None:
        runtime = _FakeRuntime([2.0] * 768)
        factory_calls: list[tuple[str, str, str | None, bool]] = []

        def factory(
            model: str,
            device: str,
            token: str | None,
            local_files_only: bool,
        ) -> _FakeRuntime:
            factory_calls.append(
                (model, device, token, local_files_only)
            )
            return runtime

        provider = DINOv3ImageEmbeddingProvider(
            model=DEFAULT_DINOV3_MODEL,
            device="cpu",
            token="secret-token",
            local_files_only=True,
            runtime_factory=factory,
        )
        self.assertEqual(factory_calls, [])

        first = provider.embed(S3_IMAGE, dimensions=768)
        second = provider.embed(S3_IMAGE, dimensions=768)

        self.assertEqual(len(factory_calls), 1)
        self.assertEqual(
            factory_calls[0],
            (
                DEFAULT_DINOV3_MODEL,
                "cpu",
                "secret-token",
                True,
            ),
        )
        self.assertEqual(runtime.calls, [S3_IMAGE, S3_IMAGE])
        self.assertEqual(first, second)
        self.assertAlmostEqual(
            math.sqrt(sum(value * value for value in first)),
            1.0,
            places=6,
        )
        self.assertEqual(
            provider.fingerprint(dimensions=768),
            (
                f"dinov3:{DEFAULT_DINOV3_MODEL}:"
                "pooler_output:l2:768"
            ),
        )

    def test_dinov3_invalid_output_and_load_failure_are_sanitized(self) -> None:
        invalid_vectors: list[list[Any]] = [
            [1.0] * 767,
            [0.0] * 768,
            [math.nan] + [1.0] * 767,
            [True] + [1.0] * 767,
        ]
        for vector in invalid_vectors:
            with self.subTest(vector_head=vector[:1], size=len(vector)):
                provider = DINOv3ImageEmbeddingProvider(
                    runtime_factory=lambda *_: _FakeRuntime(vector),
                )
                with self.assertRaisesRegex(
                    ImageEmbeddingError,
                    "invalid_provider_output",
                ):
                    provider.embed(S3_IMAGE, dimensions=768)

        def failed_factory(*args: Any) -> _FakeRuntime:
            raise RuntimeError("secret model response")

        failed = DINOv3ImageEmbeddingProvider(
            runtime_factory=failed_factory
        )
        with self.assertRaisesRegex(
            ImageEmbeddingError,
            "^model_load_error$",
        ):
            failed.embed(S3_IMAGE, dimensions=768)
        with self.assertRaisesRegex(
            ImageEmbeddingError,
            "dimension_mismatch",
        ):
            failed.embed(S3_IMAGE, dimensions=384)

    def test_builder_is_explicit_and_does_not_load_weights(self) -> None:
        factory_calls: list[object] = []

        def recording_factory(
            model: str,
            device: str,
            token: str | None,
            local_files_only: bool,
        ) -> _FakeRuntime:
            factory_calls.append(
                (model, device, token, local_files_only)
            )
            return _FakeRuntime([1.0] * 768)

        offline = build_image_embedding_provider(
            {"IMAGE_EMBEDDING_PROVIDER": "deterministic"},
            runtime_factory=recording_factory,
        )
        self.assertIsInstance(
            offline,
            DeterministicImageEmbeddingProvider,
        )
        self.assertEqual(factory_calls, [])

        configured = build_image_embedding_provider(
            {
                "IMAGE_EMBEDDING_PROVIDER": "dinov3",
                "DINOV3_MODEL": "configured-model",
                "DINOV3_DEVICE": "cpu",
                "DINOV3_LOCAL_FILES_ONLY": "true",
                "HF_TOKEN": "secret-token",
            },
            runtime_factory=recording_factory,
        )
        self.assertIsInstance(configured, DINOv3ImageEmbeddingProvider)
        self.assertEqual(factory_calls, [])
        self.assertNotIn(
            "secret-token",
            configured.fingerprint(dimensions=768),
        )

        for environ in (
            {"IMAGE_EMBEDDING_PROVIDER": "unknown"},
            {
                "IMAGE_EMBEDDING_PROVIDER": "dinov3",
                "DINOV3_DEVICE": "tpu",
            },
            {
                "IMAGE_EMBEDDING_PROVIDER": "dinov3",
                "DINOV3_LOCAL_FILES_ONLY": "maybe",
            },
            {
                "IMAGE_EMBEDDING_PROVIDER": "deterministic",
                "IMAGE_EMBEDDING_MAX_BYTES": "0",
            },
        ):
            with self.subTest(environ=environ):
                with self.assertRaises(ValueError):
                    build_image_embedding_provider(environ)

    def test_missing_runtime_dependencies_are_typed(self) -> None:
        with patch(
            "agent_workshop_demo.image_embedding.import_module",
            side_effect=ImportError("secret module path"),
        ):
            with self.assertRaisesRegex(
                ImageEmbeddingError,
                "^dependency_missing$",
            ):
                _build_transformers_runtime(
                    DEFAULT_DINOV3_MODEL,
                    "cpu",
                    "secret-token",
                    True,
                )

    def test_manifest_paths_reject_aliases_and_traversal(self) -> None:
        for value in (
            "images//source.png",
            "images/./source.png",
            "images/../source.png",
            "/images/source.png",
            "images\\source.png",
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "asset_path"):
                    _safe_manifest_asset_path(value, record_index=1)
        self.assertEqual(
            _safe_manifest_asset_path(
                "images/source.png",
                record_index=1,
            ).as_posix(),
            "images/source.png",
        )

    def test_transformers_runtime_uses_rgb_device_inference_and_pooler(self) -> None:
        batch = _FakeBatch()
        processor = MagicMock(return_value=batch)
        processor_factory = MagicMock()
        processor_factory.from_pretrained.return_value = processor
        model = MagicMock()
        model.return_value = type(
            "Outputs",
            (),
            {"pooler_output": _FakeTensor()},
        )()
        model_factory = MagicMock()
        model_factory.from_pretrained.return_value = model
        transformers = type(
            "Transformers",
            (),
            {
                "AutoImageProcessor": processor_factory,
                "AutoModel": model_factory,
            },
        )()
        image = MagicMock()
        image.convert.return_value = "rgb-image"
        image_context = MagicMock()
        image_context.__enter__.return_value = image
        image_module = MagicMock()
        image_module.open.return_value = image_context
        torch = MagicMock()
        torch.inference_mode.return_value = MagicMock()
        modules = {
            "torch": torch,
            "transformers": transformers,
            "PIL.Image": image_module,
        }

        runtime = _TransformersDINOv3Runtime(
            DEFAULT_DINOV3_MODEL,
            "cpu",
            "secret-token",
            True,
            module_loader=modules.__getitem__,
        )
        vector = runtime.embed(S3_IMAGE)

        self.assertEqual(vector, [2.0] * 768)
        processor_factory.from_pretrained.assert_called_once_with(
            DEFAULT_DINOV3_MODEL,
            local_files_only=True,
            trust_remote_code=False,
            token="secret-token",
        )
        model_factory.from_pretrained.assert_called_once_with(
            DEFAULT_DINOV3_MODEL,
            local_files_only=True,
            trust_remote_code=False,
            token="secret-token",
        )
        image.convert.assert_called_once_with("RGB")
        processor.assert_called_once_with(
            images="rgb-image",
            return_tensors="pt",
        )
        self.assertEqual(batch.moved_to, ["cpu"])
        torch.inference_mode.assert_called_once_with()
        model.assert_called_once_with(pixel_values="pixels")
        model.to.assert_called_once_with("cpu")
        model.eval.assert_called_once_with()

        image_module.open.side_effect = OSError("secret decoder detail")
        with self.assertRaisesRegex(ImageEmbeddingError, "^decode_error$"):
            runtime.embed(S3_IMAGE)
        image_module.open.side_effect = None
        model.side_effect = RuntimeError("secret inference detail")
        with self.assertRaisesRegex(ImageEmbeddingError, "^inference_error$"):
            runtime.embed(S3_IMAGE)

    def test_configured_provider_is_cached_for_process_lifetime(self) -> None:
        provider = _RecordingProvider()
        configured_image_embedding_provider.cache_clear()
        try:
            with patch(
                "agent_workshop_demo.image_embedding."
                "build_image_embedding_provider",
                return_value=provider,
            ) as builder:
                first = configured_image_embedding_provider()
                second = configured_image_embedding_provider()
            self.assertIs(first, second)
            builder.assert_called_once_with()
        finally:
            configured_image_embedding_provider.cache_clear()

    def test_ingestion_embeds_each_manifest_image_from_its_path(self) -> None:
        provider = _RecordingProvider()

        result = ingest_demo_sources(
            Path("demo/sample_data/local_docs"),
            Path("demo/sample_data/mock_s3"),
            image_embedding_provider=provider,
        )

        image_chunks = [
            chunk for chunk in result.kb_chunks if chunk.has_image_vector
        ]
        self.assertEqual(len(image_chunks), 5)
        self.assertEqual(len(provider.paths), 5)
        self.assertEqual(
            {path.name for path in provider.paths},
            {
                "s3_sync_flow.png",
                "rag_architecture.png",
                "milvus_hybrid_search.png",
                "agentic_rag_workflow.png",
                "ingestion_pipeline.png",
            },
        )
        for chunk in image_chunks:
            self.assertEqual(len(chunk.image_vector or []), 768)
            self.assertEqual(
                (chunk.metadata or {})[
                    IMAGE_EMBEDDING_FINGERPRINT_KEY
                ],
                "recording:real-file-test:l2:768",
            )
            self.assertNotIn("placeholder", str(chunk.metadata))
            asset_path = Path("demo/sample_data/local_docs") / str(
                (chunk.metadata or {})["asset_path"]
            )
            self.assertEqual(
                chunk.checksum,
                hashlib.sha256(asset_path.read_bytes()).hexdigest(),
            )

    def test_minio_manifest_image_has_stable_object_identity(self) -> None:
        provider = _RecordingProvider()

        result = ingest_minio_sources(
            Path("demo/sample_data/local_docs"),
            _FixtureMinIOAdapter(S3_IMAGE),
            image_embedding_provider=provider,
        )

        image_chunk = next(
            chunk
            for chunk in result.ingestion.kb_chunks
            if chunk.source_uri
            == "s3://test-bucket/exercises/images/minio-source.png"
        )
        self.assertTrue(image_chunk.has_image_vector)
        self.assertEqual(image_chunk.bucket, "test-bucket")
        self.assertEqual(
            image_chunk.object_key,
            "exercises/images/minio-source.png",
        )
        self.assertEqual(
            (image_chunk.metadata or {})[
                IMAGE_EMBEDDING_FINGERPRINT_KEY
            ],
            "recording:real-file-test:l2:768",
        )
        self.assertEqual(
            image_chunk.checksum,
            hashlib.sha256(S3_IMAGE.read_bytes()).hexdigest(),
        )

    def test_ingestion_and_schema_reject_failure_or_space_mismatch(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            (
                "Image embedding failed for "
                "sample_data/local_docs/images/s3_sync_flow.png: "
                "inference_error"
            ),
        ):
            ingest_demo_sources(
                Path("demo/sample_data/local_docs"),
                Path("demo/sample_data/mock_s3"),
                image_embedding_provider=_RecordingProvider(fail=True),
            )

        image_chunk = next(
            chunk for chunk in load_kb_chunks() if chunk.has_image_vector
        )
        with self.assertRaisesRegex(
            ValueError,
            "finite L2-normalized values",
        ):
            replace(image_chunk, image_vector=[0.0] * 768)
        metadata_without_fingerprint = dict(image_chunk.metadata or {})
        metadata_without_fingerprint.pop(
            IMAGE_EMBEDDING_FINGERPRINT_KEY,
            None,
        )
        with self.assertRaisesRegex(
            ValueError,
            "metadata.image_embedding_fingerprint",
        ):
            replace(
                image_chunk,
                metadata=metadata_without_fingerprint,
            )
        metadata = dict(image_chunk.metadata or {})
        metadata[IMAGE_EMBEDDING_FINGERPRINT_KEY] = "wrong-space"
        with self.assertRaisesRegex(
            ValueError,
            "image embedding fingerprint does not match",
        ):
            MilvusHybridRetriever._record_for_insert(
                replace(image_chunk, metadata=metadata)
            )

    @unittest.skipUnless(
        os.environ.get("RUN_DINOV3_SMOKE") == "1",
        "set RUN_DINOV3_SMOKE=1 after accepting the gated DINOv3 license",
    )
    def test_real_dinov3_checkpoint_embeds_all_curated_images(self) -> None:
        model = os.environ.get(
            "DINOV3_MODEL",
            DEFAULT_DINOV3_MODEL,
        )
        provider = build_image_embedding_provider(
            {
                "IMAGE_EMBEDDING_PROVIDER": "dinov3",
                "DINOV3_MODEL": model,
                "DINOV3_DEVICE": os.environ.get(
                    "DINOV3_DEVICE",
                    "cpu",
                ),
                "DINOV3_LOCAL_FILES_ONLY": os.environ.get(
                    "DINOV3_LOCAL_FILES_ONLY",
                    "false",
                ),
                "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
            }
        )
        images = sorted(IMAGE_DIR.glob("*.png"))
        self.assertEqual(len(images), 5)
        for image_path in images:
            with self.subTest(image_path=image_path):
                vector = provider.embed(image_path, dimensions=768)
                self.assertEqual(len(vector), 768)
                self.assertAlmostEqual(
                    math.sqrt(sum(value * value for value in vector)),
                    1.0,
                    places=5,
                )
        self.assertEqual(
            provider.fingerprint(dimensions=768),
            f"dinov3:{model}:pooler_output:l2:768",
        )


if __name__ == "__main__":
    unittest.main()
