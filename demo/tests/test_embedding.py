from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from typing import Any

from agent_workshop_demo.embedding import (
    DeterministicTextEmbeddingProvider,
    OpenAITextEmbeddingProvider,
    TextEmbeddingProvider,
    TextEmbeddingError,
    build_text_embedding_provider,
    embedding_metadata,
)


class RecordingEmbeddings:
    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=self.vector)]
        )


class RecordingOpenAIClient:
    def __init__(self, vector: list[float]) -> None:
        self.embeddings = RecordingEmbeddings(vector)


class FailingEmbeddings:
    def create(self, **kwargs: Any) -> SimpleNamespace:
        raise TimeoutError("secret provider response")


class FailingOpenAIClient:
    def __init__(self) -> None:
        self.embeddings = FailingEmbeddings()


class EmbeddingTests(unittest.TestCase):
    def test_openai_provider_sends_dimension_preserving_request(self) -> None:
        client = RecordingOpenAIClient([0.1, 0.2, 0.3])
        provider = OpenAITextEmbeddingProvider(
            client=client,
            model="text-embedding-3-small",
            timeout_seconds=8.5,
        )

        vector = provider.embed("Milvus 文档", dimensions=3)

        self.assertEqual(vector, [0.1, 0.2, 0.3])
        self.assertEqual(
            client.embeddings.calls,
            [
                {
                    "input": "Milvus 文档",
                    "model": "text-embedding-3-small",
                    "dimensions": 3,
                    "encoding_format": "float",
                    "timeout": 8.5,
                }
            ],
        )

    def test_openai_provider_rejects_invalid_vectors(self) -> None:
        invalid_vectors = [
            [0.1],
            [0.1, math.nan],
            [0.1, math.inf],
        ]

        for vector in invalid_vectors:
            with self.subTest(vector=vector):
                provider = OpenAITextEmbeddingProvider(
                    client=RecordingOpenAIClient(vector),
                    model="text-embedding-3-small",
                )
                with self.assertRaisesRegex(
                    TextEmbeddingError,
                    "invalid_provider_output",
                ):
                    provider.embed("valid text", dimensions=2)

    def test_openai_provider_sanitizes_provider_failure(self) -> None:
        provider = OpenAITextEmbeddingProvider(
            client=FailingOpenAIClient(),
            model="text-embedding-3-small",
        )

        with self.assertRaises(TextEmbeddingError) as raised:
            provider.embed("sensitive document", dimensions=2)

        self.assertEqual(raised.exception.reason_code, "timeout")
        self.assertIsNone(raised.exception.__cause__)
        self.assertIsNone(raised.exception.__context__)
        self.assertNotIn("secret provider response", str(raised.exception))
        self.assertNotIn("sensitive document", str(raised.exception))

    def test_providers_reject_empty_text_before_network_io(self) -> None:
        client = RecordingOpenAIClient([0.1, 0.2])
        providers: list[TextEmbeddingProvider] = [
            DeterministicTextEmbeddingProvider(),
            OpenAITextEmbeddingProvider(
                client=client,
                model="text-embedding-3-small",
            ),
        ]

        for provider in providers:
            with self.subTest(provider=provider.name):
                with self.assertRaisesRegex(ValueError, "non-empty"):
                    provider.embed("  ", dimensions=2)

        self.assertEqual(client.embeddings.calls, [])

    def test_auto_mode_without_key_uses_deterministic_provider(self) -> None:
        provider = build_text_embedding_provider(
            {"EMBEDDING_PROVIDER": "auto"}
        )

        self.assertIsInstance(provider, DeterministicTextEmbeddingProvider)

    def test_default_mode_never_uses_ambient_openai_key(self) -> None:
        client_factory_called = False

        def client_factory(api_key: str) -> RecordingOpenAIClient:
            nonlocal client_factory_called
            client_factory_called = True
            return RecordingOpenAIClient([0.1, 0.2])

        provider = build_text_embedding_provider(
            {"OPENAI_API_KEY": "ambient-key"},
            client_factory=client_factory,
        )

        self.assertIsInstance(provider, DeterministicTextEmbeddingProvider)
        self.assertFalse(client_factory_called)

    def test_auto_mode_with_key_uses_default_openai_model(self) -> None:
        client = RecordingOpenAIClient([0.1, 0.2])

        provider = build_text_embedding_provider(
            {
                "EMBEDDING_PROVIDER": "auto",
                "OPENAI_API_KEY": "test-key",
            },
            client_factory=lambda api_key: client,
        )

        provider.embed("query", dimensions=2)

        self.assertIsInstance(provider, OpenAITextEmbeddingProvider)
        self.assertEqual(
            client.embeddings.calls[0]["model"],
            "text-embedding-3-small",
        )

    def test_openai_mode_uses_explicit_configuration(self) -> None:
        client = RecordingOpenAIClient([0.1, 0.2])
        received_keys: list[str] = []

        def client_factory(api_key: str) -> RecordingOpenAIClient:
            received_keys.append(api_key)
            return client

        provider = build_text_embedding_provider(
            {
                "EMBEDDING_PROVIDER": "openai",
                "OPENAI_API_KEY": "test-key",
                "OPENAI_EMBEDDING_MODEL": "configured-model",
                "OPENAI_EMBEDDING_TIMEOUT_SECONDS": "7.5",
            },
            client_factory=client_factory,
        )

        vector = provider.embed("query", dimensions=2)

        self.assertEqual(received_keys, ["test-key"])
        self.assertEqual(vector, [0.1, 0.2])
        self.assertEqual(client.embeddings.calls[0]["model"], "configured-model")
        self.assertEqual(client.embeddings.calls[0]["timeout"], 7.5)

    def test_invalid_embedding_configuration_fails_clearly(self) -> None:
        invalid_environments = [
            {"EMBEDDING_PROVIDER": "unknown"},
            {"EMBEDDING_PROVIDER": "openai"},
            {
                "EMBEDDING_PROVIDER": "openai",
                "OPENAI_API_KEY": "key",
                "OPENAI_EMBEDDING_TIMEOUT_SECONDS": "0",
            },
            {
                "EMBEDDING_PROVIDER": "openai",
                "OPENAI_API_KEY": "key",
                "OPENAI_EMBEDDING_TIMEOUT_SECONDS": "nan",
            },
            {
                "EMBEDDING_PROVIDER": "openai",
                "OPENAI_API_KEY": "key",
                "OPENAI_EMBEDDING_TIMEOUT_SECONDS": "inf",
            },
        ]

        for environ in invalid_environments:
            with self.subTest(environ=environ):
                with self.assertRaises(ValueError):
                    build_text_embedding_provider(environ)

    def test_embedding_metadata_records_provider_model_and_dimension(
        self,
    ) -> None:
        metadata = embedding_metadata({"parser": "markdown"}, dimensions=3)

        self.assertEqual(metadata["parser"], "markdown")
        self.assertEqual(
            metadata["text_embedding_fingerprint"],
            "deterministic:sha256-token-v1:3",
        )


if __name__ == "__main__":
    unittest.main()
