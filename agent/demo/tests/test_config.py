from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent_workshop_demo.config import load_demo_env


class ConfigTests(unittest.TestCase):
    def test_loads_env_values_and_preserves_explicit_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "# demo configuration",
                        "ANSWER_GENERATOR=openai",
                        "OPENAI_API_KEY='file-key'",
                        'OPENAI_MODEL="configured-model"',
                        "MILVUS_TOKEN=",
                    ]
                ),
                encoding="utf-8",
            )
            environ = {"OPENAI_API_KEY": "process-key"}

            loaded_path = load_demo_env(env_path, environ=environ)

        self.assertEqual(loaded_path, env_path)
        self.assertEqual(environ["ANSWER_GENERATOR"], "openai")
        self.assertEqual(environ["OPENAI_API_KEY"], "process-key")
        self.assertEqual(environ["OPENAI_MODEL"], "configured-model")
        self.assertEqual(environ["MILVUS_TOKEN"], "")

    def test_override_replaces_explicit_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text("MODE=file\n", encoding="utf-8")
            environ = {"MODE": "process"}

            load_demo_env(env_path, environ=environ, override=True)

        self.assertEqual(environ["MODE"], "file")

    def test_missing_env_file_is_optional(self) -> None:
        missing = Path("/definitely/missing/demo.env")

        self.assertIsNone(load_demo_env(missing, environ={}))

    def test_invalid_env_entry_fails_with_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text("NOT VALID\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, r":1"):
                load_demo_env(env_path, environ={})


if __name__ == "__main__":
    unittest.main()
