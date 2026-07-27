"""Tests for validated terminology configuration and domain resolution."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_workshop_demo.entities import (
    entity_catalog_from_mapping,
    load_entity_catalog,
)


class EntityCatalogTests(unittest.TestCase):
    def test_go_button_aliases_resolve_to_one_reviewed_entity(self) -> None:
        catalog = load_entity_catalog()

        for surface in ["GO按钮", "跳转按钮", "领取按钮"]:
            with self.subTest(surface=surface):
                result = catalog.resolve(
                    f"产品中的{surface}是什么意思？",
                    query_type="product",
                )
                self.assertEqual(len(result.matched), 1)
                self.assertEqual(
                    result.matched[0]["entity_id"],
                    "ui.go_button",
                )
                self.assertEqual(result.ambiguous, ())

    def test_same_surface_requires_domain_context(self) -> None:
        catalog = load_entity_catalog()

        ambiguous = catalog.resolve("段位是什么意思？", query_type="unknown")
        game = catalog.resolve(
            "游戏中的段位是什么意思？",
            query_type="unknown",
        )

        self.assertEqual(ambiguous.matched, ())
        self.assertEqual(ambiguous.ambiguous[0]["status"], "ambiguous")
        self.assertEqual(game.matched[0]["entity_id"], "game.rank_tier")

    def test_loader_rejects_duplicate_ids_and_unknown_fields(self) -> None:
        duplicate = {
            "catalog_version": "1",
            "entities": [
                {
                    "entity_id": "same",
                    "entity": "A",
                    "aliases": [],
                    "comment": "first",
                    "domains": ["product"],
                },
                {
                    "entity_id": "same",
                    "entity": "B",
                    "aliases": [],
                    "comment": "second",
                    "domains": ["game"],
                },
            ],
        }
        with self.assertRaisesRegex(ValueError, "unique"):
            entity_catalog_from_mapping(duplicate)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "entities.yaml"
            path.write_text(
                json.dumps(
                    {
                        "catalog_version": "1",
                        "entities": [],
                        "unexpected": True,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unknown fields"):
                load_entity_catalog(path)

    def test_prompt_cap_does_not_hide_later_ambiguity(self) -> None:
        entities = [
            {
                "entity_id": f"general.term_{index}",
                "entity": f"术语{index}",
                "aliases": [],
                "comment": f"定义 {index}",
                "domains": ["product"],
            }
            for index in range(20)
        ]
        entities.extend(
            [
                {
                    "entity_id": "game.rank",
                    "entity": "段位",
                    "aliases": [],
                    "comment": "游戏竞技等级",
                    "domains": ["game"],
                },
                {
                    "entity_id": "hr.rank",
                    "entity": "段位",
                    "aliases": [],
                    "comment": "人才职级",
                    "domains": ["hr"],
                },
            ]
        )
        catalog = entity_catalog_from_mapping(
            {"catalog_version": "cap-test", "entities": entities}
        )
        question = " ".join(
            [*(f"术语{index}" for index in range(20)), "段位"]
        )

        resolution = catalog.resolve(question, query_type="unknown")

        self.assertEqual(len(resolution.matched), 20)
        self.assertEqual(len(resolution.ambiguous), 1)
        self.assertEqual(resolution.ambiguous[0]["status"], "ambiguous")


if __name__ == "__main__":
    unittest.main()
