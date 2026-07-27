"""Validated predefined-entity catalog and deterministic terminology resolver."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from agent_workshop_demo.config import DEMO_ROOT

DEFAULT_ENTITY_CATALOG_PATH: Final = (
    DEMO_ROOT / "config" / "predefined_entities.yaml"
)
MAX_CATALOG_ENTITIES: Final = 500
MAX_PROMPT_ENTITIES: Final = 20
MAX_ID_CHARS: Final = 128
MAX_TERM_CHARS: Final = 128
MAX_COMMENT_CHARS: Final = 512
DOMAIN_HINTS: Final[dict[str, frozenset[str]]] = {
    "product": frozenset({"product", "产品", "ui", "按钮", "界面"}),
    "game": frozenset({"game", "游戏", "玩家", "竞技"}),
    "hr": frozenset({"hr", "人力", "人才", "职级", "员工"}),
    "engineering": frozenset({"engineering", "工程", "架构", "代码"}),
    "policy": frozenset({"policy", "制度", "规则"}),
}
QUERY_TYPE_DOMAINS: Final[dict[str, frozenset[str]]] = {
    "product": frozenset({"product"}),
    "policy": frozenset({"hr", "policy"}),
    "architecture": frozenset({"engineering"}),
}


@dataclass(frozen=True)
class PredefinedEntity:
    """One reviewed domain-term definition."""

    entity_id: str
    entity: str
    aliases: tuple[str, ...]
    comment: str
    domains: tuple[str, ...]

    @property
    def surfaces(self) -> tuple[str, ...]:
        """Return canonical and alias surfaces without duplicates."""

        return tuple(dict.fromkeys((self.entity, *self.aliases)))

    def to_prompt_dict(self, *, matched_surface: str) -> dict[str, Any]:
        """Serialize the bounded prompt/trace representation."""

        return {
            "entity_id": self.entity_id,
            "matched_surface": matched_surface,
            "entity": self.entity,
            "comment": self.comment,
            "domains": list(self.domains),
            "status": "resolved",
        }


@dataclass(frozen=True)
class EntityResolution:
    """Resolved and ambiguous terminology for one question."""

    matched: tuple[dict[str, Any], ...]
    ambiguous: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class EntityCatalog:
    """Immutable, validated predefined-entity catalog."""

    catalog_version: str
    entities: tuple[PredefinedEntity, ...]

    def resolve(self, question: str, *, query_type: str) -> EntityResolution:
        """Resolve matched surfaces using explicit domain context."""

        normalized_question = question.casefold()
        grouped: dict[str, list[PredefinedEntity]] = {}
        surface_display: dict[str, str] = {}
        for entity in self.entities:
            for surface in entity.surfaces:
                normalized_surface = surface.casefold()
                if normalized_surface in normalized_question:
                    grouped.setdefault(normalized_surface, []).append(entity)
                    surface_display.setdefault(normalized_surface, surface)

        matched: list[dict[str, Any]] = []
        ambiguous: list[dict[str, Any]] = []
        question_domains = _question_domains(normalized_question, query_type)
        seen_ids: set[str] = set()
        for normalized_surface in sorted(
            grouped,
            key=lambda value: (-len(value), value),
        ):
            candidates = [
                item
                for item in grouped[normalized_surface]
                if item.entity_id not in seen_ids
            ]
            if not candidates:
                continue
            selected = _select_candidate(candidates, question_domains)
            if selected is None:
                ambiguous.append(
                    {
                        "matched_surface": surface_display[normalized_surface],
                        "candidate_entity_ids": [
                            item.entity_id for item in candidates
                        ],
                        "domains": sorted(
                            {
                                domain
                                for item in candidates
                                for domain in item.domains
                            }
                        ),
                        "status": "ambiguous",
                    }
                )
                seen_ids.update(item.entity_id for item in candidates)
                continue
            if len(matched) < MAX_PROMPT_ENTITIES:
                matched.append(
                    selected.to_prompt_dict(
                        matched_surface=surface_display[normalized_surface]
                    )
                )
            seen_ids.add(selected.entity_id)
        return EntityResolution(tuple(matched), tuple(ambiguous))


def load_entity_catalog(
    path: Path = DEFAULT_ENTITY_CATALOG_PATH,
) -> EntityCatalog:
    """Load strict JSON-compatible YAML with contextual errors."""

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Unable to read entity catalog {path}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON-compatible YAML entity catalog {path}: "
            f"line {exc.lineno}, column {exc.colno}"
        ) from exc
    try:
        return entity_catalog_from_mapping(payload)
    except ValueError as exc:
        raise ValueError(f"Invalid entity catalog {path}: {exc}") from exc


def entity_catalog_from_mapping(payload: object) -> EntityCatalog:
    """Validate an injected mapping into an immutable catalog."""

    if not isinstance(payload, Mapping):
        raise ValueError("catalog root must be an object")
    _reject_unknown_fields(payload, {"catalog_version", "entities"}, "catalog")
    catalog_version = _bounded_string(
        payload.get("catalog_version"),
        field_name="catalog_version",
        max_chars=MAX_ID_CHARS,
    )
    raw_entities = payload.get("entities")
    if not isinstance(raw_entities, list):
        raise ValueError("entities must be a list")
    if len(raw_entities) > MAX_CATALOG_ENTITIES:
        raise ValueError(
            f"entities must contain at most {MAX_CATALOG_ENTITIES} items"
        )
    entities = tuple(
        _parse_entity(item, index=index)
        for index, item in enumerate(raw_entities)
    )
    ids = [item.entity_id for item in entities]
    if len(ids) != len(set(ids)):
        raise ValueError("entity_id values must be unique")
    return EntityCatalog(catalog_version, entities)


def _parse_entity(value: object, *, index: int) -> PredefinedEntity:
    if not isinstance(value, Mapping):
        raise ValueError(f"entities[{index}] must be an object")
    _reject_unknown_fields(
        value,
        {"entity_id", "entity", "aliases", "comment", "domains"},
        f"entities[{index}]",
    )
    aliases = _string_list(
        value.get("aliases"),
        field_name=f"entities[{index}].aliases",
        max_chars=MAX_TERM_CHARS,
    )
    domains = _string_list(
        value.get("domains"),
        field_name=f"entities[{index}].domains",
        max_chars=MAX_TERM_CHARS,
    )
    if not domains:
        raise ValueError(f"entities[{index}].domains must not be empty")
    return PredefinedEntity(
        entity_id=_bounded_string(
            value.get("entity_id"),
            field_name=f"entities[{index}].entity_id",
            max_chars=MAX_ID_CHARS,
        ),
        entity=_bounded_string(
            value.get("entity"),
            field_name=f"entities[{index}].entity",
            max_chars=MAX_TERM_CHARS,
        ),
        aliases=aliases,
        comment=_bounded_string(
            value.get("comment"),
            field_name=f"entities[{index}].comment",
            max_chars=MAX_COMMENT_CHARS,
        ),
        domains=domains,
    )


def _question_domains(question: str, query_type: str) -> frozenset[str]:
    explicit = {
        domain
        for domain, hints in DOMAIN_HINTS.items()
        if any(hint in question for hint in hints)
    }
    if explicit:
        return frozenset(explicit)
    return QUERY_TYPE_DOMAINS.get(query_type, frozenset())


def _select_candidate(
    candidates: list[PredefinedEntity],
    question_domains: frozenset[str],
) -> PredefinedEntity | None:
    if len(candidates) == 1:
        return candidates[0]
    narrowed = [
        item
        for item in candidates
        if question_domains.intersection(item.domains)
    ]
    return narrowed[0] if len(narrowed) == 1 else None


def _reject_unknown_fields(
    value: Mapping[object, object],
    allowed: set[str],
    location: str,
) -> None:
    unknown = sorted(
        str(key) for key in value if not isinstance(key, str) or key not in allowed
    )
    if unknown:
        raise ValueError(f"{location} contains unknown fields: {unknown}")


def _bounded_string(
    value: object,
    *,
    field_name: str,
    max_chars: int,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > max_chars:
        raise ValueError(f"{field_name} must be at most {max_chars} characters")
    return normalized


def _string_list(
    value: object,
    *,
    field_name: str,
    max_chars: int,
) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise ValueError(f"{field_name} must be a string list")
    output = tuple(
        _bounded_string(
            item,
            field_name=f"{field_name}[{index}]",
            max_chars=max_chars,
        )
        for index, item in enumerate(value)
    )
    if len(output) != len(set(item.casefold() for item in output)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return output
