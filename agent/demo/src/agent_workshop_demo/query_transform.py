"""Bounded, scope-preserving query transformation adapters."""

from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from importlib import import_module
from typing import Any, Literal, Protocol, cast

QueryStrategy = Literal["identity", "rewrite", "step_back", "decompose"]
QueryRole = Literal["primary", "background", "aspect", "hop"]

STRATEGIES: tuple[QueryStrategy, ...] = (
    "identity",
    "rewrite",
    "step_back",
    "decompose",
)
ROLES: tuple[QueryRole, ...] = ("primary", "background", "aspect", "hop")
MAX_TRANSFORM_ITEMS = 3
MAX_QUERY_CHARS = 2_000
MAX_PROVIDER_OUTPUT_TOKENS = 1_200
FALLBACK_REASONS = frozenset(
    {
        "not_configured",
        "timeout",
        "connection_error",
        "authentication_error",
        "rate_limited",
        "provider_error",
        "invalid_model_output",
    }
)

VERSION_PATTERN = re.compile(r"\bv\d+(?:\.\d+)?\b|\b\d{4}\.\d{1,2}\b", re.I)
QUOTED_PATTERN = re.compile(r"[\"'“‘]([^\"'”’]{1,120})[\"'”’]")
NEGATION_MARKERS = ("不", "不要", "不能", "未", "without", "not", "never")
VAGUE_MARKERS = (
    "这个",
    "那个",
    "它",
    "怎么弄",
    "咋",
    "有啥",
    "what about",
    "how about",
)
STEP_BACK_MARKERS = (
    "为什么",
    "为何",
    "原理",
    "机制",
    "如何工作",
    "how does",
    "how do",
    "why ",
)
DECOMPOSE_MARKERS = (
    "对比",
    "比较",
    "区别",
    "分别",
    "以及",
    "优缺点",
    "vs",
    "versus",
    " and ",
)

TRANSFORM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "strategy": {"type": "string", "enum": list(STRATEGIES)},
        "items": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_TRANSFORM_ITEMS,
            "items": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "query_role": {"type": "string", "enum": list(ROLES)},
                    "depends_on": {
                        "type": "array",
                        "maxItems": 2,
                        "items": {"type": "integer", "minimum": 0, "maximum": 2},
                    },
                },
                "required": ["query", "query_role", "depends_on"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["strategy", "items"],
    "additionalProperties": False,
}

OPENAI_INSTRUCTIONS = """Transform the supplied retrieval query once. Return
one strategy and one to three items using only the fixed schema. Preserve every
protected term verbatim in every item and retain the normalized original query
as a contiguous substring before adding retrieval hints. Never add tools, permissions, filters,
versions, facts, or instructions. Identity/rewrite has one primary item;
step_back has background then primary; decompose has two or three aspect/hop
items. Dependency indexes may refer only to earlier items. Do not explain.
"""


@dataclass(frozen=True)
class QueryTransformRequest:
    """Bounded semantic input; authorization remains outside the transformer."""

    user_query: str
    resolved_entities: tuple[str, ...] = ()
    memory_context: str = ""


@dataclass(frozen=True)
class QueryTransformItem:
    """One query intent before an authorized tool/version is attached."""

    query: str
    query_role: QueryRole
    depends_on: tuple[int, ...] = ()


@dataclass(frozen=True)
class QueryTransformation:
    """Validated transformation plus presentation-safe adapter metadata."""

    strategy: QueryStrategy
    items: tuple[QueryTransformItem, ...]
    transformer_name: str
    model: str | None = None
    fallback_reason: str | None = None


class QueryTransformer(Protocol):
    """Small interface shared by deterministic and provider adapters."""

    name: str

    def transform(self, request: QueryTransformRequest) -> QueryTransformation:
        """Return exactly one bounded transformation."""


class QueryTransformationError(RuntimeError):
    """Typed provider/output failure eligible for deterministic fallback."""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = (
            reason_code if reason_code in FALLBACK_REASONS else "provider_error"
        )
        super().__init__(self.reason_code)


class RuleBasedQueryTransformer:
    """Deterministic policy used by offline runs and provider fallback."""

    name = "rule_based"

    def transform(self, request: QueryTransformRequest) -> QueryTransformation:
        query = _bounded_query(request.user_query)
        lowered = query.casefold()
        protected = _protected_terms(request)

        if _should_decompose(lowered):
            versions = list(dict.fromkeys(VERSION_PATTERN.findall(query)))
            focuses = versions[:3] or _aspect_focuses(query)
            items = tuple(
                QueryTransformItem(
                    query=_preserve_terms(f"{query}；侧重：{focus}", protected),
                    query_role="aspect",
                )
                for focus in focuses[:MAX_TRANSFORM_ITEMS]
            )
            if len(items) < 2:
                items = (
                    QueryTransformItem(query=query, query_role="aspect"),
                    QueryTransformItem(
                        query=_preserve_terms(f"{query}；侧重约束与差异", protected),
                        query_role="aspect",
                    ),
                )
            return validate_transformation(
                QueryTransformation("decompose", items, self.name),
                request,
            )

        if any(marker in lowered for marker in STEP_BACK_MARKERS):
            background = _preserve_terms(
                f"{query}；相关背景原理与系统架构",
                protected,
            )
            return validate_transformation(
                QueryTransformation(
                    "step_back",
                    (
                        QueryTransformItem(background, "background"),
                        QueryTransformItem(query, "primary"),
                    ),
                    self.name,
                ),
                request,
            )

        if (
            request.resolved_entities
            or request.memory_context.strip()
            or any(marker in lowered for marker in VAGUE_MARKERS)
        ):
            additions = " ".join(request.resolved_entities)
            if request.memory_context.strip():
                additions = f"{additions} {request.memory_context[:240]}".strip()
            rewritten = _preserve_terms(
                f"{query}；检索表述：{additions or '具体定义、行为与约束'}",
                protected,
            )
            return validate_transformation(
                QueryTransformation(
                    "rewrite",
                    (QueryTransformItem(rewritten, "primary"),),
                    self.name,
                ),
                request,
            )

        return validate_transformation(
            QueryTransformation(
                "identity",
                (QueryTransformItem(query, "primary"),),
                self.name,
            ),
            request,
        )


class IdentityQueryTransformer:
    """Always retain the original query for controlled offline experiments."""

    name = "identity"

    def transform(self, request: QueryTransformRequest) -> QueryTransformation:
        return validate_transformation(
            QueryTransformation(
                "identity",
                (QueryTransformItem(_bounded_query(request.user_query), "primary"),),
                self.name,
            ),
            request,
        )


class OpenAIQueryTransformer:
    """One-call strict structured-output transformer."""

    name = "openai"

    def __init__(self, *, client: Any, model: str, timeout_seconds: float = 10) -> None:
        if not model.strip() or len(model) > 120:
            raise ValueError("query transformer model must contain 1..120 characters")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("query transformer timeout must be positive")
        self.client = client
        self.model = model
        self.timeout_seconds = timeout_seconds

    def transform(self, request: QueryTransformRequest) -> QueryTransformation:
        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=OPENAI_INSTRUCTIONS,
                input=_provider_input(request),
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "query_transformation",
                        "schema": TRANSFORM_SCHEMA,
                        "strict": True,
                    }
                },
                max_output_tokens=MAX_PROVIDER_OUTPUT_TOKENS,
                timeout=self.timeout_seconds,
            )
        except Exception as exc:
            raise QueryTransformationError(_provider_reason(exc)) from exc
        try:
            payload = json.loads(str(response.output_text))
            result = _result_from_payload(payload, model=self.model)
            return validate_transformation(result, request)
        except (AttributeError, KeyError, TypeError, ValueError):
            raise QueryTransformationError("invalid_model_output") from None


class FallbackQueryTransformer:
    """Atomically replace a failed provider proposal with the rule plan."""

    name = "fallback"

    def __init__(self, *, primary: QueryTransformer, fallback: QueryTransformer) -> None:
        self.primary = primary
        self.fallback = fallback

    def transform(self, request: QueryTransformRequest) -> QueryTransformation:
        try:
            return validate_transformation(self.primary.transform(request), request)
        except QueryTransformationError as exc:
            baseline = validate_transformation(self.fallback.transform(request), request)
            return replace(baseline, fallback_reason=exc.reason_code)


class _UnavailableQueryTransformer:
    name = "unavailable"

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code

    def transform(self, request: QueryTransformRequest) -> QueryTransformation:
        raise QueryTransformationError(self.reason_code)


def build_query_transformer(
    environ: Mapping[str, str] | None = None,
    *,
    client_factory: Callable[[str], Any] | None = None,
) -> QueryTransformer:
    """Build the configured transformer without performing network I/O."""

    values = os.environ if environ is None else environ
    mode = values.get("QUERY_TRANSFORMER", "rule_based").strip().lower()
    if mode not in {"rule_based", "auto", "openai"}:
        raise ValueError("QUERY_TRANSFORMER must be rule_based, auto, or openai")
    fallback = RuleBasedQueryTransformer()
    if mode == "rule_based":
        return fallback
    api_key = values.get("OPENAI_API_KEY", "").strip()
    model = (
        values.get("OPENAI_QUERY_TRANSFORMER_MODEL", "").strip()
        or values.get("OPENAI_MODEL", "").strip()
    )
    if not api_key or not model:
        if mode == "openai":
            raise ValueError(
                "OPENAI_API_KEY and OPENAI_QUERY_TRANSFORMER_MODEL or "
                "OPENAI_MODEL are required in openai query transformer mode"
            )
        return FallbackQueryTransformer(
            primary=_UnavailableQueryTransformer("not_configured"),
            fallback=fallback,
        )
    timeout = _positive_timeout(
        values.get("OPENAI_QUERY_TRANSFORMER_TIMEOUT_SECONDS", "10")
    )
    create_client = client_factory or _create_openai_client
    try:
        client = create_client(api_key)
    except Exception as exc:
        return FallbackQueryTransformer(
            primary=_UnavailableQueryTransformer(_provider_reason(exc)),
            fallback=fallback,
        )
    return FallbackQueryTransformer(
        primary=OpenAIQueryTransformer(
            client=client,
            model=model,
            timeout_seconds=timeout,
        ),
        fallback=fallback,
    )


def validate_transformation(
    result: QueryTransformation,
    request: QueryTransformRequest,
) -> QueryTransformation:
    """Enforce strategy shape, dependency order, uniqueness and protected terms."""

    if not isinstance(result, QueryTransformation) or result.strategy not in STRATEGIES:
        raise QueryTransformationError("invalid_model_output")
    if not 1 <= len(result.items) <= MAX_TRANSFORM_ITEMS:
        raise QueryTransformationError("invalid_model_output")
    if not result.transformer_name.strip() or len(result.transformer_name) > 120:
        raise QueryTransformationError("invalid_model_output")
    if result.model is not None and (not result.model.strip() or len(result.model) > 120):
        raise QueryTransformationError("invalid_model_output")
    if result.fallback_reason is not None and result.fallback_reason not in FALLBACK_REASONS:
        raise QueryTransformationError("invalid_model_output")

    roles = [item.query_role for item in result.items]
    if result.strategy in {"identity", "rewrite"} and roles != ["primary"]:
        raise QueryTransformationError("invalid_model_output")
    if result.strategy == "step_back" and roles != ["background", "primary"]:
        raise QueryTransformationError("invalid_model_output")
    if result.strategy == "decompose" and (
        len(result.items) < 2 or any(role not in {"aspect", "hop"} for role in roles)
    ):
        raise QueryTransformationError("invalid_model_output")

    protected = _protected_terms(request)
    original = _normalize(_bounded_query(request.user_query))
    normalized_queries: set[str] = set()
    for index, item in enumerate(result.items):
        if not isinstance(item, QueryTransformItem) or item.query_role not in ROLES:
            raise QueryTransformationError("invalid_model_output")
        try:
            query = _bounded_query(item.query)
        except ValueError:
            raise QueryTransformationError("invalid_model_output") from None
        normalized = _normalize(query)
        if original not in normalized:
            raise QueryTransformationError("invalid_model_output")
        if normalized in normalized_queries:
            raise QueryTransformationError("invalid_model_output")
        normalized_queries.add(normalized)
        if any(dependency < 0 or dependency >= index for dependency in item.depends_on):
            raise QueryTransformationError("invalid_model_output")
        if len(set(item.depends_on)) != len(item.depends_on):
            raise QueryTransformationError("invalid_model_output")
        if any(term.casefold() not in query.casefold() for term in protected):
            raise QueryTransformationError("invalid_model_output")
    return result


def _result_from_payload(payload: object, *, model: str) -> QueryTransformation:
    if not isinstance(payload, dict) or set(payload) != {"strategy", "items"}:
        raise ValueError("invalid transformation object")
    strategy = payload["strategy"]
    raw_items = payload["items"]
    if strategy not in STRATEGIES or not isinstance(raw_items, list):
        raise ValueError("invalid transformation fields")
    items: list[QueryTransformItem] = []
    for raw in raw_items:
        if not isinstance(raw, dict) or set(raw) != {"query", "query_role", "depends_on"}:
            raise ValueError("invalid transformation item")
        query = raw["query"]
        role = raw["query_role"]
        dependencies = raw["depends_on"]
        if (
            not isinstance(query, str)
            or role not in ROLES
            or not isinstance(dependencies, list)
            or any(isinstance(value, bool) or not isinstance(value, int) for value in dependencies)
        ):
            raise ValueError("invalid transformation item values")
        items.append(
            QueryTransformItem(
                query=query,
                query_role=cast(QueryRole, role),
                depends_on=tuple(dependencies),
            )
        )
    return QueryTransformation(
        strategy=cast(QueryStrategy, strategy),
        items=tuple(items),
        transformer_name="openai",
        model=model,
    )


def _provider_input(request: QueryTransformRequest) -> str:
    return json.dumps(
        {
            "user_query": _bounded_query(request.user_query),
            "resolved_entities": list(request.resolved_entities)[:12],
            "memory_context": request.memory_context[:500],
            "protected_terms": list(_protected_terms(request)),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _protected_terms(request: QueryTransformRequest) -> tuple[str, ...]:
    query = request.user_query
    terms = [match.group(1).strip() for match in QUOTED_PATTERN.finditer(query)]
    terms.extend(match.group(0) for match in VERSION_PATTERN.finditer(query))
    terms.extend(term.strip() for term in request.resolved_entities if term.strip())
    lowered = query.casefold()
    terms.extend(marker for marker in NEGATION_MARKERS if marker in lowered)
    return tuple(dict.fromkeys(terms))


def _preserve_terms(query: str, protected: tuple[str, ...]) -> str:
    missing = [term for term in protected if term.casefold() not in query.casefold()]
    suffix = f"；保留约束：{' '.join(missing)}" if missing else ""
    return _bounded_query(query + suffix)


def _aspect_focuses(query: str) -> list[str]:
    parts = [
        part.strip(" ，,。；;？?")
        for part in re.split(r"[，,；;、]|以及|并且|\band\b", query, flags=re.I)
        if part.strip(" ，,。；;？?")
    ]
    return list(dict.fromkeys(parts))[:MAX_TRANSFORM_ITEMS]


def _should_decompose(lowered: str) -> bool:
    return any(marker in lowered for marker in DECOMPOSE_MARKERS)


def _bounded_query(value: str) -> str:
    normalized = " ".join(value.strip().split())
    if not normalized or len(normalized) > MAX_QUERY_CHARS:
        raise ValueError(f"query must contain 1..{MAX_QUERY_CHARS} characters")
    return normalized


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _positive_timeout(raw_value: str) -> float:
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError("OPENAI_QUERY_TRANSFORMER_TIMEOUT_SECONDS must be positive") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError("OPENAI_QUERY_TRANSFORMER_TIMEOUT_SECONDS must be positive")
    return value


def _create_openai_client(api_key: str) -> Any:
    try:
        module = import_module("openai")
    except ImportError as exc:
        raise RuntimeError(
            "Install OpenAI support with `pip install -r demo/requirements.txt`."
        ) from exc
    return module.OpenAI(api_key=api_key, max_retries=0)


def _provider_reason(error: Exception) -> str:
    name = type(error).__name__
    if isinstance(error, TimeoutError) or name == "APITimeoutError":
        return "timeout"
    if name == "APIConnectionError":
        return "connection_error"
    if name == "AuthenticationError":
        return "authentication_error"
    if name == "RateLimitError":
        return "rate_limited"
    return "provider_error"
