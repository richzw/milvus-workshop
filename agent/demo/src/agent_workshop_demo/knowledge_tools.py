"""Bounded knowledge tools and the demo permission boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

ALL_DEPARTMENTS = ("engineering", "product", "hr", "security", "general")


@dataclass(frozen=True)
class PermissionDecision:
    """Safe, traceable result from the permission tool."""

    allowed: bool
    allowed_departments: tuple[str, ...]
    reason: str
    checker_name: str = "demo-permission-tool"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": "get_user_permission",
            "checker": self.checker_name,
            "allowed": self.allowed,
            "allowed_departments": list(self.allowed_departments),
            "reason": self.reason,
        }


class PermissionChecker(Protocol):
    """Permission seam that must run before private retrieval."""

    def check(
        self,
        *,
        session_id: str,
        intent: str,
        query_type: str,
    ) -> PermissionDecision: ...


class DemoPermissionChecker:
    """Authorize only the repository's synthetic Workshop corpus."""

    def check(
        self,
        *,
        session_id: str,
        intent: str,
        query_type: str,
    ) -> PermissionDecision:
        del session_id, intent, query_type
        return PermissionDecision(
            allowed=True,
            allowed_departments=ALL_DEPARTMENTS,
            reason="Synthetic Workshop corpus access is allowed.",
        )


@dataclass(frozen=True)
class KnowledgeSearchTool:
    """One registered search tool with policy-owned metadata filters."""

    name: str
    description: str
    filters: Mapping[str, str | bool | tuple[str, ...]]
    query_hint: str

    def build_filters(
        self,
        *,
        base_filters: Mapping[str, Any],
        allowed_departments: tuple[str, ...],
    ) -> dict[str, Any]:
        output = dict(base_filters)
        for field, configured in self.filters.items():
            output[field] = (
                list(configured)
                if isinstance(configured, tuple)
                else configured
            )

        configured_departments = output.get(
            "department",
            list(allowed_departments),
        )
        requested = (
            [configured_departments]
            if isinstance(configured_departments, str)
            else list(configured_departments)
        )
        allowed = set(allowed_departments)
        output["department"] = [
            department for department in requested if department in allowed
        ]
        if not output["department"]:
            raise PermissionError(
                f"Tool {self.name!r} has no permitted department scope"
            )
        return output


SEARCH_TOOLS: dict[str, KnowledgeSearchTool] = {
    "search_policy_docs": KnowledgeSearchTool(
        name="search_policy_docs",
        description="Search HR and security policy documents.",
        filters={
            "department": ("hr", "security"),
            "doc_type": ("markdown", "pdf", "text"),
        },
        query_hint="policy rules standards eligibility limits",
    ),
    "search_product_docs": KnowledgeSearchTool(
        name="search_product_docs",
        description="Search product plans, roadmaps, and UI documents.",
        filters={
            "department": ("product",),
            "doc_type": ("markdown", "pdf", "text"),
        },
        query_hint="product roadmap features delivery coverage",
    ),
    "search_meeting_notes": KnowledgeSearchTool(
        name="search_meeting_notes",
        description="Search customer and internal meeting notes.",
        filters={
            "department": ("product",),
            "doc_type": ("markdown", "text"),
        },
        query_hint="meeting notes customer concerns feedback requests",
    ),
    "search_code_docs": KnowledgeSearchTool(
        name="search_code_docs",
        description="Search engineering, architecture, and code documents.",
        filters={
            "department": ("engineering",),
            "doc_type": ("markdown", "pdf", "text", "image"),
        },
        query_hint="engineering architecture implementation code",
    ),
}

REGISTERED_TOOL_NAMES = frozenset(
    {
        "get_user_permission",
        *SEARCH_TOOLS,
        "summarize_document",
    }
)


def summarize_document(text: str, *, max_chars: int = 600) -> str:
    """Return a bounded deterministic summary of authorized retrieved text."""

    normalized = " ".join(text.split())
    if not normalized:
        raise ValueError("summarize_document requires non-empty text")
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")
    suffix = "..." if len(normalized) > max_chars else ""
    return normalized[:max_chars] + suffix
