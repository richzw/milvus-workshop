"""Milvus Agent Chat workshop demo package."""

from agent_workshop_demo.config import load_demo_env

load_demo_env()

from agent_workshop_demo.workflow import AgenticRAGWorkflow  # noqa: E402

__all__ = ["AgenticRAGWorkflow"]
