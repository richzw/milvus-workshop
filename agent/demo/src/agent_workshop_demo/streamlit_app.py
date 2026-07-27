"""Optional Streamlit UI for the query-only Agent Chat MVP."""

# ruff: noqa: E402

from __future__ import annotations

import os
import sys
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from html import escape
from importlib import import_module
from math import isfinite
from pathlib import Path
from typing import Any

# Streamlit executes this file as a standalone script, so make the adjacent
# src-layout package importable even when PYTHONPATH is not set in this shell.
SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from agent_workshop_demo.embedding import text_embedding_fingerprint
from agent_workshop_demo.events import details_are_safe
from agent_workshop_demo.langgraph_workflow import build_milvus_workflow
from agent_workshop_demo.workflow import WorkflowStageError

TRACE_EVENT_FIELDS = {
    "query_id",
    "sequence",
    "kind",
    "stage",
    "title",
    "summary",
    "status",
    "elapsed_ms",
    "details",
}
MARKDOWN_ENTITY_ESCAPES = str.maketrans(
    {
        "\\": "&#92;",
        "`": "&#96;",
        "*": "&#42;",
        "_": "&#95;",
        "{": "&#123;",
        "}": "&#125;",
        "[": "&#91;",
        "]": "&#93;",
        "(": "&#40;",
        ")": "&#41;",
        "#": "&#35;",
        "+": "&#43;",
        "-": "&#45;",
        ".": "&#46;",
        "!": "&#33;",
        "|": "&#124;",
    }
)


def presentation_text(value: str) -> str:
    """Encode event text so neither HTML nor Markdown can interpret it."""

    return escape(value, quote=True).translate(MARKDOWN_ENTITY_ESCAPES)


def append_trace_event(
    events: list[dict[str, Any]],
    event: dict[str, Any],
    *,
    query_id: str,
) -> bool:
    """Append only an in-order event for the active query."""

    if set(event) != TRACE_EVENT_FIELDS:
        return False
    if event.get("query_id") != query_id:
        return False
    sequence = event.get("sequence")
    if (
        not isinstance(sequence, int)
        or isinstance(sequence, bool)
        or sequence != len(events) + 1
    ):
        return False
    if event.get("kind") not in {
        "stage_completed",
        "tool_completed",
        "retry_scheduled",
    }:
        return False
    if event.get("status") not in {"completed", "warning"}:
        return False
    title = event.get("title")
    if not isinstance(title, str) or not 0 < len(title) <= 80:
        return False
    summary = event.get("summary")
    if not isinstance(summary, str) or not 0 < len(summary) <= 300:
        return False
    if not isinstance(event.get("stage"), str) or not event["stage"]:
        return False
    elapsed = event.get("elapsed_ms")
    if elapsed is not None and (
        not isinstance(elapsed, (int, float))
        or isinstance(elapsed, bool)
        or elapsed < 0
        or not isfinite(float(elapsed))
    ):
        return False
    if not details_are_safe(event.get("details")):
        return False
    events.append(dict(event))
    return True


def render_trace_timeline(events: list[dict[str, Any]]) -> str:
    """Render escaped HTML without interpreting event text as Markdown."""

    lines: list[str] = []
    for event in events:
        icon = "⚠️" if event["status"] == "warning" else "✓"
        elapsed = event.get("elapsed_ms")
        timing = (
            f" · {float(elapsed):.0f} ms"
            if isinstance(elapsed, (int, float))
            else ""
        )
        lines.append(
            f"{icon} <strong>{presentation_text(event['title'])}</strong>"
            f"{timing}<br><small>"
            f"{presentation_text(event['summary'])}</small>"
        )
    return "<br><br>".join(lines)


def safe_query_error(exc: Exception) -> str:
    """Return an actionable error without exposing dependency internals."""

    if isinstance(exc, WorkflowStageError):
        return (
            f"Agent stopped during `{exc.stage}`. "
            f"Query id: `{exc.query_id}`."
        )
    if isinstance(exc, ValueError):
        return "The query or its filters did not pass validation."
    return "Agent execution failed before a terminal response was produced."


@dataclass
class StreamConsumer:
    """Validate the trace* → answer* → exactly-one-final UI protocol."""

    query_id: str
    trace_events: list[dict[str, Any]] = field(default_factory=list)
    answer_parts: list[str] = field(default_factory=list)
    final_response: dict[str, Any] | None = None

    def consume(self, envelope: dict[str, Any]) -> str | None:
        """Accept one envelope and return answer text when present."""

        event_type = envelope.get("type")
        if self.final_response is not None:
            raise RuntimeError("Workflow emitted data after final")
        if event_type == "trace_event":
            if self.answer_parts:
                raise RuntimeError("Workflow emitted trace after answer")
            event = envelope.get("event")
            if not isinstance(event, dict) or not append_trace_event(
                self.trace_events,
                event,
                query_id=self.query_id,
            ):
                raise RuntimeError("Workflow emitted an invalid trace event")
            return None
        if event_type == "answer_delta":
            text = envelope.get("text")
            if not isinstance(text, str) or not text:
                raise RuntimeError("Workflow emitted an invalid answer delta")
            self.answer_parts.append(text)
            return text
        if event_type == "final":
            response = envelope.get("response")
            if (
                not isinstance(response, dict)
                or response.get("query_id") != self.query_id
                or not self.answer_parts
                or response.get("answer") != "".join(self.answer_parts)
            ):
                raise RuntimeError("Workflow emitted an invalid final response")
            self.final_response = response
            return None
        raise RuntimeError("Workflow emitted an unknown event type")

    def finish(self) -> dict[str, Any]:
        """Return the final response or reject an incomplete stream."""

        if self.final_response is None:
            raise RuntimeError(
                "Workflow stream ended without a terminal response"
            )
        return self.final_response


def main() -> None:
    """Render the three-tab teaching UI."""

    try:
        st = import_module("streamlit")
    except ImportError as exc:
        raise RuntimeError(
            "Install UI dependencies with `pip install -r "
            "demo/requirements.txt`."
        ) from exc

    st.set_page_config(page_title="Milvus Agent Chat Demo", layout="wide")
    st.title("Milvus 3.0 Agent Chat")
    st.caption(
        "Entity-aware terminology, document-version-scoped retrieval, "
        "evidence grading, grounded generation, and citation self-check."
    )

    def cached_workflow(
        embedding_fingerprint: str,
        answer_mode: str,
        answer_model: str,
        milvus_uri: str,
        milvus_collection_name: str,
        milvus_memory_collection_name: str,
        memory_top_k: str,
        memory_ttl_seconds: str,
    ) -> Any:
        """Reuse expensive configured resources across Streamlit reruns."""

        return build_milvus_workflow()

    cached_workflow = st.cache_resource(cached_workflow)
    workflow = cached_workflow(
        text_embedding_fingerprint(),
        os.getenv("ANSWER_GENERATOR", "auto"),
        os.getenv("OPENAI_MODEL", ""),
        os.getenv("MILVUS_URI", "http://localhost:19530"),
        os.getenv("MILVUS_COLLECTION_NAME", "kb_chunks"),
        os.getenv(
            "MILVUS_MEMORY_COLLECTION_NAME",
            "conversation_memory",
        ),
        os.getenv("MEMORY_TOP_K", "3"),
        os.getenv("MEMORY_TTL_SECONDS", "86400"),
    )
    defaults: dict[str, Any] = {
        "last_response": None,
        "last_error": None,
        "session_id": f"session_{uuid.uuid4().hex}",
        "query_events": {},
        "messages": [],
        "memory_records": [],
        "memory_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    clear_clicked = st.button(
        "Clear conversation & memory",
        type="secondary",
    )
    if clear_clicked:
        try:
            workflow.clear_memory(st.session_state["session_id"])
        except (ValueError, RuntimeError) as exc:
            st.error(safe_query_error(exc))
        else:
            st.session_state["last_response"] = None
            st.session_state["last_error"] = None
            st.session_state["query_events"] = {}
            st.session_state["messages"] = []
            st.session_state["memory_records"] = []
            st.session_state["memory_error"] = None
            st.success("Current conversation and session Memory cleared.")

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input(
        "Ask a question; the Agent will select knowledge tools"
    )
    if question:
        query_id = f"query_{uuid.uuid4().hex}"
        st.session_state["messages"].append(
            {
                "role": "user",
                "content": question,
                "query_id": query_id,
            }
        )
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            consumer = StreamConsumer(query_id)
            live_events = consumer.trace_events
            agent_status = st.status(
                "Agent 正在理解问题…",
                expanded=True,
            )
            timeline = agent_status.empty()
            answer_slot = st.empty()
            try:
                def answer_stream() -> Iterator[str]:
                    for envelope in workflow.stream(
                        question,
                        session_id=st.session_state["session_id"],
                        query_id=query_id,
                    ):
                        answer_text = consumer.consume(envelope)
                        if envelope.get("type") == "trace_event":
                            timeline.markdown(
                                render_trace_timeline(live_events),
                                unsafe_allow_html=True,
                            )
                            agent_status.update(
                                label=(
                                    "Agent 正在执行 · "
                                    f"{len(live_events)} steps"
                                ),
                                state="running",
                                expanded=True,
                            )
                        if answer_text is not None:
                            yield answer_text

                answer_slot.write_stream(answer_stream())
                response = consumer.finish()
            except (ValueError, RuntimeError) as exc:
                timeline.empty()
                answer_slot.empty()
                live_events.clear()
                agent_status.update(
                    label="Agent 执行失败",
                    state="error",
                    expanded=True,
                )
                error_message = safe_query_error(exc)
                st.session_state["last_response"] = None
                st.session_state["last_error"] = error_message
                st.error(error_message)
            else:
                total_latency = response["metrics"]["latency_ms"]
                agent_status.update(
                    label=(
                        f"Agent 已完成 · {len(live_events)} steps · "
                        f"{total_latency:.0f} ms"
                    ),
                    state="complete",
                    expanded=False,
                )
                st.session_state["last_response"] = response
                st.session_state["last_error"] = None
                st.session_state["query_events"][response["query_id"]] = (
                    live_events
                )
                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": response["answer"],
                        "query_id": response["query_id"],
                    }
                )
                try:
                    st.session_state["memory_records"] = (
                        workflow.list_memories(
                            st.session_state["session_id"],
                        )
                    )
                except (ValueError, RuntimeError):
                    st.session_state["memory_error"] = (
                        "Session Memory records are temporarily unavailable."
                    )
                else:
                    st.session_state["memory_error"] = None

    response = st.session_state["last_response"]
    if st.session_state["last_error"] and not response:
        st.error(st.session_state["last_error"])
        return
    if not response:
        st.info("Try: 我们 S3 文档同步流程是怎么设计的？")
        return

    tabs = st.tabs(["Chat", "Evidence", "Agent Trace", "Memory"])
    with tabs[0]:
        st.subheader("Answer")
        if response["terminal_status"] == "abstained":
            st.warning(
                "Insufficient evidence after retries; no grounded answer."
            )
        if response["terminal_status"] == "clarification_required":
            st.warning(
                "Please clarify the industry meaning or requested versions."
            )
        generation = response["trace"]["answer_generation"]
        if generation["fallback_active"]:
            st.info(
                "Deterministic answer fallback active: "
                f"{generation['fallback_reason']}"
            )
        elif generation["generator_name"] == "openai":
            st.caption(
                "Answer generated with OpenAI model: "
                f"{generation['model']}"
            )
        st.write(response["answer"])
        st.subheader("Sources")
        for citation in response["citations"]:
            page = (
                f"page {citation['page_no']}"
                if citation["page_no"] is not None
                else "chunk"
            )
            st.markdown(
                f"**[{citation['citation_id']}] {citation['title']}** · "
                f"`{citation['chunk_id']}` · "
                f"version `{citation['doc_version']}` · {page}"
            )
            st.caption(citation["source_uri"])

    with tabs[1]:
        left, right = st.columns(2)
        with left:
            st.subheader("Tool Recall Results")
            st.dataframe(
                [
                    {
                        "tools": ", ".join(
                            entry["tool"]
                            for entry in response[
                                "retrieval_provenance"
                            ].get(item["chunk_id"], [])
                        ),
                        "rank": item["rank"],
                        "hybrid_score": item["hybrid_score"],
                        "source": item["title"],
                        "chunk": item["chunk_id"],
                        "version": item["doc_version"],
                        "updated_at": item["updated_at"],
                        "snippet": item["text_summary"],
                    }
                    for item in response["milvus_recalled"]
                ],
                use_container_width=True,
            )
        st.subheader("Evidence Coverage")
        st.json(response["trace"]["evidence_grading"])
        with right:
            st.subheader("Reranked Results")
            st.dataframe(
                [
                    {
                        "rerank": item["rerank"],
                        "rerank_score": item["rerank_score"],
                        "old_rank": item["old_rank"],
                        "source": item["title"],
                        "chunk": item["chunk_id"],
                        "version": item["doc_version"],
                        "selected": item["selected"],
                    }
                    for item in response["reranked"]
                ],
                use_container_width=True,
            )

    with tabs[2]:
        events = st.session_state["query_events"].get(
            response["query_id"],
            [],
        )
        st.subheader("Agent execution")
        if events:
            st.markdown(
                render_trace_timeline(events),
                unsafe_allow_html=True,
            )
        else:
            st.info("No presentation-safe trace events were recorded.")
        st.subheader("Metrics")
        st.json(response["metrics"])
        with st.expander("Advanced trace JSON"):
            st.json(
                {
                    "events": events,
                    "terminal_trace": response["trace"],
                }
            )

    with tabs[3]:
        memory = response["trace"]["memory"]
        st.subheader("Session Memory")
        st.caption(
            f"Status: {memory['status']} · "
            f"recalled {memory['recalled_count']} · "
            f"written {memory['written_count']} · "
            f"TTL {memory['ttl_seconds']} seconds"
        )
        if memory["status"] in {"recall_failed", "write_failed"}:
            st.warning(
                "Memory is temporarily degraded; the validated answer "
                "remains available."
            )
        if memory["recalled"]:
            st.subheader("Used for this turn")
            st.dataframe(
                memory["recalled"],
                use_container_width=True,
            )
        else:
            st.info("No prior session Memory was used for this turn.")
        if st.session_state["memory_error"]:
            st.warning(st.session_state["memory_error"])
        records = st.session_state["memory_records"]
        st.subheader("Live records in this session")
        if records:
            st.dataframe(records, use_container_width=True)
        else:
            st.info("No live Memory records are currently visible.")


if __name__ == "__main__":
    main()
