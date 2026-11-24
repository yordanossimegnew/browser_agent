"""Run memory: a LangGraph PostgreSQL checkpointer, one thread per run."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def validate_run_id(run_id: str) -> str:
    if not _SAFE_ID.match(run_id):
        raise ValueError(f"Invalid run id: {run_id!r}")
    return run_id


def open_checkpointer(database_url: str) -> PostgresSaver:
    """Connection-pooled LangGraph checkpointer; creates its tables on first use."""
    conninfo = database_url.replace("postgresql+psycopg://", "postgresql://")
    pool = ConnectionPool(conninfo, min_size=1, max_size=10, open=True,
                          kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row})
    saver = PostgresSaver(pool)
    saver.setup()
    return saver


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(
        b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text" and b.get("text")
    )


def to_transcript(messages: list[Any]) -> list[dict]:
    """LangChain messages -> plain dicts (screenshots dropped, flagged instead)."""
    out: list[dict] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            out.append({"role": "user", "text": _text_of(msg.content)})
        elif isinstance(msg, AIMessage):
            out.append({
                "role": "assistant",
                "text": _text_of(msg.content),
                "tool_calls": [{"name": c["name"], "args": c["args"]} for c in msg.tool_calls],
            })
        elif isinstance(msg, ToolMessage):
            has_image = isinstance(msg.content, list) and any(
                isinstance(b, dict) and b.get("type") == "image" for b in msg.content
            )
            out.append({
                "role": "tool",
                "name": msg.name,
                "text": _text_of(msg.content),
                "is_error": msg.status == "error",
                "has_screenshot": has_image,
            })
    return out
