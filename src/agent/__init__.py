"""Core agent package plus a factory that wires the graph to a browser."""

from __future__ import annotations

from typing import Any

from ..models import build_chat_model
from ..prompts import build_system_prompt
from ..tools import BrowserSession, default_tools
from ..utils.config import Settings
from .agent import BrowserAgent, EventFn
from .executor import ToolExecutor
from .memory import open_checkpointer, to_transcript, validate_run_id
from .state import AgentResult, AgentState, Usage


def build_agent(
    settings: Settings,
    browser: BrowserSession,
    checkpointer: Any,
    on_event: EventFn | None = None,
    llm: Any | None = None,
) -> BrowserAgent:
    executor = ToolExecutor(default_tools(), browser, on_event=on_event)
    return BrowserAgent(
        llm=llm or build_chat_model(settings, executor.definitions()),
        executor=executor,
        system_prompt=build_system_prompt(settings.knowledge_base_dir),
        checkpointer=checkpointer,
        max_steps=settings.max_steps,
        on_event=on_event,
    )


__all__ = [
    "AgentResult", "AgentState", "BrowserAgent", "ToolExecutor", "Usage", "build_agent",
    "open_checkpointer", "to_transcript", "validate_run_id",
]
