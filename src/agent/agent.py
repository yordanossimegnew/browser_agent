"""The browser agent as a LangGraph state machine.

    START ──> agent ──(tool calls)──> browser ──> agent ──> ... ──(answer)──> END

`agent` asks Claude (LangChain ChatAnthropic) for the next action; `browser` performs
it with Playwright and returns the new page (text + labelled screenshot).
"""

from __future__ import annotations

import uuid
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from ..prompts import TRUNCATED_TOOL_CALL
from ..utils.logger import get_logger
from .executor import ToolExecutor
from .memory import validate_run_id
from .state import AgentResult, AgentState, Usage

log = get_logger("agent")

EventFn = Callable[[dict], None]


def _stop_reason(msg: AIMessage) -> str | None:
    return (msg.response_metadata or {}).get("stop_reason")


def _text(msg: AIMessage) -> str:
    if isinstance(msg.content, str):
        return msg.content
    return "\n".join(
        b["text"] for b in msg.content if isinstance(b, dict) and b.get("type") == "text" and b.get("text")
    )


class BrowserAgent:
    def __init__(
        self,
        llm: Any,
        executor: ToolExecutor,
        system_prompt: str,
        checkpointer: Any,
        max_steps: int = 20,
        on_event: EventFn | None = None,
    ):
        self.llm = llm
        self.executor = executor
        self.system_prompt = system_prompt
        self.checkpointer = checkpointer
        self.max_steps = max_steps
        self.on_event = on_event or (lambda _e: None)
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ nodes
    def _agent_node(self, state: AgentState) -> dict:
        response = self.llm.invoke([SystemMessage(self.system_prompt), *state["messages"]])
        steps = state.get("steps", 0) + 1
        log.info("step=%d stop_reason=%s tool_calls=%d", steps, _stop_reason(response), len(response.tool_calls))
        if text := _text(response):
            self.on_event({"type": "thought", "step": steps, "text": text, "final": not response.tool_calls})
        for call in response.tool_calls:
            self.on_event({"type": "action", "step": steps, "tool": call["name"], "args": call["args"]})
        return {"messages": [response], "steps": steps}

    def _browser_node(self, state: AgentState) -> dict:
        last: AIMessage = state["messages"][-1]
        if _stop_reason(last) == "max_tokens":
            calls = [*last.tool_calls, *last.invalid_tool_calls]
            return {"messages": [
                ToolMessage(content=TRUNCATED_TOOL_CALL, tool_call_id=c["id"], name=c.get("name") or "", status="error")
                for c in calls if c.get("id")
            ]}
        return {"messages": [self.executor.execute(call) for call in last.tool_calls]}

    # ------------------------------------------------------------------ routing
    def _after_agent(self, state: AgentState) -> str:
        last: AIMessage = state["messages"][-1]
        stop = _stop_reason(last)
        if last.tool_calls or (stop == "max_tokens" and last.invalid_tool_calls):
            return "browser"
        if stop == "pause_turn" and state.get("steps", 0) < self.max_steps:
            return "agent"
        return END

    def _after_browser(self, state: AgentState) -> str:
        return END if state.get("steps", 0) >= self.max_steps else "agent"

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("agent", self._agent_node)
        graph.add_node("browser", self._browser_node)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", self._after_agent, ["browser", "agent", END])
        graph.add_conditional_edges("browser", self._after_browser, ["agent", END])
        return graph.compile(checkpointer=self.checkpointer)

    # ------------------------------------------------------------------ public API
    def messages(self, run_id: str) -> list[Any]:
        config = {"configurable": {"thread_id": validate_run_id(run_id)}}
        return self.graph.get_state(config).values.get("messages", [])

    def run(self, task: str, run_id: str | None = None) -> AgentResult:
        run_id = validate_run_id(run_id) if run_id else uuid.uuid4().hex[:12]
        config = {"configurable": {"thread_id": run_id}, "recursion_limit": self.max_steps * 2 + 10}
        self.graph.invoke({"messages": [HumanMessage(task)], "steps": 0}, config)

        values = self.graph.get_state(config).values
        messages = values["messages"]
        return AgentResult(
            status=self._status(messages[-1]),
            final_text=self._final_text(messages),
            steps=values.get("steps", 0),
            session_id=run_id,
            usage=Usage.from_messages(messages),
        )

    @staticmethod
    def _status(last: Any) -> str:
        if isinstance(last, ToolMessage):
            return "max_steps"
        stop = _stop_reason(last) if isinstance(last, AIMessage) else None
        return {"end_turn": "completed", "refusal": "refusal", "max_tokens": "max_tokens",
                "pause_turn": "max_steps"}.get(stop, "error")

    @staticmethod
    def _final_text(messages: list[Any]) -> str:
        last = messages[-1]
        if isinstance(last, AIMessage) and not last.tool_calls:
            return _text(last)
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and (text := _text(msg)):
                return text
        return ""
