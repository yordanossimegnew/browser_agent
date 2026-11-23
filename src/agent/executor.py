"""Runs browser tool calls and turns the resulting page into a ToolMessage.

Each result carries the page observation as text plus the labelled screenshot as
an image block, so Claude can read the page and see its layout.
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import ToolMessage

from ..tools import BrowserError, BrowserSession, Tool
from ..utils.logger import get_logger

log = get_logger("executor")

EventFn = Callable[[dict], None]


class ToolExecutor:
    def __init__(self, tools: list[Tool], browser: BrowserSession, on_event: EventFn | None = None):
        self.tools = {tool.name: tool for tool in tools}
        self.browser = browser
        self.on_event = on_event or (lambda _e: None)
        self.actions = 0

    def definitions(self) -> list[dict]:
        return [tool.to_api() for tool in self.tools.values()]

    def execute(self, call: dict) -> ToolMessage:
        """Run one tool call ({"name", "args", "id"}). Never raises."""
        name, args, call_id = call["name"], call.get("args") or {}, call["id"]
        self.actions += 1
        tool = self.tools.get(name)
        error: str | None = None
        output: Any = None

        if tool is None:
            error = f"Unknown tool: {name}"
        else:
            try:
                output = tool.handler(self.browser, **args)
            except BrowserError as exc:
                error = str(exc)
            except TypeError as exc:
                error = f"Invalid arguments: {exc}"
            except Exception as exc:  # keep the agent alive on unexpected browser failures
                log.exception("%s crashed", name)
                error = f"{type(exc).__name__}: {exc}"

        if tool is not None and not tool.observes and error is None:
            message = ToolMessage(content=str(output), tool_call_id=call_id, name=name)
            self._emit(name, args, error, observation=None, extra_text=str(output))
            return message

        # Fresh observation after every action (also after errors, so the model sees where it is).
        try:
            obs = self.browser.observe(screenshot=True)
        except Exception as exc:  # page may be mid-navigation or crashed
            log.warning("observe failed: %s", exc)
            obs = None

        text = obs.to_text() if obs else "(could not read the page)"
        if error:
            text = f"ERROR: {error}\n\n{text}"
        content: list[dict] = [{"type": "text", "text": text}]
        if obs and obs.screenshot_b64:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": obs.screenshot_b64},
            })
        self._emit(name, args, error, observation=obs)
        return ToolMessage(
            content=content, tool_call_id=call_id, name=name, status="error" if error else "success"
        )

    def _emit(self, name, args, error, observation, extra_text: str | None = None) -> None:
        self.on_event({
            "type": "observation",
            "step": self.actions,
            "tool": name,
            "args": args,
            "is_error": error is not None,
            "message": error or (extra_text[:300] + "…" if extra_text and len(extra_text) > 300 else extra_text),
            "url": observation.url if observation else (self.browser.page.url if self.browser.page else ""),
            "title": observation.title if observation else "",
            "screenshot": observation.screenshot_b64 if observation else None,
        })
