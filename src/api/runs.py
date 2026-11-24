"""Background runs: each task gets its own thread + browser; events are kept for streaming.

Playwright's sync API must stay on the thread that created it, so the agent runs on
a dedicated worker thread and publishes events. HTTP handlers only read those events.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import anthropic

from ..agent import build_agent
from ..tools import BrowserSession
from ..utils.config import Settings
from ..utils.logger import get_logger

log = get_logger("runs")

FINISHED = {"completed", "max_steps", "max_tokens", "refusal", "error"}
MAX_KEPT_RUNS = 50


@dataclass
class Run:
    id: str
    task: str
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    events: list[dict] = field(default_factory=list)
    result: dict | None = None
    cond: threading.Condition = field(default_factory=threading.Condition, repr=False)

    @property
    def finished(self) -> bool:
        return self.status in FINISHED


class RunManager:
    def __init__(
        self,
        settings: Settings,
        checkpointer: Any,
        llm_factory: Callable[[], Any] | None = None,
        browser_factory: Callable[[Settings], BrowserSession] = BrowserSession,
    ):
        self.settings = settings
        self.checkpointer = checkpointer
        self.llm_factory = llm_factory
        self.browser_factory = browser_factory
        self._slots = threading.BoundedSemaphore(settings.max_concurrent_runs)
        self._runs: dict[str, Run] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ public
    def start(self, task: str) -> Run:
        run = Run(id=uuid.uuid4().hex[:12], task=task)
        with self._lock:
            self._runs[run.id] = run
            for old in sorted(self._runs.values(), key=lambda r: r.created_at)[:-MAX_KEPT_RUNS]:
                if old.finished:
                    del self._runs[old.id]
        threading.Thread(target=self._worker, args=(run,), name=f"run-{run.id}", daemon=True).start()
        return run

    def get(self, run_id: str) -> Run | None:
        return self._runs.get(run_id)

    def list(self) -> list[Run]:
        return sorted(self._runs.values(), key=lambda r: r.created_at, reverse=True)

    def wait(self, run: Run, timeout: float | None = None) -> bool:
        with run.cond:
            return run.cond.wait_for(lambda: run.finished, timeout=timeout)

    def stream(self, run: Run, start: int = 0, keepalive: float = 15.0) -> Iterator[dict | None]:
        """Yield events from `start` as they arrive; None means 'no news' (send a keepalive)."""
        index = start
        while True:
            with run.cond:
                run.cond.wait_for(lambda: len(run.events) > index or run.finished, timeout=keepalive)  # noqa: B023 (called before index changes)
                batch = run.events[index:]
                done = run.finished
            if not batch:
                if done:
                    return
                yield None
                continue
            for event in batch:
                yield event
            index += len(batch)

    # ------------------------------------------------------------------ worker
    def _emit(self, run: Run, event: dict) -> None:
        with run.cond:
            event = {"seq": len(run.events), "time": round(time.time() - run.created_at, 2), **event}
            run.events.append(event)
            run.cond.notify_all()

    def _finish(self, run: Run, status: str, result: dict) -> None:
        # Publish the done event first, then mark finished, so streams never end before it.
        self._emit(run, {"type": "done", "status": status, "result": result})
        with run.cond:
            run.result = result
            run.status = status
            run.cond.notify_all()

    def _worker(self, run: Run) -> None:
        with self._slots:
            run.status = "running"
            self._emit(run, {"type": "status", "status": "running"})
            try:
                with self.browser_factory(self.settings) as browser:
                    agent = build_agent(
                        self.settings,
                        browser,
                        self.checkpointer,
                        on_event=lambda e: self._emit(run, e),
                        llm=self.llm_factory() if self.llm_factory else None,
                    )
                    result = agent.run(run.task, run_id=run.id)
                self._finish(run, result.status, {
                    "run_id": run.id,
                    "status": result.status,
                    "final_text": result.final_text,
                    "steps": result.steps,
                    "actions": agent.executor.actions,
                    "usage": vars(result.usage),
                })
            except anthropic.AuthenticationError:
                self._fail(run, "Authentication failed: set ANTHROPIC_API_KEY in .env.")
            except anthropic.RateLimitError:
                self._fail(run, "Rate limited by the model API; retry later.")
            except anthropic.APIError as exc:
                self._fail(run, f"Model API error: {exc}")
            except Exception as exc:
                log.exception("run %s crashed", run.id)
                self._fail(run, f"{type(exc).__name__}: {exc}")

    def _fail(self, run: Run, message: str) -> None:
        self._emit(run, {"type": "error", "message": message})
        self._finish(run, "error", {"run_id": run.id, "status": "error", "final_text": message,
                                    "steps": 0, "actions": 0, "usage": None})
