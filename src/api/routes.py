"""FastAPI app: start browser runs, stream their steps (SSE), and fetch results."""

from __future__ import annotations

import json
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse

from ..agent import open_checkpointer, to_transcript, validate_run_id
from ..tools import BrowserSession
from ..utils.config import Settings, load_settings
from .runs import Run, RunManager
from .schemas import HealthResponse, RunCreated, RunDetail, RunRequest, RunResult, RunSummary


def create_app(
    settings: Settings | None = None,
    llm_factory: Callable[[], Any] | None = None,
    browser_factory: Callable[[Settings], BrowserSession] = BrowserSession,
) -> FastAPI:
    settings = settings or load_settings()
    app = FastAPI(title="Browser Agent", version="1.0.0")
    checkpointer = open_checkpointer(settings.sessions_dir)
    runs = RunManager(settings, checkpointer, llm_factory=llm_factory, browser_factory=browser_factory)
    app.state.runs = runs

    def get_run(run_id: str) -> Run:
        run = runs.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return run

    def summary(run: Run) -> dict:
        return {"run_id": run.id, "task": run.task, "status": run.status, "created_at": run.created_at}

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(model=settings.model, search_url=settings.search_url, headless=settings.headless)

    @app.post("/runs", response_model=RunCreated, status_code=202)
    def start_run(request: RunRequest) -> RunCreated:
        """Start a run in the background. Follow it at events_url (Server-Sent Events)."""
        run = runs.start(request.task)
        return RunCreated(run_id=run.id, status=run.status, events_url=f"/runs/{run.id}/events")

    @app.post("/run", response_model=RunResult)
    def run_sync(request: RunRequest, timeout: float = Query(600, ge=1, le=3600)) -> RunResult:
        """Run a task and wait for the answer (no streaming)."""
        run = runs.start(request.task)
        if not runs.wait(run, timeout=timeout):
            raise HTTPException(status_code=504, detail=f"Still running; follow /runs/{run.id}")
        return RunResult(**run.result)

    @app.get("/runs", response_model=list[RunSummary])
    def list_runs() -> list[dict]:
        return [summary(r) for r in runs.list()]

    @app.get("/runs/{run_id}", response_model=RunDetail)
    def run_detail(run_id: str, screenshots: bool = True) -> dict:
        run = get_run(run_id)
        events = run.events if screenshots else [{**e, "screenshot": None} for e in run.events]
        return {**summary(run), "result": run.result, "events": events}

    @app.get("/runs/{run_id}/events")
    def run_events(run_id: str, start: int = Query(0, ge=0)) -> StreamingResponse:
        """Server-Sent Events: one `data:` line per step until the run is done."""
        run = get_run(run_id)

        def sse():
            for event in runs.stream(run, start=start):
                yield ": keepalive\n\n" if event is None else f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/runs/{run_id}/transcript")
    def transcript(run_id: str) -> dict:
        """The full LangGraph message history of a run (screenshots omitted)."""
        try:
            validate_run_id(run_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        checkpoint = checkpointer.get({"configurable": {"thread_id": run_id}})
        if not checkpoint:
            raise HTTPException(status_code=404, detail="Run not found")
        return {"run_id": run_id, "messages": to_transcript(checkpoint["channel_values"].get("messages", []))}

    return app
