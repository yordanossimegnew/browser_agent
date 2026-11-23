"""Request/response models for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=4000, description="What to research on the web.")


class RunCreated(BaseModel):
    run_id: str
    status: str
    events_url: str


class UsageOut(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int


class RunResult(BaseModel):
    run_id: str
    status: str
    final_text: str
    steps: int
    actions: int
    usage: UsageOut | None


class RunSummary(BaseModel):
    run_id: str
    task: str
    status: str
    created_at: float


class RunDetail(RunSummary):
    result: RunResult | None
    events: list[dict]


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str
    search_url: str
    headless: bool
