"""FastAPI endpoints, including the Server-Sent Events stream."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from src.api import create_app
from tests.fakes import FakeLLM, response, text, tool_use


def client_for(settings, site_url):
    def script():
        return FakeLLM([
            response(tool_use("t1", "navigate", url=site_url + "/pricing.html"), stop_reason="tool_use"),
            response(text("Pro is $20/month.")),
        ])
    return TestClient(create_app(settings, llm_factory=script))


def test_health(settings, site_url):
    body = client_for(settings, site_url).get("/health").json()
    assert body["status"] == "ok" and body["model"] == settings.model


def test_sync_run(settings, site_url):
    res = client_for(settings, site_url).post("/run", json={"task": "Acme Pro price?"})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "completed" and body["final_text"] == "Pro is $20/month."
    assert body["actions"] == 1 and body["usage"]["input_tokens"] == 20


def test_streamed_run_events(settings, site_url):
    client = client_for(settings, site_url)
    created = client.post("/runs", json={"task": "Acme Pro price?"})
    assert created.status_code == 202
    run_id = created.json()["run_id"]

    events = []
    with client.stream("GET", f"/runs/{run_id}/events") as stream:
        for line in stream.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    kinds = [e["type"] for e in events]
    assert kinds == ["status", "action", "observation", "thought", "done"]
    assert events[2]["screenshot"] and events[-1]["result"]["status"] == "completed"
    assert [e["seq"] for e in events] == list(range(len(events)))

    # Replay from an offset, list, detail and transcript
    with client.stream("GET", f"/runs/{run_id}/events?start=3") as stream:
        replay = [json.loads(l[6:]) for l in stream.iter_lines() if l.startswith("data: ")]
    assert [e["type"] for e in replay] == ["thought", "done"]
    assert client.get("/runs").json()[0]["run_id"] == run_id
    detail = client.get(f"/runs/{run_id}", params={"screenshots": False}).json()
    assert detail["status"] == "completed" and all(e.get("screenshot") is None for e in detail["events"])
    roles = [m["role"] for m in client.get(f"/runs/{run_id}/transcript").json()["messages"]]
    assert roles == ["user", "assistant", "tool", "assistant"]


def test_errors(settings, site_url):
    client = client_for(settings, site_url)
    assert client.get("/runs/missing").status_code == 404
    assert client.get("/runs/../x/transcript").status_code in (400, 404)
    assert client.post("/runs", json={"task": ""}).status_code == 422
