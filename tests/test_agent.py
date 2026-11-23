"""The LangGraph agent driving a real browser, with a scripted model."""

from __future__ import annotations

from langchain_core.messages import ToolMessage

from src.agent import build_agent, open_checkpointer
from src.tools import BrowserSession
from tests.fakes import FakeLLM, response, text, tool_use


def run_agent(settings, script):
    events = []
    llm = FakeLLM(script)
    with BrowserSession(settings) as browser:
        agent = build_agent(settings, browser, open_checkpointer(settings.sessions_dir),
                            on_event=events.append, llm=llm)
        result = agent.run("What does Acme Cloud Pro cost?")
    return result, llm, events, agent


def test_agent_browses_and_answers(settings, site_url):
    result, llm, events, agent = run_agent(settings, [
        response(text("Searching first."), tool_use("t1", "web_search", query="acme pricing"), stop_reason="tool_use"),
        response(tool_use("t2", "navigate", url=site_url + "/pricing.html"), stop_reason="tool_use"),
        response(tool_use("t3", "read_page", part=1), stop_reason="tool_use"),
        response(text("Pro costs **$20/month**.\n\nSources: " + site_url + "/pricing.html")),
    ])

    assert result.status == "completed"
    assert "$20/month" in result.final_text
    assert result.steps == 4 and agent.executor.actions == 3

    # Browser actions come back as text + screenshot; read_page as plain text.
    nav_result = llm.calls[2][-1]
    assert isinstance(nav_result, ToolMessage)
    assert [b["type"] for b in nav_result.content] == ["text", "image"]
    assert "Pricing - Acme Cloud" in nav_result.content[0]["text"]
    assert "contact sales" in llm.calls[3][-1].content

    kinds = [e["type"] for e in events]
    assert kinds[:3] == ["thought", "action", "observation"]
    shots = [e for e in events if e["type"] == "observation" and e["tool"] == "navigate"]
    assert shots[0]["screenshot"] and shots[0]["url"].endswith("/pricing.html")


def test_errors_include_current_page(settings, site_url):
    result, llm, events, _ = run_agent(settings, [
        response(tool_use("t1", "navigate", url=site_url + "/index.html"), stop_reason="tool_use"),
        response(tool_use("t2", "click", element_id=999), stop_reason="tool_use"),
        response(text("Could not find it.")),
    ])
    err = llm.calls[2][-1]
    assert err.status == "error"
    assert err.content[0]["text"].startswith("ERROR: No element [999]")
    assert err.content[1]["type"] == "image"  # still shows where the agent is
    assert any(e["type"] == "observation" and e["is_error"] for e in events)


def test_step_limit(settings, site_url):
    loop = [response(tool_use(f"t{i}", "scroll", direction="down"), stop_reason="tool_use") for i in range(20)]
    result, *_ = run_agent(settings, [response(tool_use("t0", "navigate", url=site_url + "/pricing.html"),
                                               stop_reason="tool_use"), *loop])
    assert result.status == "max_steps" and result.steps == settings.max_steps
