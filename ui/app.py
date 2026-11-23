"""Streamlit front end for the Browser Agent. Streams each step live from the FastAPI backend.

    uvicorn main:app            # terminal 1
    streamlit run ui/app.py     # terminal 2
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import requests
import streamlit as st

API_URL = os.getenv("AGENT_API_URL", "http://localhost:8000")
EXAMPLES = json.loads((Path(__file__).resolve().parents[1] / "data" / "examples.json").read_text())
TOOL_ICONS = {"web_search": "🔎", "navigate": "🧭", "click": "👆", "type_text": "⌨️",
              "scroll": "📜", "go_back": "↩️", "read_page": "📖"}

st.set_page_config(page_title="Browser Agent", page_icon="🌐", layout="wide")
st.markdown("""<style>
.step {padding:6px 10px;border-left:3px solid #e8590c;margin:4px 0;background:#fff7f0;border-radius:4px;font-size:0.9rem}
.step.err {border-left-color:#c92a2a;background:#fff5f5}
.step.thought {border-left-color:#5c7cfa;background:#f3f5ff}
.url {font-family:monospace;font-size:0.8rem;color:#666;word-break:break-all}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- API helpers
def api_get(endpoint: str, params: dict | None = None):
    try:
        r = requests.get(f"{API_URL}{endpoint}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def stream_events(run_id: str):
    with requests.get(f"{API_URL}/runs/{run_id}/events", stream=True, timeout=(5, 120)) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                yield json.loads(line[6:])


def describe(event: dict) -> str:
    args = event.get("args") or {}
    tool = event.get("tool", "")
    detail = args.get("query") or args.get("url") or args.get("text") or args.get("direction")
    if "element_id" in args:
        detail = f"[{args['element_id']}]" + (f' "{args["text"]}"' if args.get("text") else "")
    return f"{TOOL_ICONS.get(tool, '•')} <b>{tool}</b> {detail or ''}"


def render_step(container, event: dict) -> None:
    kind = event["type"]
    if kind == "thought" and not event.get("final"):
        container.markdown(f'<div class="step thought">💭 {event["text"]}</div>', unsafe_allow_html=True)
    elif kind == "observation":
        css = "step err" if event["is_error"] else "step"
        extra = f'<br>⚠️ {event["message"]}' if event["is_error"] else ""
        container.markdown(
            f'<div class="{css}">{describe(event)}{extra}<div class="url">{event["url"]}</div></div>',
            unsafe_allow_html=True,
        )
    elif kind == "error":
        container.error(event["message"])


def render_result(result: dict) -> None:
    badge = {"completed": "🟢", "refusal": "🟠", "max_steps": "🟡"}.get(result["status"], "🔴")
    st.markdown(f"### {badge} {result['status'].replace('_', ' ').title()}")
    usage = result.get("usage") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Browser actions", result.get("actions", 0))
    c2.metric("Model calls", result.get("steps", 0))
    c3.metric("Input tokens", f"{usage.get('input_tokens', 0):,}")
    c4.metric("Output tokens", f"{usage.get('output_tokens', 0):,}")
    st.markdown(result["final_text"] or "_(no answer)_")


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🌐 Browser Agent")
    st.caption("LangGraph + Playwright research agent on Claude")
    health = api_get("/health")
    if health:
        st.success("API connected", icon="✅")
        st.markdown(f"**Model:** `{health['model']}`")
        st.markdown(f"**Search:** `{health['search_url'].split('/')[2]}`")
    else:
        st.error(f"API not reachable at {API_URL}. Start it with `uvicorn main:app`.", icon="🚫")


tab_new, tab_history = st.tabs(["🔎 New research", "🗂️ Run history"])

# ----------------------------------------------------------------------------- new run
with tab_new:
    example = st.selectbox("Start from an example", ["—", *[e["name"] for e in EXAMPLES]])
    default = next((e["task"] for e in EXAMPLES if e["name"] == example), "")
    task = st.text_area("What should the agent find out?", value=default, height=90,
                        placeholder="e.g. What are the current free tier limits of Vercel and Netlify?")
    start = st.button("Start browsing", type="primary", disabled=not (task.strip() and health))

    if start:
        created = requests.post(f"{API_URL}/runs", json={"task": task}, timeout=10).json()
        st.session_state["run_id"] = created["run_id"]
        left, right = st.columns([3, 2])
        with left:
            st.markdown("**Live browser**")
            url_box = st.empty()
            screen = st.empty()
            screen.info("Launching browser…")
        with right:
            st.markdown("**Steps**")
            steps = st.container(height=520)
        final = st.container()
        try:
            for event in stream_events(created["run_id"]):
                if event["type"] == "observation" and event.get("screenshot"):
                    screen.image(base64.b64decode(event["screenshot"]), width="stretch")
                    url_box.markdown(f'<div class="url">🔒 {event["url"]}</div>', unsafe_allow_html=True)
                if event["type"] == "done":
                    with final:
                        render_result(event["result"])
                else:
                    render_step(steps, event)
        except requests.RequestException as exc:
            st.error(f"Lost connection to the API: {exc}")

    elif st.session_state.get("run_id"):
        detail = api_get(f"/runs/{st.session_state['run_id']}", {"screenshots": False})
        if detail and detail.get("result"):
            render_result(detail["result"])

# ----------------------------------------------------------------------------- history
with tab_history:
    runs = api_get("/runs") or []
    if not runs:
        st.info("No runs yet. Start one in the New research tab.")
    else:
        labels = {f"{r['task'][:70]} · {r['status']}": r["run_id"] for r in runs}
        chosen = st.selectbox("Run", list(labels))
        detail = api_get(f"/runs/{labels[chosen]}")
        if detail:
            if detail.get("result"):
                render_result(detail["result"])
            st.markdown("#### Step by step")
            for event in detail["events"]:
                if event["type"] != "observation":
                    continue
                title = f"Step {event['step']} · {event['tool']} · {event['title'] or event['url']}"
                with st.expander(("⚠️ " if event["is_error"] else "") + title):
                    st.markdown(describe(event), unsafe_allow_html=True)
                    if event.get("message"):
                        st.caption(event["message"])
                    if event.get("screenshot"):
                        st.image(base64.b64decode(event["screenshot"]), width="stretch")
