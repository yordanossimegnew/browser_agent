"""The browser actions the model can take."""

from __future__ import annotations

from .base import Tool, schema
from .browser import BrowserSession

ELEMENT_ID = {"type": "integer", "description": "Element number from the latest observation, e.g. 7 for [7]."}

TOOLS = [
    Tool(
        name="web_search",
        description="Search the web and open the results page. Start research here unless you already know the URL.",
        input_schema=schema({"query": {"type": "string", "description": "Search query."}}),
        handler=lambda b, query: b.web_search(query),
    ),
    Tool(
        name="navigate",
        description="Open a URL directly (http or https).",
        input_schema=schema({"url": {"type": "string", "description": "Full URL to open."}}),
        handler=lambda b, url: b.navigate(url),
    ),
    Tool(
        name="click",
        description="Click an element (link, button, tab...) by its number from the latest observation.",
        input_schema=schema({"element_id": ELEMENT_ID}),
        handler=lambda b, element_id: b.click(element_id),
    ),
    Tool(
        name="type_text",
        description="Type text into an input or search box by its number, optionally pressing Enter to submit.",
        input_schema=schema({
            "element_id": ELEMENT_ID,
            "text": {"type": "string", "description": "Text to type (replaces current value)."},
            "press_enter": {"type": "boolean", "description": "Press Enter afterwards to submit."},
        }),
        handler=lambda b, element_id, text, press_enter: b.type_text(element_id, text, press_enter),
    ),
    Tool(
        name="scroll",
        description="Scroll the page to reveal more content and elements.",
        input_schema=schema({"direction": {"type": "string", "enum": ["down", "up"]}}),
        handler=lambda b, direction: b.scroll(direction),
    ),
    Tool(
        name="go_back",
        description="Go back to the previous page.",
        input_schema=schema({}),
        handler=lambda b: b.go_back(),
    ),
    Tool(
        name="read_page",
        description=(
            "Return the full text of the current page (no screenshot), in parts of ~20k characters. "
            "Use it to extract details from long articles, docs or tables instead of scrolling. "
            "Start with part 1; the result says how many parts exist."
        ),
        input_schema=schema({"part": {"type": "integer", "description": "Which part to read, starting at 1."}}),
        handler=lambda b, part: b.read_page(part),
        observes=False,
    ),
]


def default_tools() -> list[Tool]:
    return list(TOOLS)


__all__ = ["BrowserSession", "TOOLS", "default_tools"]
