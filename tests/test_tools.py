"""Browser tools against a real headless Chromium and a local test site."""

from __future__ import annotations

import base64
from dataclasses import replace

import pytest

from src.tools import BrowserError, BrowserSession, default_tools


@pytest.fixture
def browser(settings):
    with BrowserSession(settings) as b:
        yield b


def _id(obs, label: str) -> int:
    return next(e["id"] for e in obs.elements if e["label"] == label)


def test_observation_has_numbered_elements_and_screenshot(browser, site_url):
    browser.navigate(site_url + "/index.html")
    obs = browser.observe()
    assert obs.title == "Acme Cloud"
    labels = {e["label"] for e in obs.elements}
    assert {"Pricing", "About", "Search docs", "Search"} <= labels
    assert base64.b64decode(obs.screenshot_b64)[:2] == b"\xff\xd8"  # JPEG
    text = obs.to_text()
    assert f'[{_id(obs, "Pricing")}] link "Pricing"' in text and "30 regions" in text


def test_click_link_navigates(browser, site_url):
    browser.navigate(site_url + "/index.html")
    browser.click(_id(browser.observe(), "Pricing"))
    assert browser.page.title() == "Pricing - Acme Cloud"


def test_type_and_submit_search(browser, site_url):
    browser.navigate(site_url + "/index.html")
    browser.type_text(_id(browser.observe(), "Search docs"), "pricing", press_enter=True)
    obs = browser.observe()
    assert obs.title == "Search: pricing"
    assert "Acme Cloud Pricing" in {e["label"] for e in obs.elements}


def test_web_search_uses_configured_engine(browser):
    browser.web_search("free tier")
    assert browser.page.title() == "Search: free tier"


def test_scroll_back_and_read_page(browser, site_url):
    browser.navigate(site_url + "/index.html")
    browser.navigate(site_url + "/pricing.html")
    assert "more below" in browser.observe().scroll
    browser.scroll("down")
    assert browser.page.evaluate("scrollY") > 0
    page = browser.read_page(1)  # full text, not just the viewport
    assert "contact sales" in page and "Part 1 of 1." in page
    with pytest.raises(BrowserError, match="has 1 part"):
        browser.read_page(2)
    browser.go_back()
    assert browser.page.title() == "Acme Cloud"


def test_unknown_element_is_reported(browser, site_url):
    browser.navigate(site_url + "/index.html")
    browser.observe()
    with pytest.raises(BrowserError, match="No element \\[999\\]"):
        browser.click(999)


@pytest.mark.parametrize("url, reason", [
    ("file:///C:/Windows/win.ini", "Only http"),
    ("http://localhost:8000/", "private/local"),
    ("http://127.0.0.1/", "private/local"),
    ("http://169.254.169.254/latest/meta-data/", "private/local"),
])
def test_ssrf_guard_blocks_local_targets(settings, url, reason):
    with BrowserSession(replace(settings, allow_private_hosts=False)) as b:
        with pytest.raises(BrowserError, match=reason):
            b.navigate(url)


def test_guard_also_blocks_clicked_links(settings, site_url):
    """Navigations started by clicks/redirects are checked too, not just navigate()."""
    with BrowserSession(replace(settings, allow_private_hosts=False)) as b:
        b.page.set_content(f'<a href="{site_url}/index.html">internal</a>')
        b.click(_id(b.observe(), "internal"))
        assert b.page.title() != "Acme Cloud"


def test_blocked_domains(settings):
    with BrowserSession(replace(settings, blocked_domains=("example.com",))) as b:
        with pytest.raises(BrowserError, match="blocked domain"):
            b.navigate("https://docs.example.com/")


def test_tool_schemas_are_strict():
    for tool in default_tools():
        api = tool.to_api()
        assert api["strict"] is True and api["input_schema"]["additionalProperties"] is False
