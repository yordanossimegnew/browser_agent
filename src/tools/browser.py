"""Playwright browser session with numbered-element perception.

Every observation gives the model three views of the page:
  1. a numbered list of interactive elements ([7] link "Pricing"),
  2. the visible page text, and
  3. a screenshot with the same numbers drawn on the elements ("set-of-marks").
The model then acts by element number: click(7), type_text(3, "...").

Playwright's sync API is bound to the thread that started it, so one
BrowserSession must be created, used and closed on a single thread.
"""

from __future__ import annotations

import base64
import ipaddress
import socket
from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import quote_plus, urlparse

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page, sync_playwright

from ..utils.config import Settings
from ..utils.helpers import clean_text, truncate
from ..utils.logger import get_logger

log = get_logger("browser")

MAX_ELEMENTS = 120
OBSERVATION_TEXT_CHARS = 6_000
READ_PAGE_CHARS = 20_000

# Label interactive elements in (or just below) the viewport with data-agent-id.
_LABEL_JS = """
(maxElements) => {
  document.querySelectorAll('[data-agent-id]').forEach(e => e.removeAttribute('data-agent-id'));
  const selector = 'a[href], button, input:not([type=hidden]), textarea, select, summary, ' +
    '[role=button], [role=link], [role=tab], [role=menuitem], [role=checkbox], [contenteditable=true]';
  const out = [];
  let id = 0;
  for (const el of document.querySelectorAll(selector)) {
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) continue;
    if (r.bottom < 0 || r.top > window.innerHeight * 1.5) continue;
    const st = getComputedStyle(el);
    if (st.visibility === 'hidden' || st.display === 'none' || st.opacity === '0') continue;
    id += 1;
    el.setAttribute('data-agent-id', String(id));
    const tag = el.tagName.toLowerCase();
    const role = el.getAttribute('role') ||
      (tag === 'a' ? 'link' : tag === 'input' ? 'input:' + (el.type || 'text') : tag);
    const label = (el.innerText || el.value || el.getAttribute('aria-label') ||
      el.getAttribute('placeholder') || el.getAttribute('title') || el.getAttribute('alt') || '')
      .trim().replace(/\\s+/g, ' ').slice(0, 80);
    out.push({id, role, label, href: tag === 'a' ? el.href : null,
              inView: r.top >= 0 && r.top < window.innerHeight,
              box: [r.left, r.top, r.width, r.height]});
    if (id >= maxElements) break;
  }
  return out;
}
"""

# Draw numbered boxes over labelled elements that are inside the viewport.
_MARKS_JS = """
(elements) => {
  const layer = document.createElement('div');
  layer.id = '__agent_marks__';
  layer.style.cssText = 'position:fixed;inset:0;pointer-events:none;z-index:2147483647';
  for (const e of elements) {
    if (!e.inView) continue;
    const [x, y, w, h] = e.box;
    const box = document.createElement('div');
    box.style.cssText = `position:fixed;left:${x}px;top:${y}px;width:${w}px;height:${h}px;` +
      'border:2px solid #e8590c;border-radius:3px;box-sizing:border-box';
    const tag = document.createElement('div');
    tag.textContent = e.id;
    tag.style.cssText = `position:fixed;left:${x}px;top:${Math.max(0, y - 14)}px;` +
      'background:#e8590c;color:#fff;font:bold 11px/14px sans-serif;padding:0 3px;border-radius:2px';
    layer.append(box, tag);
  }
  document.documentElement.appendChild(layer);
}
"""

_UNMARK_JS = "() => document.getElementById('__agent_marks__')?.remove()"


class BrowserError(Exception):
    """A browser action failed in a way the model should hear about."""


@dataclass
class Observation:
    url: str
    title: str
    elements: list[dict]
    text: str
    scroll: str
    screenshot_b64: str | None = None

    def to_text(self) -> str:
        lines = [f"URL: {self.url}", f"Title: {self.title}", f"Scroll: {self.scroll}", "",
                 "Interactive elements (use the number to click or type):"]
        for e in self.elements:
            label = f' "{e["label"]}"' if e["label"] else ""
            lines.append(f'[{e["id"]}] {e["role"]}{label}')
        if not self.elements:
            lines.append("(none visible)")
        lines += ["", "Visible page text:", self.text or "(no text)"]
        return "\n".join(lines)


@lru_cache(maxsize=512)
def _resolves_to_private(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False  # let the browser report the DNS failure itself
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return True
    return False


class BrowserSession:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._pw = None
        self._browser = None
        self.page: Page | None = None

    # ------------------------------------------------------------------ lifecycle
    def __enter__(self) -> "BrowserSession":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def start(self) -> None:
        s = self.settings
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=s.headless)
        context = self._browser.new_context(
            viewport={"width": s.viewport_width, "height": s.viewport_height},
            locale="en-US",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
            ),
        )
        context.set_default_timeout(s.page_timeout_ms)
        # Guard every document request (navigations, redirects, iframes), not just goto().
        context.route("**/*", self._route_guard)
        self.page = context.new_page()

    def close(self) -> None:
        for closer in (self._browser, self._pw):
            try:
                if closer is not None:
                    closer.close() if closer is self._browser else closer.stop()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
        self._browser = self._pw = self.page = None

    # ------------------------------------------------------------------ URL safety
    def check_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise BrowserError(f"Only http(s) URLs are allowed, got '{parsed.scheme or url}'.")
        host = (parsed.hostname or "").lower()
        if not host:
            raise BrowserError(f"URL has no host: {url}")
        for blocked in self.settings.blocked_domains:
            if host == blocked or host.endswith("." + blocked):
                raise BrowserError(f"{host} is on the blocked domain list.")
        if not self.settings.allow_private_hosts and (host == "localhost" or _resolves_to_private(host)):
            raise BrowserError(f"{host} is a private/local address, which this agent may not visit.")

    def _route_guard(self, route) -> None:
        request = route.request
        if request.resource_type == "document":
            try:
                self.check_url(request.url)
            except BrowserError as exc:
                log.warning("Blocked navigation: %s", exc)
                route.abort("blockedbyclient")
                return
        route.continue_()

    # ------------------------------------------------------------------ actions
    def navigate(self, url: str) -> None:
        if "://" not in url:
            url = "https://" + url
        self.check_url(url)
        self._goto(url)

    def web_search(self, query: str) -> None:
        self._goto(self.settings.search_url.format(query=quote_plus(query)))

    def click(self, element_id: int) -> None:
        locator = self._element(element_id)
        try:
            locator.click()
        except PlaywrightError as exc:
            raise BrowserError(f"Could not click [{element_id}]: {_short(exc)}") from exc
        self._settle()

    def type_text(self, element_id: int, text: str, press_enter: bool) -> None:
        locator = self._element(element_id)
        try:
            locator.fill(text)
            if press_enter:
                locator.press("Enter")
        except PlaywrightError as exc:
            raise BrowserError(f"Could not type into [{element_id}]: {_short(exc)}") from exc
        self._settle()

    def scroll(self, direction: str) -> None:
        delta = self.settings.viewport_height * 0.8 * (1 if direction == "down" else -1)
        self.page.mouse.wheel(0, delta)
        self.page.wait_for_timeout(400)

    def go_back(self) -> None:
        if self.page.go_back() is None:
            raise BrowserError("There is no previous page in the history.")
        self._settle()

    # ------------------------------------------------------------------ perception
    def observe(self, screenshot: bool = True) -> Observation:
        page = self.page
        elements = page.evaluate(_LABEL_JS, MAX_ELEMENTS)
        shot = None
        if screenshot:
            page.evaluate(_MARKS_JS, elements)
            try:
                raw = page.screenshot(type="jpeg", quality=60)
            finally:
                page.evaluate(_UNMARK_JS)
            shot = base64.b64encode(raw).decode("ascii")
        metrics = page.evaluate(
            "() => [Math.round(scrollY), Math.max(document.body?.scrollHeight || 0, innerHeight), innerHeight]"
        )
        y, height, view = metrics
        pct = 100 if height <= view else round(100 * (y + view) / height)
        return Observation(
            url=page.url,
            title=page.title(),
            elements=elements,
            text=truncate(clean_text(page.evaluate(_VISIBLE_TEXT_JS)), OBSERVATION_TEXT_CHARS),
            scroll=f"{pct}% of page seen" + (" (more below)" if pct < 100 else ""),
            screenshot_b64=shot,
        )

    def read_page(self, part: int = 1) -> str:
        """Full page text, split into parts of READ_PAGE_CHARS so long pages lose nothing."""
        text = clean_text(self.page.evaluate("() => document.body ? document.body.innerText : ''"))
        total = max(1, -(-len(text) // READ_PAGE_CHARS))
        if not 1 <= part <= total:
            raise BrowserError(f"Part {part} does not exist; this page has {total} part(s).")
        chunk = text[(part - 1) * READ_PAGE_CHARS: part * READ_PAGE_CHARS]
        more = f" Call read_page with part={part + 1} for more." if part < total else ""
        return f"URL: {self.page.url}\nTitle: {self.page.title()}\nPart {part} of {total}.{more}\n\n{chunk}"

    # ------------------------------------------------------------------ internals
    def _goto(self, url: str) -> None:
        try:
            self.page.goto(url, wait_until="domcontentloaded")
        except PlaywrightError as exc:
            raise BrowserError(f"Could not open {url}: {_short(exc)}") from exc
        self._settle()

    def _settle(self) -> None:
        try:
            self.page.wait_for_load_state("networkidle", timeout=3000)
        except PlaywrightError:
            pass  # busy pages never go idle; the DOM is usable anyway

    def _element(self, element_id: int):
        locator = self.page.locator(f'[data-agent-id="{int(element_id)}"]')
        if locator.count() == 0:
            raise BrowserError(f"No element [{element_id}] on the current page. Use the latest numbers.")
        return locator.first


# Text of elements currently on screen (keeps observations short and relevant).
_VISIBLE_TEXT_JS = """
() => {
  const walker = document.createTreeWalker(document.body || document.documentElement, NodeFilter.SHOW_TEXT);
  const lines = [];
  let node;
  while ((node = walker.nextNode())) {
    const text = node.textContent.trim();
    if (!text) continue;
    const el = node.parentElement;
    if (!el || ['SCRIPT', 'STYLE', 'NOSCRIPT'].includes(el.tagName)) continue;
    const r = el.getBoundingClientRect();
    if (r.bottom < 0 || r.top > innerHeight || r.width === 0) continue;
    lines.push(text.replace(/\\s+/g, ' '));
  }
  return lines.join('\\n');
}
"""


def _short(exc: Exception) -> str:
    return str(exc).splitlines()[0][:200]
