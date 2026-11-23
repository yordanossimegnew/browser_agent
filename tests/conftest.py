"""Fixtures: a local test website and settings that let the browser visit it."""

from __future__ import annotations

import functools
import http.server
import threading
from pathlib import Path

import pytest

from src.utils.config import Settings

SITE = Path(__file__).parent / "site"


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # keep pytest output clean
        pass


@pytest.fixture(scope="session")
def site_url():
    handler = functools.partial(_QuietHandler, directory=str(SITE))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.fixture
def settings(tmp_path, site_url) -> Settings:
    return Settings(
        allow_private_hosts=True,  # the test site runs on localhost
        search_url=site_url + "/search.html?q={query}",
        sessions_dir=tmp_path / "sessions",
        knowledge_base_dir=tmp_path / "kb",
        log_dir=tmp_path / "logs",
        max_steps=6,
        page_timeout_ms=10000,
    )
