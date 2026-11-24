"""Fixtures: a local test website and settings that let the browser visit it."""

from __future__ import annotations

import functools
import http.server
import os
import threading
from pathlib import Path

import psycopg
import pytest
from dotenv import load_dotenv

from src.utils.config import Settings

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")


def fresh_database() -> str:
    """The test database with LangGraph's checkpoint tables emptied, so every test starts clean."""
    if not TEST_DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL is not set: point it at an empty PostgreSQL database (see .env.example)")
    with psycopg.connect(TEST_DATABASE_URL, autocommit=True) as conn:
        conn.execute("DO $$ BEGIN IF to_regclass('checkpoints') IS NOT NULL THEN "
                     "TRUNCATE checkpoints, checkpoint_blobs, checkpoint_writes; END IF; END $$")
    return TEST_DATABASE_URL

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
        database_url=fresh_database(),
        knowledge_base_dir=tmp_path / "kb",
        log_dir=tmp_path / "logs",
        max_steps=6,
        page_timeout_ms=10000,
    )
