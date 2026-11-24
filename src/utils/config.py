"""Configuration loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str) -> tuple[str, ...]:
    return tuple(x.strip().lower() for x in os.getenv(name, "").split(",") if x.strip())


@dataclass(frozen=True)
class Settings:
    model: str = "claude-opus-5-5"
    effort: str = "high"  # low | medium | high | xhigh | max
    max_tokens: int = 16000
    max_steps: int = 20
    headless: bool = True
    viewport_width: int = 1280
    viewport_height: int = 800
    page_timeout_ms: int = 20000
    search_url: str = "https://search.brave.com/search?q={query}"
    allow_private_hosts: bool = False  # block localhost / LAN addresses (SSRF guard)
    blocked_domains: tuple[str, ...] = ()
    max_concurrent_runs: int = 2
    log_level: str = "INFO"
    log_dir: Path = PROJECT_ROOT / "logs"
    database_url: str = "postgresql://postgres:postgres@localhost:5432/browser_agent"
    knowledge_base_dir: Path = PROJECT_ROOT / "data" / "knowledge_base"


def load_settings(**overrides) -> Settings:
    """Build Settings from env vars, then apply any non-None keyword overrides."""
    load_dotenv(PROJECT_ROOT / ".env")
    settings = Settings(
        model=os.getenv("AGENT_MODEL", Settings.model),
        effort=os.getenv("AGENT_EFFORT", Settings.effort),
        max_tokens=int(os.getenv("AGENT_MAX_TOKENS", Settings.max_tokens)),
        max_steps=int(os.getenv("AGENT_MAX_STEPS", Settings.max_steps)),
        headless=_env_bool("BROWSER_HEADLESS", Settings.headless),
        page_timeout_ms=int(os.getenv("BROWSER_TIMEOUT_MS", Settings.page_timeout_ms)),
        search_url=os.getenv("SEARCH_URL", Settings.search_url),
        allow_private_hosts=_env_bool("BROWSER_ALLOW_PRIVATE_HOSTS", Settings.allow_private_hosts),
        blocked_domains=_env_list("BROWSER_BLOCKED_DOMAINS"),
        max_concurrent_runs=int(os.getenv("MAX_CONCURRENT_RUNS", Settings.max_concurrent_runs)),
        log_level=os.getenv("LOG_LEVEL", Settings.log_level),
        database_url=os.getenv("DATABASE_URL", Settings.database_url),
    )
    return replace(settings, **{k: v for k, v in overrides.items() if v is not None})
