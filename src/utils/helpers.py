"""Small shared helpers."""

from __future__ import annotations

import re


def truncate(text: str, limit: int) -> str:
    """Keep the head and tail of long text so the model sees both ends."""
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n\n... [{len(text) - limit} characters omitted] ...\n\n{text[-half:]}"


def clean_text(text: str) -> str:
    """Collapse runs of blank lines and trailing spaces in scraped page text."""
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
