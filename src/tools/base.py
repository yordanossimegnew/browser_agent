"""Tool definition shared by all browser tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Any]  # handler(browser, **args)
    observes: bool = True  # return a fresh page observation (text + screenshot) afterwards

    def to_api(self) -> dict[str, Any]:
        """Claude tool definition (strict schema validation on)."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "strict": True,
        }


def schema(properties: dict[str, dict]) -> dict[str, Any]:
    """Strict JSON schema where every listed property is required."""
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
