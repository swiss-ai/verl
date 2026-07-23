from __future__ import annotations

from typing import Any

import numpy as np


def normalize_agentic_forced_tokens(value: Any) -> list[dict[str, Any]]:
    """Normalize backend forced-token metadata into a list of dictionaries."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def merge_agentic_forced_tokens(*values: Any) -> list[dict[str, Any]]:
    """Concatenate normalized forced-token metadata from generation chunks."""
    merged: list[dict[str, Any]] = []
    for value in values:
        merged.extend(normalize_agentic_forced_tokens(value))
    return merged


def is_degeneration_stopped(value: Any) -> bool:
    """Return whether metadata contains an authoritative ``degen_stop`` marker."""
    return any(item.get("reason") == "degen_stop" for item in normalize_agentic_forced_tokens(value))


__all__ = [
    "is_degeneration_stopped",
    "merge_agentic_forced_tokens",
    "normalize_agentic_forced_tokens",
]
