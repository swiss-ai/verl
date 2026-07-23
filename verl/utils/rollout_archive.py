from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def json_safe(value: Any) -> Any:
    """Recursively convert tensors, arrays, and scalar wrappers to JSON-safe values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class JsonlArchive:
    """Append records to one JSONL file without overwriting existing contents."""

    def __init__(self, path: str | Path):
        """Create an archive that appends to ``path``, creating its parent directory."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, records: Iterable[dict[str, Any]]) -> None:
        """Serialize and append records, writing one JSON object per line."""
        payload = "".join(json.dumps(json_safe(record), ensure_ascii=False) + "\n" for record in records)
        if not payload:
            return
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(payload)


__all__ = ["JsonlArchive", "json_safe"]
