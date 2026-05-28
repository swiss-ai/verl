# Copyright 2026 The VERL Team and individual contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def extract_choice_letter(solution_str: str) -> str | None:
    text = _normalize_text(solution_str).upper()
    patterns = [
        r"\\BOXED\{\s*([A-Z])\s*\}",
        r"\bANSWER\s*[:\-]\s*([A-Z])\b",
        r"\bFINAL\s+ANSWER\s*[:\-]?\s*([A-Z])\b",
        r"\bTHE\s+ANSWER\s+IS\s+([A-Z])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    matches = re.findall(r"\b([A-Z])\b", text)
    return matches[-1] if matches else None


def compute_score(solution_str: str, ground_truth: Any, format_score: float = 0.0, score: float = 1.0) -> float:
    predicted = extract_choice_letter(solution_str)
    gold = _normalize_text(ground_truth).upper()
    if predicted is None:
        return format_score
    return score if predicted == gold else 0.0
