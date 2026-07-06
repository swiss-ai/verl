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

_ANSWER_PATTERN = re.compile(
    r"\b(?:FINAL\s+)?ANSWER(?:\s+IS)?\s*[:\-]?\s*([A-Z])\b",
    flags=re.IGNORECASE,
)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _completion_text(solution_str: str) -> str:
    text = _normalize_text(solution_str)
    for marker in ("<|assistant_start|>", "<|im_start|>assistant", "assistant\n"):
        if marker in text:
            text = text.rsplit(marker, 1)[-1]
    for marker in ("<|assistant_end|>", "<|im_end|>"):
        if marker in text:
            text = text.split(marker, 1)[0]
    return re.sub(r"(?:<pad>|\s)+$", "", text, flags=re.IGNORECASE).strip()


def extract_choice_letter(solution_str: str) -> str | None:
    """Extract an answer following the MCQA prompt contract.

    Prefer the last explicit ``Answer: C`` declaration so a later correction
    wins. Also accept a bare final answer line and completions beginning with
    an option-style answer such as ``C. option text``.
    """
    text = _completion_text(solution_str).upper()

    explicit_answers = _ANSWER_PATTERN.findall(text)
    if explicit_answers:
        return explicit_answers[-1]

    final_line = text.rsplit("\n", 1)[-1]
    bare_answer = re.fullmatch(r"\s*([A-Z])\s*[\).]?\s*", final_line)
    if bare_answer:
        return bare_answer.group(1)

    option_start = re.match(r"^\s*([A-Z])\s*[\).:\-]\s+\S", text)
    return option_start.group(1) if option_start else None


def compute_score(
    solution_str: str,
    ground_truth: Any,
    format_score: float = 0.0,
    score: float = 1.0,
    data_source: str | None = None,
) -> float:
    del data_source
    predicted = extract_choice_letter(solution_str)
    gold = _normalize_text(ground_truth).upper()
    if predicted is None or re.fullmatch(r"[A-Z]", gold) is None:
        return format_score
    return score if predicted == gold else 0.0
