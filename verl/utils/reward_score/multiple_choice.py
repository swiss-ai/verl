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

_ANSWER_MARKER = re.compile(
    r"""
    \b(?:
        (?:the\s+)?(?:(?:final|correct)\s+)?answer
        | (?:the\s+)?correct\s+(?:option|choice)
    )
    (?:\s+is)?\s*[:\-]?\s*
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)
_WRAPPER_PREFIX = re.compile(
    r"""
    ^(
        \s+
        | [*_`~$"']+
        | \\(?:boxed|text)\s*\{\s*
        | \\?[\[\(\{]\s*
    )+
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)
_CHOICE_LETTER = re.compile(
    r"(?i)^([A-Z])(?=$|[^\w])"
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


def _choice_from_fragment(fragment: str) -> str | None:
    """Extract a choice letter after common Markdown/LaTeX wrappers."""
    fragment = _strip_wrapper_prefix(fragment)

    match = _CHOICE_LETTER.match(fragment)
    return match.group(1).upper() if match else None


def _strip_wrapper_prefix(fragment: str) -> str:
    """Remove common wrappers that precede an answer letter."""
    for _ in range(8):
        cleaned = _WRAPPER_PREFIX.sub("", fragment, count=1)
        if cleaned == fragment:
            break
        fragment = cleaned
    return fragment


def _explicit_choice_candidates(text: str) -> list[tuple[int, str]]:
    candidates = []
    for match in _ANSWER_MARKER.finditer(text):
        letter = _choice_from_fragment(text[match.end() :])
        if letter is not None:
            candidates.append((match.start(), letter))
    return candidates


def extract_choice_letter(solution_str: str) -> str | None:
    """Extract an answer following the MCQA prompt contract.

    Prefer the last valid explicit answer declaration in the final response
    region so a later correction wins. Common Markdown/LaTeX wrappers around
    the answer letter are accepted. Also accept a bare final answer line and
    completions beginning with an option-style answer such as ``C. option
    text``.
    """
    text = _completion_text(solution_str)

    # Keep old reasoning from dominating the result while allowing long final
    # explanations and answer blocks. Invalid markers such as ``Answer:
    # $letter`` are ignored, allowing a later valid declaration to win.
    final_region = text[-2048:]
    explicit_answers = _explicit_choice_candidates(final_region)
    if not explicit_answers:
        # Preserve compatibility with long completions whose explicit answer
        # occurs before the final response region.
        explicit_answers = _explicit_choice_candidates(text)
    if explicit_answers:
        return explicit_answers[-1][1]

    final_line = text.rsplit("\n", 1)[-1]
    bare_line = _strip_wrapper_prefix(final_line)
    bare_answer = re.fullmatch(
        r"([A-Z])\s*[\).]?\s*[*_`~$\\\]\}]*",
        bare_line,
        flags=re.IGNORECASE,
    )
    if bare_answer:
        return bare_answer.group(1).upper()

    option_start = re.match(r"^\s*([A-Z])\s*[\).:\-]\s+\S", text)
    return option_start.group(1).upper() if option_start else None


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
