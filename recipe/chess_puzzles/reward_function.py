"""
Reward function for chess puzzles (ApertusChessGym/ChessPuzzles).

The dataset expects the model to respond with exactly one UCI move token,
like "e2e4" or "e7e8q". We extract the last valid UCI token from the
model output and compare to the ground truth.
"""

import re
from typing import Optional


def extract_solution(solution_str: str, method: str = "strict") -> Optional[str]:
    """Extract the last UCI move token from the solution string.

    Accepts 4 or 5 character UCI moves, optional promotion piece at the end.
    Examples: e2e4, e7e8q
    methods:
    - strict: match the last token after ####, e.g., `#### e2e4` or `#### e7e8q`
    - flexible: match the last UCI-like token anywhere in the text
    """
    assert method in ("strict", "flexible")

    if solution_str is None:
        return None
    text = solution_str.strip()

    uci_pattern = r"([a-h][1-8][a-h][1-8][qrbn]?)"

    if method == "strict":
        # Require #### followed by the UCI token
        strict_re = re.compile(r"####\s*" + uci_pattern, re.IGNORECASE)
        matches = strict_re.findall(text)
    else:
        # Fallback: find any UCI token
        flex_re = re.compile(uci_pattern, re.IGNORECASE)
        matches = flex_re.findall(text)

    if not matches:
        return None
    return matches[-1].lower()


def compute_score(
    data_source: str | None = None,
    solution_str: str = "",
    ground_truth: str = "",
    extra_info: dict | None = None,
    method: str = "strict",
    **kwargs,
) -> float:
    """Config-based reward entrypoint.
    1.0 if '#### <uci>' matches ground_truth, else 0.0.
    """
    answer = extract_solution(solution_str, method=method)
    gt = (ground_truth or "").strip().lower()

    return 1.0 if answer == gt else 0.0