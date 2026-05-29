from verl.utils.reward_score.multiple_choice import compute_score, extract_choice_letter


def test_extracts_choice_prefixed_answer_with_explanation():
    response = """B. Gastroenteritis in children

Explanation: Rotavirus is a major cause of gastroenteritis, particularly in children."""

    assert extract_choice_letter(response) == "B"
    assert compute_score(response, "B") == 1.0


def test_extracts_tagged_answer():
    assert extract_choice_letter("<answer>C</answer>\nBecause ...") == "C"
    assert compute_score("<answer>C</answer>\nBecause ...", "C") == 1.0


def test_extracts_answer_letter_followed_by_text():
    response = "A possible answer: D because the passage directly supports that option."

    assert extract_choice_letter(response) == "D"
    assert compute_score(response, "D") == 1.0


def test_ignores_prompt_options_when_decoded_text_contains_assistant_marker():
    decoded = """Question?

A. First
B. Second
C. Third
D. Fourth

<|assistant_start|>B.
Second<|assistant_end|><pad><pad>"""

    assert extract_choice_letter(decoded) == "B"
    assert compute_score(decoded, "B") == 1.0


def test_does_not_guess_from_prompt_options_without_answer():
    prompt_only = """Question?

A. First
B. Second
C. Third
D. Fourth

Answer with the letter of the correct option."""

    assert extract_choice_letter(prompt_only) is None
    assert compute_score(prompt_only, "B") == 0.0
