import json
import os
import time

import pytest

from examples.data_preprocess import apertus_demo_rl as preprocess
from verl.utils.reward_score import default_compute_score


def taco_call_based_example():
    return {
        "question": "Write a function that adds two integers.",
        "solutions": ["def add(a, b):\n    return a + b\n"],
        "starter_code": "def add(a, b):",
        "input_output": json.dumps(
            {
                "fn_name": "add",
                "inputs": [[1, 2], [5, 7]],
                "outputs": [[3], [12]],
            }
        ),
        "difficulty": "EASY",
        "source": "unit",
        "name": "add",
        "url": None,
    }


def humaneval_example():
    return {
        "task_id": "HumanEval/Unit",
        "prompt": 'def add(a: int, b: int) -> int:\n    """Return the sum of a and b."""\n',
        "canonical_solution": "    return a + b\n",
        "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(-1, 1) == 0\n",
        "entry_point": "add",
    }


def test_taco_preprocessing_is_prime_code_compatible():
    config = next(config for config in preprocess.TRAIN_DATASETS if config.name == "taco_verified")
    row = preprocess.adapt_taco(taco_call_based_example(), 0, "train", config)

    assert row["data_source"] == "taco"
    assert row["ability"] == "code"
    test_cases = json.loads(row["reward_model"]["ground_truth"])
    assert test_cases == {
        "fn_name": "add",
        "inputs": ["1\n2", "5\n7"],
        "outputs": ["[3]", "[12]"],
    }


def test_taco_prime_code_verification_works():
    config = next(config for config in preprocess.TRAIN_DATASETS if config.name == "taco_verified")
    row = preprocess.adapt_taco(taco_call_based_example(), 0, "train", config)
    score = default_compute_score(
        row["data_source"],
        "def add(a, b):\n    return a + b\n",
        row["reward_model"]["ground_truth"],
    )
    assert score == 1.0


def test_humaneval_preprocessing_is_prime_code_compatible():
    config = next(config for config in preprocess.EVAL_DATASETS if config.name == "openai_humaneval")
    row = preprocess.adapt_humaneval(humaneval_example(), 0, "test", config)

    assert row["data_source"] == "humaneval"
    assert row["ability"] == "code"
    test_cases = json.loads(row["reward_model"]["ground_truth"])
    assert set(test_cases) == {"prompt", "test", "entry_point"}
    assert test_cases["prompt"].endswith("\n")


def test_humaneval_prime_code_verification_works():
    config = next(config for config in preprocess.EVAL_DATASETS if config.name == "openai_humaneval")
    row = preprocess.adapt_humaneval(humaneval_example(), 0, "test", config)
    score = default_compute_score(
        row["data_source"],
        humaneval_example()["canonical_solution"],
        row["reward_model"]["ground_truth"],
    )
    assert score == 1.0


def test_prime_code_verification_speed_smoke():
    taco_config = next(config for config in preprocess.TRAIN_DATASETS if config.name == "taco_verified")
    humaneval_config = next(config for config in preprocess.EVAL_DATASETS if config.name == "openai_humaneval")
    cases = [
        (
            preprocess.adapt_taco(taco_call_based_example(), 0, "train", taco_config),
            "def add(a, b):\n    return a + b\n",
        ),
        (
            preprocess.adapt_humaneval(humaneval_example(), 0, "test", humaneval_config),
            humaneval_example()["canonical_solution"],
        ),
        (
            preprocess.adapt_humaneval(humaneval_example(), 1, "test", humaneval_config),
            "```python\n    return a + b\n```",
        ),
    ]

    started = time.perf_counter()
    scores = [
        default_compute_score(row["data_source"], solution, row["reward_model"]["ground_truth"])
        for row, solution in cases
    ]
    elapsed = time.perf_counter() - started

    assert scores == [1.0, 1.0, 1.0]
    print(f"prime_code verified {len(cases)} examples in {elapsed:.3f}s ({elapsed / len(cases):.3f}s/example)")


def test_humaneval_prime_code_matches_hf_evaluate_code_eval():
    if os.environ.get("HF_ALLOW_CODE_EVAL") != "1":
        pytest.skip("HF evaluate code_eval requires HF_ALLOW_CODE_EVAL=1")
    evaluate = pytest.importorskip("evaluate")

    config = next(config for config in preprocess.EVAL_DATASETS if config.name == "openai_humaneval")
    row = preprocess.adapt_humaneval(humaneval_example(), 0, "test", config)
    completion = humaneval_example()["canonical_solution"]
    prime_code_score = default_compute_score(row["data_source"], completion, row["reward_model"]["ground_truth"])

    metric = evaluate.load("code_eval")
    candidate = humaneval_example()["prompt"] + completion
    pass_at_k, _ = metric.compute(
        references=[humaneval_example()["test"]],
        predictions=[[candidate]],
        k=[1],
    )

    assert prime_code_score == 1.0
    assert pass_at_k["pass@1"] == 1.0
