import base64
import json
import pickle
import zlib

from experiments.code.preprocess_livecodebench import build_rows, build_test_cases, decode_test_cases, split_test_cases
from verl.utils.reward_score import default_compute_score


def _compressed_payload(value):
    return base64.b64encode(zlib.compress(pickle.dumps(json.dumps(value)))).decode("utf-8")


def test_decode_json_and_compressed_test_cases():
    public_cases = [{"input": "1 2\n", "output": "3\n"}]
    private_cases = [{"input": "4 5\n", "output": "9\n"}]

    assert decode_test_cases(json.dumps(public_cases)) == public_cases
    assert decode_test_cases(_compressed_payload(private_cases)) == private_cases


def test_build_test_cases_for_stdin_and_functional_examples():
    stdin_example = {
        "public_test_cases": json.dumps([{"input": "1 2\n", "output": "3\n"}]),
        "private_test_cases": json.dumps([]),
        "generated_test_cases": json.dumps([]),
        "metadata": json.dumps({}),
    }
    stdin_cases = build_test_cases(stdin_example)
    assert stdin_cases["testtype"] == "stdin"
    assert stdin_cases["fn_name"] == ""

    functional_example = {
        "public_test_cases": json.dumps([{"input": [1, 2], "output": 3}]),
        "private_test_cases": json.dumps([]),
        "generated_test_cases": json.dumps([]),
        "metadata": json.dumps({"func_name": "add"}),
    }
    functional_cases = build_test_cases(functional_example)
    assert functional_cases["testtype"] == "functional"
    assert functional_cases["fn_name"] == "add"
    assert functional_cases["inputs"] == [[1, 2]]
    assert functional_cases["outputs"] == [3]


def test_split_test_cases_is_deterministic():
    test_cases = {
        "inputs": ["0\n", "1\n", "2\n", "3\n"],
        "outputs": ["0\n", "1\n", "2\n", "3\n"],
        "fn_name": "",
        "testtype": "stdin",
        "time_limit": 1,
    }
    split_a, stats_a = split_test_cases(test_cases, fraction=0.5, seed=7, stable_id="abc")
    split_b, stats_b = split_test_cases(test_cases, fraction=0.5, seed=7, stable_id="abc")

    assert split_a == split_b
    assert stats_a == stats_b
    assert stats_a["visible_test_count"] == 2
    assert stats_a["heldout_test_count"] == 2


def test_build_rows_filters_difficulty_and_writes_visible_train_tests():
    examples = [
        {
            "question_id": "hard-1",
            "question_content": "Add two numbers.",
            "difficulty": "hard",
            "platform": "codeforces",
            "contest_date": "2025-01-01T00:00:00",
            "starter_code": "",
            "public_test_cases": json.dumps(
                [
                    {"input": "1 2\n", "output": "3\n"},
                    {"input": "3 4\n", "output": "7\n"},
                    {"input": "5 6\n", "output": "11\n"},
                    {"input": "7 8\n", "output": "15\n"},
                ]
            ),
            "private_test_cases": json.dumps([]),
            "generated_test_cases": json.dumps([]),
            "metadata": json.dumps({}),
        },
        {
            "question_id": "easy-1",
            "question_content": "Print one.",
            "difficulty": "easy",
            "public_test_cases": json.dumps([{"input": "", "output": "1\n"}]),
            "private_test_cases": json.dumps([]),
            "generated_test_cases": json.dumps([]),
            "metadata": json.dumps({}),
        },
    ]

    train_rows, test_rows, eval_rows, stats = build_rows(
        examples,
        difficulty="hard",
        train_test_fraction=0.5,
        seed=42,
        max_samples=-1,
    )

    assert stats["processed"] == 1
    assert stats["filtered_difficulty"] == 1
    assert len(train_rows) == len(test_rows) == len(eval_rows) == 1
    train_gt = json.loads(train_rows[0]["reward_model"]["ground_truth"])
    test_gt = json.loads(test_rows[0]["reward_model"]["ground_truth"])
    assert len(train_gt["inputs"]) == 2
    assert len(test_gt["inputs"]) == 4
    assert train_rows[0]["data_source"] == "livecodebench"
    assert train_rows[0]["extra_info"]["visible_test_count"] == 2
    assert test_rows[0]["extra_info"]["split"] == "test"
    assert eval_rows[0]["extra_info"]["split"] == "eval"


def test_livecodebench_reward_correct_wrong_runtime_and_format():
    ground_truth = json.dumps(
        {
            "inputs": ["1 2\n", "5 7\n"],
            "outputs": ["3\n", "12\n"],
            "fn_name": "",
            "testtype": "stdin",
            "time_limit": 1,
        }
    )
    correct = "```python\nimport sys\nprint(sum(map(int, sys.stdin.read().split())))\n```"
    wrong = "```python\nprint(0)\n```"
    runtime_error = "```python\nraise ValueError('boom')\n```"

    correct_score = default_compute_score("livecodebench", correct, ground_truth, {"split": "test"})
    wrong_score = default_compute_score("livecodebench", wrong, ground_truth, {"split": "test"})
    format_score = default_compute_score("livecodebench", "no code block", ground_truth, {"split": "test"})
    runtime_score = default_compute_score("livecodebench", runtime_error, ground_truth, {"split": "test"})

    assert correct_score["score"] == 1.0
    assert correct_score["acc"] == 1.0
    assert correct_score["pass_fraction"] == 1.0
    assert wrong_score["score"] == 0.0
    assert format_score["incorrect_format"] == 1
    assert runtime_score["runtime_error"] == 1


def test_livecodebench_reward_functional_solution():
    ground_truth = json.dumps(
        {
            "inputs": ["1 2"],
            "outputs": ["3"],
            "fn_name": "add",
            "testtype": "functional",
            "time_limit": 1,
        }
    )
    solution = "```python\ndef add(a, b):\n    return a + b\n```"

    score = default_compute_score("livecodebench/code_generation_lite-v6", solution, ground_truth, {"split": "test"})

    assert score["score"] == 1.0
    assert score["acc"] == 1.0


def test_livecodebench_reward_functional_native_args():
    ground_truth = json.dumps(
        {
            "inputs": [["hello world", "!"]],
            "outputs": ["hello world!"],
            "fn_name": "join",
            "testtype": "functional",
            "time_limit": 1,
        }
    )
    solution = "```python\ndef join(a, b):\n    return a + b\n```"

    score = default_compute_score("livecodebench", solution, ground_truth, {"split": "test"})

    assert score["score"] == 1.0
    assert score["acc"] == 1.0


def test_livecodebench_reward_timeout():
    ground_truth = json.dumps(
        {
            "inputs": ["\n"],
            "outputs": ["\n"],
            "fn_name": "",
            "testtype": "stdin",
            "time_limit": 0.01,
        }
    )
    solution = "```python\nwhile True:\n    pass\n```"

    score = default_compute_score("livecodebench", solution, ground_truth, {"split": "test"})

    assert score["score"] == 0.0
    assert score["timed_out"] == 1
