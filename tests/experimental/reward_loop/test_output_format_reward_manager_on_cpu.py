from __future__ import annotations

import asyncio

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.output_format import OutputFormatRewardManager


def _config(*, format_bonus=0.05):
    values = {
        "output_format": {
            "enabled": True,
            "success_threshold": 0.7,
            "format_penalty": 0.1,
            "format_bonus": format_bonus,
        },
        "reward": {"reward_kwargs": {"overlong_buffer_cfg": None, "max_resp_len": None}},
    }
    return OmegaConf.create(values)


def _data(*, response_text="answer", format_valid=True, forced_tokens=None, validate=False):
    data = DataProto.from_dict(
        {
            "responses": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }
    )
    data.non_tensor_batch = {
        "data_source": np.array(["test/source"], dtype=object),
        "reward_model": np.array([{"ground_truth": "answer"}], dtype=object),
        "extra_info": np.array([{}], dtype=object),
        "tool_extra_fields": np.array(
            [
                {
                    "response_text": [response_text],
                    "format_valid": format_valid,
                    "agentic_forced_tokens": forced_tokens or [],
                    "validate": validate,
                }
            ],
            dtype=object,
        ),
    }
    return data


@pytest.mark.parametrize("format_valid", [False, True])
def test_degeneration_hard_zero_skips_verifier_and_penalties(format_valid):
    calls = 0

    def verifier(**kwargs):
        nonlocal calls
        calls += 1
        return {"score": 1.0, "acc": 1.0}

    manager = OutputFormatRewardManager(_config(), tokenizer=object(), compute_score=verifier)
    result = asyncio.run(
        manager.run_single(
            _data(
                format_valid=format_valid,
                forced_tokens=[{"reason": "degen_stop", "token_id": 2}],
            )
        )
    )
    assert calls == 0
    assert result["reward_score"] == 0.0
    assert result["reward_extra_info"] == {
        "score": 0.0,
        "task_score": 0.0,
        "raw_task_score": 0.0,
        "task_success": 0.0,
        "acc": 0.0,
        "format_valid": float(format_valid),
        "format_penalty": 0.0,
        "format_bonus": 0.0,
        "optimization_reward": 0.0,
        "degeneration_stopped": 1.0,
        "verifier_skipped": 1.0,
    }


def test_degeneration_hard_zero_also_skips_validation_verifier():
    calls = 0

    def verifier(**kwargs):
        nonlocal calls
        calls += 1
        return {"score": 1.0}

    manager = OutputFormatRewardManager(_config(), tokenizer=object(), compute_score=verifier)
    result = asyncio.run(
        manager.run_single(
            _data(
                forced_tokens=[{"reason": "degen_stop", "token_id": 2}],
                validate=True,
            )
        )
    )

    assert calls == 0
    assert result["reward_score"] == 0.0
    assert result["reward_extra_info"]["degeneration_stopped"] == 1.0
    assert result["reward_extra_info"]["verifier_skipped"] == 1.0


def test_valid_output_keeps_penalty_free_accuracy():
    async def verifier(**kwargs):
        assert kwargs["solution_str"] == "answer"
        return {"score": 1.0, "acc": 1.0, "detail": "ok"}

    manager = OutputFormatRewardManager(_config(), tokenizer=object(), compute_score=verifier)
    result = asyncio.run(manager.run_single(_data(format_valid=True)))
    extras = result["reward_extra_info"]
    assert result["reward_score"] == 1.0
    assert extras["task_score"] == 1.0
    assert extras["raw_task_score"] == 1.0
    assert extras["task_success"] == 1.0
    assert extras["acc"] == 1.0
    assert extras["format_penalty"] == 0.0
    assert extras["format_bonus"] == 0.0
    assert extras["optimization_reward"] == 1.0


def test_malformed_nondegenerate_output_gets_binary_format_penalty():
    async def verifier(**kwargs):
        return {"score": 1.0, "acc": 1.0}

    manager = OutputFormatRewardManager(_config(), tokenizer=object(), compute_score=verifier)
    result = asyncio.run(manager.run_single(_data(format_valid=False)))
    extras = result["reward_extra_info"]
    assert extras["task_score"] == 1.0
    assert extras["acc"] == 1.0
    assert extras["format_penalty"] == pytest.approx(-0.1)
    assert extras["format_bonus"] == 0.0
    assert result["reward_score"] == pytest.approx(0.9)
    assert extras["optimization_reward"] == pytest.approx(0.9)


@pytest.mark.parametrize(
    ("raw_score", "format_valid", "task_success", "expected_reward", "expected_penalty", "expected_bonus"),
    [
        (0.0, False, 0.0, 0.0, 0.0, 0.0),
        (0.0, True, 0.0, 0.05, 0.0, 0.05),
        (0.667, False, 0.0, 0.0, 0.0, 0.0),
        (0.667, True, 0.0, 0.05, 0.0, 0.05),
        (0.7, False, 1.0, 0.9, -0.1, 0.0),
        (0.7, True, 1.0, 1.0, 0.0, 0.0),
        (0.75, False, 1.0, 0.9, -0.1, 0.0),
        (0.75, True, 1.0, 1.0, 0.0, 0.0),
        (1.0, False, 1.0, 0.9, -0.1, 0.0),
        (1.0, True, 1.0, 1.0, 0.0, 0.0),
    ],
)
def test_training_binarizes_raw_score_and_applies_format_shaping(
    raw_score, format_valid, task_success, expected_reward, expected_penalty, expected_bonus
):
    async def verifier(**kwargs):
        return {"score": raw_score, "acc": 0.123}

    manager = OutputFormatRewardManager(_config(), tokenizer=object(), compute_score=verifier)
    result = asyncio.run(manager.run_single(_data(format_valid=format_valid)))
    extras = result["reward_extra_info"]
    assert extras["score"] == pytest.approx(raw_score)
    assert extras["task_score"] == pytest.approx(raw_score)
    assert extras["raw_task_score"] == pytest.approx(raw_score)
    assert extras["task_success"] == task_success
    assert extras["acc"] == task_success
    assert extras["format_penalty"] == pytest.approx(expected_penalty)
    assert extras["format_bonus"] == pytest.approx(expected_bonus)
    assert result["reward_score"] == pytest.approx(expected_reward)
    assert extras["optimization_reward"] == pytest.approx(expected_reward)


def test_validation_scores_raw_output_without_format_penalty():
    raw_response = "### Reasoning\nwork\n\n### Response\nanswer"

    async def verifier(**kwargs):
        assert kwargs["solution_str"] == raw_response
        return {"score": 0.75, "acc": 0.25}

    manager = OutputFormatRewardManager(_config(), tokenizer=object(), compute_score=verifier)
    result = asyncio.run(
        manager.run_single(
            _data(
                response_text=raw_response,
                format_valid=False,
                validate=True,
            )
        )
    )
    extras = result["reward_extra_info"]
    assert result["reward_score"] == 0.75
    assert extras["task_score"] == 0.75
    assert extras["raw_task_score"] == 0.75
    assert extras["task_success"] == 1.0
    assert extras["acc"] == 1.0
    assert extras["optimization_reward"] == 0.75
    assert "format_valid" not in extras
    assert "format_penalty" not in extras
    assert "format_bonus" not in extras


def test_natural_eos_metadata_does_not_trigger_zeroing():
    calls = 0

    async def verifier(**kwargs):
        nonlocal calls
        calls += 1
        return 0.5

    manager = OutputFormatRewardManager(_config(), tokenizer=object(), compute_score=verifier)
    result = asyncio.run(manager.run_single(_data(forced_tokens=[{"reason": "some_other_forcing", "token_id": 2}])))
    assert calls == 1
    assert result["reward_score"] == 0.05
    assert result["reward_extra_info"]["raw_task_score"] == 0.5
    assert result["reward_extra_info"]["task_success"] == 0.0
    assert result["reward_extra_info"]["degeneration_stopped"] == 0.0
