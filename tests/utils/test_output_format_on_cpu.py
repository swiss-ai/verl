from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    AgentLoopWorker,
)
from verl.experimental.fully_async_policy.detach_utils import (
    RolloutSample,
    should_keep_async_filter_group,
    should_keep_output_format_group,
    validate_async_filter_groups_config,
)
from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
from verl.utils.generation_metadata import is_degeneration_stopped, merge_agentic_forced_tokens
from verl.utils.output_format import (
    output_formatting_config,
    validate_output_format_agent_config,
    validate_output_format_reward_config,
)
from verl.utils.output_formatting import (
    MARKDOWN_PARSER,
    USER_PROMPT_ROLE,
    XML_PARSER,
    XML_THINK_PARSER,
    OutputFormattingConfig,
    add_formatting_instruction,
    parse_formatted_output,
)
from verl.utils.rollout_archive import JsonlArchive


def _agent_config(*, enabled=True, enable_thinking=False, parser=MARKDOWN_PARSER, prompt_role="system"):
    return OmegaConf.create(
        {
            "output_format": {
                "enabled": enabled,
                "parser": parser,
                "prompt": None,
                "prompt_role": prompt_role,
            },
            "data": {
                "force_thinking_prefix": False,
                "apply_chat_template_kwargs": {"enable_thinking": enable_thinking},
            },
            "actor_rollout_ref": {
                "rollout": {
                    "multi_turn": {"enable": False},
                    "agent": {"default_agent_loop": "single_turn_agent"},
                }
            },
        }
    )


def test_output_format_rejects_no_format_instead_of_overriding_it():
    with pytest.raises(ValueError, match="incompatible with NO_FORMAT=true"):
        validate_output_format_agent_config(_agent_config(), no_format=True)


def test_agent_loop_worker_rejects_no_format_with_output_format(monkeypatch):
    monkeypatch.setenv("NO_FORMAT", "true")
    with pytest.raises(ValueError, match="incompatible with NO_FORMAT=true"):
        AgentLoopWorker(_agent_config(), llm_client=None)


def test_no_format_remains_valid_when_output_format_is_disabled():
    validate_output_format_agent_config(_agent_config(enabled=False), no_format=True)


def test_enable_thinking_is_independent_from_output_formatting():
    validate_output_format_agent_config(_agent_config(enable_thinking=True), no_format=False)


def test_output_format_requires_an_enabled_formatter():
    with pytest.raises(ValueError, match="parser='none'"):
        validate_output_format_agent_config(_agent_config(parser="none"), no_format=False)


def test_removed_system_prompt_config_is_rejected():
    config = _agent_config()
    config.output_format.system_prompt = "legacy"

    with pytest.raises(ValueError, match="system_prompt was removed"):
        validate_output_format_agent_config(config, no_format=False)


def test_validation_prompt_does_not_receive_output_format_instruction():
    original = [{"role": "system", "content": "Original"}, {"role": "user", "content": "Question"}]
    agent_loop = SimpleNamespace(
        output_format_enabled=True,
        output_formatting=OutputFormattingConfig(parser=MARKDOWN_PARSER, prompt="Output template"),
    )

    prepared = AgentLoopBase.prepare_messages(agent_loop, original, validate=True)

    assert prepared == original
    assert prepared is not original
    assert prepared[0] is not original[0]


def test_validation_response_bypasses_output_format_parser():
    raw_response = "### Reasoning\nwork\n\n### Response\nanswer"
    worker = SimpleNamespace(
        config=_agent_config(),
        output_formatting=output_formatting_config(_agent_config()),
        tokenizer=SimpleNamespace(decode=lambda *_args, **_kwargs: raw_response),
    )
    output = AgentLoopOutput(
        prompt_ids=[],
        response_ids=[1],
        response_mask=[1],
        metrics=AgentLoopMetrics(),
    )

    AgentLoopWorker._set_response_text_fields(worker, output, validate=True)

    assert output.extra_fields["raw_response_text"] == raw_response
    assert output.extra_fields["response_text"] == [raw_response]
    assert "reasoning_text" not in output.extra_fields
    assert "final_response_text" not in output.extra_fields
    assert "format_valid" not in output.extra_fields
    assert "parser_outcome" not in output.extra_fields


@pytest.mark.parametrize(
    ("parser", "raw_response", "reasoning", "answer"),
    [
        (
            MARKDOWN_PARSER,
                "### Reasoning\nwork\n### Response\nanswer",
            "work",
            "answer",
        ),
        (
            XML_PARSER,
            "<reasoning>work</reasoning><answer>answer</answer>",
            "work",
            "answer",
        ),
        (
            XML_THINK_PARSER,
            "<think>work</think>answer",
            "work",
            "answer",
        ),
    ],
)
def test_training_response_uses_selected_formatter(parser, raw_response, reasoning, answer):
    config = _agent_config(parser=parser)
    worker = SimpleNamespace(
        config=config,
        output_formatting=output_formatting_config(config),
        tokenizer=SimpleNamespace(decode=lambda *_args, **_kwargs: raw_response),
    )
    output = AgentLoopOutput(
        prompt_ids=[],
        response_ids=[1],
        response_mask=[1],
        metrics=AgentLoopMetrics(),
    )

    AgentLoopWorker._set_response_text_fields(worker, output, validate=False)

    assert output.extra_fields["reasoning_text"] == reasoning
    assert output.extra_fields["final_response_text"] == answer
    assert output.extra_fields["response_text"] == [answer]
    assert output.extra_fields["format_valid"] is True
    assert output.extra_fields["output_format_parser"] == parser


def test_user_prompt_role_appends_instruction_to_last_user_message():
    original = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "question"},
    ]
    agent_loop = SimpleNamespace(
        output_format_enabled=True,
        output_formatting=OutputFormattingConfig(
            parser=XML_PARSER,
            prompt="Use XML.",
            prompt_role=USER_PROMPT_ROLE,
        ),
    )

    prepared = AgentLoopBase.prepare_messages(agent_loop, original, validate=False)

    assert prepared[-1]["content"] == "question\n\nUse XML."
    assert original[-1]["content"] == "question"


def test_strict_semantic_template_parsing():
    parsed = parse_formatted_output(
        "### Reasoning\n2 + 2 is 4.\n\n### Response\n4",
        OutputFormattingConfig(parser=MARKDOWN_PARSER),
    )
    assert parsed.format_valid is True
    assert parsed.outcome == "valid"
    assert parsed.reasoning == "2 + 2 is 4."
    assert parsed.final_response == "4"
    assert parsed.verifier_response == "4"


def test_malformed_template_recovers_last_nonempty_final_section():
    parsed = parse_formatted_output(
        "preface\n### Response\nwrong\n### Response\nanswer",
        OutputFormattingConfig(parser=MARKDOWN_PARSER),
    )
    assert parsed.format_valid is False
    assert parsed.outcome == "recovered_final_response"
    assert parsed.verifier_response == "answer"


def test_missing_or_empty_final_section_falls_back_to_raw_output():
    text = "### Reasoning\nwork\n### Response\n"
    parsed = parse_formatted_output(text, OutputFormattingConfig(parser=MARKDOWN_PARSER))
    assert parsed.format_valid is False
    assert parsed.outcome == "raw_fallback"
    assert parsed.final_response == ""
    assert parsed.verifier_response == text.strip()


def test_heading_with_extra_text_is_not_strictly_valid():
    text = "### Reasoning but not a heading\nwork\n### Response\nanswer"
    parsed = parse_formatted_output(text, OutputFormattingConfig(parser=MARKDOWN_PARSER))
    assert parsed.format_valid is False
    assert parsed.outcome == "recovered_final_response"
    assert parsed.verifier_response == "answer"


def test_prompt_injection_copies_and_preserves_existing_system_content():
    original = [{"role": "system", "content": "Original"}, {"role": "user", "content": "Question"}]
    injected = add_formatting_instruction(
        original,
        OutputFormattingConfig(parser=MARKDOWN_PARSER, prompt="Template"),
    )
    assert original[0]["content"] == "Original"
    assert injected[0]["content"] == "Original\n\nTemplate"
    assert injected[1] == original[1]


def test_only_explicit_degen_stop_marker_is_authoritative():
    natural_eos = [{"reason": "forced_suffix", "token_id": 2}]
    degen = [{"reason": "degen_stop", "token_id": 2}]
    merged = merge_agentic_forced_tokens(natural_eos, degen)
    assert is_degeneration_stopped(None) is False
    assert is_degeneration_stopped(natural_eos) is False
    assert is_degeneration_stopped(merged) is True


def test_jsonl_archive_appends_without_overwriting_on_resume(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    archive = JsonlArchive(path)
    archive.append([{"id": 1, "text": "x" * 40}])
    archive.append([{"id": 2, "text": "y" * 40}])

    resumed = JsonlArchive(path)
    resumed.append([{"id": 3}])

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["id"] for record in records] == [1, 2, 3]


def test_prefilter_archive_preserves_raw_score_and_binary_reward_metadata(tmp_path):
    batch = DataProto.from_dict({"rm_scores": torch.tensor([[1.0]])})
    batch.non_tensor_batch = {
        "data_source": np.array(["swiss-ai/if-rl-singleturn-prompts"], dtype=object),
        "reward_model": np.array([{"ground_truth": "answer"}], dtype=object),
        "raw_task_score": np.array([0.75]),
        "task_score": np.array([0.75]),
        "task_success": np.array([1.0]),
        "acc": np.array([1.0]),
        "format_valid": np.array([True]),
        "format_penalty": np.array([0.0]),
        "format_bonus": np.array([0.0]),
        "optimization_reward": np.array([1.0]),
        "output_format_parser": np.array(["xml"], dtype=object),
        "output_format_prompt_role": np.array(["user"], dtype=object),
    }
    rollout_sample = RolloutSample(
        full_batch=batch,
        sample_id="sample_1",
        epoch=0,
        rollout_status={},
    )
    rollouter_class = FullyAsyncRollouter.__ray_metadata__.modified_class
    rollouter = rollouter_class.__new__(rollouter_class)
    rollouter._prefilter_archive = JsonlArchive(tmp_path / "prefilter.jsonl")

    rollouter._archive_output_format_prefilter_group(rollout_sample, kept=True)

    record = json.loads((tmp_path / "prefilter.jsonl").read_text().strip())
    assert record["raw_task_score"] == 0.75
    assert record["task_score"] == 0.75
    assert record["task_success"] == 1.0
    assert record["format_bonus"] == 0.0
    assert record["output_format_parser"] == "xml"
    assert record["output_format_prompt_role"] == "user"


def test_group_filter_uses_penalty_free_accuracy_not_optimization_reward():
    batch = DataProto.from_dict({"rm_scores": torch.tensor([[0.9], [0.9]])})
    batch.non_tensor_batch = {
        "acc": np.array([1.0, 0.0]),
        "optimization_reward": np.array([0.9, 0.9]),
    }
    config = OmegaConf.create({"metric": "acc", "min": 0.0, "max": 1.0})
    assert should_keep_async_filter_group(batch, config) is True


def test_all_degeneration_zero_accuracy_group_is_filtered():
    batch = DataProto.from_dict({"rm_scores": torch.zeros((2, 1))})
    batch.non_tensor_batch = {"acc": np.zeros(2), "degeneration_stopped": np.ones(2)}
    config = OmegaConf.create({"metric": "acc", "min": 0.0, "max": 1.0})
    assert should_keep_async_filter_group(batch, config) is False


def _output_format_filter_batch(task_success, format_valid):
    batch = DataProto.from_dict({"rm_scores": torch.zeros((len(task_success), 1))})
    batch.non_tensor_batch = {
        "task_success": np.asarray(task_success, dtype=np.float64),
        "format_valid": np.asarray(format_valid, dtype=bool),
    }
    return batch


@pytest.mark.parametrize(
    ("task_success", "format_valid", "expected"),
    [
        ([0, 0, 0, 0], [False, True, False, True], False),
        ([1, 0, 0, 0], [False, True, False, True], True),
        ([1, 1, 1, 1], [True, False, True, True], True),
        ([1, 1, 1, 1], [True, True, True, True], False),
        ([1, 1, 1, 1], [False, False, False, False], False),
    ],
)
def test_output_format_group_filter(task_success, format_valid, expected):
    batch = _output_format_filter_batch(task_success, format_valid)
    assert should_keep_output_format_group(batch) is expected


def test_nonbootstrap_filter_validation_keeps_existing_acc_metric():
    config = OmegaConf.create(
        {
            "output_format": {"enabled": False},
            "algorithm": {
                "use_kl_in_reward": False,
                "filter_groups": {"enable": True, "metric": "acc", "min": 0.0, "max": 1.0},
            },
        }
    )
    validate_async_filter_groups_config(config)


def test_output_format_filter_validation_requires_task_success_metric():
    config = OmegaConf.create(
        {
            "output_format": {"enabled": True},
            "algorithm": {
                "use_kl_in_reward": False,
                "filter_groups": {"enable": True, "metric": "acc", "min": 0.0, "max": 1.0},
            },
        }
    )
    with pytest.raises(ValueError, match="metric=task_success"):
        validate_async_filter_groups_config(config)


def test_output_format_reward_validation_is_noop_when_disabled():
    config = OmegaConf.create(
        {
            "output_format": {
                "enabled": False,
                "success_threshold": 2.0,
                "format_penalty": 2.0,
                "format_bonus": 2.0,
            },
            "actor_rollout_ref": {"rollout": {"n": 32}},
        }
    )
    validate_output_format_reward_config(config)
