from __future__ import annotations

from typing import Any

from verl.utils.output_formatting import (
    MARKDOWN_PARSER,
    SYSTEM_PROMPT_ROLE,
    OutputFormattingConfig,
)


def output_format_config(config: Any) -> Any:
    """Return the output-format subsection, or an empty mapping when absent."""
    if config is None:
        return {}
    if hasattr(config, "get"):
        return config.get("output_format", {}) or {}
    return {}


def output_format_enabled(config: Any) -> bool:
    """Return whether output-format training is enabled."""
    formatting = output_format_config(config)
    return bool(formatting.get("enabled", False)) if hasattr(formatting, "get") else False


def output_formatting_config(config: Any) -> OutputFormattingConfig:
    """Return and validate the selected output formatter."""
    if not output_format_enabled(config):
        return OutputFormattingConfig()

    output_format = output_format_config(config)
    if "system_prompt" in output_format:
        raise ValueError("output_format.system_prompt was removed; use output_format.prompt and prompt_role")
    prompt = output_format.get("prompt")
    formatting = OutputFormattingConfig(
        parser=str(output_format.get("parser", MARKDOWN_PARSER)),
        prompt=None if prompt is None else str(prompt),
        prompt_role=str(output_format.get("prompt_role", SYSTEM_PROMPT_ROLE)),
    )
    formatting.validate(require_enabled=True)
    return formatting


def validate_output_format_reward_config(config: Any) -> None:
    """Validate format reward shaping without affecting other runs."""
    if not output_format_enabled(config):
        return

    output_formatting_config(config)
    output_format = output_format_config(config)
    success_threshold = float(output_format.get("success_threshold", 0.7))
    format_penalty = float(output_format.get("format_penalty", 0.1))
    format_bonus = float(output_format.get("format_bonus", 0.05))

    if not 0.0 <= success_threshold <= 1.0:
        raise ValueError("output_format.success_threshold must be in [0, 1]")
    if not 0.0 <= format_penalty < 1.0:
        raise ValueError("output_format.format_penalty must be in [0, 1)")
    if format_bonus < 0.0:
        raise ValueError("output_format.format_bonus must be non-negative")


def validate_output_format_agent_config(config: Any, *, no_format: bool) -> None:
    """Reject rollout settings that conflict with output-format training.

    Validation is a no-op when output-format training is disabled. Native thinking-prefix
    forcing, legacy ``NO_FORMAT``, multi-turn rollout, and non-single-turn agents
    are incompatible because this mode configures its own plain-text structure.
    """
    if not output_format_enabled(config):
        return
    output_formatting_config(config)
    if no_format:
        raise ValueError(
            "output_format.enabled=true is incompatible with NO_FORMAT=true. "
            "Output-format mode has its own semantic format; leave NO_FORMAT=false."
        )

    data_config = config.get("data", {}) or {}
    if bool(data_config.get("force_thinking_prefix", False)):
        raise ValueError(
            "output_format.enabled=true requires data.force_thinking_prefix=false because output-format training "
            "must not append native inner-token delimiters."
        )

    rollout_config = config.actor_rollout_ref.rollout
    if bool(rollout_config.multi_turn.enable):
        raise ValueError("output_format.enabled=true requires actor_rollout_ref.rollout.multi_turn.enable=false.")
    if rollout_config.agent.default_agent_loop != "single_turn_agent":
        raise ValueError(
            "output_format.enabled=true requires "
            "actor_rollout_ref.rollout.agent.default_agent_loop=single_turn_agent."
        )


__all__ = [
    "output_format_config",
    "output_format_enabled",
    "output_formatting_config",
    "validate_output_format_reward_config",
    "validate_output_format_agent_config",
]
