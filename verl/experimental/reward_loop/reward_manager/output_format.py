from __future__ import annotations

import inspect

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.generation_metadata import is_degeneration_stopped
from verl.utils.output_format import output_format_config, validate_output_format_reward_config
from verl.utils.reward_score import default_compute_score


@register("output_format")
class OutputFormatRewardManager(RewardManagerBase):
    """Binary task reward with output-format shaping and unshaped validation."""

    def __init__(
        self,
        config,
        tokenizer,
        compute_score,
        reward_router_address=None,
        reward_model_tokenizer=None,
    ):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

        validate_output_format_reward_config(config)
        format_config = output_format_config(config)
        self.success_threshold = float(format_config.get("success_threshold", 0.7))
        self.format_penalty_value = float(format_config.get("format_penalty", 0.1))
        self.format_bonus_value = float(format_config.get("format_bonus", 0.05))

        self.overlong_buffer_cfg = config.reward.get("reward_kwargs", {}).get("overlong_buffer_cfg", None)
        self.max_resp_len = config.reward.get("reward_kwargs", {}).get("max_resp_len", None)
        if self.overlong_buffer_cfg is not None:
            if self.max_resp_len is None:
                raise ValueError("reward.reward_kwargs.max_resp_len is required with overlong_buffer_cfg")
            if self.max_resp_len < self.overlong_buffer_cfg.len:
                raise ValueError("max_resp_len must be at least overlong_buffer_cfg.len")

    async def _call_verifier(self, *, data_source, response_str, ground_truth, extra_info, extra_reward_kwargs):
        if self.is_async_reward_score:
            return await self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            )
        return await self.loop.run_in_executor(
            None,
            lambda: self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            ),
        )

    async def run_single(self, data: DataProto) -> dict:
        """Score one rollout, applying optimization penalties only during training."""
        data_item = data[-1:][0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = int(data_item.batch["attention_mask"][-response_length:].sum().item())

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = dict(data_item.non_tensor_batch.get("extra_info", {}) or {})
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        if isinstance(tool_extra_fields, dict):
            extra_info.update(tool_extra_fields)

        validate = bool(extra_info.get("validate", False))
        format_valid = bool(extra_info.get("format_valid", False))
        degeneration_stopped = is_degeneration_stopped(extra_info.get("agentic_forced_tokens"))
        if degeneration_stopped:
            reward_extra_info = {
                "score": 0.0,
                "task_score": 0.0,
                "raw_task_score": 0.0,
                "task_success": 0.0,
                "acc": 0.0,
                "optimization_reward": 0.0,
                "degeneration_stopped": 1.0,
                "verifier_skipped": 1.0,
            }
            if not validate:
                reward_extra_info.update(
                    {
                        "format_valid": float(format_valid),
                        "format_penalty": 0.0,
                        "format_bonus": 0.0,
                    }
                )
            if (
                not validate
                and self.overlong_buffer_cfg is not None
                and self.overlong_buffer_cfg.enable
                and self.overlong_buffer_cfg.log
            ):
                reward_extra_info.update({"overlong_reward": 0.0, "overlong": False})
            return {"reward_score": 0.0, "reward_extra_info": reward_extra_info}

        response_text = extra_info.get("response_text")
        if response_text is None:
            raise KeyError("response_text is required for reward verification")
        response_strs = response_text or [""]
        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )
        results = [
            await self._call_verifier(
                data_source=data_source,
                response_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                extra_reward_kwargs=extra_reward_kwargs,
            )
            for response_str in response_strs
        ]
        result = (
            min(results, key=lambda item: item["score"] if isinstance(item, dict) else item)
            if len(results) > 1
            else results[0]
        )

        if isinstance(result, dict):
            raw_task_score = float(result["score"])
        else:
            raw_task_score = float(result)
        task_success = float(raw_task_score >= self.success_threshold)
        # Keep the reward-extra schema fixed across task families and across
        # verifier-skipped samples. Variable verifier diagnostics would make
        # async DataProto concatenation depend on which rollout finished first.
        reward_extra_info = {"score": raw_task_score}

        if validate:
            reward_extra_info.update(
                {
                    "task_score": raw_task_score,
                    "raw_task_score": raw_task_score,
                    "task_success": task_success,
                    "acc": task_success,
                    "optimization_reward": raw_task_score,
                    "degeneration_stopped": 0.0,
                    "verifier_skipped": 0.0,
                }
            )
            return {
                "reward_score": raw_task_score,
                "reward_extra_info": reward_extra_info,
            }

        format_penalty = -self.format_penalty_value if task_success and not format_valid else 0.0
        format_bonus = self.format_bonus_value if not task_success and format_valid else 0.0
        optimization_reward = task_success + format_penalty + format_bonus

        if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
            overlong_buffer_len = self.overlong_buffer_cfg.len
            expected_len = self.max_resp_len - overlong_buffer_len
            exceed_len = valid_response_length - expected_len
            overlong_reward = min(-exceed_len / overlong_buffer_len * self.overlong_buffer_cfg.penalty_factor, 0)
            optimization_reward += overlong_reward
            if self.overlong_buffer_cfg.log:
                reward_extra_info["overlong_reward"] = overlong_reward
                reward_extra_info["overlong"] = overlong_reward < 0

        reward_extra_info.update(
            {
                "task_score": raw_task_score,
                "raw_task_score": raw_task_score,
                "task_success": task_success,
                "acc": task_success,
                "format_valid": float(format_valid),
                "format_penalty": format_penalty,
                "format_bonus": format_bonus,
                "optimization_reward": optimization_reward,
                "degeneration_stopped": 0.0,
                "verifier_skipped": 0.0,
            }
        )
        return {"reward_score": optimization_reward, "reward_extra_info": reward_extra_info}


__all__ = ["OutputFormatRewardManager"]
