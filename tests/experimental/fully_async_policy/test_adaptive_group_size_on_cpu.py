from __future__ import annotations

import asyncio
import concurrent.futures
import time
from collections import defaultdict

import numpy as np
import pytest
import ray
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.experimental.agent_loop.agent_loop import get_generation_request_id
from verl.experimental.fully_async_policy.detach_utils import (
    RolloutSample,
    adaptive_group_has_success,
    assemble_batch_from_rollout_samples,
    pad_adaptive_minibatch,
    partition_adaptive_prompt_minibatches,
    should_keep_async_filter_group,
    speculative_prompt_concurrency_enabled,
    validate_adaptive_group_size_config,
    validate_inverse_batch,
)
from verl.experimental.fully_async_policy.fully_async_rollouter import (
    FullyAsyncRollouter,
    _generate_sequences_inverse_batch,
    _max_concurrent_prompt_groups,
)
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.trainer.ppo.core_algos import compute_rloo_vectorized_outcome_advantage
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


def _metric_batch(metric: str, values) -> DataProto:
    values = np.asarray(values)
    batch = DataProto.from_dict({"dummy": torch.arange(len(values)).unsqueeze(-1)})
    batch.non_tensor_batch[metric] = values
    return batch


@pytest.mark.parametrize("metric", ["acc", "task_success"])
@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([0.0, 1.0], True),
        ([0.0, 0.75], False),
        ([0.0, 0.8], True),
        ([0.0, 0.999], True),
        ([0.0, np.nan], False),
        ([0.0, np.inf], False),
        ([1.0, 1.0], True),
    ],
)
def test_adaptive_success_metric(metric, values, expected):
    config = OmegaConf.create({"metric": metric, "min": 0.0, "max": 1.0})
    assert adaptive_group_has_success(_metric_batch(metric, values), config) is expected


def test_adaptive_success_metric_rejects_missing_field():
    config = OmegaConf.create({"metric": "acc", "min": 0.0, "max": 1.0})
    with pytest.raises(ValueError, match="was not found"):
        adaptive_group_has_success(_metric_batch("other", [0.0, 1.0]), config)


def test_all_correct_group_stops_but_is_variance_filtered():
    config = OmegaConf.create({"metric": "acc", "min": 0.0, "max": 1.0})
    batch = _metric_batch("acc", [1.0, 1.0, 1.0, 1.0])
    assert adaptive_group_has_success(batch, config)
    assert not should_keep_async_filter_group(batch, config)


class _RecordingManager:
    def __init__(self):
        self.calls = []

    async def generate_sequences_single(self, batch):
        self.calls.append(len(batch))
        return batch


def test_inverse_chunks_are_sequential_and_annotated():
    manager = _RecordingManager()
    batch = DataProto.from_dict({"dummy": torch.arange(6).unsqueeze(-1)})
    output = asyncio.run(
        _generate_sequences_inverse_batch(
            manager,
            batch,
            2,
            sampling_round=3,
            scheduled_version=7,
        )
    )

    assert manager.calls == [2, 2, 2]
    assert output.non_tensor_batch["sampling_round"].tolist() == [3] * 6
    assert output.non_tensor_batch["inverse_chunk"].tolist() == [1, 1, 2, 2, 3, 3]
    assert output.non_tensor_batch["round_scheduled_version"].tolist() == [7] * 6


def test_nonadaptive_single_chunk_preserves_direct_generation_path():
    manager = _RecordingManager()
    batch = DataProto.from_dict({"dummy": torch.arange(4).unsqueeze(-1)})
    output = asyncio.run(_generate_sequences_inverse_batch(manager, batch, 4))
    assert output is batch
    assert manager.calls == [4]


@pytest.mark.parametrize(
    "rollout_config",
    [
        {"n": 4, "n_per_round": 2, "adaptive_group_size": {"enabled": False}},
        {"n": 4, "n_per_round": 4, "adaptive_group_size": {"enabled": True}},
    ],
)
def test_inverse_and_adaptive_rollouts_use_uid_for_sticky_routing(rollout_config):
    config = OmegaConf.create(rollout_config)
    assert get_generation_request_id(config, {"uid": "sticky"}) == "sticky"


class _AdaptiveManager:
    def __init__(self, chunk_scores, *, on_call=None):
        self.chunk_scores = chunk_scores
        self.on_call = on_call
        self.calls = []

    async def generate_sequences_single(self, batch):
        call_index = len(self.calls)
        self.calls.append(
            {
                "size": len(batch),
                "uids": np.asarray(
                    batch.non_tensor_batch["uid"], dtype=object
                ).tolist(),
            }
        )
        if self.on_call is not None:
            self.on_call(call_index)
        scores = self.chunk_scores[call_index]
        rm_scores = torch.zeros((len(batch), 2), dtype=torch.float32)
        rm_scores[:, -1] = torch.as_tensor(scores, dtype=torch.float32)
        output = DataProto.from_dict(
            {
                "attention_mask": torch.ones((len(batch), 3), dtype=torch.long),
                "input_ids": torch.zeros((len(batch), 3), dtype=torch.long),
                "position_ids": torch.arange(3).repeat(len(batch), 1),
                "prompts": torch.zeros((len(batch), 1), dtype=torch.long),
                "responses": torch.zeros((len(batch), 2), dtype=torch.long),
                "response_mask": torch.ones((len(batch), 2), dtype=torch.long),
                "rm_scores": rm_scores,
                "rollout_log_probs": torch.zeros((len(batch), 2), dtype=torch.float32),
                "dummy": torch.arange(len(batch)).unsqueeze(-1),
            }
        )
        output.non_tensor_batch.update(
            {
                "acc": np.asarray(scores, dtype=np.float64),
                "min_global_steps": np.zeros(len(batch), dtype=np.int64),
                "max_global_steps": np.full(
                    len(batch), 1 if call_index else 0, dtype=np.int64
                ),
            }
        )
        output.meta_info["metrics"] = [
            {"generate_sequences": 0.1, "tool_calls": 0.0} for _ in range(len(batch))
        ]
        return output


class _CaptureQueue:
    def __init__(self):
        self.samples = []

    async def put_sample(self, sample):
        self.samples.append(ray.cloudpickle.loads(sample))
        return True

    async def get_queue_size(self):
        return len(self.samples)

    async def get_statistics(self):
        return {"queue_size": len(self.samples)}


def _adaptive_rollouter(chunk_scores, *, max_num_rounds, on_call=None, speculative=False):
    rollouter_class = FullyAsyncRollouter.__ray_metadata__.modified_class
    rollouter = rollouter_class.__new__(rollouter_class)
    rollouter.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "adaptive_group_size": {
                        "enabled": True,
                        "max_num_rounds": max_num_rounds,
                        "speculative_prompt_concurrency": {
                            "enabled": speculative,
                            "target_inflight_trajectories_per_replica": 40,
                        },
                    }
                }
            },
            "algorithm": {
                "filter_groups": {
                    "enable": True,
                    "metric": "acc",
                    "min": 0.0,
                    "max": 1.0,
                }
            },
            "output_format": {"enabled": False},
        }
    )
    rollouter.n_per_round = 2
    rollouter.async_rollout_manager = _AdaptiveManager(chunk_scores, on_call=on_call)
    rollouter.message_queue_client = _CaptureQueue()
    rollouter.lock = asyncio.Lock()
    rollouter._resume_event = asyncio.Event()
    rollouter._resume_event.set()
    rollouter.current_param_version = 0
    rollouter.speculative_prompt_concurrency_enabled = speculative
    rollouter.target_inflight_trajectories_per_replica = 40
    rollouter.acceptance_window_open = True
    rollouter.num_rollout_replicas = 1
    rollouter.paused = False
    rollouter.max_required_samples = 100
    rollouter.max_queue_size = 100
    rollouter.max_concurrent_samples = 100
    rollouter.staleness_samples = 1
    rollouter.processed_sample_count = 0
    rollouter.total_generated_samples = 0
    rollouter.total_generated_trajectories = 0
    rollouter.dropped_stale_samples = 0
    rollouter.filtered_group_samples = 0
    rollouter.filtered_group_trajectories = 0
    rollouter.adaptive_rounds_attempted = 0
    rollouter.adaptive_trajectories_attempted = 0
    rollouter.adaptive_successful_groups = 0
    rollouter.adaptive_version_boundary_drops = 0
    rollouter.adaptive_max_round_drops = 0
    rollouter.adaptive_filter_drops = 0
    rollouter.adaptive_discarded_trajectories = 0
    rollouter.adaptive_discarded_tokens = 0
    rollouter.adaptive_speculative_budget_drops = 0
    rollouter.adaptive_speculative_version_drops = 0
    rollouter.adaptive_speculative_queue_drops = 0
    rollouter.adaptive_speculative_discarded_trajectories = 0
    rollouter.adaptive_speculative_discarded_tokens = 0
    rollouter.adaptive_success_round_counts = defaultdict(int)
    rollouter.adaptive_final_group_size_counts = defaultdict(int)
    rollouter._output_format_metric_totals = defaultdict(float)
    rollouter._output_format_source_metric_totals = {}
    rollouter.active_tasks = set()
    rollouter.step_start_time = time.time()
    rollouter.idle_start_time = rollouter.step_start_time
    rollouter._record_output_format_prefilter_metrics = lambda _batch: None
    rollouter._archive_output_format_prefilter_group = lambda _sample, *, kept: None

    async def _statistics():
        return {}

    rollouter.get_statistics = _statistics
    return rollouter


def _rollout_sample() -> RolloutSample:
    batch = DataProto.from_dict({"prompt": torch.arange(4).unsqueeze(-1)})
    batch.non_tensor_batch.update(
        {
            "data_source": np.asarray(["test"] * 4, dtype=object),
            "reward_model": np.asarray([{} for _ in range(4)], dtype=object),
        }
    )
    return RolloutSample(
        full_batch=batch, sample_id="sticky", epoch=0, rollout_status={}
    )


@pytest.mark.parametrize(
    ("chunk_scores", "max_rounds", "expected_size", "expected_success_round"),
    [
        ([[0, 0], [1, 0]], 3, 4, 1),
        ([[0, 0], [0, 0], [0, 0], [1, 0]], 3, 8, 2),
    ],
)
def test_adaptive_rollout_retains_completed_rounds(
    chunk_scores, max_rounds, expected_size, expected_success_round
):
    rollouter = _adaptive_rollouter(chunk_scores, max_num_rounds=max_rounds)
    asyncio.run(rollouter._process_single_sample_streaming(_rollout_sample()))

    assert len(rollouter.message_queue_client.samples) == 1
    queued = rollouter.message_queue_client.samples[0].full_batch
    assert len(queued) == expected_size
    assert set(queued.non_tensor_batch["uid"]) == {"uid_sticky"}
    assert set(queued.non_tensor_batch["success_round"]) == {expected_success_round}
    assert len(rollouter.async_rollout_manager.calls) == expected_size // 2
    assert all(
        set(call["uids"]) == {"uid_sticky"}
        for call in rollouter.async_rollout_manager.calls
    )


def test_adaptive_max_round_exhaustion_enqueues_nothing():
    rollouter = _adaptive_rollouter([[0, 0]] * 4, max_num_rounds=2)
    asyncio.run(rollouter._process_single_sample_streaming(_rollout_sample()))

    assert not rollouter.message_queue_client.samples
    assert rollouter.adaptive_max_round_drops == 1
    assert len(rollouter.async_rollout_manager.calls) == 4


def test_version_change_finishes_started_round_but_prevents_next_round():
    rollouter = None

    def change_version(call_index):
        if call_index == 0:
            rollouter.current_param_version = 1

    rollouter = _adaptive_rollouter(
        [[0, 0], [0, 0]], max_num_rounds=3, on_call=change_version
    )
    asyncio.run(rollouter._process_single_sample_streaming(_rollout_sample()))

    assert not rollouter.message_queue_client.samples
    assert rollouter.adaptive_version_boundary_drops == 1
    assert len(rollouter.async_rollout_manager.calls) == 2


def test_successful_cross_version_round_is_accepted():
    rollouter = None

    def change_version(call_index):
        if call_index == 0:
            rollouter.current_param_version = 1

    rollouter = _adaptive_rollouter(
        [[0, 0], [1, 0]], max_num_rounds=3, on_call=change_version
    )
    asyncio.run(rollouter._process_single_sample_streaming(_rollout_sample()))

    assert len(rollouter.message_queue_client.samples) == 1
    queued = rollouter.message_queue_client.samples[0].full_batch
    assert set(queued.non_tensor_batch["group_start_version"]) == {0}
    assert set(queued.non_tensor_batch["group_end_version"]) == {1}


@pytest.mark.parametrize(
    ("n_per_round", "expected"),
    [(8, 320), (4, 640)],
)
def test_speculative_concurrency_targets_trajectories_per_replica(n_per_round, expected):
    assert (
        _max_concurrent_prompt_groups(
            num_replicas=64,
            rollout_n=n_per_round,
            n_per_round=n_per_round,
            max_required_samples=307,
            speculative_enabled=True,
            target_inflight_trajectories_per_replica=40,
        )
        == expected
    )


def test_speculative_concurrency_never_reduces_accepted_budget():
    assert (
        _max_concurrent_prompt_groups(
            num_replicas=2,
            rollout_n=32,
            n_per_round=32,
            max_required_samples=307,
            speculative_enabled=True,
            target_inflight_trajectories_per_replica=40,
        )
        == 307
    )


def test_speculative_success_claims_one_acceptance_credit():
    rollouter = _adaptive_rollouter([[0, 0], [1, 0]], max_num_rounds=2, speculative=True)
    asyncio.run(rollouter._process_single_sample_streaming(_rollout_sample()))

    assert len(rollouter.message_queue_client.samples) == 1
    assert rollouter.staleness_samples == 2


def test_speculative_filter_drop_consumes_no_acceptance_credit():
    rollouter = _adaptive_rollouter([[1, 1], [1, 1]], max_num_rounds=2, speculative=True)
    asyncio.run(rollouter._process_single_sample_streaming(_rollout_sample()))

    assert not rollouter.message_queue_client.samples
    assert rollouter.staleness_samples == 1
    assert rollouter.adaptive_filter_drops == 1


def test_speculative_budget_drop_does_not_mutate_queue():
    rollouter = _adaptive_rollouter([[0, 0], [1, 0]], max_num_rounds=2, speculative=True)
    sentinel = object()
    rollouter.message_queue_client.samples.append(sentinel)
    rollouter.staleness_samples = rollouter.max_required_samples

    asyncio.run(rollouter._process_single_sample_streaming(_rollout_sample()))

    assert rollouter.message_queue_client.samples == [sentinel]
    assert rollouter.adaptive_speculative_budget_drops == 1
    assert rollouter.adaptive_speculative_discarded_trajectories == 4


def test_speculative_concurrent_publication_cannot_overspend_credits():
    rollouter = _adaptive_rollouter([[0, 0], [1, 0]], max_num_rounds=2, speculative=True)
    rollouter.staleness_samples = 0
    rollouter.max_required_samples = 2
    rollouter.max_queue_size = 2

    async def publish_all():
        return await asyncio.gather(
            *[
                rollouter._try_publish_speculative_group(
                    _rollout_sample(),
                    group_start_version=0,
                )
                for _ in range(5)
            ]
        )

    results = asyncio.run(publish_all())

    assert results.count(None) == 2
    assert results.count("acceptance_budget") == 3
    assert rollouter.staleness_samples == 2
    assert len(rollouter.message_queue_client.samples) == 2


def test_speculative_success_after_version_boundary_is_discarded():
    rollouter = None

    def change_version(call_index):
        if call_index == 0:
            rollouter.current_param_version = 1

    rollouter = _adaptive_rollouter(
        [[0, 0], [1, 0]],
        max_num_rounds=2,
        on_call=change_version,
        speculative=True,
    )
    asyncio.run(rollouter._process_single_sample_streaming(_rollout_sample()))

    assert not rollouter.message_queue_client.samples
    assert rollouter.adaptive_speculative_version_drops == 1


def test_speculative_sync_closes_then_resets_acceptance_from_queue_only():
    rollouter = _adaptive_rollouter([[0, 0], [1, 0]], max_num_rounds=2, speculative=True)
    rollouter.message_queue_client.samples.extend([object(), object(), object()])
    rollouter.active_tasks = {object(), object()}
    rollouter.staleness_samples = 50

    asyncio.run(rollouter.begin_parameter_sync(next_param_version=3))
    assert not rollouter.acceptance_window_open
    assert rollouter.paused
    assert rollouter.current_param_version == 3

    asyncio.run(rollouter.reset_staleness(current_param_version=3))
    assert rollouter.acceptance_window_open
    assert not rollouter.paused
    assert rollouter.staleness_samples == 3


def _heterogeneous_trainer_batch() -> DataProto:
    uids = np.asarray(["a"] * 2 + ["b"] * 4 + ["c"] * 6 + ["d"] * 2, dtype=object)
    size = len(uids)
    response_mask = torch.ones((size, 3), dtype=torch.float32)
    batch = DataProto.from_dict(
        {
            "trajectory_index": torch.arange(size).unsqueeze(-1),
            "input_ids": torch.arange(5).repeat(size, 1),
            "position_ids": torch.arange(5).repeat(size, 1),
            "prompts": torch.arange(2).repeat(size, 1),
            "responses": torch.arange(3).repeat(size, 1),
            "attention_mask": torch.ones((size, 5), dtype=torch.long),
            "response_mask": response_mask,
            "advantages": torch.ones_like(response_mask),
            "returns": torch.ones_like(response_mask),
            "token_level_rewards": torch.ones_like(response_mask),
            "rollout_log_probs": torch.ones_like(response_mask),
        }
    )
    batch.non_tensor_batch["uid"] = uids
    return batch


def test_fixed_prompt_minibatches_cover_every_real_trajectory_once():
    batch = _heterogeneous_trainer_batch()
    minibatches = partition_adaptive_prompt_minibatches(
        batch, prompts_per_minibatch=2, num_minibatches=2, seed=11
    )

    assert len(minibatches) == 2
    assert all(
        len(set(minibatch.non_tensor_batch["uid"])) == 2 for minibatch in minibatches
    )
    observed = torch.cat(
        [minibatch.batch["trajectory_index"] for minibatch in minibatches]
    ).flatten()
    assert sorted(observed.tolist()) == list(range(len(batch)))


def test_adaptive_dp_padding_is_divisible_and_zero_loss():
    batch = _heterogeneous_trainer_batch().select_idxs(list(range(10)))
    batch.meta_info["trajectory_param_versions"] = np.arange(len(batch))
    padded, padding_size = pad_adaptive_minibatch(batch, batch_multiple=4)

    assert padding_size == 2
    assert len(padded) == 12
    assert len(padded) % 4 == 0
    assert padded.non_tensor_batch["is_padding"].tolist() == [False] * 10 + [True, True]
    assert padded.non_tensor_batch["uid"][-2:].tolist() == ["__adaptive_padding__"] * 2
    for key in (
        "response_mask",
        "advantages",
        "returns",
        "token_level_rewards",
        "rollout_log_probs",
    ):
        assert torch.count_nonzero(padded.batch[key][-2:]) == 0
    assert torch.all(padded.batch["advantages"][:10] == 1)
    assert torch.all(padded.batch["attention_mask"][-2:] == 1)
    np.testing.assert_array_equal(
        padded.meta_info["trajectory_param_versions"], np.arange(len(batch))
    )

    unpadded = left_right_2_no_padding(padded.to_tensordict())
    restored = no_padding_2_padding(unpadded["input_ids"], unpadded)
    assert restored.shape == padded.batch["responses"].shape
    assert torch.count_nonzero(unpadded["loss_mask"][-2:]) == 0


def test_adaptive_trainer_makes_one_actor_call_per_prompt_minibatch():
    trainer_class = FullyAsyncTrainer.__ray_metadata__.modified_class
    trainer = trainer_class.__new__(trainer_class)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {"adaptive_group_size": {"enabled": True}},
                "actor": {"ppo_mini_batch_size": 2, "data_loader_seed": 17},
            },
            "trainer": {"critic_warmup": 0, "balance_batch": False},
        }
    )
    trainer.global_steps = 3
    trainer.require_batches = 2
    trainer.actor_rollout_wg = object()
    trainer.metrics = {}
    trainer.timing_raw = {}
    trainer._get_dp_size = lambda _worker_group, _role: 4
    actor_inputs = []

    def update_actor(actor_batch):
        actor_inputs.append(actor_batch)
        return DataProto.from_single_dict(data={}, meta_info={"metrics": {"actor/test": [len(actor_batch)]}})

    trainer._update_actor = update_actor
    unpadded = _heterogeneous_trainer_batch()
    returned = trainer._fit_update_actor(unpadded)

    assert returned is unpadded
    assert len(actor_inputs) == 2
    seen_indices = []
    for actor_batch in actor_inputs:
        real_mask = ~actor_batch.non_tensor_batch["is_padding"]
        assert len(set(actor_batch.non_tensor_batch["uid"][real_mask])) == 2
        assert len(actor_batch) % 4 == 0
        seen_indices.extend(actor_batch.batch["trajectory_index"][real_mask].flatten().tolist())
    assert sorted(seen_indices) == list(range(len(unpadded)))
    assert trainer.metrics["adaptive/trainer/actor_calls"] == 2
    assert trainer.metrics["adaptive/trainer/real_trajectories"] == len(unpadded)


def test_adaptive_fully_async_cpu_smoke():
    four_rollouter = _adaptive_rollouter([[0, 0], [1, 0]], max_num_rounds=2)
    eight_rollouter = _adaptive_rollouter(
        [[0, 0], [0, 0], [0, 0], [1, 0]], max_num_rounds=2
    )
    asyncio.run(four_rollouter._process_single_sample_streaming(_rollout_sample()))
    asyncio.run(eight_rollouter._process_single_sample_streaming(_rollout_sample()))

    queue_samples = []
    for index, source in enumerate(
        [
            four_rollouter.message_queue_client.samples[0],
            eight_rollouter.message_queue_client.samples[0],
            four_rollouter.message_queue_client.samples[0],
            eight_rollouter.message_queue_client.samples[0],
        ]
    ):
        sample = ray.cloudpickle.loads(ray.cloudpickle.dumps(source))
        sample.full_batch.non_tensor_batch["uid"] = np.full(
            len(sample.full_batch), f"smoke_{index}", dtype=object
        )
        sample.rollout_status = {}
        queue_samples.append(sample)

    batch = assemble_batch_from_rollout_samples(queue_samples, tokenizer=None, config=None)
    advantages, returns = compute_rloo_vectorized_outcome_advantage(
        token_level_rewards=batch.batch["rm_scores"],
        response_mask=batch.batch["response_mask"],
        index=batch.non_tensor_batch["uid"],
    )
    batch.batch["advantages"] = advantages
    batch.batch["returns"] = returns

    trainer_class = FullyAsyncTrainer.__ray_metadata__.modified_class
    trainer = trainer_class.__new__(trainer_class)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {"adaptive_group_size": {"enabled": True}},
                "actor": {"ppo_mini_batch_size": 2, "data_loader_seed": 5},
            },
            "trainer": {"critic_warmup": 0, "balance_batch": False},
        }
    )
    trainer.global_steps = 1
    trainer.require_batches = 2
    trainer.actor_rollout_wg = object()
    trainer.metrics = {}
    trainer.timing_raw = {}
    trainer._get_dp_size = lambda _worker_group, _role: 4
    actor_calls = []

    def update_actor(actor_batch):
        actor_calls.append(actor_batch)
        return DataProto.from_single_dict(data={}, meta_info={"metrics": {"actor/smoke": [1.0]}})

    trainer._update_actor = update_actor
    assert trainer._fit_update_actor(batch) is batch
    assert len(actor_calls) == 2
    assert sorted(len(set(call.non_tensor_batch["uid"]) - {"__adaptive_padding__"}) for call in actor_calls) == [2, 2]

    asyncio.run(four_rollouter.reset_staleness(current_param_version=1))
    assert four_rollouter.current_param_version == 1


class _CompletedRemoteCall:
    def __init__(self, result):
        self._future = concurrent.futures.Future()
        self._future.set_result(result)

    def future(self):
        return self._future


class _RecordingRemoteMethod:
    def __init__(self, events, name, result=None):
        self.events = events
        self.name = name
        self.result = result

    def remote(self, **_kwargs):
        self.events.append(self.name)
        return _CompletedRemoteCall(self.result)


class _RecordingCheckpointManager:
    def __init__(self, events):
        self.events = events

    async def update_weights(self, global_steps):
        self.events.append(f"weights:{global_steps}")


def test_speculative_trainer_closes_acceptance_before_weight_sync():
    trainer_class = FullyAsyncTrainer.__ray_metadata__.modified_class
    trainer = trainer_class.__new__(trainer_class)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "adaptive_group_size": {
                        "enabled": True,
                        "speculative_prompt_concurrency": {"enabled": True},
                    }
                }
            },
            "global_profiler": {"steps": None},
        }
    )
    trainer.local_trigger_step = 1
    trainer.current_param_version = 4
    trainer.timing_raw = {}
    events = []
    trainer.rollouter = type(
        "RollouterStub",
        (),
        {
            "begin_parameter_sync": _RecordingRemoteMethod(events, "begin"),
            "reset_staleness": _RecordingRemoteMethod(events, "reset", {}),
        },
    )()
    trainer.checkpoint_manager = _RecordingCheckpointManager(events)

    asyncio.run(trainer._fit_update_weights())

    assert events == ["begin", "weights:4", "reset"]


@pytest.mark.parametrize(
    ("n", "n_per_round"),
    [(4, 0), (4, 5), (4, 3)],
)
def test_inverse_chunk_validation_rejects_invalid_values(n, n_per_round):
    with pytest.raises(ValueError):
        validate_inverse_batch(OmegaConf.create({"n": n, "n_per_round": n_per_round}))


def test_adaptive_config_rejects_invalid_round_count_before_other_checks():
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "adaptive_group_size": {"enabled": True, "max_num_rounds": 0}
                }
            }
        }
    )
    with pytest.raises(ValueError, match="max_num_rounds must be >= 1"):
        validate_adaptive_group_size_config(config)


def _valid_adaptive_config():
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {"adaptive_group_size": {"enabled": True, "max_num_rounds": 2}},
                "actor": {"use_prefix_grouper": False, "use_kl_loss": False},
            },
            "algorithm": {
                "adv_estimator": "rloo_vectorized",
                "use_kl_in_reward": False,
                "filter_groups": {"enable": True, "metric": "acc", "min": 0.0, "max": 1.0},
                "rollout_correction": {"bypass_mode": True},
            },
            "async_training": {"use_rollout_log_probs": True},
            "critic": {"enable": False},
            "distillation": {"enabled": False},
        }
    )


def test_valid_adaptive_config_is_accepted():
    validate_adaptive_group_size_config(_valid_adaptive_config())


def test_speculative_config_is_opt_in_and_validated():
    config = _valid_adaptive_config()
    OmegaConf.update(
        config,
        "actor_rollout_ref.rollout.adaptive_group_size.speculative_prompt_concurrency",
        {"enabled": True, "target_inflight_trajectories_per_replica": 40},
    )
    OmegaConf.update(config, "actor_rollout_ref.rollout.max_num_seqs", 256)

    assert speculative_prompt_concurrency_enabled(config)
    validate_adaptive_group_size_config(config)

    OmegaConf.update(
        config,
        "actor_rollout_ref.rollout.adaptive_group_size.speculative_prompt_concurrency."
        "target_inflight_trajectories_per_replica",
        257,
    )
    with pytest.raises(ValueError, match="must be <= rollout.max_num_seqs"):
        validate_adaptive_group_size_config(config)


def test_speculative_config_requires_adaptive_group_size():
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "adaptive_group_size": {
                        "enabled": False,
                        "speculative_prompt_concurrency": {"enabled": True},
                    }
                }
            }
        }
    )
    with pytest.raises(ValueError, match="requires adaptive_group_size.enabled=True"):
        validate_adaptive_group_size_config(config)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        ("algorithm.filter_groups.enable", False, "filter_groups.enable"),
        ("algorithm.filter_groups.metric", "score", "supports filter_groups.metric"),
        ("algorithm.adv_estimator", "grpo", "requires algorithm.adv_estimator"),
        ("algorithm.rollout_correction.bypass_mode", False, "bypass_mode=True"),
        ("async_training.use_rollout_log_probs", False, "use_rollout_log_probs=True"),
        ("actor_rollout_ref.actor.use_prefix_grouper", True, "use_prefix_grouper=True"),
        ("critic.enable", True, "does not support critic"),
        ("actor_rollout_ref.actor.use_kl_loss", True, "reference policy"),
        ("distillation.enabled", True, "distillation"),
    ],
)
def test_adaptive_config_rejects_unsupported_modes(path, value, match):
    config = _valid_adaptive_config()
    OmegaConf.update(config, path, value)
    with pytest.raises(ValueError, match=match):
        validate_adaptive_group_size_config(config)
