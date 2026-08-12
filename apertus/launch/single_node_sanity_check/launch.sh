#!/bin/bash
set -xeuo pipefail

export DATA_PATH="/capstor/store/cscs/swissai/infra01/reasoning/users/atazza/data_verl"
export VERL_DIR=$(realpath ../../../)
export MEGATRON_CKPT_PATH="/capstor/scratch/cscs/atazza/megatron_checkpoints/iter_0000000/"

run_driver() {
  VERL_LOGGING_LEVEL=INFO HYDRA_FULL_ERROR=1 python3 -m verl.experimental.fully_async_policy.fully_async_main \
    --config-name="async_single_node" \
    data.train_files="${DATA_PATH}/train.parquet" \
    data.val_files="${DATA_PATH}/test.parquet" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="${VERL_DIR}:${PYTHONPATH:-}" \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing="true" \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path="${MEGATRON_CKPT_PATH}"
}

run_driver