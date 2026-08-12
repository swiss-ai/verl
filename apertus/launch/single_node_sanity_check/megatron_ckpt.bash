#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --job-name=verl_single_node_sanity
set -xeuo pipefail

export DATA_PATH="/capstor/store/cscs/swissai/infra01/reasoning/users/atazza/data_verl"
export VERL_DIR=$(realpath ../../../)
export CONFIG_PATH=$(realpath .)
export ENV_PATH=$(realpath ../vllm_env.toml)
export MEGATRON_PATH=""

run_driver() {
  export PYTHONPATH="/vllm:${VERL_DIR}:${PYTHONPATH:-}"
  echo $PYTHONPATH
  VERL_LOGGING_LEVEL=INFO HYDRA_FULL_ERROR=1 python3 -m verl.experimental.fully_async_policy.fully_async_main \
    --config-path=${CONFIG_PATH} \
    --config-name="async_single_node" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="${PYTHONPATH}" \
    data.train_files="${DATA_PATH}/train.parquet" \
    data.val_files="${DATA_PATH}/test.parquet" \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing="true" \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path="${MEGATRON_CKPT_PATH}"
}

export -f run_driver

srun --environment=$ENV_PATH --container-writable \
  bash -c "run_driver"