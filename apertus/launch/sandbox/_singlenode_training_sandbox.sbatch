#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --container-writable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --environment=reasoning
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --output=slurm_logs/singlenode_sandbox_%j.out
#SBATCH --error=slurm_logs/singlenode_sandbox_%j.err

set -euo pipefail

: "${SCHEDULER_URL:?SCHEDULER_URL must be set by singlenode_sandbox_split.sh}"

WORKING_DIR=/iopsstor/scratch/cscs/msantelmo/apertus_rl
model_path=/capstor/scratch/cscs/msantelmo/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
run_name=sandbox_debug_$(date +%Y%m%d-%H%M%S)
wandb_project=apertus-rl-tests
DEFAULT_TIMEOUT=10
DEFAULT_MEMORY_LIMIT_MB=512
CODEGYM_REWARD_MAX_CONCURRENT=128

output_dir="./outputs/${wandb_project}/${run_name}_$(date +%Y%m%d-%H%M%S)/"

export PYTHONUNBUFFERED="1"
export RAY_DEDUP_LOGS="0"
export NLTK_DATA="${WORKING_DIR}/.cache/nltk_data"
export SCHEDULER_URL
export DEFAULT_TIMEOUT
export DEFAULT_MEMORY_LIMIT_MB
export CODEGYM_REWARD_MAX_CONCURRENT

cd "${WORKING_DIR}"

pip install --no-deps --no-cache-dir --force-reinstall -e .
pip install math-verify langdetect immutabledict nltk absl-py requests
mkdir -p "${NLTK_DATA}"
python -m nltk.downloader -d "${NLTK_DATA}" punkt punkt_tab

HYDRA_FULL_ERROR=1 python -m verl.trainer.main_ppo \
  --config-name="1p5_grpo" \
  hydra.run.dir="${output_dir}" \
  hydra.output_subdir=".hydra" \
  trainer.nnodes=1 \
  data.apply_chat_template_kwargs.enable_thinking=false \
  trainer.experiment_name="${run_name}" \
  trainer.project_name="${wandb_project}" \
  trainer.rollout_data_dir="${output_dir}/rollout/" \
  actor_rollout_ref.model.path="${model_path}" \
  data.train_files="['./data/code/train.parquet']" \
  data.val_files="['./data/code/val.parquet']" \
  actor_rollout_ref.actor.use_dynamic_bsz=true \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
  critic.use_dynamic_bsz=true \
  critic.ppo_max_token_len_per_gpu=16384 \
  reward_model.sandbox_fusion.url="${SCHEDULER_URL}" \
  reward_model.sandbox_fusion.memory_limit_mb="${DEFAULT_MEMORY_LIMIT_MB}" \
  reward_model.sandbox_fusion.max_concurrent="${CODEGYM_REWARD_MAX_CONCURRENT}" \
  "$@"
