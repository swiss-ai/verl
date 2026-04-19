#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --container-writable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --environment=reasoning
#SBATCH --job-name=mixed-policy
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -xeuo pipefail
ulimit -c 0

WORKING_DIR=/iopsstor/scratch/cscs/msantelmo/inverse_batch/RL-policy-mix
cd "${WORKING_DIR}"

PROJECT_NAME=RLVR-policy-mix

TRAIN_FILE=./data/hendrycks_math/train.parquet
VAL_FILE=./data/hendrycks_math/test.parquet

STUDENT_MODEL_PATH=meta-llama/Llama-3.2-1B-Instruct
TEACHER_MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct

BASE_SEED=42
REPEATS=3

# EASD hyperparameters
K_ON=7
K_OFF=1
ENTROPY_TOP_K=50
ENTROPY_AWARE_MIXING=geometric
ENTROPY_AWARE_ALPHA=linear
# Train batch size
TRAIN_BATCH_SIZE=512

EXPERIMENT_NAME="math-$(basename "${STUDENT_MODEL_PATH}")-$(basename "${TEACHER_MODEL_PATH}")_${K_ON}on${K_OFF}off_EASD-${ENTROPY_TOP_K}topk-${ENTROPY_AWARE_MIXING}-${ENTROPY_AWARE_ALPHA}"
OUTPUT_DIR=./outputs/${PROJECT_NAME}/${EXPERIMENT_NAME}
# Slurm reads #SBATCH headers before this script runs, so rename the running
# job here once EXPERIMENT_NAME has been computed.
if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
  scontrol update JobId="${SLURM_JOB_ID}" JobName="${EXPERIMENT_NAME}" || true
fi

# --------------------------------------------------------------------------
HF_HUB_CACHE_DIR="/capstor/scratch/cscs/msantelmo/huggingface/hub"
resolve_model_path() {
  local model_path="$1"
  local resolved_model="${model_path}"
  local cache_root="${HF_HUB_CACHE_DIR}/models--${model_path//\//--}"

  if [ ! -d "${model_path}" ] && [ -f "${cache_root}/refs/main" ]; then
    local commit
    commit="$(cat "${cache_root}/refs/main")"
    resolved_model="${cache_root}/snapshots/${commit}"
  fi

  printf '%s' "${resolved_model}"
}
# --------------------------------------------------------------------------

# Set HF cache dirs and offline mode
export HF_HOME="${HF_HOME:-/iopsstor/scratch/cscs/msantelmo/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HUB_CACHE_DIR:-${HF_HOME}/hub}}"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
# Set WANDB environment variables
export WANDB_NAME="${EXPERIMENT_NAME}"
# Set vLLM cache dir
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_CACHE_ROOT="/iopsstor/scratch/cscs/msantelmo/.cache/vllm"

# Patch vLLM to use entropy-aware speculative decoding
cp -f /users/msantelmo/scratch/vllm/vllm/config/speculative.py /usr/local/lib/python3.12/dist-packages/vllm/config/speculative.py
cp -f /users/msantelmo/scratch/vllm/vllm/engine/arg_utils.py /usr/local/lib/python3.12/dist-packages/vllm/engine/arg_utils.py
cp -f /users/msantelmo/scratch/vllm/vllm/v1/sample/rejection_sampler.py /usr/local/lib/python3.12/dist-packages/vllm/v1/sample/rejection_sampler.py
cp -f /users/msantelmo/scratch/vllm/vllm/v1/spec_decode/eagle.py /usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/eagle.py
cp -f /users/msantelmo/scratch/vllm/vllm/v1/spec_decode/draft_model.py /usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/draft_model.py
cp -f /users/msantelmo/scratch/vllm/vllm/v1/worker/gpu_model_runner.py /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py
cp -f /users/msantelmo/scratch/vllm/vllm/compilation/decorators.py /usr/local/lib/python3.12/dist-packages/vllm/compilation/decorators.py

# Install verl
python3 -m pip install  --no-deps --no-cache-dir --force-reinstall -e .

# (Re)install problematic deps
python3 -m pip install --user --ignore-installed --no-cache-dir "cupy-cuda13x==13.6.0"
python3 -m pip install --no-deps --no-cache-dir "numpy<=2.2.0"

python3 -m verl.trainer.main_ppo \
  --config-name=mixed_policy \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.mixed_policy.enable=True \
  algorithm.mixed_policy.k_on=${K_ON} \
  algorithm.mixed_policy.k_off=${K_OFF} \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size=${TRAIN_BATCH_SIZE} \
  actor_rollout_ref.model.path="$(resolve_model_path "${STUDENT_MODEL_PATH}")" \
  actor_rollout_ref.actor.checkpoint.save_contents=['hf_model'] \
  actor_rollout_ref.actor.data_loader_seed=${SEED}" \
  actor_rollout_ref.mixed_policy.teacher_model_path="$(resolve_model_path "${TEACHER_MODEL_PATH}")" \
  actor_rollout_ref.mixed_policy.speculative.use_entropy_aware_mixing=True \
  actor_rollout_ref.mixed_policy.speculative.entropy_top_k=${ENTROPY_TOP_K} \
  actor_rollout_ref.mixed_policy.speculative.entropy_aware_mixing=${ENTROPY_AWARE_MIXING} \
  actor_rollout_ref.mixed_policy.speculative.entropy_aware_alpha=${ENTROPY_AWARE_ALPHA} \
  actor_rollout_ref.rollout.n=$((K_ON + K_OFF)) \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.use_legacy_worker_impl=enable \
  critic.checkpoint.save_contents=[] \
  trainer.critic_warmup=0 \
  trainer.save_freq=20 \
  trainer.test_freq=5 \
  trainer.max_actor_ckpt_to_keep=null \
  trainer.max_critic_ckpt_to_keep=null \
  trainer.val_before_train=True \
  trainer.default_local_dir=${OUTPUT_DIR} \
  trainer.validation_data_dir=${OUTPUT_DIR}/validation \
  hydra.run.dir=${OUTPUT_DIR} \
  hydra.output_subdir=.hydra \
  "$@"
