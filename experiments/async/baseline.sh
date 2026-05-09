#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --container-writable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --environment=reasoning
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -xeuo pipefail

# Prevent large core_nid* dumps when native libs crash inside the job.
ulimit -c 0

WORKING_DIR=/iopsstor/scratch/cscs/msantelmo/inverse_batch/async-ada
HF_HUB_CACHE_DIR=/capstor/scratch/cscs/msantelmo/huggingface/hub
cd "${WORKING_DIR}"

DATA_DIR=${WORKING_DIR}/data/math-12k
MODEL_NAME_OR_PATH=meta-llama/Llama-3.2-3B-Instruct
PROJECT_NAME=async-rl
CONFIG_NAME=grpo_math_base

ROLLOUT_N=8
SEED=42

TRAIN_FILE=${DATA_DIR}/train.parquet
TEST_FILE=${DATA_DIR}/test.parquet
OUTPUT_ROOT=${WORKING_DIR}/outputs/${PROJECT_NAME}
RUN_NAME="${RUN_NAME:-}"

resolve_run_name() {
  local model_tag
  model_tag="$(basename "${MODEL_NAME_OR_PATH}" | tr '/:.' '-' | tr -c '[:alnum:]_-' '-')"
  local date_time
  date_time="$(date +'%Y%m%dT%H%M%S')"
  if [ -z "${RUN_NAME}" ]; then
    RUN_NAME="${PROJECT_NAME}__grpo__${model_tag}__DAPO__seed${SEED}__${date_time}"
  fi
  RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
  mkdir -p "${RUN_DIR}"
}


RESOLVED_MODEL="${MODEL_NAME_OR_PATH}"
if [ ! -d "${MODEL_NAME_OR_PATH}" ] && [ -f "${HF_HUB_CACHE_DIR}/models--${MODEL_NAME_OR_PATH//\//--}/refs/main" ]; then
  commit="$(cat "${HF_HUB_CACHE_DIR}/models--${MODEL_NAME_OR_PATH//\//--}/refs/main")"
  RESOLVED_MODEL="${HF_HUB_CACHE_DIR}/models--${MODEL_NAME_OR_PATH//\//--}/snapshots/${commit}"
fi

resolve_run_name

build_overrides() {
  overrides=(
    "--config-name=${CONFIG_NAME}"
    "actor_rollout_ref.model.path=${RESOLVED_MODEL}"
    "data.train_files=['${TRAIN_FILE}']"
    "data.val_files=['${TEST_FILE}']"
    "data.seed=${SEED}"
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.experiment_name=${RUN_NAME}"
    "trainer.default_local_dir=${RUN_DIR}"
    "trainer.validation_data_dir=${RUN_DIR}/validation"
    "trainer.val_before_train=false"
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
    "hydra.run.dir=${RUN_DIR}"
    "algorithm.filter_groups.enable=true"
    "hydra.output_subdir=.hydra"
    "actor_rollout_ref.actor.data_loader_seed=${SEED}"
    "actor_rollout_ref.actor.checkpoint.save_contents=['hf_model']"
    "critic.checkpoint.save_contents=[]"
    "trainer.save_freq=10"
    "trainer.test_freq=10"
    "trainer.max_actor_ckpt_to_keep=null"
    "trainer.max_critic_ckpt_to_keep=null"
    "actor_rollout_ref.actor.strategy=fsdp2"
    "critic.strategy=fsdp2"
    "actor_rollout_ref.actor.use_dynamic_bsz=True"
    "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True"
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True"
    "trainer.resume_mode=disable"
  )
}


# Shared HuggingFace cache (prevents repeated downloads across runs/workers).
export HF_HOME="${HF_HOME:-/iopsstor/scratch/cscs/msantelmo/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HUB_CACHE_DIR:-${HF_HOME}/hub}}"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export PYTHONUNBUFFERED="1"
export RAY_DEDUP_LOGS="0"



export WANDB_RUN_GROUP="${RUN_NAME%__seed${SEED}}"
export WANDB_NAME="${RUN_NAME}"

pip install --no-deps --no-cache-dir --force-reinstall -e .

build_overrides

HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
  "${overrides[@]}" \
  "$@"
