#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --container-writable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --environment=reasoning
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -xeuo pipefail

WORKING_DIR="${WORKING_DIR:-/capstor/scratch/cscs/msantelmo/inverse_batch/verl}"
cd "${WORKING_DIR}"

ALGO="${ALGO:-grpo}" # grpo|maxrl|f_grpo|rl_ada
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-3B-Instruct}"
DATA_DIR="${DATA_DIR:-${WORKING_DIR}/data/hard_math}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKING_DIR}/outputs/hard_ablations}"
PROJECT_NAME="${PROJECT_NAME:-hard-ablation}"
DATASET_TAG="${DATASET_TAG:-hard}"

# For baselines
ROLLOUT_N="${ROLLOUT_N:-8}"

# For Adaptive Sampling
ROUND_SAMPLES="${ROUND_SAMPLES:-2}"
MAX_ROUNDS="${MAX_ROUNDS:-8}"
MIN_POS="${MIN_POS:-1}"
MIN_NEG="${MIN_NEG:-1}"
APPLY_DOWNSAMPLING="${APPLY_DOWNSAMPLING:-true}"
APPLY_INV_PASS_RATE_WEIGHT="${APPLY_INV_PASS_RATE_WEIGHT:-true}"
APPLY_PROMPT_INVERSE_GROUP_WEIGHT="${APPLY_PROMPT_INVERSE_GROUP_WEIGHT:-false}"
APPLY_WITHIN_PROMPT_MASS_BALANCE="${APPLY_WITHIN_PROMPT_MASS_BALANCE:-false}"

# DAPO-like filtering
ENABLE_FILTER_GROUPS="${ENABLE_FILTER_GROUPS:-false}"
FILTER_GROUPS_BATCH_TARGET="${FILTER_GROUPS_BATCH_TARGET:-prompts}"

REPEAT_IDX="${REPEAT_IDX:-1}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-}"

TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train.parquet}"
TEST_FILE="${TEST_FILE:-${DATA_DIR}/test.parquet}"

resolve_run_name() {
  local model_tag algo_tag
  model_tag="$(basename "${MODEL_NAME_OR_PATH}" | tr '/:.' '-' | tr -c '[:alnum:]_-' '-')"
  algo_tag="$(echo "${ALGO}" | tr '/:.' '-' | tr -c '[:alnum:]_-' '-')"
  if [ -z "${RUN_NAME}" ]; then
    RUN_NAME="${DATASET_TAG}__${algo_tag}__${model_tag}__${BUDGET_TAG}__rep${REPEAT_IDX}"
  fi
  RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
  mkdir -p "${RUN_DIR}"
}

build_overrides() {
  overrides=(
    "--config-name=${CONFIG_NAME}"
    "actor_rollout_ref.model.path=${MODEL_NAME_OR_PATH}"
    "data.train_files=['${TRAIN_FILE}']"
    "data.val_files=['${TEST_FILE}']"
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.experiment_name=${RUN_NAME}"
    "trainer.default_local_dir=${RUN_DIR}"
    "trainer.validation_data_dir=${RUN_DIR}/validation"
    "hydra.run.dir=${RUN_DIR}"
    "hydra.output_subdir=.hydra"
    "actor_rollout_ref.actor.data_loader_seed=${SEED}"
    "actor_rollout_ref.actor.checkpoint.save_contents=['hf_model']"
    "critic.checkpoint.save_contents=[]"
    "trainer.save_freq=10"
    "trainer.test_freq=20"
    "trainer.max_actor_ckpt_to_keep=1"
    "trainer.max_critic_ckpt_to_keep=1"
    "algorithm.filter_groups.enable=${ENABLE_FILTER_GROUPS}"
    "algorithm.filter_groups.batch_target=${FILTER_GROUPS_BATCH_TARGET}"
  )

  if [ "${ALGO}" = "rl_ada" ]; then
    overrides+=(
      "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
      "algorithm.adaptive_group_sampling.enable=true"
      "algorithm.adaptive_group_sampling.rollouts_per_round=${ROUND_SAMPLES}"
      "algorithm.adaptive_group_sampling.max_rounds=${MAX_ROUNDS}"
      "algorithm.adaptive_group_sampling.min_positive_samples=${MIN_POS}"
      "algorithm.adaptive_group_sampling.min_negative_samples=${MIN_NEG}"
      "algorithm.adaptive_group_sampling.apply_downsampling=${APPLY_DOWNSAMPLING}"
      "algorithm.adaptive_group_sampling.apply_inverse_pass_rate_weight=${APPLY_INV_PASS_RATE_WEIGHT}"
      "algorithm.adaptive_group_sampling.apply_prompt_inverse_group_weight=${APPLY_PROMPT_INVERSE_GROUP_WEIGHT}"
      "algorithm.adaptive_group_sampling.apply_within_prompt_mass_balance=${APPLY_WITHIN_PROMPT_MASS_BALANCE}"
    )
  else
    overrides+=(
      "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
      "algorithm.adaptive_group_sampling.enable=false"
    )
  fi
}

write_metadata() {
  local meta_file="${RUN_DIR}/run_meta.txt"
  {
    echo "RUN_NAME=${RUN_NAME}"
    echo "ALGO=${ALGO}"
    echo "MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
    echo "CONFIG_NAME=${CONFIG_NAME}"
    echo "TRAIN_FILE=${TRAIN_FILE}"
    echo "TEST_FILE=${TEST_FILE}"
    echo "SEED=${SEED}"
    echo "REPEAT_IDX=${REPEAT_IDX}"
    echo "BUDGET_TAG=${BUDGET_TAG}"
    echo "ENABLE_FILTER_GROUPS=${ENABLE_FILTER_GROUPS}"
    echo "FILTER_GROUPS_BATCH_TARGET=${FILTER_GROUPS_BATCH_TARGET}"
    echo "APPLY_PROMPT_INVERSE_GROUP_WEIGHT=${APPLY_PROMPT_INVERSE_GROUP_WEIGHT}"
    echo "APPLY_WITHIN_PROMPT_MASS_BALANCE=${APPLY_WITHIN_PROMPT_MASS_BALANCE}"
    echo "DATE=$(date --iso-8601=seconds)"
  } > "${meta_file}"
}

case "${ALGO}" in
  grpo|maxrl|f_grpo)
    CONFIG_NAME="${ALGO}_math_base"
    BUDGET_TAG="n${ROLLOUT_N}"
    ;;
  rl_ada)
    CONFIG_NAME="grpo_math_adaptive"
    BUDGET_TAG="${ROUND_SAMPLES}x${MAX_ROUNDS}_${MIN_POS}-${MIN_NEG}_k${ROLLOUT_N}_d${APPLY_DOWNSAMPLING}_w${APPLY_INV_PASS_RATE_WEIGHT}_pk${APPLY_PROMPT_INVERSE_GROUP_WEIGHT}_mb${APPLY_WITHIN_PROMPT_MASS_BALANCE}"
    ;;
  *)
    echo "Unsupported ALGO=${ALGO}. Use grpo|maxrl|f_grpo|rl_ada"
    exit 1
    ;;
esac

# Shared HuggingFace cache (prevents repeated downloads across runs/workers).
export HF_HOME="${HF_HOME:-/capstor/scratch/cscs/msantelmo/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HUB_CACHE_DIR:-${HF_HOME}/hub}}"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"

resolve_run_name

export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${DATASET_TAG}}"
export WANDB_NAME="${RUN_NAME}"

pip install --no-deps --no-cache-dir --force-reinstall -e .

build_overrides
write_metadata

HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
  "${overrides[@]}" \
  "$@"
