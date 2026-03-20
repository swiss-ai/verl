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

WORKING_DIR="${WORKING_DIR:-/capstor/scratch/cscs/msantelmo/inverse_batch}"
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
    "algorithm.use_kl_in_reward=false"
    "actor_rollout_ref.actor.use_kl_loss=false"
    "algorithm.norm_adv_by_std_in_grpo=false"
    "algorithm.filter_groups.enable=false"
    "algorithm.rollout_correction.rollout_rs=null"
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.experiment_name=${RUN_NAME}"
    "trainer.default_local_dir=${RUN_DIR}"
    "trainer.validation_data_dir=${RUN_DIR}/validation"
    "hydra.run.dir=${RUN_DIR}"
    "hydra.output_subdir=.hydra"
    "actor_rollout_ref.actor.data_loader_seed=${SEED}"
  )

  if [ "${ALGO}" = "rl_ada" ]; then
    overrides+=(
      "actor_rollout_ref.rollout.n=${ROUND_SAMPLES}"
      "algorithm.adaptive_group_sampling.enable=true"
      "algorithm.adaptive_group_sampling.rollouts_per_round=${ROUND_SAMPLES}"
      "algorithm.adaptive_group_sampling.max_rounds=${MAX_ROUNDS}"
      "algorithm.adaptive_group_sampling.min_positive_samples=${MIN_POS}"
      "algorithm.adaptive_group_sampling.min_negative_samples=${MIN_NEG}"
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
    BUDGET_TAG="${ROUND_SAMPLES}x${MAX_ROUNDS}_${MIN_POS}-${MIN_NEG}"
    ;;
  *)
    echo "Unsupported ALGO=${ALGO}. Use grpo|maxrl|f_grpo|rl_ada"
    exit 1
    ;;
esac

resolve_run_name

export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${DATASET_TAG}}"
export WANDB_NAME="${RUN_NAME}"

cd verl
pip install --no-deps --no-cache-dir --force-reinstall -e .

build_overrides
write_metadata

HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
  "${overrides[@]}" \
  "$@"
