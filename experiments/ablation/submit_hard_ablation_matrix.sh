#!/bin/bash
set -euo pipefail

# --------------------------------------------------------------------
# Self-contained experiment matrix
# --------------------------------------------------------------------
WORKING_DIR=/capstor/scratch/cscs/msantelmo/inverse_batch/verl
cd "${WORKING_DIR}"

PROJECT_NAME="RLVR-Ada-Math"
DATA_DIR="${WORKING_DIR}/data/hard_math"
OUTPUT_ROOT="${WORKING_DIR}/outputs/${PROJECT_NAME}"

DATASET_TAG="hard"
REPEATS=3
START_SEED=42
HF_HUB_CACHE_DIR="${HF_HUB_CACHE_DIR:-/capstor/scratch/cscs/msantelmo/huggingface/hub}"

MODELS=(
  "swiss-ai/Apertus-8B-Instruct-2509"
  # "meta-llama/Llama-3.2-3B-Instruct"
)

BASELINE_ALGOS=()	# grpo maxrl f_grpo)
BASELINE_NS=(8 32)
# RL-Ada configs as tuples:
# "<round_samples> <max_rounds> <min_pos> <min_neg> <downsample_n> <apply_inv_pass_rate_weight>"
RL_ADA_CONFIGS=(
  "2 8 1 1 2 true"
  "2 16 1 1 2 true"
  "4 8 1 1 4 true"
  "4 8 2 2 4 true"
	"2 64 1 1 2 true"
)

mkdir -p "${OUTPUT_ROOT}"

submit_job() {
  local algo="$1"
  local model="$2"
  local budget_tag="$3"
  local rep="$4"
  local seed="$5"
  local rollout_n="${6:-}"
  local round_samples="${7:-}"
  local max_rounds="${8:-}"
  local min_pos="${9:-1}"
  local min_neg="${10:-1}"
  local apply_inv_pass_rate_weight="${11:-true}"

  local model_tag
  model_tag="$(basename "${model}" | tr '/:.' '-' | tr -c '[:alnum:]_-' '-')"
  local resolved_model="${model}"
  if [ ! -d "${model}" ] && [ -f "${HF_HUB_CACHE_DIR}/models--${model//\//--}/refs/main" ]; then
    local commit
    commit="$(cat "${HF_HUB_CACHE_DIR}/models--${model//\//--}/refs/main")"
    resolved_model="${HF_HUB_CACHE_DIR}/models--${model//\//--}/snapshots/${commit}"
  fi
  local run_name="${DATASET_TAG}__${algo}__${model_tag}__${budget_tag}__rep${rep}"
  local wandb_group="${run_name%__rep${rep}}"
  local run_dir="${OUTPUT_ROOT}/${run_name}"
  mkdir -p "${run_dir}"

  WORKING_DIR="${WORKING_DIR}" \
  DATA_DIR="${DATA_DIR}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  PROJECT_NAME="${PROJECT_NAME}" \
  DATASET_TAG="${DATASET_TAG}" \
  HF_HUB_CACHE_DIR="${HF_HUB_CACHE_DIR}" \
  RUN_NAME="${run_name}" \
  WANDB_RUN_GROUP="${wandb_group}" \
  ALGO="${algo}" \
  MODEL_NAME_OR_PATH="${resolved_model}" \
  REPEAT_IDX="${rep}" \
  SEED="${seed}" \
  ROLLOUT_N="${rollout_n}" \
  ROUND_SAMPLES="${round_samples}" \
  MAX_ROUNDS="${max_rounds}" \
  MIN_POS="${min_pos}" \
  MIN_NEG="${min_neg}" \
  APPLY_INV_PASS_RATE_WEIGHT="${apply_inv_pass_rate_weight}" \
  sbatch \
    --job-name="${run_name}" \
    --output="${run_dir}/slurm.out" \
    --error="${run_dir}/slurm.err" \
    --export=ALL \
    "${WORKING_DIR}/experiments/ablation/run_hard_ablation_job.sh"

  sleep 5
}

for model in "${MODELS[@]}"; do
  for rep in $(seq 1 "${REPEATS}"); do
    seed=$((START_SEED + rep - 1))

    # Baselines: GRPO, MAXRL, F-GRPO with n in {8, 32}
    for n in "${BASELINE_NS[@]}"; do
      for algo in "${BASELINE_ALGOS[@]}"; do
        submit_job "${algo}" "${model}" "n${n}" "${rep}" "${seed}" "${n}"
      done
    done

    # RL-Ada: 2 x rounds with min pos/neg requirements
    for cfg in "${RL_ADA_CONFIGS[@]}"; do
      read -r round_samples max_rounds min_pos min_neg downsample_n apply_inv_pass_rate_weight <<< "${cfg}"
      submit_job \
        "rl_ada" \
        "${model}" \
        "${round_samples}x${max_rounds}_${min_pos}-${min_neg}_k${downsample_n}_w${apply_inv_pass_rate_weight}" \
        "${rep}" \
        "${seed}" \
        "${downsample_n}" \
        "${round_samples}" \
        "${max_rounds}" \
        "${min_pos}" \
        "${min_neg}" \
        "${apply_inv_pass_rate_weight}"
    done
  done
done

echo "Submitted hard-ablation matrix to SLURM."
