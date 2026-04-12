#!/bin/bash
set -euo pipefail

# --------------------------------------------------------------------
# Self-contained experiment matrix
# --------------------------------------------------------------------
WORKING_DIR=/iopsstor/scratch/cscs/msantelmo/inverse_batch/verl
cd "${WORKING_DIR}"

PROJECT_NAME="RL-Ada-MATH-12k"
DATA_DIR="${WORKING_DIR}/data/math-12k"
OUTPUT_ROOT="${WORKING_DIR}/outputs/${PROJECT_NAME}"

REPEATS=3
START_SEED=42

# DAPO-like filtering settings
ENABLE_FILTER_GROUPS=true
FILTER_GROUPS_BATCH_TARGET=prompts

MODELS=(
  # "swiss-ai/Apertus-8B-Instruct-2509"
  # "meta-llama/Llama-3.2-1B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct"
)

BASELINE_ALGOS=() # grpo maxrl f_grpo)  # to include baselines.
BASELINE_NS=(8)

# RL-Ada configs as tuples:
# "<round_samples> <max_rounds> <min_pos> <min_neg> <n> <apply_downsampling>"
RL_ADA_CONFIGS=(
  "4 4 1 1 8 false"
  # "4 8 1 1 16 false"
)
# Additional adaptive weighting switches (crossed with every RL_ADA_CONFIGS entry):
# - apply_inv_pass_rate_weight: inverse pass-rate weighting.
# - apply_prompt_inverse_group_weight: 1 / K_i prompt-level weighting.
# - apply_within_prompt_mass_balance: within-prompt +/- mass balancing.
INV_PASS_RATE_WEIGHT_OPTIONS=(true)
PROMPT_INVERSE_GROUP_WEIGHT_OPTIONS=(false)
WITHIN_PROMPT_MASS_BALANCE_OPTIONS=(false)

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
  local apply_downsampling="${11:-true}"
  local apply_inv_pass_rate_weight="${12:-true}"
  local apply_prompt_inverse_group_weight="${13:-false}"
  local apply_within_prompt_mass_balance="${14:-false}"
  
  HF_HUB_CACHE_DIR="/capstor/scratch/cscs/msantelmo/huggingface/hub"
  local model_tag
  model_tag="$(basename "${model}" | tr '/:.' '-' | tr -c '[:alnum:]_-' '-')"
  local resolved_model="${model}"
  if [ ! -d "${model}" ] && [ -f "${HF_HUB_CACHE_DIR}/models--${model//\//--}/refs/main" ]; then
    local commit
    commit="$(cat "${HF_HUB_CACHE_DIR}/models--${model//\//--}/refs/main")"
    resolved_model="${HF_HUB_CACHE_DIR}/models--${model//\//--}/snapshots/${commit}"
  fi

  local filter_suffix=""
  if [ "${ENABLE_FILTER_GROUPS}" = "true" ]; then
    filter_suffix="__DAPO-${FILTER_GROUPS_BATCH_TARGET}"
  fi
  local run_name="${algo}__${model_tag}${filter_suffix}__${budget_tag}__rep${rep}"
  local wandb_group="${run_name%__rep${rep}}"
  local run_dir="${OUTPUT_ROOT}/${run_name}"
  mkdir -p "${run_dir}"

  WORKING_DIR="${WORKING_DIR}" \
  DATA_DIR="${DATA_DIR}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  PROJECT_NAME="${PROJECT_NAME}" \
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
  APPLY_DOWNSAMPLING="${apply_downsampling}" \
  APPLY_INV_PASS_RATE_WEIGHT="${apply_inv_pass_rate_weight}" \
  APPLY_PROMPT_INVERSE_GROUP_WEIGHT="${apply_prompt_inverse_group_weight}" \
  APPLY_WITHIN_PROMPT_MASS_BALANCE="${apply_within_prompt_mass_balance}" \
  ENABLE_FILTER_GROUPS="${ENABLE_FILTER_GROUPS}" \
  FILTER_GROUPS_METRIC="acc" \
  FILTER_GROUPS_MAX_NUM_GEN_BATCHES=0 \
  FILTER_GROUPS_BATCH_TARGET="${FILTER_GROUPS_BATCH_TARGET}" \
  sbatch \
    --job-name="${run_name}" \
    --output="${run_dir}/slurm.out" \
    --error="${run_dir}/slurm.err" \
    --export=ALL \
    "${WORKING_DIR}/experiments/ablation/train/run_hard_ablation_job.sh"

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

    # RL-Ada: round-based adaptive sampling configs
    for cfg in "${RL_ADA_CONFIGS[@]}"; do
      read -r round_samples max_rounds min_pos min_neg downsample_n apply_downsampling <<< "${cfg}"
      for apply_inv_pass_rate_weight in "${INV_PASS_RATE_WEIGHT_OPTIONS[@]}"; do
        for apply_prompt_inverse_group_weight in "${PROMPT_INVERSE_GROUP_WEIGHT_OPTIONS[@]}"; do
          for apply_within_prompt_mass_balance in "${WITHIN_PROMPT_MASS_BALANCE_OPTIONS[@]}"; do
            submit_job \
              "rl_ada" \
              "${model}" \
              "${round_samples}x${max_rounds}_${min_pos}-${min_neg}_k${downsample_n}_d${apply_downsampling}_w${apply_inv_pass_rate_weight}_pk${apply_prompt_inverse_group_weight}_mb${apply_within_prompt_mass_balance}" \
              "${rep}" \
              "${seed}" \
              "${downsample_n}" \
              "${round_samples}" \
              "${max_rounds}" \
              "${min_pos}" \
              "${min_neg}" \
              "${apply_downsampling}" \
              "${apply_inv_pass_rate_weight}" \
              "${apply_prompt_inverse_group_weight}" \
              "${apply_within_prompt_mass_balance}"
          done
        done
      done
    done
  done
done

echo "Submitted hard-ablation matrix to SLURM."
