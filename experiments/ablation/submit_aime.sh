#!/bin/bash
set -euo pipefail

WORKING_DIR=/iopsstor/scratch/cscs/msantelmo/inverse_batch/RL-policy-mix
cd "${WORKING_DIR}"

#####################################################################

PROJECT_NAME=policy-mix-aime
OUTPUT_ROOT=${WORKING_DIR}/outputs/${PROJECT_NAME}

TRAIN_FILE=./data/aime/train.parquet
VAL_FILE=./data/aime/test.parquet

REPEATS=3
START_SEED=42
MAX_CONCURRENT_RUNS=10

STUDENT_MODEL_PATH=Qwen/Qwen3-1.7B-Base
TEACHER_MODEL_PATH=Qwen/Qwen3-14B-FP8 # Qwen/Qwen3-14B-FP8 Qwen/Qwen3-8B Qwen/Qwen3-32B-FP8

# Training parameters
TRAIN_BATCH_SIZE=64
TOTAL_EPOCHS=16
MAX_RESPONSE_LENGTH=8192
PPO_MINI_BATCH_SIZE=64
PPO_MICRO_BATCH_SIZE_PER_GPU=8
LOG_TRAIN_ROLLOUTS=false

# EASD parameters
ENTROPY_TOP_K=32
ENTROPY_AWARE_MIXING=geometric
ENTROPY_AWARE_ALPHA=sqrt

USE_ENTROPY_AWARE_MIXING_OPTIONS=(
	true
	false
)

USE_ROLLOUT_CORRECTION_OPTIONS=(true)
FILTER_GROUPS_ENABLE_OPTIONS=(true)
FILTER_NEGATIVE_OFF_POLICY_ADVANTAGE=true

MIXED_SPLIT_PAIRS=(
	"7 1"
	# "6 2"
)

GRPO_ROLLOUT_NS=(8)

#####################################################################

STUDENT_MODEL_TAG="$(basename "${STUDENT_MODEL_PATH}" | tr '/:.' '-' | tr -c '[:alnum:]_-' '-')"
TEACHER_MODEL_TAG="$(basename "${TEACHER_MODEL_PATH}" | tr '/:.' '-' | tr -c '[:alnum:]_-' '-')"

mkdir -p "${OUTPUT_ROOT}"

submitted_job_ids=()

normalize_job_id() {
  local raw_job_id="$1"
  printf '%s' "${raw_job_id%%;*}"
}

submit_job() {
  local algo="$1"
  local rep="$2"
  local seed="$3"
  local k_on="$4"
  local k_off="$5"
  local rollout_n="$6"
  local use_entropy_aware_mixing="${7:-true}"
  local use_rollout_correction="${8:-true}"
  local filter_groups_enable="${9:-false}"

  local filter_tag=""
  if [ "${filter_groups_enable}" = "true" ]; then
    filter_tag="__DAPO"
  fi
  if [ "${algo}" = "mixed_policy" ] && [ "${filter_groups_enable}" = "true" ] && [ "${FILTER_NEGATIVE_OFF_POLICY_ADVANTAGE}" = "true" ]; then
    filter_tag="${filter_tag}-noNegOff"
  fi
  local run_name
  if [ "${algo}" = "mixed_policy" ]; then
    if [ "${use_entropy_aware_mixing}" = "true" ]; then
      run_name="mixed__${STUDENT_MODEL_TAG}--${TEACHER_MODEL_TAG}__bs${TRAIN_BATCH_SIZE}__${k_on}on${k_off}off__rc-${use_rollout_correction}${filter_tag}__EASD-topk${ENTROPY_TOP_K}_${ENTROPY_AWARE_MIXING}_${ENTROPY_AWARE_ALPHA}__seed${seed}"
    else
      run_name="mixed__${STUDENT_MODEL_TAG}--${TEACHER_MODEL_TAG}__bs${TRAIN_BATCH_SIZE}__${k_on}on${k_off}off__rc-${use_rollout_correction}${filter_tag}__SD__seed${seed}"
    fi
  else
    run_name="${algo}__${STUDENT_MODEL_TAG}__bs${TRAIN_BATCH_SIZE}__n${rollout_n}${filter_tag}__seed${seed}"
  fi
  local wandb_group="${run_name%__seed${seed}}"
  local run_dir="${OUTPUT_ROOT}/${run_name}"
  mkdir -p "${run_dir}"

  local dependency_arg=()
  local current_idx="${#submitted_job_ids[@]}"
  if (( current_idx >= MAX_CONCURRENT_RUNS )); then
    local dep_idx=$((current_idx - MAX_CONCURRENT_RUNS))
    dependency_arg=(--dependency="afterany:${submitted_job_ids[dep_idx]}")
  fi

  local sbatch_output
  sbatch_output="$(
    sbatch \
    --parsable \
    "${dependency_arg[@]}" \
    --job-name="${run_name}" \
    --output="${run_dir}/slurm.out" \
    --error="${run_dir}/slurm.err" \
    --export=ALL,WORKING_DIR="${WORKING_DIR}",PROJECT_NAME="${PROJECT_NAME}",OUTPUT_ROOT="${OUTPUT_ROOT}",RUN_NAME="${run_name}",WANDB_RUN_GROUP="${wandb_group}",ALGO="${algo}",REPEAT_IDX="${rep}",SEED="${seed}",TRAIN_FILE="${TRAIN_FILE}",VAL_FILE="${VAL_FILE}",STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH}",TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH}",TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}",TOTAL_EPOCHS="${TOTAL_EPOCHS}",K_ON="${k_on}",K_OFF="${k_off}",ROLLOUT_N="${rollout_n}",ENTROPY_TOP_K="${ENTROPY_TOP_K}",ENTROPY_AWARE_MIXING="${ENTROPY_AWARE_MIXING}",ENTROPY_AWARE_ALPHA="${ENTROPY_AWARE_ALPHA}",USE_ENTROPY_AWARE_MIXING="${use_entropy_aware_mixing}",USE_ROLLOUT_CORRECTION="${use_rollout_correction}",FILTER_GROUPS_ENABLE="${filter_groups_enable}",FILTER_NEGATIVE_OFF_POLICY_ADVANTAGE="${FILTER_NEGATIVE_OFF_POLICY_ADVANTAGE}",LOG_TRAIN_ROLLOUTS="${LOG_TRAIN_ROLLOUTS}",MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH}",PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE}",PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    "${WORKING_DIR}/experiments/ablation/run_single_job.sh"
  )"

  local job_id
  job_id="$(normalize_job_id "${sbatch_output}")"
  submitted_job_ids+=("${job_id}")

  sleep 2
}

submitted=0
for rep in $(seq 1 "${REPEATS}"); do
  seed=$((START_SEED + rep - 1))

  # Mixed policy runs over explicit (k_on, k_off) pairs.
  for split in "${MIXED_SPLIT_PAIRS[@]}"; do
    read -r k_on k_off <<< "${split}"
    rollout_n=$((k_on + k_off))

    # Mixed policy with (k_on, k_off), for each mixing/correction mode.
    for filter_groups_enable in "${FILTER_GROUPS_ENABLE_OPTIONS[@]}"; do
      for use_entropy_aware_mixing in "${USE_ENTROPY_AWARE_MIXING_OPTIONS[@]}"; do
        for use_rollout_correction in "${USE_ROLLOUT_CORRECTION_OPTIONS[@]}"; do
          submit_job "mixed_policy" "${rep}" "${seed}" "${k_on}" "${k_off}" "${rollout_n}" "${use_entropy_aware_mixing}" "${use_rollout_correction}" "${filter_groups_enable}"
          submitted=$((submitted + 1))
        done
      done
    done
  done

  # GRPO baseline runs over explicit rollout budgets n.
  for rollout_n in "${GRPO_ROLLOUT_NS[@]}"; do
    # k_on and k_off are ignored for GRPO; pass placeholders.
    for filter_groups_enable in "${FILTER_GROUPS_ENABLE_OPTIONS[@]}"; do
      submit_job "grpo" "${rep}" "${seed}" 0 0 "${rollout_n}" "true" "true" "${filter_groups_enable}"
      submitted=$((submitted + 1))
    done
  done
done

echo "Submitted ${submitted} jobs."
echo "Max concurrent runs: ${MAX_CONCURRENT_RUNS}"
echo "Methods: mixed_policy + grpo"
echo "Repeats: ${REPEATS}, start seed: ${START_SEED}, split pairs: ${MIXED_SPLIT_PAIRS[*]}, mixing options: ${USE_ENTROPY_AWARE_MIXING_OPTIONS[*]}, rollout correction options: ${USE_ROLLOUT_CORRECTION_OPTIONS[*]}, filter groups options: ${FILTER_GROUPS_ENABLE_OPTIONS[*]}, filter negative off-policy advantage: ${FILTER_NEGATIVE_OFF_POLICY_ADVANTAGE}, filter metric: seq_reward, max gen batches: 0, grpo n list: ${GRPO_ROLLOUT_NS[*]}"
