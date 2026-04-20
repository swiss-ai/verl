#!/bin/bash
set -euo pipefail

WORKING_DIR=/iopsstor/scratch/cscs/msantelmo/inverse_batch/RL-policy-mix
cd "${WORKING_DIR}"

#####################################################################

PROJECT_NAME=RLVR-policy-mix
OUTPUT_ROOT=${WORKING_DIR}/outputs/${PROJECT_NAME}

TRAIN_FILE=./data/hendrycks_math/train.parquet
VAL_FILE=./data/hendrycks_math/test.parquet

REPEATS=3
START_SEED=42

STUDENT_MODEL_PATH=meta-llama/Llama-3.2-1B-Instruct
TEACHER_MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct	# nvidia/OpenMath2-Llama3.1-8B

TRAIN_BATCH_SIZE=512
TOTAL_EPOCHS=10
# EASD parameters
ENTROPY_TOP_K=128
ENTROPY_AWARE_MIXING=geometric
ENTROPY_AWARE_ALPHA=linear
# Sweep over speculative mixing mode in mixed policy:
# - true: entropy-aware mixture of student/teacher (EASD)
# - false: fully off-policy teacher drafts
USE_ENTROPY_AWARE_MIXING_OPTIONS=(
	true
	false
)
# K_on/K_off pairs for mixed policy. GRPO will use n = k_on + k_off.
MIXED_SPLIT_PAIRS=(
	"7 1"
	"6 2"
)
# Rollout budgets to run GRPO on.
GRPO_ROLLOUT_NS=(
	8
)

#####################################################################

STUDENT_MODEL_TAG="$(basename "${STUDENT_MODEL_PATH}" | tr '/:.' '-' | tr -c '[:alnum:]_-' '-')"
TEACHER_MODEL_TAG="$(basename "${TEACHER_MODEL_PATH}" | tr '/:.' '-' | tr -c '[:alnum:]_-' '-')"

mkdir -p "${OUTPUT_ROOT}"

submit_job() {
  local algo="$1"
  local rep="$2"
  local seed="$3"
  local k_on="$4"
  local k_off="$5"
  local rollout_n="$6"
  local use_entropy_aware_mixing="${7:-true}"

  local budget_tag
  local run_name
  if [ "${algo}" = "mixed_policy" ]; then
    if [ "${use_entropy_aware_mixing}" = "true" ]; then
      run_name="mixed__${STUDENT_MODEL_TAG}--${TEACHER_MODEL_TAG}__bs${TRAIN_BATCH_SIZE}__${k_on}on${k_off}off__EASD-topk${ENTROPY_TOP_K}_${ENTROPY_AWARE_MIXING}_${ENTROPY_AWARE_ALPHA}__rep${rep}"
    else
      run_name="mixed__${STUDENT_MODEL_TAG}--${TEACHER_MODEL_TAG}__bs${TRAIN_BATCH_SIZE}__${k_on}on${k_off}off__SD__rep${rep}"
    fi
  else
    run_name="${algo}__${STUDENT_MODEL_TAG}__bs${TRAIN_BATCH_SIZE}__n${rollout_n}__rep${rep}"
  fi
  local wandb_group="${run_name%__rep${rep}}"
  local run_dir="${OUTPUT_ROOT}/${run_name}"
  mkdir -p "${run_dir}"

  sbatch \
    --job-name="${run_name}" \
    --output="${run_dir}/slurm.out" \
    --error="${run_dir}/slurm.err" \
    --export=ALL,WORKING_DIR="${WORKING_DIR}",PROJECT_NAME="${PROJECT_NAME}",OUTPUT_ROOT="${OUTPUT_ROOT}",RUN_NAME="${run_name}",WANDB_RUN_GROUP="${wandb_group}",ALGO="${algo}",REPEAT_IDX="${rep}",SEED="${seed}",TRAIN_FILE="${TRAIN_FILE}",VAL_FILE="${VAL_FILE}",STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH}",TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH}",TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}",TOTAL_EPOCHS="${TOTAL_EPOCHS}",K_ON="${k_on}",K_OFF="${k_off}",ROLLOUT_N="${rollout_n}",ENTROPY_TOP_K="${ENTROPY_TOP_K}",ENTROPY_AWARE_MIXING="${ENTROPY_AWARE_MIXING}",ENTROPY_AWARE_ALPHA="${ENTROPY_AWARE_ALPHA}",USE_ENTROPY_AWARE_MIXING="${use_entropy_aware_mixing}" \
    "${WORKING_DIR}/experiments/ablation/run_single_job.sh"

  sleep 2
}

submitted=0
for rep in $(seq 1 "${REPEATS}"); do
  seed=$((START_SEED + rep - 1))

  # Mixed policy runs over explicit (k_on, k_off) pairs.
  for split in "${MIXED_SPLIT_PAIRS[@]}"; do
    read -r k_on k_off <<< "${split}"
    rollout_n=$((k_on + k_off))

    # Mixed policy with (k_on, k_off), for each mixing mode.
    for use_entropy_aware_mixing in "${USE_ENTROPY_AWARE_MIXING_OPTIONS[@]}"; do
      submit_job "mixed_policy" "${rep}" "${seed}" "${k_on}" "${k_off}" "${rollout_n}" "${use_entropy_aware_mixing}"
      submitted=$((submitted + 1))
    done
  done

  # GRPO baseline runs over explicit rollout budgets n.
  for rollout_n in "${GRPO_ROLLOUT_NS[@]}"; do
    # k_on and k_off are ignored for GRPO; pass placeholders.
    submit_job "grpo" "${rep}" "${seed}" 0 0 "${rollout_n}" "true"
    submitted=$((submitted + 1))
  done
done

echo "Submitted ${submitted} jobs."
echo "Methods: mixed_policy + grpo"
echo "Repeats: ${REPEATS}, start seed: ${START_SEED}, split pairs: ${MIXED_SPLIT_PAIRS[*]}, mixing options: ${USE_ENTROPY_AWARE_MIXING_OPTIONS[*]}, grpo n list: ${GRPO_ROLLOUT_NS[*]}"
