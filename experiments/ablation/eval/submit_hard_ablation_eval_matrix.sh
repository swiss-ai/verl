#!/bin/bash
set -euo pipefail

# --------------------------------------------------------------------
# Submit one Slurm latest-checkpoint evaluation job per run directory.
# --------------------------------------------------------------------
WORKING_DIR="/iopsstor/scratch/cscs/msantelmo/inverse_batch/RL-policy-mix"
cd "${WORKING_DIR}"

PROJECT_NAME="policy-mix-math"
MAX_CONCURRENT_RUNS=10

OUTPUT_ROOT="${WORKING_DIR}/outputs/${PROJECT_NAME}"
EVAL_DATA_DIR="${WORKING_DIR}/data/eval_benchmarks"
EVAL_OUTPUT_SUBDIR="eval"

# grep -E uses POSIX ERE (no lookaheads), so keep filters ERE-compatible.
RUN_NAME_REGEX="^mixed_.*Llama.*$"
# RUN_NAME_REGEX="^(maxrl__Llama)"
EXCLUDE_REGEX="bs128"

TASKS_CSV="math500,aime2024,gsm8k"  # aime2025,amc23
FORCE=false
SAVE_PREDICTIONS=false
RECORD_CONF=false
ALL_CHECKPOINTS=false

N=64
MAX_NEW_TOKENS=8192
TEMPERATURE=0.6
TOP_K=""
TOP_P=0.95
SEED=42
DATA_PARALLEL_SIZE=4

# Leave to default
DTYPE="${DTYPE:-}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"

SLEEP_BETWEEN_SUBMITS="${SLEEP_BETWEEN_SUBMITS:-2}"

if ! [[ "${MAX_CONCURRENT_RUNS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_CONCURRENT_RUNS must be a positive integer, got: ${MAX_CONCURRENT_RUNS}"
  exit 1
fi

if [[ ! -d "${OUTPUT_ROOT}" ]]; then
  echo "Output root does not exist: ${OUTPUT_ROOT}"
  exit 1
fi
if [[ ! -d "${EVAL_DATA_DIR}" ]]; then
  echo "Eval data directory does not exist: ${EVAL_DATA_DIR}"
  exit 1
fi

submit_eval_job() {
  local run_dir="$1"
  local run_name
  run_name="$(basename "${run_dir}")"

  local job_name="eval_${run_name}"
  local eval_dir="${run_dir}/${EVAL_OUTPUT_SUBDIR}"
  mkdir -p "${eval_dir}"

  local dependency_arg=()
  local current_idx="${#submitted_job_ids[@]}"
  if (( current_idx >= MAX_CONCURRENT_RUNS )); then
    local dep_idx=$((current_idx - MAX_CONCURRENT_RUNS))
    dependency_arg=(--dependency="afterany:${submitted_job_ids[dep_idx]}")
  fi

  local sbatch_output
  sbatch_output="$(
  WORKING_DIR="${WORKING_DIR}" \
  RUN_DIR="${run_dir}" \
  EVAL_DATA_DIR="${EVAL_DATA_DIR}" \
  EVAL_OUTPUT_SUBDIR="${EVAL_OUTPUT_SUBDIR}" \
  TASKS_CSV="${TASKS_CSV}" \
  ALL_CHECKPOINTS="${ALL_CHECKPOINTS}" \
  FORCE="${FORCE}" \
  SAVE_PREDICTIONS="${SAVE_PREDICTIONS}" \
  RECORD_CONF="${RECORD_CONF}" \
  N="${N}" \
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
  TEMPERATURE="${TEMPERATURE}" \
  TOP_K="${TOP_K}" \
  TOP_P="${TOP_P}" \
  SEED="${SEED}" \
  DTYPE="${DTYPE}" \
  TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE}" \
  DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE}" \
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
  sbatch \
    --parsable \
    "${dependency_arg[@]}" \
    --job-name="${job_name}" \
    --output="${eval_dir}/slurm_eval.out" \
    --error="${eval_dir}/slurm_eval.err" \
    --export=ALL \
    "${WORKING_DIR}/experiments/ablation/eval/run_hard_ablation_eval_job.sh"
  )"

  local job_id="${sbatch_output%%;*}"
  submitted_job_ids+=("${job_id}")

  echo "[submitted] ${run_name}"
  sleep "${SLEEP_BETWEEN_SUBMITS}"
}

submitted=0
submitted_job_ids=()
while IFS= read -r run_dir; do
  run_name="$(basename "${run_dir}")"

  if ! echo "${run_name}" | grep -Eq "${RUN_NAME_REGEX}"; then
    continue
  fi
  if [[ -n "${EXCLUDE_REGEX}" ]] && echo "${run_name}" | grep -Eq "${EXCLUDE_REGEX}"; then
    continue
  fi

  if ! find "${run_dir}" -maxdepth 3 -type d -path "${run_dir}/*/actor/huggingface" | grep -q .; then
    echo "[skip] ${run_name}: no checkpoint found"
    continue
  fi

  submit_eval_job "${run_dir}"
  submitted=$((submitted + 1))
done < <(find "${OUTPUT_ROOT}" -maxdepth 1 -mindepth 1 -type d | sort)

echo "Submitted ${submitted} latest-checkpoint evaluation jobs."
echo "Max concurrent runs: ${MAX_CONCURRENT_RUNS}"
