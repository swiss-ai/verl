#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --container-writable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --environment=sdpo
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -xeuo pipefail

WORKING_DIR="${WORKING_DIR:-/capstor/scratch/cscs/msantelmo/inverse_batch/verl}"
RUN_DIR="${RUN_DIR:-}"
MODEL_PATH="${MODEL_PATH:-}"
MODEL_NAME="${MODEL_NAME:-}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-${WORKING_DIR}/data/eval_benchmarks}"
EVAL_OUTPUT_SUBDIR="${EVAL_OUTPUT_SUBDIR:-offline_eval}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKING_DIR}/outputs/base_model_eval}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-}"

TASKS_CSV="${TASKS_CSV:-all}"
EVALUATION_LOG_FILE="${EVALUATION_LOG_FILE:-}"

N="${N:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
TEMPERATURE="${TEMPERATURE:-}"
TOP_K="${TOP_K:-}"
TOP_P="${TOP_P:-}"
SEED="${SEED:-}"
DTYPE="${DTYPE:-}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"

FORCE="${FORCE:-false}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-false}"

if [[ ! -d "${EVAL_DATA_DIR}" ]]; then
  echo "Eval data directory does not exist: ${EVAL_DATA_DIR}"
  exit 1
fi

cd "${WORKING_DIR}"

echo "[setup] Reinstalling local verl"
pip install --no-deps --no-cache-dir --force-reinstall -e .

append_common_overrides() {
  local -n cmd_ref="$1"
  [[ -n "${N}" ]] && cmd_ref+=(--n "${N}")
  [[ -n "${MAX_NEW_TOKENS}" ]] && cmd_ref+=(--max-new-tokens "${MAX_NEW_TOKENS}")
  [[ -n "${TEMPERATURE}" ]] && cmd_ref+=(--temperature "${TEMPERATURE}")
  [[ -n "${TOP_K}" ]] && cmd_ref+=(--top-k "${TOP_K}")
  [[ -n "${TOP_P}" ]] && cmd_ref+=(--top-p "${TOP_P}")
  [[ -n "${SEED}" ]] && cmd_ref+=(--seed "${SEED}")
  [[ -n "${DTYPE}" ]] && cmd_ref+=(--dtype "${DTYPE}")
  [[ -n "${TENSOR_PARALLEL_SIZE}" ]] && cmd_ref+=(--tensor-parallel-size "${TENSOR_PARALLEL_SIZE}")
  [[ -n "${GPU_MEMORY_UTILIZATION}" ]] && cmd_ref+=(--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}")
  [[ -n "${MAX_MODEL_LEN}" ]] && cmd_ref+=(--max-model-len "${MAX_MODEL_LEN}")
  [[ -n "${MAX_NUM_SEQS}" ]] && cmd_ref+=(--max-num-seqs "${MAX_NUM_SEQS}")
  [[ "${SAVE_PREDICTIONS}" == "true" ]] && cmd_ref+=(--save-predictions)
}

normalize_model_tag() {
  local raw="$1"
  local tag
  tag="$(printf '%s' "${raw}" | tr '/:.' '-' | tr -c '[:alnum:]_-' '-')"
  tag="${tag%-}"
  echo "${tag}"
}

run_checkpoint_eval() {
  if [[ -z "${RUN_DIR}" ]]; then
    echo "RUN_DIR is required when MODEL_PATH is not set."
    exit 1
  fi
  if [[ ! -d "${RUN_DIR}" ]]; then
    echo "Run directory does not exist: ${RUN_DIR}"
    exit 1
  fi

  local output_dir="${EVAL_OUTPUT_DIR:-${RUN_DIR}/${EVAL_OUTPUT_SUBDIR}}"
  local eval_log="${EVALUATION_LOG_FILE:-${output_dir}/evaluation_log.jsonl}"
  mkdir -p "${output_dir}"

  cmd=(
    "${WORKING_DIR}/experiments/ablation/eval/evaluate_checkpoints.sh"
    --run-dir "${RUN_DIR}"
    --eval-data-dir "${EVAL_DATA_DIR}"
    --output-dir "${output_dir}"
    --tasks "${TASKS_CSV}"
    --evaluation-log-file "${eval_log}"
  )
  append_common_overrides cmd
  [[ "${FORCE}" == "true" ]] && cmd+=(--force)
  "${cmd[@]}"
}

# Evaluation of base model
run_base_model_eval() {
  if [[ -z "${MODEL_PATH}" ]]; then
    echo "MODEL_PATH is required for base-model mode."
    exit 1
  fi

  local model_name="${MODEL_NAME}"
  if [[ -z "${model_name}" ]]; then
    model_name="$(basename "${MODEL_PATH}")"
  fi
  model_name="$(normalize_model_tag "${model_name}")"

  local output_dir="${EVAL_OUTPUT_DIR:-${OUTPUT_ROOT}/${model_name}}"
  local eval_log="${EVALUATION_LOG_FILE:-${output_dir}/evaluation_log.jsonl}"
  local eval_script="${WORKING_DIR}/experiments/ablation/eval/evaluate_checkpoint.py"
  mkdir -p "${output_dir}"

  cmd=(
    python3 "${eval_script}"
    --model-path "${MODEL_PATH}"
    --eval-data-dir "${EVAL_DATA_DIR}"
    --tasks "${TASKS_CSV}"
    --output-dir "${output_dir}"
    --evaluation-log-file "${eval_log}"
  )
  append_common_overrides cmd
  [[ "${FORCE}" == "true" ]] && cmd+=(--force)
  echo "[run] ${model_name} tasks=${TASKS_CSV}"
  "${cmd[@]}"

  echo "Base-model evaluation complete."
}

if [[ -n "${MODEL_PATH}" ]]; then
  run_base_model_eval
else
  run_checkpoint_eval
fi
