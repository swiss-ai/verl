#!/bin/bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage:
  $0 --run-dir <path> --eval-data-dir <path> [options]

Required:
  --run-dir <path>            Training run directory containing checkpoints.
  --eval-data-dir <path>      Directory with per-task parquet files.

Optional:
  --output-dir <path>         Output root (default: <run-dir>/offline_eval).
  --tasks <csv>               Comma-separated task names (default: all parquet files).
  --evaluation-log-file <p>   Shared append-only JSONL log (default: <output-dir>/evaluation_log.jsonl).
  --n <int>                   Override responses per prompt.
  --max-new-tokens <int>      Override max new output tokens.
  --temperature <float>       Override sampling temperature.
  --top-k <int>               Override top-k.
  --top-p <float>             Override top-p.
  --seed <int>                Override seed.
  --dtype <str>               Override vLLM dtype.
  --tensor-parallel-size <n>  Override tensor parallel size.
  --gpu-memory-utilization <f>Override vLLM memory utilization.
  --max-model-len <int>       Override vLLM max_model_len.
  --max-num-seqs <int>        Override vLLM max_num_seqs.
  --save-predictions          Save predictions.jsonl (default: off).
  --force                     Re-run even if metrics.json exists.
USAGE
}

RUN_DIR=""
EVAL_DATA_DIR=""
OUTPUT_DIR=""
TASKS_CSV="all"
EVALUATION_LOG_FILE=""

N=""
MAX_NEW_TOKENS=""
TEMPERATURE=""
TOP_K=""
TOP_P=""
SEED=""
DTYPE=""
TENSOR_PARALLEL_SIZE=""
GPU_MEMORY_UTILIZATION=""
MAX_MODEL_LEN=""
MAX_NUM_SEQS=""

SAVE_PREDICTIONS=false
FORCE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --eval-data-dir) EVAL_DATA_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --tasks) TASKS_CSV="$2"; shift 2 ;;
    --evaluation-log-file) EVALUATION_LOG_FILE="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --top-k) TOP_K="$2"; shift 2 ;;
    --top-p) TOP_P="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --tensor-parallel-size) TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
    --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --max-num-seqs) MAX_NUM_SEQS="$2"; shift 2 ;;
    --save-predictions) SAVE_PREDICTIONS=true; shift ;;
    --force) FORCE=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${RUN_DIR}" || -z "${EVAL_DATA_DIR}" ]]; then
  echo "--run-dir and --eval-data-dir are required."
  usage
  exit 1
fi
if [[ ! -d "${RUN_DIR}" ]]; then
  echo "Run directory does not exist: ${RUN_DIR}"
  exit 1
fi
if [[ ! -d "${EVAL_DATA_DIR}" ]]; then
  echo "Eval data directory does not exist: ${EVAL_DATA_DIR}"
  exit 1
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${RUN_DIR}/offline_eval"
fi
mkdir -p "${OUTPUT_DIR}"

if [[ -z "${EVALUATION_LOG_FILE}" ]]; then
  EVALUATION_LOG_FILE="${OUTPUT_DIR}/evaluation_log.jsonl"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/evaluate_checkpoint.py"
if [[ ! -f "${EVAL_SCRIPT}" ]]; then
  echo "Evaluator script not found: ${EVAL_SCRIPT}"
  exit 1
fi

resolve_latest_checkpoint_dir() {
  local run_dir="$1"
  local tracker="${run_dir}/latest_checkpointed_iteration.txt"
  local latest_dir=""

  if [[ -f "${tracker}" ]]; then
    local step
    step="$(tr -d '[:space:]' < "${tracker}")"
    if [[ "${step}" =~ ^[0-9]+$ ]]; then
      local candidate="${run_dir}/global_step_${step}"
      if [[ -d "${candidate}" ]]; then
        latest_dir="${candidate}"
      fi
    fi
  fi

  if [[ -z "${latest_dir}" ]]; then
    latest_dir="$(find "${run_dir}" -maxdepth 1 -mindepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1)"
  fi

  # Fallback for non-standard checkpoint folder names:
  # pick the latest run_dir/* that actually contains actor/huggingface.
  if [[ -z "${latest_dir}" ]]; then
    latest_dir="$(
      find "${run_dir}" -maxdepth 1 -mindepth 1 -type d -exec test -d "{}/actor/huggingface" ';' -print \
        | sort -V \
        | tail -n 1
    )"
  fi

  if [[ -z "${latest_dir}" || ! -d "${latest_dir}" ]]; then
    return 1
  fi

  echo "${latest_dir}"
}

latest_ckpt_dir="$(resolve_latest_checkpoint_dir "${RUN_DIR}")" || {
  echo "No checkpoint found in ${RUN_DIR}"
  exit 1
}

ckpt_name="$(basename "${latest_ckpt_dir}")"
model_path="${latest_ckpt_dir}/actor/huggingface"
if [[ ! -d "${model_path}" ]]; then
  echo "Latest checkpoint is missing actor/huggingface: ${model_path}"
  exit 1
fi

declare -a TASK_FILES=()
if [[ "${TASKS_CSV}" == "all" ]]; then
  while IFS= read -r task_file; do
    TASK_FILES+=("${task_file}")
  done < <(find "${EVAL_DATA_DIR}" -maxdepth 1 -type f -name "*.parquet" | sort)
else
  IFS=',' read -r -a TASKS <<< "${TASKS_CSV}"
  for task in "${TASKS[@]}"; do
    task_file="${EVAL_DATA_DIR}/${task}.parquet"
    if [[ ! -f "${task_file}" ]]; then
      echo "Task parquet not found: ${task_file}"
      exit 1
    fi
    TASK_FILES+=("${task_file}")
  done
fi
if [[ ${#TASK_FILES[@]} -eq 0 ]]; then
  echo "No task parquet files found in ${EVAL_DATA_DIR}"
  exit 1
fi

for task_file in "${TASK_FILES[@]}"; do
  task_name="$(basename "${task_file}" .parquet)"
  task_output_dir="${OUTPUT_DIR}/${ckpt_name}/${task_name}"
  metrics_file="${task_output_dir}/metrics.json"

  if [[ -f "${metrics_file}" && "${FORCE}" != "true" ]]; then
    echo "[skip] ${ckpt_name} ${task_name}: metrics already exist"
    continue
  fi

  mkdir -p "${task_output_dir}"
  cmd=(
    python3 "${EVAL_SCRIPT}"
    --model-path "${model_path}"
    --data-file "${task_file}"
    --output-dir "${task_output_dir}"
    --task-name "${task_name}"
    --evaluation-log-file "${EVALUATION_LOG_FILE}"
  )

  [[ -n "${N}" ]] && cmd+=(--n "${N}")
  [[ -n "${MAX_NEW_TOKENS}" ]] && cmd+=(--max-new-tokens "${MAX_NEW_TOKENS}")
  [[ -n "${TEMPERATURE}" ]] && cmd+=(--temperature "${TEMPERATURE}")
  [[ -n "${TOP_K}" ]] && cmd+=(--top-k "${TOP_K}")
  [[ -n "${TOP_P}" ]] && cmd+=(--top-p "${TOP_P}")
  [[ -n "${SEED}" ]] && cmd+=(--seed "${SEED}")
  [[ -n "${DTYPE}" ]] && cmd+=(--dtype "${DTYPE}")
  [[ -n "${TENSOR_PARALLEL_SIZE}" ]] && cmd+=(--tensor-parallel-size "${TENSOR_PARALLEL_SIZE}")
  [[ -n "${GPU_MEMORY_UTILIZATION}" ]] && cmd+=(--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}")
  [[ -n "${MAX_MODEL_LEN}" ]] && cmd+=(--max-model-len "${MAX_MODEL_LEN}")
  [[ -n "${MAX_NUM_SEQS}" ]] && cmd+=(--max-num-seqs "${MAX_NUM_SEQS}")
  [[ "${SAVE_PREDICTIONS}" == "true" ]] && cmd+=(--save-predictions)

  echo "[run] ${ckpt_name} ${task_name}"
  "${cmd[@]}"
done

echo "Latest-checkpoint evaluation complete."
