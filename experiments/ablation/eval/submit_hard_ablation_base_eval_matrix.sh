#!/bin/bash
set -euo pipefail

# --------------------------------------------------------------------
# Submit Slurm jobs to evaluate base models (no training checkpoints).
# --------------------------------------------------------------------
WORKING_DIR="/iopsstor/scratch/cscs/msantelmo/inverse_batch/verl"
cd "${WORKING_DIR}"

EVAL_DATA_DIR="${WORKING_DIR}/data/eval_benchmarks"
OUTPUT_ROOT="${WORKING_DIR}/outputs/base_model_eval"
HF_HUB_CACHE_DIR="/capstor/scratch/cscs/msantelmo/huggingface/hub"

MODELS=(
  "meta-llama/Llama-3.2-3B-Instruct"
  "meta-llama/Llama-3.2-1B-Instruct"
  # "Qwen/Qwen2.5-7B-Instruct"
)

TASKS_CSV="math500,aime2024,aime2025,amc23,gsm8k"
FORCE=false
SAVE_PREDICTIONS=true

N=64
MAX_NEW_TOKENS=8192
TEMPERATURE=0.6
TOP_K=""
TOP_P=0.95
SEED=42

# Leave to default
DTYPE="${DTYPE:-}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"

SLEEP_BETWEEN_SUBMITS="${SLEEP_BETWEEN_SUBMITS:-2}"

if [[ ! -d "${EVAL_DATA_DIR}" ]]; then
  echo "Eval data directory does not exist: ${EVAL_DATA_DIR}"
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}"

resolve_model_path() {
  local model="$1"
  if [[ -d "${model}" ]]; then
    echo "${model}"
    return
  fi

  local ref_file="${HF_HUB_CACHE_DIR}/models--${model//\//--}/refs/main"
  if [[ -f "${ref_file}" ]]; then
    local commit
    commit="$(cat "${ref_file}")"
    echo "${HF_HUB_CACHE_DIR}/models--${model//\//--}/snapshots/${commit}"
    return
  fi

  echo "${model}"
}

normalize_model_tag() {
  local raw="$1"
  local tag
  tag="$(printf '%s' "${raw}" | tr '/:.' '-' | tr -c '[:alnum:]_-' '-')"
  tag="${tag%-}"
  echo "${tag}"
}

submit_base_eval_job() {
  local model="$1"
  local resolved_model
  resolved_model="$(resolve_model_path "${model}")"

  local model_tag
  model_tag="$(normalize_model_tag "$(basename "${model}")")"
  local out_dir="${OUTPUT_ROOT}/${model_tag}"
  mkdir -p "${out_dir}"

  WORKING_DIR="${WORKING_DIR}" \
  MODEL_PATH="${resolved_model}" \
  MODEL_NAME="${model_tag}" \
  EVAL_DATA_DIR="${EVAL_DATA_DIR}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  TASKS_CSV="${TASKS_CSV}" \
  FORCE="${FORCE}" \
  SAVE_PREDICTIONS="${SAVE_PREDICTIONS}" \
  N="${N}" \
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
  TEMPERATURE="${TEMPERATURE}" \
  TOP_K="${TOP_K}" \
  TOP_P="${TOP_P}" \
  SEED="${SEED}" \
  DTYPE="${DTYPE}" \
  TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE}" \
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
  sbatch \
    --job-name="eval_base_${model_tag}" \
    --output="${out_dir}/slurm_eval.out" \
    --error="${out_dir}/slurm_eval.err" \
    --export=ALL \
    "${WORKING_DIR}/experiments/ablation/eval/run_hard_ablation_eval_job.sh"

  echo "[submitted] base model: ${model} -> ${model_tag}"
  sleep "${SLEEP_BETWEEN_SUBMITS}"
}

submitted=0
for model in "${MODELS[@]}"; do
  submit_base_eval_job "${model}"
  submitted=$((submitted + 1))
done

echo "Submitted ${submitted} base-model evaluation jobs."
