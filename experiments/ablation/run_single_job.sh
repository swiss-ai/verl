#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --container-writable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --environment=reasoning

set -xeuo pipefail
ulimit -c 0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKING_DIR="${WORKING_DIR:-${REPO_ROOT}}"
cd "${WORKING_DIR}"

ALGO="${ALGO:-mixed_policy}" # mixed_policy | grpo
PROJECT_NAME="${PROJECT_NAME:-RLVR-policy-mix}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKING_DIR}/outputs/${PROJECT_NAME}}"

TRAIN_FILE="${TRAIN_FILE:-./data/hendrycks_math/train.parquet}"
VAL_FILE="${VAL_FILE:-./data/hendrycks_math/test.parquet}"

STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-meta-llama/Llama-3.2-1B-Instruct}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"

REPEAT_IDX="${REPEAT_IDX:-1}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-}"

K_ON="${K_ON:-7}"
K_OFF="${K_OFF:-1}"
ROLLOUT_N="${ROLLOUT_N:-$((K_ON + K_OFF))}"

ENTROPY_TOP_K="${ENTROPY_TOP_K:-50}"
ENTROPY_AWARE_MIXING="${ENTROPY_AWARE_MIXING:-geometric}"
ENTROPY_AWARE_ALPHA="${ENTROPY_AWARE_ALPHA:-linear}"
USE_ENTROPY_AWARE_MIXING="${USE_ENTROPY_AWARE_MIXING:-true}"
USE_ROLLOUT_CORRECTION="${USE_ROLLOUT_CORRECTION:-true}"
FILTER_GROUPS_ENABLE="${FILTER_GROUPS_ENABLE:-false}"

LOG_TRAIN_ROLLOUTS="${LOG_TRAIN_ROLLOUTS:-false}"
TRAIN_ROLLOUT_LOG_FREQ=5
TRAIN_ROLLOUT_LOG_MAX_SAMPLES=32

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-10}"

# Throughput-oriented rollout knobs (kept configurable via env/submit script)
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-16384}"
ROLLOUT_ENABLE_CHUNKED_PREFILL="${ROLLOUT_ENABLE_CHUNKED_PREFILL:-false}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-16384}"
ROLLOUT_CUDAGRAPH_MODE="${ROLLOUT_CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}"

VLLM_PATCH_ROOT="${VLLM_PATCH_ROOT:-/users/msantelmo/scratch/vllm/vllm}"

HF_HUB_CACHE_DIR="${HF_HUB_CACHE_DIR:-/capstor/scratch/cscs/msantelmo/huggingface/hub}"

resolve_model_path() {
  local model_path="$1"
  local resolved_model="${model_path}"
  local cache_root="${HF_HUB_CACHE_DIR}/models--${model_path//\//--}"

  if [ ! -d "${model_path}" ] && [ -f "${cache_root}/refs/main" ]; then
    local commit
    commit="$(cat "${cache_root}/refs/main")"
    resolved_model="${cache_root}/snapshots/${commit}"
  fi

  printf '%s' "${resolved_model}"
}

if [ -z "${RUN_NAME}" ]; then
  echo "RUN_NAME must be provided by submit script."
  exit 1
fi
if [ -z "${WANDB_RUN_GROUP}" ]; then
  WANDB_RUN_GROUP="${RUN_NAME%__rep${REPEAT_IDX}}"
fi

RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

export HF_HOME="${HF_HOME:-/iopsstor/scratch/cscs/msantelmo/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HUB_CACHE_DIR}}"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export PYTHONNOUSERSITE=1
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/iopsstor/scratch/cscs/msantelmo/.cache/vllm}"

export WANDB_NAME="${RUN_NAME}"
export WANDB_RUN_GROUP

# Apply local vLLM patch
cp -f "${VLLM_PATCH_ROOT}/config/speculative.py" /usr/local/lib/python3.12/dist-packages/vllm/config/speculative.py
cp -f "${VLLM_PATCH_ROOT}/engine/arg_utils.py" /usr/local/lib/python3.12/dist-packages/vllm/engine/arg_utils.py
cp -f "${VLLM_PATCH_ROOT}/v1/sample/rejection_sampler.py" /usr/local/lib/python3.12/dist-packages/vllm/v1/sample/rejection_sampler.py
cp -f "${VLLM_PATCH_ROOT}/v1/spec_decode/eagle.py" /usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/eagle.py
cp -f "${VLLM_PATCH_ROOT}/v1/spec_decode/draft_model.py" /usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/draft_model.py
cp -f "${VLLM_PATCH_ROOT}/v1/worker/gpu_model_runner.py" /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py
cp -f "${VLLM_PATCH_ROOT}/compilation/decorators.py" /usr/local/lib/python3.12/dist-packages/vllm/compilation/decorators.py

python3 -m pip install --no-deps --no-cache-dir --force-reinstall -e .
python3 -m pip install --ignore-installed --no-cache-dir "cupy-cuda13x==13.6.0"
python3 -m pip install --no-deps --no-cache-dir "numpy<=2.2.0"

STUDENT_RESOLVED="$(resolve_model_path "${STUDENT_MODEL_PATH}")"
TEACHER_RESOLVED="$(resolve_model_path "${TEACHER_MODEL_PATH}")"

overrides=(
  "data.train_files=${TRAIN_FILE}"
  "data.val_files=${VAL_FILE}"
  "data.train_batch_size=${TRAIN_BATCH_SIZE}"
  "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
  "data.max_response_length=${MAX_RESPONSE_LENGTH}"
  "actor_rollout_ref.model.path=${STUDENT_RESOLVED}"
  "actor_rollout_ref.actor.data_loader_seed=${SEED}"
  "actor_rollout_ref.actor.checkpoint.save_contents=['hf_model']"
  "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
  "trainer.project_name=${PROJECT_NAME}"
  "trainer.experiment_name=${RUN_NAME}"
  "trainer.default_local_dir=${RUN_DIR}"
  "trainer.validation_data_dir=${RUN_DIR}/validation"
  "trainer.total_epochs=${TOTAL_EPOCHS}"
  "trainer.use_legacy_worker_impl=enable"
  "trainer.critic_warmup=0"
  "trainer.save_freq=20"
  "trainer.test_freq=5"
  "trainer.max_actor_ckpt_to_keep=null"
  "trainer.max_critic_ckpt_to_keep=null"
  "trainer.val_before_train=true"
  "critic.checkpoint.save_contents=[]"
  "++algorithm.filter_groups.enable=${FILTER_GROUPS_ENABLE}"
  "++algorithm.filter_groups.metric=seq_reward"
  "++algorithm.filter_groups.max_num_gen_batches=0"
  "hydra.run.dir=${RUN_DIR}"
  "hydra.output_subdir=.hydra"
)

# If the teacher is a non-Base Qwen3 model, disable thinking and add rollout optimizations.
if [[ "${TEACHER_MODEL_PATH}" == *"Qwen3"* && "${TEACHER_MODEL_PATH}" != *"Base"* ]]; then
  echo "Rollout speed knobs:"
  echo "  ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN}"
  echo "  ROLLOUT_ENABLE_CHUNKED_PREFILL=${ROLLOUT_ENABLE_CHUNKED_PREFILL}"
  echo "  ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
  echo "  ROLLOUT_CUDAGRAPH_MODE=${ROLLOUT_CUDAGRAPH_MODE}"
  echo "Disabled thinking in chat template"

  overrides+=(
    "actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN}"
    "++actor_rollout_ref.rollout.enable_chunked_prefill=${ROLLOUT_ENABLE_CHUNKED_PREFILL}"
    "++actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
    "++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=${ROLLOUT_CUDAGRAPH_MODE}"
    "++data.apply_chat_template_kwargs.enable_thinking=false"
  )
fi

if [ "${LOG_TRAIN_ROLLOUTS}" = "true" ]; then
  echo "Training rollout debug logging enabled:"
  echo "  dir=${RUN_DIR}/train_rollouts"
  echo "  freq=${TRAIN_ROLLOUT_LOG_FREQ}"
  echo "  max_samples=${TRAIN_ROLLOUT_LOG_MAX_SAMPLES}"
  overrides+=(
    "trainer.rollout_data_dir=${RUN_DIR}/train_rollouts"
    "++trainer.rollout_data_freq=${TRAIN_ROLLOUT_LOG_FREQ}"
    "++trainer.rollout_data_max_samples=${TRAIN_ROLLOUT_LOG_MAX_SAMPLES}"
  )
fi

case "${ALGO}" in
  mixed_policy)
    CONFIG_NAME="mixed_policy"
    overrides+=(
      "algorithm.adv_estimator=grpo"
      "algorithm.use_kl_in_reward=false"
      "algorithm.mixed_policy.enable=true"
      "algorithm.mixed_policy.k_on=${K_ON}"
      "algorithm.mixed_policy.k_off=${K_OFF}"
      "actor_rollout_ref.mixed_policy.teacher_model_path=${TEACHER_RESOLVED}"
      "actor_rollout_ref.mixed_policy.speculative.use_entropy_aware_mixing=${USE_ENTROPY_AWARE_MIXING}"
      "actor_rollout_ref.mixed_policy.speculative.entropy_top_k=${ENTROPY_TOP_K}"
      "actor_rollout_ref.mixed_policy.speculative.entropy_aware_mixing=${ENTROPY_AWARE_MIXING}"
      "actor_rollout_ref.mixed_policy.speculative.entropy_aware_alpha=${ENTROPY_AWARE_ALPHA}"
    )
    if [ "${USE_ROLLOUT_CORRECTION}" = "false" ]; then
      overrides+=(
        "algorithm.rollout_correction.rollout_is=null"
        "algorithm.rollout_correction.rollout_is_threshold=null"
        "algorithm.rollout_correction.rollout_rs=null"
        "algorithm.rollout_correction.rollout_rs_threshold=null"
        "algorithm.rollout_correction.bypass_mode=true"
        "algorithm.rollout_correction.loss_type=ppo_clip"
      )
    fi
    ;;
  grpo)
    CONFIG_NAME="grpo"
    overrides+=(
      "algorithm.adv_estimator=grpo"
      "algorithm.use_kl_in_reward=false"
    )
    ;;
  *)
    echo "Unsupported ALGO=${ALGO}. Use mixed_policy|grpo"
    exit 1
    ;;
esac

HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
  --config-name="${CONFIG_NAME}" \
  "${overrides[@]}" \
  "$@"
