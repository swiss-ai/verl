#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --container-writable
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --environment=reasoning_fixed
#SBATCH --time=12:00:00
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -xeuo pipefail

# Prevent large core_nid* dumps when native libs crash inside the job.
ulimit -c 0

WORKING_DIR=/users/jgarcagi/iopsstor/projects/verl
HF_HOME=/users/jgarcagi/iopsstor/huggingface
HF_HUB_CACHE_DIR=${HF_HOME}/hub
NLTK_DATA_DIR=/users/jgarcagi/iopsstor/nltk_data
PY_DEPS_DIR=/users/jgarcagi/iopsstor/python_deps/py312
IF_REWARD_PIP_PACKAGES="immutabledict nltk langdetect absl-py emoji syllapy"

MODEL_NAME_OR_PATH=swiss-ai/Apertus-8B-Instruct-2509
# MODEL_NAME_OR_PATH=/users/jgarcagi/iopsstor/checkpoints/Apertus-1p5-8B-sft-capfilter-lr6e-5-constant-innovator-fix-it23409
TOKENIZER_NAME_OR_PATH=swiss-ai/Apertus-8B-Instruct-2509
# TOKENIZER_NAME_OR_PATH=/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok_instruct_thinking_token_fixed
PROJECT_NAME=apertus-rl-tests
CONFIG_NAME=grpo_math_base

ROLLOUT_N=8
SEED=85

NNODES="${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-2}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-4}}}"
GPUS_PER_NODE="${GPUS_PER_NODE%%(*}"
GPUS_PER_NODE="${GPUS_PER_NODE##*:}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-64}"

DATA_DIR=${WORKING_DIR}/data/apertus_demo_rl
TRAIN_FILE=${DATA_DIR}/train.parquet
TEST_FILE=${DATA_DIR}/val.parquet
OUTPUT_ROOT=${WORKING_DIR}/outputs/${PROJECT_NAME}
RUN_NAME="${RUN_NAME:-}"


####################################################################################################

ENABLE_THINKING=false
FORCE_THINKING=false
if [ "${FORCE_THINKING}" = true ]; then
  FORCE_THINKING_FLAGS=(
    "data.force_thinking_prefix=true"
    "data.thinking_prefix_token='<think>'"
  )
else
  FORCE_THINKING_FLAGS=()
fi

resolve_run_name() {
  local model_tag
  model_tag="$(basename "${MODEL_NAME_OR_PATH}" | tr '/:.' '-' | tr -c '[:alnum:]_-' '-')"
  local date_time
  date_time="$(date +'%Y%m%dT%H%M%S')"
  if [ -z "${RUN_NAME}" ]; then
    RUN_NAME="${PROJECT_NAME}__${model_tag}"
  fi
  # if force thinking, add a tag to the run name
  if [ "${FORCE_THINKING}" = true ]; then
    RUN_NAME="${RUN_NAME}_force-think"
  fi
  RUN_NAME="${RUN_NAME}_no-think"
  RUN_NAME="${RUN_NAME}__sd${SEED}_${date_time}"
  RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
  mkdir -p "${RUN_DIR}"
}


resolve_run_name

build_overrides() {
  overrides=(
    "--config-name=${CONFIG_NAME}"
    "actor_rollout_ref.model.path=${MODEL_NAME_OR_PATH}"
    "actor_rollout_ref.model.tokenizer_path=${TOKENIZER_NAME_OR_PATH}"
    "data.train_files=['${TRAIN_FILE}']"
    "data.val_files=['${TEST_FILE}']"
    "data.seed=${SEED}"
    "+data.apply_chat_template_kwargs.enable_thinking=${ENABLE_THINKING}"
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.experiment_name=${RUN_NAME}"
    "trainer.default_local_dir=${RUN_DIR}"
    "trainer.validation_data_dir=${RUN_DIR}/validation"
    "trainer.val_before_train=false"
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
    "trainer.total_epochs=10"
    # "trainer.total_training_steps=1000000"
    "hydra.run.dir=${RUN_DIR}"
    "algorithm.filter_groups.enable=true"
    "hydra.output_subdir=.hydra"
    "actor_rollout_ref.actor.data_loader_seed=${SEED}"
    "actor_rollout_ref.actor.checkpoint.save_contents=['hf_model']"
    "critic.checkpoint.save_contents=[]"
    "trainer.save_freq=-1"
    "trainer.test_freq=-1"
    "trainer.max_actor_ckpt_to_keep=null"
    "trainer.max_critic_ckpt_to_keep=null"
    "actor_rollout_ref.actor.strategy=fsdp2"
    "critic.strategy=fsdp2"
    "actor_rollout_ref.actor.use_dynamic_bsz=True"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384"
    "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True"
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True"
    "trainer.n_gpus_per_node=${GPUS_PER_NODE}"
    "trainer.nnodes=${NNODES}"
    "ray_kwargs.ray_init.num_cpus=null"
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=${PY_DEPS_DIR}:${WORKING_DIR}:${PYTHONPATH:-}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.NLTK_DATA=${NLTK_DATA_DIR}"
    "trainer.resume_mode=disable"
    "${FORCE_THINKING_FLAGS[@]}"
  )
}

cleanup_ray() {
  set +e
  srun --overlap --nodes="${NNODES}" --ntasks="${NNODES}" ray stop --force
}
trap cleanup_ray EXIT

wait_for_ray_cluster() {
  local attempt
  for attempt in {1..60}; do
    if python3 -c 'import os, ray
ray.init(address=os.environ["ip_head"], ignore_reinit_error=True)
alive = sum(1 for node in ray.nodes() if node.get("Alive"))
expected = int(os.environ["NNODES"])
print(f"Ray nodes alive: {alive}/{expected}")
ray.shutdown()
raise SystemExit(0 if alive >= expected else 1)'; then
      return 0
    fi
    sleep 5
  done
  echo "Timed out waiting for Ray workers to join ${ip_head}" >&2
  return 1
}

resolve_cached_hf_path() {
  local candidate="$1"
  local cache_entry="${HF_HUB_CACHE_DIR}/models--${candidate//\//--}"
  local commit

  if [ -d "${candidate}" ]; then
    printf '%s\n' "${candidate}"
    return 0
  fi
  if [ -f "${cache_entry}/refs/main" ]; then
    commit="$(cat "${cache_entry}/refs/main")"
    printf '%s\n' "${cache_entry}/snapshots/${commit}"
    return 0
  fi
  printf '%s\n' "${candidate}"
}

# Shared HuggingFace cache (prevents repeated downloads across runs/workers).
export HF_HOME
export HF_HUB_CACHE="${HF_HUB_CACHE_DIR}"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export NLTK_DATA="${NLTK_DATA_DIR}"
export PYTHONPATH="${PY_DEPS_DIR}:${WORKING_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="1"
export RAY_DEDUP_LOGS="0"
export NNODES

MODEL_NAME_OR_PATH="$(resolve_cached_hf_path "${MODEL_NAME_OR_PATH}")"
TOKENIZER_NAME_OR_PATH="$(resolve_cached_hf_path "${TOKENIZER_NAME_OR_PATH}")"
echo "Using model path: ${MODEL_NAME_OR_PATH}"
echo "Using tokenizer path: ${TOKENIZER_NAME_OR_PATH}"

set +x
export HF_TOKEN=""
export WANDB_API_KEY=""
set -x

export WANDB_RUN_GROUP="${RUN_NAME%__seed${SEED}}"
export WANDB_NAME="${RUN_NAME}"

nodes_array=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
if [ "${#nodes_array[@]}" -lt "${NNODES}" ]; then
  echo "Expected ${NNODES} Slurm nodes, found ${#nodes_array[@]} in ${SLURM_JOB_NODELIST}" >&2
  exit 1
fi

for node_i in "${nodes_array[@]}"; do
  srun --nodes=1 --ntasks=1 -w "${node_i}" \
    bash -lc "cd \"${WORKING_DIR}\" && export PYTHONPATH=\"${PY_DEPS_DIR}:${WORKING_DIR}:\${PYTHONPATH:-}\" && export NLTK_DATA=\"${NLTK_DATA_DIR}\" && mkdir -p \"${PY_DEPS_DIR}\" \"${NLTK_DATA_DIR}\" && python3 -m pip install --no-cache-dir --upgrade --target \"${PY_DEPS_DIR}\" ${IF_REWARD_PIP_PACKAGES} && python3 -c \"import nltk; [nltk.download(pkg, download_dir='${NLTK_DATA_DIR}', quiet=True) for pkg in ('punkt', 'punkt_tab', 'stopwords', 'averaged_perceptron_tagger_eng')]\" && python3 -m pip install --no-deps --no-cache-dir --force-reinstall -e . && python3 -c \"import importlib.metadata as m; import absl, emoji, immutabledict, langdetect, math_verify, nltk, omegaconf, syllapy, verl; print('Using python deps from', '${PY_DEPS_DIR}'); print('Using verl from', verl.__file__); print('Using math_verify from', math_verify.__file__); print('Using nltk from', nltk.__file__); print('Using immutabledict from', immutabledict.__file__); print('Using omegaconf from', omegaconf.__file__); print('antlr4 runtime', m.version('antlr4-python3-runtime'))\""
done

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "${head_node}" hostname --ip-address)

if [[ "${head_node_ip}" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<<"${head_node_ip}"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    head_node_ip=${ADDR[1]}
  else
    head_node_ip=${ADDR[0]}
  fi
  echo "IPV6 address detected. Using IPV4 address ${head_node_ip}"
fi

port=6379
ip_head=${head_node_ip}:${port}
export ip_head
export MASTER_ADDR="${head_node_ip}"
export MASTER_PORT="${port}"

echo "Starting Ray head at ${head_node} (${ip_head})"
srun --nodes=1 --ntasks=1 -w "${head_node}" \
  ray start --head --node-ip-address="${head_node_ip}" --port="${port}" \
  --dashboard-host=0.0.0.0 \
  --num-cpus "${CPUS_PER_TASK}" --num-gpus "${GPUS_PER_NODE}" --block &

sleep 10

worker_num=$((NNODES - 1))
for ((i = 1; i <= worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "Starting Ray worker ${i} at ${node_i}"
  srun --nodes=1 --ntasks=1 -w "${node_i}" \
    ray start --address "${ip_head}" \
    --num-cpus "${CPUS_PER_TASK}" --num-gpus "${GPUS_PER_NODE}" --block &
  sleep 5
done

wait_for_ray_cluster

build_overrides

HYDRA_FULL_ERROR=1 srun --overlap --nodes=1 --ntasks=1 -w "${head_node}" \
  python3 -m verl.trainer.main_ppo \
  "${overrides[@]}" \
  "$@"
