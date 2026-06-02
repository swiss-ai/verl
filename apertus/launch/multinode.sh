#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --container-writable
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --environment=reasoning_fixed
#SBATCH --time=09:00:00
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -xeuo pipefail

# Prevent large core_nid* dumps when native libs crash inside the job.
ulimit -c 0

WORKING_DIR=/iopsstor/scratch/cscs/msantelmo/apertus_rl
HF_HOME=/capstor/scratch/cscs/msantelmo/huggingface
HF_HUB_CACHE_DIR=${HF_HOME}/hub

cd "${WORKING_DIR}"

MODEL_NAME_OR_PATH=/iopsstor/scratch/cscs/msantelmo/checkpoints/Apertus-1p5-8B-sft-capfilter-linear-it8816
TOKENIZER_NAME_OR_PATH="" # Use the same as the model by default
PROJECT_NAME=apertus-rl-tests
CONFIG_NAME=1p5_gmpo-loo  # 1p5_grpo

ROLLOUT_N=8
SEED=85
ENABLE_THINKING=false
FORCE_THINKING=false
THINK_PREFIX_TOKEN="<think>"

PY_DEPS_DIR=/iopsstor/scratch/cscs/msantelmo/python_deps/apertus_reward_py312
MATH_VERIFY_DEPS_DIR=/iopsstor/scratch/cscs/msantelmo/python_deps/math_verify_py312
NLTK_DATA_DIR=/iopsstor/scratch/cscs/msantelmo/nltk_data
DEPS_STAMP_DIR=/iopsstor/scratch/cscs/msantelmo/python_deps/.stamps
DEPS_STAMP=${DEPS_STAMP_DIR}/apertus_reward_py312.ready
DEPS_LOCK=${DEPS_STAMP_DIR}/apertus_reward_py312.lock
IF_REWARD_PIP_PACKAGES="immutabledict nltk langdetect absl-py emoji syllapy antlr4-python3-runtime==4.9.3"
MATH_VERIFY_PIP_PACKAGES="math_verify"

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

if [ "${FORCE_THINKING}" = true ]; then
  FORCE_THINKING_FLAGS=(
    "data.force_thinking_prefix=true"
    "data.thinking_prefix_token='${THINK_PREFIX_TOKEN}'"
  )
else
  FORCE_THINKING_FLAGS=()
fi

resolve_run_name() {
  local model_tag
  model_tag="$(basename "${MODEL_NAME_OR_PATH}" | tr '/:.' '-' | tr -c '[:alnum:]_-' '-')"
  if [ -z "${RUN_NAME}" ]; then
    RUN_NAME="${model_tag}"
  fi
  RUN_NAME="${RUN_NAME}_${CONFIG_NAME}_${NNODES}nodes"

  # if force thinking, add a tag to the run name
  if [ "${FORCE_THINKING}" = true ]; then
    RUN_NAME="${RUN_NAME}_force-think"
  fi
  RUN_NAME="${RUN_NAME}__seed${SEED}"
  RUN_NAME="${RUN_NAME}__$(date +%Y%m%d-%H%M%S)"
  RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
  mkdir -p "${RUN_DIR}"
}

resolve_run_name

build_overrides() {
  overrides=(
    "--config-name=${CONFIG_NAME}"
    "actor_rollout_ref.model.path=${MODEL_NAME_OR_PATH}"
    "data.train_files=['${TRAIN_FILE}']"
    "data.val_files=['${TEST_FILE}']"
    "data.seed=${SEED}"
    "data.apply_chat_template_kwargs.enable_thinking=${ENABLE_THINKING}"
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.experiment_name=${RUN_NAME}"
    "trainer.default_local_dir=${RUN_DIR}"
    "trainer.validation_data_dir=${RUN_DIR}/validation"
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
    "hydra.run.dir=${RUN_DIR}"
    "hydra.output_subdir=.hydra"
    "actor_rollout_ref.actor.data_loader_seed=${SEED}"
    "trainer.n_gpus_per_node=${GPUS_PER_NODE}"
    "trainer.nnodes=${NNODES}"
    "ray_kwargs.ray_init.num_cpus=null"
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=${PY_DEPS_DIR}:${WORKING_DIR}:${PYTHONPATH:-}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.NLTK_DATA=${NLTK_DATA_DIR}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.MATH_VERIFY_PYTHONPATH=${MATH_VERIFY_DEPS_DIR}"
  )

  if [ -n "${TOKENIZER_NAME_OR_PATH}" ]; then
    overrides+=("actor_rollout_ref.model.tokenizer_path=${TOKENIZER_NAME_OR_PATH}")
  fi

  if [ "${#FORCE_THINKING_FLAGS[@]}" -gt 0 ]; then
    overrides+=("${FORCE_THINKING_FLAGS[@]}")
  fi
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

prepare_shared_python_deps() {
  mkdir -p "${PY_DEPS_DIR}" "${MATH_VERIFY_DEPS_DIR}" "${NLTK_DATA_DIR}" "${DEPS_STAMP_DIR}"

  if [ -f "${DEPS_STAMP}" ]; then
    return 0
  fi

  while ! mkdir "${DEPS_LOCK}" 2>/dev/null; do
    if [ -f "${DEPS_STAMP}" ]; then
      return 0
    fi
    echo "Waiting for shared Python deps to be prepared by another process..."
    sleep 10
  done
  trap 'rmdir "${DEPS_LOCK}" 2>/dev/null || true; cleanup_ray' EXIT

  if [ ! -f "${DEPS_STAMP}" ]; then
    python3 -m pip install --no-cache-dir --upgrade --target "${PY_DEPS_DIR}" ${IF_REWARD_PIP_PACKAGES}
    python3 -m pip install --no-cache-dir --upgrade --target "${MATH_VERIFY_DEPS_DIR}" ${MATH_VERIFY_PIP_PACKAGES}
    python3 -c "import nltk; [nltk.download(pkg, download_dir='${NLTK_DATA_DIR}', quiet=True) for pkg in ('punkt', 'punkt_tab', 'stopwords', 'averaged_perceptron_tagger_eng')]"
    python3 -c "import sys; sys.path[:0] = ['${PY_DEPS_DIR}', '${WORKING_DIR}']; import absl, emoji, immutabledict, langdetect, nltk, syllapy, verl; print('Using shared deps from', '${PY_DEPS_DIR}'); print('Using verl from', verl.__file__)"
    python3 -c "import sys; sys.path[:0] = ['${MATH_VERIFY_DEPS_DIR}']; import math_verify; print('Using math_verify from', math_verify.__file__)"
    touch "${DEPS_STAMP}"
  fi

  rmdir "${DEPS_LOCK}" 2>/dev/null || true
  trap cleanup_ray EXIT
}


# Shared HuggingFace cache (prevents repeated downloads across runs/workers).
export HF_HOME
export HF_HUB_CACHE="${HF_HUB_CACHE_DIR}"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export NLTK_DATA="${NLTK_DATA_DIR}"
export PYTHONPATH="${PY_DEPS_DIR}:${WORKING_DIR}:${PYTHONPATH:-}"
export MATH_VERIFY_PYTHONPATH="${MATH_VERIFY_DEPS_DIR}"
export PYTHONUNBUFFERED="1"
export RAY_DEDUP_LOGS="0"
export NNODES

export WANDB_RUN_GROUP="${RUN_NAME%__seed${SEED}}"
export WANDB_NAME="${RUN_NAME}"

nodes_array=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
if [ "${#nodes_array[@]}" -lt "${NNODES}" ]; then
  echo "Expected ${NNODES} Slurm nodes, found ${#nodes_array[@]} in ${SLURM_JOB_NODELIST}" >&2
  exit 1
fi

prepare_shared_python_deps

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
