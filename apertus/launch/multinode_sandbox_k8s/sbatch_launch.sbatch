#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=32
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --job-name=verl_multinode_k8s
set -xeuo

export LAUNCH_SCRIPT_DIR=$(realpath .)
export DATA_PATH="/capstor/store/cscs/swissai/infra01/reasoning/users/atazza/data_verl"
export ENV_PATH=$(realpath ../sglang_env.toml)
export CONFIG_PATH=$(realpath .)
export WORKING_DIR=$(realpath .)
export VERL_DIR=$(realpath ../../../)
export NNODES=$SLURM_JOB_NUM_NODES
export TRAIN_NODES=16
export USE_TOOLS=1
if [ $USE_TOOLS -eq 1]; then
  NO_FORMAT=0
else
  NO_FORMAT=1
fi
# TODO: fix this, the issue is that NO_FORMAT is necessary if tool usage is not specified, because 
# data is generated with appended tool calls regardless and it's necessary to specify how the chat template
# gets applied. The ENV_VAR format is a bit shaky and prone to become a pain to maintain
export NO_FORMAT

export ENABLE_THINKING="false"
export ROLLOUT_NODES=$((NNODES - TRAIN_NODES))
export MEGATRON_CKPT_PATH="/capstor/scratch/cscs/atazza/megatron_checkpoints/iter_0000000/"
export EXPERIMENT_NAME="GRPO-k8s-$(date +'%Y%m%dT%H%M%S')"
echo $EXPERIMENT_NAME

# startup
nodes_array=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
if [ "${#nodes_array[@]}" -lt "${NNODES}" ]; then
  echo "Expected ${NNODES} Slurm nodes, found ${#nodes_array[@]} in ${SLURM_JOB_NODELIST}" >&2
  exit 1
fi

head_node="${nodes_array[0]}"
head_node_ip="$(srun --environment=$ENV_PATH --nodes=1 --ntasks=1 -w "${head_node}" \
  bash -lc "cd '${WORKING_DIR}' && hostname --ip-address")"

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
export RAY_ADDRESS="${ip_head}"
export MASTER_ADDR="${head_node_ip}"
export MASTER_PORT="${port}"

echo "Starting Ray head at ${head_node} (${ip_head})"
srun --environment=$ENV_PATH --container-writable --nodes=1 --ntasks=1 -w "${head_node}" \
  bash -lc "cd '${WORKING_DIR}' && ray start --head --node-ip-address='${head_node_ip}' --port='${port}' --dashboard-host=0.0.0.0 --num-gpus 4 --block" &

sleep 10

worker_num=$((NNODES - 1))
worker_nodes=("${nodes_array[@]:1}")
for ((i = 1; i <= worker_num; i++)); do
  node_i="${nodes_array[$i]}"
  echo "Starting Ray worker ${i} at ${node_i}"
  srun --environment=$ENV_PATH --container-writable --nodes=1 --ntasks=1 -w "${node_i}" \
    bash -lc "cd '${WORKING_DIR}' && ray start --address '${ip_head}' --num-gpus 4 --block" &
  sleep 5
done

wait_for_ray_cluster() {
  local attempt
  for attempt in {1..60}; do
    if srun --overlap --environment=$ENV_PATH --container-writable --nodes=1 --ntasks=1 -w "${head_node}" \
      bash -lc 'python3 -c '"'"'import os, ray
ray.init(address=os.environ["ip_head"], ignore_reinit_error=True)
alive = sum(1 for node in ray.nodes() if node.get("Alive"))
expected = int(os.environ["NNODES"])
print(f"Ray nodes alive: {alive}/{expected}")
ray.shutdown()
raise SystemExit(0 if alive >= expected else 1)'"'"''; then
      return 0
    fi
    sleep 5
  done
  echo "Timed out waiting for Ray workers to join ${ip_head}" >&2
  return 1
}

wait_for_ray_cluster

srun --overlap --environment=$ENV_PATH --container-writable --nodes=1 --ntasks=1 -w "${head_node}" \
  bash ${WORKING_DIR}/trainer_script.bash
