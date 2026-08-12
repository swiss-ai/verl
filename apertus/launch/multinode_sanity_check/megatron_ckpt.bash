#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --job-name=multinode_verl_sanity

set -xeuo pipefail

export DATA_PATH="/capstor/store/cscs/swissai/infra01/reasoning/users/atazza/data_verl"
export ENV_PATH=$(realpath ../vllm_env_debug.toml)
export CONFIG_PATH=$(realpath .)
export WORKING_DIR=$(realpath .)
export VERL_DIR=$(realpath ../../../)
export NNODES=$SLURM_JOB_NUM_NODES
export TRAIN_NODES=8
export ROLLOUT_NODES=$((NNODES - TRAIN_NODES))
export MEGATRON_CKPT_PATH="/capstor/scratch/cscs/atazza/megatron_checkpoints/iter_0000000/"
export EXPERIMENT_NAME="GRPO-$(date +'%Y%m%dT%H%M%S')"
echo $EXPERIMENT_NAME

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

run_driver() {
  export PYTHONPATH="/vllm:${VERL_DIR}:${PYTHONPATH:-}"
  echo $PYTHONPATH
  NCCL_SOCKET_IFNAME=hsn VERL_LOGGING_LEVEL=INFO HYDRA_FULL_ERROR=1 python3 -m verl.experimental.fully_async_policy.fully_async_main \
    --config-path=${CONFIG_PATH} \
    --config-name="mn_sanity_check.yaml" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="${PYTHONPATH}" \
    data.train_files="${DATA_PATH}/train.parquet" \
    data.val_files="${DATA_PATH}/test.parquet" \
    trainer.nnodes="${TRAIN_NODES}" \
    rollout.nnodes="${ROLLOUT_NODES}" \
    actor_rollout_ref.rollout.nnodes="${ROLLOUT_NODES}" \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing="true" \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path="${MEGATRON_CKPT_PATH}" \
    trainer.experiment_name="${EXPERIMENT_NAME}"
}

export -f run_driver

wait_for_ray_cluster

srun --overlap --environment=$ENV_PATH --container-writable --nodes=1 --ntasks=1 -w "${head_node}" \
  bash -c "run_driver"
