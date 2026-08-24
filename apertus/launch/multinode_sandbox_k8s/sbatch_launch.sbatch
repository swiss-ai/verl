#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=64
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --job-name=verl_multinode_k8s
set -xeuo pipefail

export LAUNCH_SCRIPT_DIR=$(realpath .)
export DATA_PATH="/capstor/store/cscs/swissai/infra01/reasoning/apertus_megatron/data_think"
# export MEGATRON_CKPT_PATH="/capstor/store/cscs/swissai/infra01/reasoning/apertus_megatron/megatron_checkpoints/iter_0000000"
export ENV_PATH=$(realpath ../sglang_env.toml)
export CONFIG_PATH=$(realpath .)
export WORKING_DIR=$(realpath .)
export VERL_DIR=$(realpath ../../../)
export NNODES=$SLURM_JOB_NUM_NODES
export TRAIN_NODES=32
export USE_TOOLS=0
if [ $USE_TOOLS -eq 1 ]; then
  NO_FORMAT=0
else
  NO_FORMAT=1
fi
# TODO: fix this, the issue is that NO_FORMAT is necessary if tool usage is not specified, because 
# data is generated with appended tool calls regardless and it's necessary to specify how the chat template
# gets applied. The ENV_VAR format is a bit shaky and prone to become a pain to maintain
export NO_FORMAT

export ENABLE_THINKING="true"
export ROLLOUT_NODES=$((NNODES - TRAIN_NODES))
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


srun --nodes="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 -u --environment=$ENV_PATH --container-writable bash -lc '
  set -ueox pipefail
  unset HTTPS_PROXY HTTP_PROXY http_proxy https_proxy no_proxy
  export RAY_TMP_DIR="/tmp/ray_tmp_${SLURM_JOB_ID}"
  mkdir -p "$RAY_TMP_DIR"
  cd ${WORKING_DIR}
  if [[ "$SLURM_NODEID" == "0" ]]; then
    ray start --head \
      --node-ip-address="$MASTER_ADDR" \
      --port="$MASTER_PORT" \
      --temp-dir="$RAY_TMP_DIR" \
      --disable-usage-stats \
      --labels="actor=true,trainer_head=true"
    until [ "$(ray status 2>/dev/null | awk "/Active:/{flag=1;next}/Pending:/{flag=0}flag" | grep -c "node_" || true)" -ge "$SLURM_JOB_NUM_NODES" ]; do
      echo "Waiting for all nodes to join..."; sleep 1
    done
    bash trainer_script.bash
  else
    until (echo > /dev/tcp/"$MASTER_ADDR"/"$MASTER_PORT") >/dev/null 2>&1; do sleep 1; done
    if [[ $SLURM_NODEID -eq ${TRAIN_NODES} ]]; then
      labels="rollout=true,rollout_head=true"
    elif [[ $SLURM_NODEID -gt ${TRAIN_NODES} ]]; then
      labels="rollout=true"
    else
      labels="actor=true"
    fi
    ray start \
      --address="${RAY_ADDRESS}" \
      --node-ip-address="$(hostname -i | awk "{print \$1}")" \
      --temp-dir="$RAY_TMP_DIR" \
      --disable-usage-stats \
      --labels="$labels" \
      --block
  fi
'
