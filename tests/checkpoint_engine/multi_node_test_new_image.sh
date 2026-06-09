#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --container-writable
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:20:00
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --output=slurm_logs/test_%j.out
#SBATCH --error=slurm_logs/test_%j.err

set -xeuo pipefail

# Prevent large core_nid* dumps when native libs crash inside the job.
ulimit -c 0

export WORKING_DIR=/capstor/store/cscs/swissai/infra01/reasoning/users/atazza/
export NNODES="${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-2}}"

# fabric flags, TODO: include in toml
export FI_PROVIDER="cxi"
export FI_CXI_DISABLE_HOST_REGISTER="1" 
export FI_CXI_RDZV_PROTO="alt_read" 
export FI_CXI_RDZV_EAGER_SIZE="0" 
export FI_CXI_RDZV_GET_MIN="0" 
export FI_CXI_RDZV_THRESHOLD="0" 
export FI_CXI_RX_MATCH_MODE="hybrid" 
export FI_MR_CACHE_MONITOR="userfaultfd"
export FI_MR_CACHE_MAX_COUNT="4096" 
export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD="16777216"
export RAY_DEDUP_LOGS=0

install_deps() {
  cd ${WORKING_DIR}/sgl-test/verl && pip install --no-deps -e .
}

run_driver() {
  cd ${WORKING_DIR}/sgl-test/verl/tests/checkpoint_engine
  python multi_node_cxi_test.py
}

unset HTTPS_PROXY HTTP_PROXY http_proxy https_proxy no_proxy
export MASTER_PORT="${RAY_PORT:-6379}"
export MASTER_ADDR="$(hostname -i | awk '{print $1}')"
export RAY_ADDRESS="${MASTER_ADDR}:${MASTER_PORT}"
export -f run_driver install_deps

# === END RAY SETUP ===
srun --nodes="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 -u --environment=/capstor/scratch/cscs/atazza/async_rl.toml --container-writable bash -lc '
  unset HTTPS_PROXY HTTP_PROXY http_proxy https_proxy no_proxy
  install_deps

  unset ROCR_VISIBLE_DEVICES
  export RAY_DEDUP_LOGS=0
  export RAY_TMP_DIR="/tmp/ray_tmp_${SLURM_JOB_ID}"
  mkdir -p "$RAY_TMP_DIR"
  if [[ "$SLURM_PROCID" == "0" ]]; then
    ray start --head \
      --node-ip-address="$MASTER_ADDR" \
      --port="$MASTER_PORT" \
      --num-cpus="${SLURM_CPUS_ON_NODE:-288}" \
      --num-gpus="${SLURM_GPUS_ON_NODE:-4}" \
      --temp-dir="$RAY_TMP_DIR" \
      --disable-usage-stats
    until [ "$(ray status 2>/dev/null | awk "/Active:/{flag=1;next}/Pending:/{flag=0}flag" | grep -c "node_" || true)" -ge "$SLURM_JOB_NUM_NODES" ]; do
      echo "Waiting for all nodes to join..."; sleep 1
    done

    run_driver
  else
    until (echo > /dev/tcp/"$MASTER_ADDR"/"$MASTER_PORT") >/dev/null 2>&1; do sleep 1; done
    ray start \
      --address="${RAY_ADDRESS}" \
      --node-ip-address="$(hostname -i | awk "{print \$1}")" \
      --num-cpus="${SLURM_CPUS_ON_NODE:-288}" \
      --num-gpus="${SLURM_GPUS_ON_NODE:-4}" \
      --temp-dir="$RAY_TMP_DIR" \
      --disable-usage-stats \
      --block
  fi
'