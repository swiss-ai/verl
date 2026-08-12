#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --job-name=verl_single_node_k8s
set -xeuo

export DATA_PATH="/capstor/store/cscs/swissai/infra01/reasoning/users/atazza/data_verl"
export ENV_PATH=$(realpath ../vllm_env.toml)
export WORKING_DIR=$(realpath .)
export LAUNCH_SCRIPT_DIR=$(realpath .)

srun --environment=$ENV_PATH --container-writable \
  bash ${WORKING_DIR}/trainer_script.bash