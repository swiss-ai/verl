#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --container-writable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --environment=reasoning
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#SBATCH --job-name=gsm8k

set -euo pipefail

USERNAME=$(whoami)
WORKING_DIR=/iopsstor/scratch/cscs/${USERNAME}/apertus_rl
cd ${WORKING_DIR}

WANDB_PROJECT=apertus-rl-debugging
CONFIG_NAME=gsm8k_reproducibility

MODEL_PATH=/iopsstor/scratch/cscs/$(whoami)/checkpoints/Apertus-1p5-8B-sft-capfilter-linear-it8816-thinking-token-fixed
TOKENIZER_PATH=""

RUN_NAME="gsm8k_debug_$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR=./outputs/${WANDB_PROJECT}/${RUN_NAME}/

# Optional tokenizer path
tokenizer_flag=""
if [ -n "$TOKENIZER_PATH" ]; then
  tokenizer_flag="actor_rollout_ref.model.tokenizer_path=${TOKENIZER_PATH}"
fi

export PYTHONUNBUFFERED=1

# Install verl
pip install --no-deps --no-cache-dir --force-reinstall -e .

# Launch training 
datetime=$(date +%Y-%m-%d_%H-%M-%S)
echo "Starting training at ${datetime}"

HYDRA_FULL_ERROR=1 python -m verl.trainer.main_ppo \
	--config-name="${CONFIG_NAME}" \
	hydra.run.dir="${OUTPUT_DIR}" \
	hydra.output_subdir=".hydra" \
	trainer.nnodes=1 \
	trainer.experiment_name="${RUN_NAME}" \
	trainer.project_name="${WANDB_PROJECT}" \
	actor_rollout_ref.model.path="${MODEL_PATH}" \
	${tokenizer_flag} \
	"$@"
