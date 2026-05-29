#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --container-writable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --environment=reasoning
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

set -euo pipefail

# Defaults
WORKING_DIR=/iopsstor/scratch/cscs/msantelmo/apertus_rl
model_path="/capstor/scratch/cscs/msantelmo/huggingface/hub/models--swiss-ai--Apertus-8B-Instruct-2509/snapshots/50761a511195fde9d958f62f3b6344329d4bd191"
run_name="debug_$(date +%Y%m%d-%H%M%S)"
wandb_project="apertus-rl-tests"

######################################################################
output_dir="./outputs/${wandb_project}/${run_name}_$(date +%Y%m%d-%H%M%S)/"

######################################################################
export PYTHONUNBUFFERED="1"
export RAY_DEDUP_LOGS="0"
export NLTK_DATA="${WORKING_DIR}/.cache/nltk_data"

cd ${WORKING_DIR}

# Prepare environment
pip install --no-deps --no-cache-dir --force-reinstall -e .
pip install math-verify langdetect immutabledict nltk absl-py requests
mkdir -p "${NLTK_DATA}"
python -m nltk.downloader -d "${NLTK_DATA}" punkt punkt_tab


# Run with Hydra output directory set to validation data directory
HYDRA_FULL_ERROR=1 python -m verl.trainer.main_ppo \
	--config-name="apertus" \
	hydra.run.dir="${output_dir}" \
	hydra.output_subdir=".hydra" \
	trainer.nnodes=1 \
	trainer.experiment_name="${run_name}" \
	trainer.project_name="${wandb_project}" \
	actor_rollout_ref.model.path="${model_path}" \
	data.train_files="['./data/debug/train.parquet']" \
	data.val_files="['./data/debug/val.parquet']" \
	actor_rollout_ref.actor.use_dynamic_bsz=true \
	actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
	critic.use_dynamic_bsz=true \
	critic.ppo_max_token_len_per_gpu=16384 \
	"$@"
