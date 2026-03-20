#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --container-writable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --environment=reasoning
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

set -xeuo pipefail

WORKING_DIR=/capstor/scratch/cscs/msantelmo/inverse_batch/verl
cd "${WORKING_DIR}"
echo "Working directory: ${WORKING_DIR}"

# Model
model_name_or_path=swiss-ai/Apertus-8B-Instruct-2509

# Training data
train_path=./data/math/train.parquet
test_path=./data/math/test.parquet

# Training setting
rollout_n=4

# Wandb setting
project_name=Reinforce-Ada
datetime=$(date +'%Y%m%d-%H%M%S')
model_name=$(basename "${model_name_or_path}" | tr '/' '-')
exp_name=GRPO-${model_name}_n${rollout_n}_${datetime}

# Output
ckpts_dir="./outputs/${project_name}/${exp_name}"
mkdir -p "${ckpts_dir}/logs"

#################################################################
# Install verl
pip install --no-deps --no-cache-dir --force-reinstall -e .

HYDRA_FULL_ERROR=1 python -m verl.trainer.main_ppo \
					--config-name="grpo_math_base" \
					actor_rollout_ref.model.path="${model_name_or_path}" \
					data.train_files="['${train_path}']" \
					data.val_files="['${test_path}']" \
					actor_rollout_ref.rollout.n="${rollout_n}" \
					trainer.project_name="${project_name}" \
					trainer.experiment_name="${exp_name}" \
					trainer.default_local_dir="${ckpts_dir}" \
					trainer.validation_data_dir="${ckpts_dir}/validation/" \
					"$@"
