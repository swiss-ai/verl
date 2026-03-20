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

echo "Working directory: ${WORKING_DIR}"

# Model
model_name_or_path=swiss-ai/Apertus-8B-Instruct-2509

# Training data
train_path=./data/math/train.parquet
test_path=./data/math/test.parquet

# Training setting
rollout_n=4
max_response_length=2048
data_train_batch_size=1024
actor_ppo_mini_batch_size=256

## Reinforce-Ada setting
max_rounds=8
n_per_round=4
n_pos=1
n_neg=1

# Wandb setting
project_name=Reinforce-Ada
datetime=$(date +'%Y%m%d-%H%M%S')
model_name=$(basename "${model_name_or_path}" | tr '/' '-')
exp_name=${model_name}_ags-${adaptive_group_sampling_enable}_npr-${n_per_round}_R-${max_rounds}_npos-${n_pos}_nneg-${n_neg}_${datetime}

# Output
ckpts_dir="./outputs/${project_name}/${exp_name}"
mkdir -p "${ckpts_dir}/logs"

#################################################################
# Install verl
pip install --no-deps --no-cache-dir --force-reinstall -e .

HYDRA_FULL_ERROR=1 python -m verl.trainer.main_ppo \
					--config-name="grpo_math_adaptive" \
					actor_rollout_ref.model.path="${model_name_or_path}" \
					data.train_files="['${train_path}']" \
					data.val_files="['${test_path}']" \
					data.train_batch_size="${data_train_batch_size}" \
					data.max_response_length="${max_response_length}" \
					algorithm.adaptive_group_sampling.min_positive_samples="${n_pos}" \
					algorithm.adaptive_group_sampling.min_negative_samples="${n_neg}" \
					algorithm.adaptive_group_sampling.max_rounds="${max_rounds}" \
					algorithm.adaptive_group_sampling.rollouts_per_round="${n_per_round}" \
					actor_rollout_ref.actor.ppo_mini_batch_size="${actor_ppo_mini_batch_size}" \
					actor_rollout_ref.rollout.n="${rollout_n}" \
					trainer.project_name="${project_name}" \
					trainer.experiment_name="${exp_name}" \
					trainer.default_local_dir="${ckpts_dir}" \
					trainer.validation_data_dir="${ckpts_dir}/validation/" \
					"$@"
