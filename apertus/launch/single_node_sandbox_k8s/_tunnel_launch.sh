set -ueox

export RAY_DEDUP_LOGS=0
export DATA_PATH="/capstor/store/cscs/swissai/infra01/reasoning/apertus_megatron/data_think"
export ENV_PATH=$(realpath ../sglang_env.toml)
export WORKING_DIR=$(realpath .)
export LAUNCH_SCRIPT_DIR=$(realpath .)
export CONFIG_PATH=$(realpath .)
export VERL_DIR=$(realpath ../../../)
export USE_TOOLS=0


ray start --head --labels="rollout=true,actor=true,trainer_head=true,rollout_head=true"
bash trainer_script.bash
ray stop
