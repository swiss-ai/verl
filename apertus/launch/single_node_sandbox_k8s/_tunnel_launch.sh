set -ueox

export DATA_PATH="/capstor/store/cscs/swissai/infra01/reasoning/users/atazza/data_verl"
export ENV_PATH=$(realpath ../sglang_env.toml)
export WORKING_DIR=$(realpath .)
export LAUNCH_SCRIPT_DIR=$(realpath .)
export CONFIG_PATH=$(realpath .)
export VERL_DIR=$(realpath ../../../)
export USE_TOOLS=0

bash trainer_script.bash