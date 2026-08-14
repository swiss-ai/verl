#!/usr/bin/env bash
#
# Submit a multi-node VERL training job with either a code-gym scheduler or a
# Kubernetes sandbox service configured for code reward evaluation.
# 
# Credits: https://github.com/swiss-ai/code-gym/tree/main

export USERNAME="$(whoami)"

export PROJECT_NAME="${PROJECT_NAME:-apertus-rl-tests}"
export SCRATCH_HOME="${SCRATCH_HOME:-/iopsstor/scratch/cscs/${USER}}"
# export WORKING_DIR="${WORKING_DIR:-${SCRATCH_HOME}/projects/verl}"
export HOME="${SCRATCH_HOME}"
export HF_HOME="${HF_HOME:-${SCRATCH_HOME}/huggingface}"
export CONFIG_PATH=$LAUNCH_SCRIPT_DIR
export VERL_DIR=$(realpath $LAUNCH_SCRIPT_DIR/../../../) # it's kinda ugly
# hacky tool-gym pythonpath, otherwise tool calls don't work. TODO: refactor package structure, imports are broken
export PYTHONPATH="/vllm:${VERL_DIR}:$(python3 -c "import tool_gym; print(tool_gym.__path__[0])"):${PYTHONPATH:-}"
echo "PYTHONPATH is: ${PYTHONPATH}"

export MODEL_NAME_OR_PATH="/capstor/store/cscs/swissai/infra01/apertus_1p5/hf_checkpoints/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_4200"
export TOKENIZERS_ROOT="${TOKENIZERS_ROOT:-/capstor/store/cscs/swissai/infra01/reasoning/models/tokenizers}"
export MULTIMODAL="false"
export TOKENIZER_NAME_OR_PATH="/capstor/store/cscs/swissai/infra01/apertus_1p5/hf_checkpoints/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_4200"

export TRAINING_DATA_DIR="${TRAINING_DATA_DIR:-/capstor/scratch/cscs/atazza/data_rl}"
# export TRAINING_DATA_DIR="${TRAINING_DATA_DIR:-/capstor/store/cscs/swissai/infra01/reasoning/data/RL-prod/apertus_1p5_ratio}"
export TRAIN_FILE="${TRAINING_DATA_DIR}/train.parquet"
export VAL_FILE="${TRAINING_DATA_DIR}/val.parquet"

export FORCE_THINKING="${FORCE_THINKING:-false}"
export THINK_PREFIX_TOKEN="${THINK_PREFIX_TOKEN:-<|inner_prefix|>}"
export ENABLE_THINKING="${ENABLE_THINKING:-false}"
export SEED="${SEED:-85}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export N_PER_ROUND="${N_PER_ROUND:-${ROLLOUT_N}}"
export USE_GROUP_FILTERING="${USE_GROUP_FILTERING:-false}"
export JOB_NAME="${JOB_NAME:-debug}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"

export ACTOR_PPO_MINI_BATCH_SIZE="${ACTOR_PPO_MINI_BATCH_SIZE:-}"
export ROLLOUT_TOTAL_ROLLOUT_STEPS="${ROLLOUT_TOTAL_ROLLOUT_STEPS:-}"
export TRAINER_TEST_FREQ="${TRAINER_TEST_FREQ:-}"
export TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-}"
export ASYNC_REQUIRE_BATCHES="${ASYNC_REQUIRE_BATCHES:-}"
export ASYNC_TRIGGER_PARAMETER_SYNC_STEP="${ASYNC_TRIGGER_PARAMETER_SYNC_STEP:-}"
export ASYNC_STALENESS_THRESHOLD="${ASYNC_STALENESS_THRESHOLD:-}"
export ASYNC_STEADY_WARMUP_STEPS="${ASYNC_STEADY_WARMUP_STEPS:-}"

# Sandbox configuration

export SANDBOX_BACKEND="kubernetes"
export KUBERNETES_SANDBOX_URL="https://sandbox-dev.swissai.svc.cscs.ch"
export PORT="${PORT:-8000}"
export POLL_SECS="${POLL_SECS:-3}"
export MAX_WAIT="${MAX_WAIT:-$((60 * 10))}"
export NO_FORMAT="${NO_FORMAT:-false}"  # disable tool-formatting (legacy plain-text rollouts)
export LONG_CONTEXT="${LONG_CONTEXT:-false}"  # enable QA-gym data and long-context config parameters
export SANDBOX_REWARD_CONTINUOUS="${SANDBOX_REWARD_CONTINUOUS:-false}" # default is binary reward
export QA_GYM_RERANKER_URL="${QA_GYM_RERANKER_URL:-https://api.swissai.svc.cscs.ch/v1/score}"
export DEFAULT_MEMORY_LIMIT_MB=512
export SANDBOX_REWARD_MAX_CONCURRENT=64

# meta

export RUN_NAME="smoke_test_sandbox"