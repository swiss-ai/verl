#!/bin/bash

set -ueo

# tie tool gym tools to local path so it isn't all over the place
export TOOL_GYM_FUNCTION_TOOL_PATH=$(realpath ../../apertus_functional_tools.py )
echo $TOOL_GYM_FUNCTION_TOOL_PATH
export LAUNCH_SCRIPT_DIR=$(realpath .)

export RUN_DIR=$(realpath ./outputs)

source ./environment.sh

build_overrides() {
  overrides=(
    "--config-dir=${VERL_DIR}/apertus/launch/single_node_sandbox_k8s/"
    "--config-name=async_single_node"
    "data.train_files=['${TRAIN_FILE}']"
    "data.val_files=['${VAL_FILE}']"
    "data.seed=${SEED}"
    "+data.force_thinking_prefix=${FORCE_THINKING}"
    "+data.thinking_prefix_token='${THINK_PREFIX_TOKEN}'"
    "data.apply_chat_template_kwargs.enable_thinking=${ENABLE_THINKING}"
    "actor_rollout_ref.model.path=${MODEL_NAME_OR_PATH}"
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
    "actor_rollout_ref.rollout.n_per_round=${N_PER_ROUND}"
    "data.tool_config_path=null"
    "actor_rollout_ref.actor.data_loader_seed=${SEED}"
    "actor_rollout_ref.actor.calculate_entropy=true"
    "actor_rollout_ref.rollout.skip_tokenizer_init=False"
    "+actor_rollout_ref.rollout.engine_kwargs.sglang.grammar_backend=llguidance"
    "actor_rollout_ref.rollout.reasoning_format=apertus2509"
    "rollout.nnodes=1"
    "rollout.n_gpus_per_node=2"
    "reward.sandbox_fusion.memory_limit_mb=${DEFAULT_MEMORY_LIMIT_MB}"
    "reward.sandbox_fusion.max_concurrent=${SANDBOX_REWARD_MAX_CONCURRENT}"
    "+reward.sandbox_fusion.continuous=${SANDBOX_REWARD_CONTINUOUS}"
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.experiment_name=${RUN_NAME}"
    "trainer.default_local_dir=${RUN_DIR}"
    "trainer.validation_data_dir=${RUN_DIR}/validation"
    "trainer.val_before_train=${VAL_BEFORE_TRAIN}"
    "trainer.nnodes=1"
    "trainer.n_gpus_per_node=2"
    "ray_kwargs.ray_init.num_cpus=null"
    "+ray_kwargs.ray_init.runtime_env.env_vars.HOME=${HOME}"
    # "+ray_kwargs.ray_init.runtime_env.env_vars.NLTK_DATA=${NLTK_DATA_DIR}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.RAY_enable_open_telemetry='false'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.OTEL_SDK_DISABLED='true'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.OTEL_METRICS_EXPORTER='none'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.OTEL_TRACES_EXPORTER='none'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.OTEL_LOGS_EXPORTER='none'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.NO_FORMAT='${NO_FORMAT}'"
  )

# TODO: later
#   if is_truthy "${DEGENERATION_EARLY_STOP}" && [ -n "${DEGEN_STOP_CUSTOM_LOGIT_PROCESSOR:-}" ]; then
#     overrides+=(
#       "+actor_rollout_ref.rollout.engine_kwargs.sglang.enable_custom_logit_processor=true"
#       "+actor_rollout_ref.rollout.custom.custom_logit_processor='${DEGEN_STOP_CUSTOM_LOGIT_PROCESSOR}'"
#       "+actor_rollout_ref.rollout.custom.custom_params.ngram_n=2"
#       "+actor_rollout_ref.rollout.custom.custom_params.window=512"
#       "+actor_rollout_ref.rollout.custom.custom_params.ttr_threshold=0.2"
#       "+actor_rollout_ref.rollout.custom.custom_params.min_output_tokens=512"
#       "+actor_rollout_ref.rollout.custom.custom_params.stride=${DEGENERATION_EARLY_STOP_STRIDE}"
#       "+actor_rollout_ref.rollout.custom.custom_params.eos_token_id=${DEGEN_STOP_EOS_TOKEN_ID}"
#     )
#   fi

  if [[ "${NO_FORMAT}" ]]; then
    overrides+=(
      "actor_rollout_ref.rollout.multi_turn.enable=false"
      "actor_rollout_ref.rollout.agent.default_agent_loop=single_turn_agent"
      "data.function_tool_path=null"
      "actor_rollout_ref.rollout.multi_turn.tool_config_path=null"
      "actor_rollout_ref.rollout.multi_turn.function_tool_path=null"
      "actor_rollout_ref.rollout.multi_turn.terminal_tool_names=[]"
    )
  else
    overrides+=(
      "actor_rollout_ref.rollout.multi_turn.enable=true"
      "actor_rollout_ref.rollout.multi_turn.format=apertus2509"
      "actor_rollout_ref.rollout.multi_turn.max_assistant_turns=1"
      "+actor_rollout_ref.rollout.multi_turn.terminal_tool_names=[display_answers]"
      "actor_rollout_ref.rollout.multi_turn.tool_config_path=null"
      "actor_rollout_ref.rollout.multi_turn.function_tool_path=${TOOL_GYM_FUNCTION_TOOL_PATH}"
      "actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent"
      "data.function_tool_path=${TOOL_GYM_FUNCTION_TOOL_PATH}"
    )
  fi

#   if [ -n "${reward_sandbox_url}" ]; then
#     overrides+=(
#       "reward.sandbox_fusion.url='${reward_sandbox_url}'"
#     )
#   fi
  overrides+=("+ray_kwargs.ray_init.runtime_env.env_vars.KUBERNETES_SANDBOX_URL='${KUBERNETES_SANDBOX_URL}'")
  excluded_abilities+=(long_context_qa)

  if [ "${#excluded_abilities[@]}" -gt 0 ]; then
    excluded_abilities_override="$(IFS=,; printf '%s' "${excluded_abilities[*]}")"
    overrides+=("+data.exclude_abilities=[${excluded_abilities_override}]")
  fi

#   if [ -n "${RESOLVED_TOKENIZER}" ]; then
#     overrides+=("actor_rollout_ref.model.tokenizer_path=${RESOLVED_TOKENIZER}")
#     overrides+=("actor_rollout_ref.model.lazy_tokenizer=true")
#   fi

  # overrides+=("actor_rollout_ref.model.load_processor=false")
  # overrides+=("actor_rollout_ref.actor.ppo_mini_batch_size=${ACTOR_PPO_MINI_BATCH_SIZE}")
  # overrides+=("rollout.total_rollout_steps=${ROLLOUT_TOTAL_ROLLOUT_STEPS}")
  # overrides+=("trainer.test_freq=${TRAINER_TEST_FREQ}")
  # overrides+=("trainer.save_freq=${TRAINER_SAVE_FREQ}")
  # overrides+=("async_training.require_batches=${ASYNC_REQUIRE_BATCHES}")
  # overrides+=("async_training.trigger_parameter_sync_step=${ASYNC_TRIGGER_PARAMETER_SYNC_STEP}")
  # overrides+=("async_training.staleness_threshold=${ASYNC_STALENESS_THRESHOLD}")
  # overrides+=("async_training.steady_warmup_steps=${ASYNC_STEADY_WARMUP_STEPS}")
  overrides+=(
    "algorithm.filter_groups.enable=false"
    "algorithm.filter_groups.metric=null"
  )
}

build_overrides
python3 check_dataset.py \
  "${overrides[@]}" \
  "$@"

