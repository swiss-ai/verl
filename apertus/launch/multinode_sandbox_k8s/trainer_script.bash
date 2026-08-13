#!/bin/bash

set -ueox

# tie tool gym tools to local path so it isn't all over the place
export TOOL_GYM_FUNCTION_TOOL_PATH=$(realpath $VERL_DIR/apertus/apertus_function_tools.py )
echo $TOOL_GYM_FUNCTION_TOOL_PATH

export RUN_DIR=$WORKING_DIR/outputs/$EXPERIMENT_NAME

source ${WORKING_DIR}/environment.sh
echo $PYTHONPATH

build_overrides() {
  overrides=(
    "--config-name=mn_k8s"
    "--config-path=${CONFIG_PATH}"
    "trainer.nnodes=${TRAIN_NODES}"
    "rollout.nnodes=${ROLLOUT_NODES}"
    "data.train_files=['${TRAIN_FILE}']"
    "data.val_files=['${VAL_FILE}']"
    "data.seed=${SEED}"
    "+data.force_thinking_prefix=${FORCE_THINKING}"
    "+data.thinking_prefix_token='${THINK_PREFIX_TOKEN}'"
    "data.apply_chat_template_kwargs.enable_thinking=${ENABLE_THINKING}"
    "++actor_rollout_ref.model.path=${MODEL_NAME_OR_PATH}"
    "++actor_rollout_ref.model.path=${TOKENIZER_NAME_OR_PATH}"
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
    "actor_rollout_ref.rollout.n_per_round=${N_PER_ROUND}"
    "data.tool_config_path=null"
    "actor_rollout_ref.actor.data_loader_seed=${SEED}"
    "actor_rollout_ref.actor.calculate_entropy=true"
    "actor_rollout_ref.rollout.skip_tokenizer_init=False"
    "+actor_rollout_ref.rollout.engine_kwargs.sglang.grammar_backend=llguidance"
    "actor_rollout_ref.rollout.reasoning_format=apertus2509"
    "reward.sandbox_fusion.memory_limit_mb=${DEFAULT_MEMORY_LIMIT_MB}"
    "reward.sandbox_fusion.max_concurrent=${SANDBOX_REWARD_MAX_CONCURRENT}"
    "+reward.sandbox_fusion.continuous=${SANDBOX_REWARD_CONTINUOUS}"
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.experiment_name=${EXPERIMENT_NAME}"
    "trainer.default_local_dir=${RUN_DIR}"
    "trainer.validation_data_dir=${RUN_DIR}/validation"
    "trainer.val_before_train=${VAL_BEFORE_TRAIN}"
    "ray_kwargs.ray_init.num_cpus=null"
    "+ray_kwargs.ray_init.runtime_env.env_vars.HOME=${HOME}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.RAY_enable_open_telemetry='false'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.OTEL_SDK_DISABLED='true'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.OTEL_METRICS_EXPORTER='none'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.OTEL_TRACES_EXPORTER='none'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.OTEL_LOGS_EXPORTER='none'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.NO_FORMAT='${NO_FORMAT}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH='${PYTHONPATH}'"
  )

  if [ $USE_TOOLS -eq 1 ]; then
    overrides+=(
      "actor_rollout_ref.rollout.multi_turn.enable=true"
      "+actor_rollout_ref.rollout.multi_turn.strict=true"
      "actor_rollout_ref.rollout.multi_turn.format=apertus2509"
      "actor_rollout_ref.rollout.multi_turn.max_assistant_turns=1"
      "+actor_rollout_ref.rollout.multi_turn.terminal_tool_names=[display_answers]"
      "actor_rollout_ref.rollout.multi_turn.tool_config_path=null"
      "actor_rollout_ref.rollout.multi_turn.function_tool_path=${TOOL_GYM_FUNCTION_TOOL_PATH}"
      "actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent"
      "data.function_tool_path=${TOOL_GYM_FUNCTION_TOOL_PATH}"
    )
  else
    overrides+=(
      "actor_rollout_ref.rollout.multi_turn.enable=false"
      "+actor_rollout_ref.rollout.multi_turn.strict=false"
      "actor_rollout_ref.rollout.agent.default_agent_loop=single_turn_agent"
      "data.function_tool_path=null"
      "actor_rollout_ref.rollout.multi_turn.tool_config_path=null"
      "actor_rollout_ref.rollout.multi_turn.function_tool_path=null"
      "actor_rollout_ref.rollout.multi_turn.terminal_tool_names=[]"
    )
  fi

  overrides+=("+ray_kwargs.ray_init.runtime_env.env_vars.KUBERNETES_SANDBOX_URL='${KUBERNETES_SANDBOX_URL}'")
  excluded_abilities+=(long_context_qa)

  if [ "${#excluded_abilities[@]}" -gt 0 ]; then
    excluded_abilities_override="$(IFS=,; printf '%s' "${excluded_abilities[*]}")"
    overrides+=("+data.exclude_abilities=[${excluded_abilities_override}]")
  fi

  overrides+=(
    "algorithm.filter_groups.enable=false"
    "algorithm.filter_groups.metric=null"
  )
}

build_overrides
cd $WORKING_DIR && python3 -m verl.experimental.fully_async_policy.fully_async_main \
  "${overrides[@]}" \
  "$@"

