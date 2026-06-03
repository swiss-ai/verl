#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash apertus/launch/merge_fsdp_to_hf.sh <fsdp_actor_ckpt_dir> [hf_target_dir] [extra verl.model_merger args...]

Example:
  bash apertus/launch/merge_fsdp_to_hf.sh \
    outputs/apertus-rl-tests/Apertus-1p5-8B-sft-capfilter-linear-it8816-_16nodes__seed85/global_step_180/actor

By default, hf_target_dir is <fsdp_actor_ckpt_dir>/hf.
Pass extra verl.model_merger args after the target dir, or directly after
fsdp_actor_ckpt_dir when using the default target, for example: --trust-remote-code.
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if [ "$#" -lt 1 ]; then
  usage >&2
  exit 1
fi

INVOCATION_DIR="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKING_DIR="${WORKING_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

LOCAL_DIR="$1"
shift

if [ ! -d "${LOCAL_DIR}" ]; then
  echo "FSDP checkpoint directory does not exist: ${LOCAL_DIR}" >&2
  exit 1
fi
LOCAL_DIR="$(cd "${LOCAL_DIR}" && pwd)"

if [ "$#" -gt 0 ] && [[ "${1}" != -* ]]; then
  TARGET_DIR="$1"
  shift
else
  TARGET_DIR="${LOCAL_DIR%/}/hf"
fi

if [[ "${TARGET_DIR}" != /* ]]; then
  TARGET_DIR="${INVOCATION_DIR}/${TARGET_DIR}"
fi

if [ ! -f "${LOCAL_DIR}/fsdp_config.json" ]; then
  echo "Missing ${LOCAL_DIR}/fsdp_config.json; expected an actor FSDP checkpoint directory." >&2
  exit 1
fi

if ! compgen -G "${LOCAL_DIR}/model_world_size_*_rank_*.pt" >/dev/null; then
  echo "No model_world_size_*_rank_*.pt files found in ${LOCAL_DIR}" >&2
  exit 1
fi

if [ ! -d "${LOCAL_DIR}/huggingface" ]; then
  echo "Missing ${LOCAL_DIR}/huggingface; verl model_merger needs the saved HF config/tokenizer there." >&2
  exit 1
fi

cd "${WORKING_DIR}"
mkdir -p "${TARGET_DIR}"

echo "Merging FSDP checkpoint:"
echo "  local_dir:  ${LOCAL_DIR}"
echo "  target_dir: ${TARGET_DIR}"

python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${LOCAL_DIR}" \
  --target_dir "${TARGET_DIR}" \
  "$@"

echo "Merged Hugging Face checkpoint written to ${TARGET_DIR}"
