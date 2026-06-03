#!/usr/bin/env bash
#
# Submit a code-gym sandbox scheduler job and, once reachable, submit the
# single-node VERL training job with SCHEDULER_URL injected.
# 
# Credits: https://github.com/swiss-ai/code-gym/tree/main
#
# Usage:
#   bash apertus/launch/singlenode_sandbox_split.sh [training overrides...]
#   bash apertus/launch/singlenode_sandbox_split.sh http://nidXXXXXX:8000 [training overrides...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCHED_SCRIPT="${SCRIPT_DIR}/_sandbox_scheduler.sbatch"
TRAIN_SCRIPT="${SCRIPT_DIR}/_singlenode_training_sandbox.sbatch"

CODE_GYM_DIR=/users/msantelmo/scratch/code-gym
PORT=8000
POLL_SECS=3
MAX_WAIT=600
GIVEN_URL="${SCHEDULER_URL:-}"

if [[ $# -gt 0 && "$1" =~ ^https?:// ]]; then
  GIVEN_URL="${1%/}"
  shift
fi

mkdir -p slurm_logs

log() {
  echo -e "$*" >&2
}

probe_ok() {
  local host="$1"
  local port="$2"
  local url="$3"
  if command -v nc >/dev/null 2>&1; then
    timeout 2s nc -z "${host}" "${port}"
  elif command -v curl >/dev/null 2>&1; then
    timeout 2s curl -sS --max-time 2 "${url}" >/dev/null
  else
    (exec 3<>/dev/tcp/"${host}"/"${port}") 2>/dev/null
  fi
}

if [[ -z "${GIVEN_URL}" ]]; then
  [[ -f "${SCHED_SCRIPT}" ]] || { echo "Missing ${SCHED_SCRIPT}" >&2; exit 1; }

  log "\n[1/4] Submit sandbox scheduler"
  SCHED_SUBMIT="$(sbatch --export=ALL,CODE_GYM_DIR="${CODE_GYM_DIR}",PORT="${PORT}" "${SCHED_SCRIPT}")"
  SCHED_ID="$(awk '{print $NF}' <<<"${SCHED_SUBMIT}")"
  [[ "${SCHED_ID}" =~ ^[0-9]+$ ]] || { echo "Failed to parse scheduler job id: ${SCHED_SUBMIT}" >&2; exit 1; }
  log "  -> Scheduler JobID: ${SCHED_ID}"

  log "\n[2/4] Wait for scheduler node"
  elapsed=0
  state=""
  node=""
  while :; do
    read -r state node < <(squeue -h -j "${SCHED_ID}" -o "%T %N" || true)
    [[ -n "${state}" ]] || state="PENDING"
    [[ -n "${node}" ]] || node="n/a"
    log "  state=${state} node=${node}"
    if [[ "${state}" == "RUNNING" && "${node}" != "n/a" ]]; then
      break
    fi
    if [[ "${state}" =~ (FAILED|CANCELLED|TIMEOUT|COMPLETED) ]]; then
      echo "Scheduler ended early: ${state}" >&2
      exit 1
    fi
    (( elapsed += POLL_SECS ))
    (( elapsed > MAX_WAIT )) && { echo "Timeout waiting for scheduler RUNNING" >&2; exit 1; }
    sleep "${POLL_SECS}"
  done

  NODE_CLEAN="$(sed -E 's/[\[\],]//g; s/ .*//g' <<<"${node}")"
  URL="http://${NODE_CLEAN}:${PORT}"
else
  log "\n[1/4 & 2/4] Reusing scheduler ${GIVEN_URL}"
  URL="${GIVEN_URL%/}"
  SCHED_ID="skipped"
  HOST_PORT="${URL#*://}"
  NODE_CLEAN="${HOST_PORT%:*}"
  PORT_FROM_URL="${HOST_PORT##*:}"
  if [[ "${PORT_FROM_URL}" != "${HOST_PORT}" ]]; then
    PORT="${PORT_FROM_URL}"
  fi
fi

log "\n[3/4] Probe scheduler ${NODE_CLEAN}:${PORT}"
elapsed=0
until probe_ok "${NODE_CLEAN}" "${PORT}" "${URL}"; do
  (( elapsed += POLL_SECS ))
  (( elapsed > MAX_WAIT )) && { echo "Port never opened: ${NODE_CLEAN}:${PORT}" >&2; exit 1; }
  sleep "${POLL_SECS}"
done
log "  -> Scheduler reachable at ${URL}"

log "\n[4/4] Submit VERL training"
TRAIN_SUBMIT="$(sbatch --export=ALL,SCHEDULER_URL="${URL}" "${TRAIN_SCRIPT}" "$@")"
TRAIN_ID="$(awk '{print $NF}' <<<"${TRAIN_SUBMIT}")"
[[ "${TRAIN_ID}" =~ ^[0-9]+$ ]] || { echo "Failed to parse training job id: ${TRAIN_SUBMIT}" >&2; exit 1; }
log "  -> Training JobID: ${TRAIN_ID}"

echo
echo "Monitor:"
if [[ "${SCHED_ID}" != "skipped" ]]; then
  echo "  squeue -j ${SCHED_ID},${TRAIN_ID}"
  echo "  tail -f slurm_logs/sandbox_scheduler_${SCHED_ID}.out"
else
  echo "  squeue -j ${TRAIN_ID}"
fi
echo "  tail -f slurm_logs/singlenode_sandbox_${TRAIN_ID}.out"
