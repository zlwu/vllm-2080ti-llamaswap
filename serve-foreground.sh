#!/usr/bin/env bash
# Run the vLLM 2080 Ti Definitive launcher's resolved server in the foreground.
#
# llama-swap supervises exactly one process per model group: it starts `cmd`,
# waits for `checkEndpoint`, and terminates it again when another group is
# requested. `launcher.sh` deliberately daemonises (nohup + setsid + pid file)
# and exits, which llama-swap would read as "model died". This wrapper therefore
# reuses the launcher's configuration resolution (profiles, mode defaults, SM75
# runtime env, speculative config, CUDA graph sizing) and then `exec`s the
# server itself so the process llama-swap tracks *is* vLLM.
#
# It intentionally mirrors the sequence used by launcher.sh's `run_start_flow`:
#   collect_config_env -> apply_mode -> set_sm75_runtime_env -> launch_server
# and skips only the parts that belong to a human-facing service manager
# (menu, pid file, log file, readiness polling, smoke test).
#
# Usage: serve-foreground.sh --model-dir <dir> --profile <profile.env> \
#          --mode fast --gpu-devices 0,1 --tp-size 2 --pp-size 1 --port <port>

set -euo pipefail

RUNTIME_TREE=${RUNTIME_TREE:-/opt/vllm-2080ti}
cd "$RUNTIME_TREE"

# shellcheck source=/dev/null
# Sourcing is safe: launcher.sh only calls main() when executed directly.
source "$RUNTIME_TREE/launcher.sh"

parse_launcher_args "$@"
register_env_config_overrides
apply_launcher_path_defaults
collect_config_env
apply_mode
set_sm75_runtime_env
print_review
configure_dflash_download_route
check_checkpoint_mmap_policy

# launch_server() injects this into the child env; we exec directly instead.
if [[ -n "${HF_ACTIVE_ENDPOINT:-}" ]]; then
  export HF_ENDPOINT="$HF_ACTIVE_ENDPOINT"
fi

# llama-swap proxies to http://127.0.0.1:${PORT}; never bind the LAN here.
build_args 127.0.0.1

printf -v args_text '%q ' "${VLLM_ARGS[@]}"
{
  echo "----------------------------------------------------------------"
  echo "foreground vLLM launch: $(date '+%F %T %Z')"
  echo "runtime tree:  $RUNTIME_TREE"
  echo "model:         $MODEL_DIR"
  echo "draft model:   ${SPECULATIVE_MODEL:-none}"
  echo "profile:       ${PROFILE:-manual}   mode: $MODE"
  echo "served name:   $SERVED_NAME"
  echo "port:          $PORT (loopback only)"
  echo "command:       $RUNTIME_TREE/.venv/bin/python -m vllm.entrypoints.openai.api_server $args_text"
  echo "----------------------------------------------------------------"
} >&2

exec "$RUNTIME_TREE/.venv/bin/python" -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}"
