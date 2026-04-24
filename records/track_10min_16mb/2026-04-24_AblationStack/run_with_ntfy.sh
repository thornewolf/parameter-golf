#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_with_ntfy.sh --run-id RUN_ID --config CONFIG_NAME -- COMMAND [ARGS...]

Runs COMMAND, tees stdout/stderr to a log file, and sends start/completion
notifications to ntfy.

Environment:
  NTFY_TOPIC       ntfy topic, default: thornewolf
  NTFY_URL         ntfy base URL, default: https://ntfy.sh
  NTFY_TOKEN       optional bearer token for private ntfy topics
  NTFY_CLICK       optional URL opened when tapping notification
  LOG_DIR          log output dir, default: logs/ntfy_runs

Example:
  NTFY_TOPIC=thornewolf ./run_with_ntfy.sh \
    --run-id A130_seed42 \
    --config A130_full_ttt \
    -- bash -lc 'DATA_DIR=./data RUN_ID=A130_seed42 SEED=42 TTT_ENABLED=1 torchrun --standalone --nproc_per_node=8 records/track_10min_16mb/2026-04-24_AblationStack/train_gpt_common_stack.py'
EOF
}

ntfy_url="${NTFY_URL:-https://ntfy.sh}"
ntfy_topic="${NTFY_TOPIC:-thornewolf}"
ntfy_token="${NTFY_TOKEN:-}"
ntfy_click="${NTFY_CLICK:-}"
log_dir="${LOG_DIR:-logs/ntfy_runs}"
run_id="${RUN_ID:-}"
config_name="${CONFIG_NAME:-manual}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      run_id="${2:?missing value for --run-id}"
      shift 2
      ;;
    --config)
      config_name="${2:?missing value for --config}"
      shift 2
      ;;
    --topic)
      ntfy_topic="${2:?missing value for --topic}"
      shift 2
      ;;
    --url)
      ntfy_url="${2:?missing value for --url}"
      shift 2
      ;;
    --log-dir)
      log_dir="${2:?missing value for --log-dir}"
      shift 2
      ;;
    --click)
      ntfy_click="${2:?missing value for --click}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -eq 0 ]]; then
  usage >&2
  exit 64
fi

if [[ -z "$run_id" ]]; then
  run_id="run_$(date -u +%Y%m%dT%H%M%SZ)"
fi

mkdir -p "$log_dir"
safe_run_id="$(printf '%s' "$run_id" | tr -c 'A-Za-z0-9_.=-' '_')"
log_file="$log_dir/${safe_run_id}.log"
started_epoch="$(date +%s)"
started_at="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
host="$(hostname 2>/dev/null || printf unknown)"
cwd="$(pwd)"
branch="$(git branch --show-current 2>/dev/null || printf unknown)"
commit="$(git rev-parse --short HEAD 2>/dev/null || printf unknown)"

send_ntfy() {
  local status="$1"
  local title="$2"
  local priority="$3"
  local tags="$4"
  local message="$5"

  local curl_args=(-fsS -X POST -H "Content-Type: application/json" --data-binary @- "$ntfy_url/")
  if [[ -n "$ntfy_token" ]]; then
    curl_args=(-fsS -X POST -H "Content-Type: application/json" -H "Authorization: Bearer $ntfy_token" --data-binary @- "$ntfy_url/")
  fi

  python3 - "$ntfy_topic" "$title" "$priority" "$tags" "$message" "$ntfy_click" <<'PY' | curl "${curl_args[@]}" >/dev/null 2>&1 || true
import json
import sys

topic, title, priority, tags, message, click = sys.argv[1:7]
payload = {
    "topic": topic,
    "title": title,
    "message": message,
    "priority": int(priority),
    "tags": [tag for tag in tags.split(",") if tag],
    "markdown": True,
}
if click:
    payload["click"] = click
print(json.dumps(payload), end="")
PY
}

extract_metrics() {
  if [[ ! -s "$log_file" ]]; then
    printf 'no log output captured'
    return
  fi
  grep -E 'val_loss|val_bpb|final_|quantized|Serialized model|Total submission size|artifact|stopping_early|Traceback|RuntimeError|CUDA out of memory' "$log_file" | tail -n 20 || true
}

start_message=$(
  cat <<EOF
Config: \`$config_name\`
Host: \`$host\`
Branch: \`$branch\` @ \`$commit\`
CWD: \`$cwd\`
Log: \`$log_file\`
Started: \`$started_at\`
EOF
)
send_ntfy "started" "Parameter Golf run started: $run_id" 3 "hourglass_flowing_sand,computer" "$start_message"

printf '[%s] starting run_id=%s config=%s host=%s branch=%s commit=%s\n' \
  "$started_at" "$run_id" "$config_name" "$host" "$branch" "$commit" | tee "$log_file"

set +e
"$@" 2>&1 | tee -a "$log_file"
status="${PIPESTATUS[0]}"
set -e

finished_epoch="$(date +%s)"
finished_at="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
elapsed="$((finished_epoch - started_epoch))"
metrics="$(extract_metrics)"

if [[ "$status" -eq 0 ]]; then
  title="Parameter Golf run completed: $run_id"
  priority=3
  tags="heavy_check_mark,computer"
  outcome="completed"
else
  title="Parameter Golf run failed: $run_id"
  priority=5
  tags="warning,computer"
  outcome="failed"
fi

completion_message=$(
  cat <<EOF
Status: **$outcome**
Exit code: \`$status\`
Config: \`$config_name\`
Host: \`$host\`
Branch: \`$branch\` @ \`$commit\`
Started: \`$started_at\`
Finished: \`$finished_at\`
Elapsed: \`${elapsed}s\`
Log: \`$log_file\`

\`\`\`
$metrics
\`\`\`
EOF
)
send_ntfy "$outcome" "$title" "$priority" "$tags" "$completion_message"

printf '[%s] finished run_id=%s status=%s exit_code=%s elapsed=%ss log=%s\n' \
  "$finished_at" "$run_id" "$outcome" "$status" "$elapsed" "$log_file" | tee -a "$log_file"

exit "$status"
