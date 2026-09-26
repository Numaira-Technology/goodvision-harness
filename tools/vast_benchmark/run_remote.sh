#!/usr/bin/env bash
# Start one prepared cloud worker at a time and run the local full-pipeline test.
set -euo pipefail
BENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$BENCH_ROOT"
BENCH_MODEL="${1:---help}"
if [ "$BENCH_MODEL" = --help ]; then
  cat <<'HELP'
Usage: bash tools/vast_benchmark/run_remote.sh MODEL --host HOST --port PORT --identity KEY
MODEL: all, rfdetr-n/s/m/l, yolov8-n/s/m/l, or yolo26-n/s/m/l (one exact name).
Requires prepared goodvision-seg supervisor services on the cloud instance.
Optional --check only downloads/warms models and checks readiness; no VLM calls.
Prompts privately for OPENROUTER_API_KEY if it is not already exported.
Runs each model twice using the fixed main display configuration (0 s buffer).
Optional --delays 0.5,1,1.5,2 scans buffers, stopping each model at its first
two-repeat >=70% result; writes one batch report under outputs/seg-delay-sweep/.
Real VLM calls incur OpenRouter charges. This command starts the real benchmark.
HELP
  exit 0
fi
shift
BENCH_HOST=""
BENCH_PORT=""
BENCH_IDENTITY=""
BENCH_CHECK_ONLY=0
BENCH_DELAYS=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --check) BENCH_CHECK_ONLY=1; shift ;;
    --host|--port|--identity|--delays)
      [ "$#" -ge 2 ] || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in
        --host) BENCH_HOST="$2" ;;
        --port) BENCH_PORT="$2" ;;
        --identity) BENCH_IDENTITY="$2" ;;
        --delays) BENCH_DELAYS="$2" ;;
      esac
      shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$BENCH_MODEL" in
  all) BENCH_MODELS=(yolov8-n yolov8-s yolov8-m yolov8-l yolo26-n yolo26-s yolo26-m yolo26-l rfdetr-n rfdetr-s rfdetr-m rfdetr-l) ;;
  yolov8-[nsml]|yolo26-[nsml]|rfdetr-[nsml]) BENCH_MODELS=("$BENCH_MODEL") ;;
  *) echo "Unknown model: $BENCH_MODEL" >&2; exit 2 ;;
esac
[[ "$BENCH_PORT" =~ ^[0-9]+$ ]] && [ "$BENCH_PORT" -gt 0 ] && [ "$BENCH_PORT" -le 65535 ] || { echo 'A valid --port is required' >&2; exit 2; }
[ -n "$BENCH_HOST" ] && [ -f "$BENCH_IDENTITY" ] || { echo '--host and an existing --identity are required' >&2; exit 2; }
BENCH_PYTHON="$BENCH_ROOT/.venv-harness-benchmark/bin/python"
[ -x "$BENCH_PYTHON" ] || { echo 'Run tools/seg_pipeline/setup_client.sh first' >&2; exit 2; }
BENCH_RUN_ARGS=(-m tools.seg_pipeline)
BENCH_REPORT_ROOT="$BENCH_ROOT/outputs/seg-pipeline"
if [ -n "$BENCH_DELAYS" ]; then
  [ "$BENCH_CHECK_ONLY" -eq 0 ] || { echo '--check and --delays cannot be combined' >&2; exit 2; }
  "$BENCH_PYTHON" -c 'from tools.seg_pipeline.sweep import parse_delays; import sys; parse_delays(sys.argv[1])' "$BENCH_DELAYS"
  BENCH_REPORT_ROOT="$BENCH_ROOT/outputs/seg-delay-sweep/$(date -u +%Y%m%dT%H%M%SZ)-$$"
  BENCH_EXPECTED="$(IFS=,; echo "${BENCH_MODELS[*]}")"
  BENCH_RUN_ARGS=(-m tools.seg_pipeline.sweep --delays "$BENCH_DELAYS" --batch-dir "$BENCH_REPORT_ROOT" --expected-models "$BENCH_EXPECTED")
fi
if [ "$BENCH_CHECK_ONLY" -eq 0 ]; then
  if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    read -r -s -p 'OpenRouter API key (hidden): ' OPENROUTER_API_KEY
    printf '\n'
  fi
  [ -n "${OPENROUTER_API_KEY:-}" ] || { echo 'An OpenRouter key is required' >&2; exit 2; }
  export OPENROUTER_API_KEY
fi
export PYTHON_DOTENV_DISABLED=1

BENCH_SSH=(ssh)
if [ "$(uname -s)" = Darwin ]; then
  BENCH_SSH=(/usr/bin/ssh -o UseKeychain=yes)
fi
BENCH_TEMP="$(mktemp -d "${TMPDIR:-/tmp}/gv-seg.XXXXXX")"
BENCH_SOCKET="$BENCH_TEMP/ssh"
BENCH_SSH+=(-F /dev/null -i "$BENCH_IDENTITY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -S "$BENCH_SOCKET" -p "$BENCH_PORT" "root@$BENCH_HOST")
BENCH_ACTIVE=""
BENCH_TUNNEL_PID=""
cleanup() {
  if [ -n "$BENCH_ACTIVE" ]; then
    "${BENCH_SSH[@]}" "supervisorctl stop goodvision-seg:seg-$BENCH_ACTIVE" >/dev/null 2>&1 || true
  fi
  if [ -n "$BENCH_TUNNEL_PID" ]; then
    kill "$BENCH_TUNNEL_PID" 2>/dev/null || true
    wait "$BENCH_TUNNEL_PID" 2>/dev/null || true
  fi
  rm -f "$BENCH_SOCKET"
  rmdir "$BENCH_TEMP" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# Options after the destination would become a remote command; insert these
# before the destination by building the tunnel command from the same prefix.
BENCH_LAST=$((${#BENCH_SSH[@]} - 1))
"${BENCH_SSH[@]:0:$BENCH_LAST}" -M -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8092:127.0.0.1:8092 "${BENCH_SSH[$BENCH_LAST]}" &
BENCH_TUNNEL_PID=$!
for ((BENCH_ATTEMPT=0; BENCH_ATTEMPT<100; BENCH_ATTEMPT++)); do
  [ -S "$BENCH_SOCKET" ] && break
  kill -0 "$BENCH_TUNNEL_PID" 2>/dev/null || { echo 'SSH tunnel failed' >&2; exit 1; }
  sleep 0.2
done
[ -S "$BENCH_SOCKET" ] || { echo 'SSH tunnel timed out' >&2; exit 1; }
BENCH_FAILURES=0
for BENCH_CURRENT in "${BENCH_MODELS[@]}"; do
  printf '\nPreparing %s on the RTX 4090...\n' "$BENCH_CURRENT"
  "${BENCH_SSH[@]}" "supervisorctl stop 'goodvision-seg:*'" >/dev/null 2>&1 || true
  BENCH_ACTIVE="$BENCH_CURRENT"
  if "${BENCH_SSH[@]}" "supervisorctl start goodvision-seg:seg-$BENCH_CURRENT" && \
      "$BENCH_PYTHON" - "$BENCH_CURRENT" <<'PY'
import sys, time
import httpx
expected = sys.argv[1]
deadline = time.monotonic() + 600
attempt = 0
with httpx.Client(trust_env=False, timeout=3) as client:
    while time.monotonic() < deadline:
        try:
            response = client.get('http://127.0.0.1:8092/benchmark/info')
            response.raise_for_status()
            info = response.json()
            if info.get('ready') is True and info.get('model') == expected:
                print(f"Ready: {expected} on {info.get('gpu')}", flush=True)
                break
        except (httpx.HTTPError, ValueError):
            pass
        if attempt % 6 == 0:
            print('Waiting for model download/loading...', flush=True)
        attempt += 1
        time.sleep(5)
    else:
        raise SystemExit('Worker not ready after 10 minutes; inspect the cloud worker log.')
PY
  then
    if [ "$BENCH_CHECK_ONLY" -eq 1 ]; then
      printf 'Readiness passed: %s (no video benchmark or VLM request).\n' "$BENCH_CURRENT"
    elif ! "$BENCH_PYTHON" "${BENCH_RUN_ARGS[@]}" --model "$BENCH_CURRENT" --endpoint http://127.0.0.1:8092 --repeats 2 --allow-face-download; then
      BENCH_FAILURES=$((BENCH_FAILURES + 1))
    fi
  else
    BENCH_FAILURES=$((BENCH_FAILURES + 1))
    "${BENCH_SSH[@]}" "tail -n 30 /workspace/goodvision-4090-benchmark/outputs/worker-logs/$BENCH_CURRENT.log" || true
  fi
  "${BENCH_SSH[@]}" "supervisorctl stop goodvision-seg:seg-$BENCH_CURRENT" || true
  BENCH_ACTIVE=""
  if [ "$BENCH_FAILURES" -gt 0 ]; then
    printf '\nFAILED: %s did not produce valid trials; stopping remaining models. Inspect its report and logs.\n' "$BENCH_CURRENT" >&2
    break
  fi
done
if [ "$BENCH_CHECK_ONLY" -eq 1 ]; then
  printf '\nReadiness checks finished; failures: %s.\n' "$BENCH_FAILURES"
else
  printf '\nReports: %s/\n' "$BENCH_REPORT_ROOT"
  if [ -n "$BENCH_DELAYS" ]; then
    printf 'Combined delay-scan report: %s/report.html\n' "$BENCH_REPORT_ROOT"
  fi
fi
[ "$BENCH_FAILURES" -eq 0 ]
