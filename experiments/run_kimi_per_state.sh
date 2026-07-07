#!/usr/bin/env bash
# Run the Kimi pilot one state at a time, with a fresh Python subprocess per
# state. Each Python invocation is short-lived and releases all its memory
# when it exits, so accumulated working set never grows. Use this on
# memory-constrained hosts where the long-running pilot crashes.
#
# Each iteration calls run_kimi_baseline.py with --resume and --max-states N,
# which:
#   1. Reads existing trace files (cheap, ~10ms) to reconstruct rows for the
#      states we've already finished.
#   2. Runs exactly ONE new state via the API.
#   3. Writes the full CSV (now N rows) to disk and exits.
# The next iteration repeats with N+1, picking up the row we just wrote.
#
# Usage (n=10 pilot, find-violations on default export):
#   ./experiments/run_kimi_per_state.sh \
#      --total 10 \
#      --output artifacts/kimi_results.csv \
#      --traces-dir artifacts/kimi_traces
#
# Usage (n=50 expansion, find-violations):
#   ./experiments/run_kimi_per_state.sh \
#      --total 100 \
#      --export-dir artifacts/relational_export_n50 \
#      --prompt-mode find_violations \
#      --output artifacts/kimi_n50_findv_results.csv \
#      --traces-dir artifacts/kimi_n50_findv_traces
#
# Usage (verify-legality control, legal-only):
#   ./experiments/run_kimi_per_state.sh \
#      --total 50 \
#      --export-dir artifacts/relational_export_n50 \
#      --prompt-mode verify_legality --label-filter legal \
#      --output artifacts/kimi_n50_verify_results.csv \
#      --traces-dir artifacts/kimi_n50_verify_traces

set -euo pipefail

PYTHON="${PYTHON:-python3}"
TOTAL=10
MAX_TOKENS=65536
OUTPUT="artifacts/kimi_results.csv"
TRACES_DIR="artifacts/kimi_traces"
EXPORT_DIR=""
PROMPT_MODE=""
LABEL_FILTER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --total)         TOTAL="$2"; shift 2;;
    --max-tokens)    MAX_TOKENS="$2"; shift 2;;
    --output)        OUTPUT="$2"; shift 2;;
    --traces-dir)    TRACES_DIR="$2"; shift 2;;
    --export-dir)    EXPORT_DIR="$2"; shift 2;;
    --prompt-mode)   PROMPT_MODE="$2"; shift 2;;
    --label-filter)  LABEL_FILTER="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Build the optional-flag tail once.
EXTRA=()
[[ -n "$EXPORT_DIR"   ]] && EXTRA+=(--export-dir "$EXPORT_DIR")
[[ -n "$PROMPT_MODE"  ]] && EXTRA+=(--prompt-mode "$PROMPT_MODE")
[[ -n "$LABEL_FILTER" ]] && EXTRA+=(--label-filter "$LABEL_FILTER")

echo "per-state pilot: total=$TOTAL  max_tokens=$MAX_TOKENS"
echo "  output=$OUTPUT"
echo "  traces=$TRACES_DIR"
[[ ${#EXTRA[@]} -gt 0 ]] && echo "  extra: ${EXTRA[*]}"
echo

for ((n=1; n<=TOTAL; n++)); do
  # Reclaimable RAM check. macOS keeps most of its RAM in "inactive" and
  # "speculative" file cache pages that the kernel reclaims on demand. The
  # naive "Pages free" metric drops to single-digit MB on a healthy machine.
  # The accurate metric is free + inactive + speculative -- that's the pool
  # macOS will hand to a new process before triggering OOM. On Linux the
  # `vm_stat` command is absent and this check is skipped.
  if command -v vm_stat >/dev/null 2>&1; then
    reclaim_mb=$(vm_stat | awk '
      /Pages free/        {f=$3}
      /Pages inactive/    {i=$3}
      /Pages speculative/ {s=$3}
      END {print int((f + i + s) * 16 / 1024)}')
    echo "--- iter $n/$TOTAL  reclaimable RAM ${reclaim_mb} MB ---"
    if [[ "$reclaim_mb" -lt 200 ]]; then
      echo "ERROR: reclaimable RAM below 200 MB; aborting to avoid OOM. Close apps and rerun." >&2
      exit 3
    fi
  else
    echo "--- iter $n/$TOTAL ---"
  fi

  # Each invocation: fresh Python process, processes states 1..n with --resume,
  # which means n-1 are read from traces and 1 runs fresh.
  "$PYTHON" -u experiments/run_kimi_baseline.py \
    --max-states "$n" \
    --max-tokens "$MAX_TOKENS" \
    --output "$OUTPUT" \
    --traces-dir "$TRACES_DIR" \
    --resume \
    "${EXTRA[@]}"

  # Brief pause for the OS to reclaim memory before next spawn.
  sleep 2
done

echo
echo "DONE: ran $TOTAL states. Final CSV at $OUTPUT"
