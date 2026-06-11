#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs
LOG_FILE="logs/run_latest.log"

{
  echo "Started: $(date -Is)"
  echo "Working directory: $ROOT_DIR"
  echo "Command: conda run -n tiktok --no-capture-output python Scripts/07_generate_paper_outputs.py $*"
  conda run -n tiktok --no-capture-output python Scripts/07_generate_paper_outputs.py "$@"
  echo "Finished: $(date -Is)"
} 2>&1 | tee "$LOG_FILE"
