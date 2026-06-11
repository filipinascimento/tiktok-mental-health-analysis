#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="mentalhealth2024"
RUN_VIDEO_COLLECTION=0
RUN_COMMENT_COLLECTION=0
SKIP_PREPROCESS=0
SKIP_SCORING=0
SKIP_TOPIC=0
SKIP_ANALYSIS=0
REUSE=1
ANALYSIS_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --collect-videos)
      RUN_VIDEO_COLLECTION=1
      shift
      ;;
    --collect-comments)
      RUN_COMMENT_COLLECTION=1
      shift
      ;;
    --no-reuse)
      REUSE=0
      shift
      ;;
    --skip-preprocess)
      SKIP_PREPROCESS=1
      shift
      ;;
    --skip-scoring)
      SKIP_SCORING=1
      shift
      ;;
    --skip-topic)
      SKIP_TOPIC=1
      shift
      ;;
    --skip-analysis)
      SKIP_ANALYSIS=1
      shift
      ;;
    --skip-umap|--skip-paper-render)
      ANALYSIS_ARGS+=("$1")
      shift
      ;;
    --umap-epochs|--max-comments-per-video|--random-state)
      ANALYSIS_ARGS+=("$1" "$2")
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p logs
LOG_FILE="logs/full_pipeline_${DATASET}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1

echo "Started: $(date -Is)"
echo "Working directory: $ROOT_DIR"
echo "Dataset: $DATASET"
echo "Python: $PYTHON_BIN"

if [[ "$RUN_VIDEO_COLLECTION" == "1" ]]; then
  "$PYTHON_BIN" Scripts/01_collect_videos.py --dataset "$DATASET"
  "$PYTHON_BIN" Scripts/02_prepare_comment_ids.py --dataset "$DATASET" --shuffle
fi

if [[ "$RUN_COMMENT_COLLECTION" == "1" ]]; then
  "$PYTHON_BIN" Scripts/03_collect_comments.py --dataset "$DATASET"
fi

if [[ "$SKIP_PREPROCESS" == "0" ]]; then
  "$PYTHON_BIN" Scripts/04_preprocess.py --dataset "$DATASET"
fi

if [[ "$SKIP_SCORING" == "0" ]]; then
  SCORE_ARGS=()
  if [[ "$REUSE" == "1" ]]; then
    SCORE_ARGS+=(--reuse)
  fi
  "$PYTHON_BIN" Scripts/05_score_sentiment_toxicity.py --dataset "$DATASET" "${SCORE_ARGS[@]}"
fi

if [[ "$SKIP_TOPIC" == "0" ]]; then
  TOPIC_ARGS=()
  if [[ "$REUSE" == "1" ]]; then
    TOPIC_ARGS+=(--reuse-embeddings --reuse-model)
  fi
  "$PYTHON_BIN" Scripts/06_embed_and_topic_model.py --dataset "$DATASET" "${TOPIC_ARGS[@]}"
fi

if [[ "$SKIP_ANALYSIS" == "0" ]]; then
  "$PYTHON_BIN" Scripts/07_generate_paper_outputs.py --dataset "$DATASET" "${ANALYSIS_ARGS[@]}"
fi

echo "Finished: $(date -Is)"
echo "Log: $LOG_FILE"
