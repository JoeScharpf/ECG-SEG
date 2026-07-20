#!/usr/bin/env bash
# Read-only Pix2Seq diagnostics (plan Step 1-2). Loads a trained checkpoint and
# prints failure-mode signals; does not train or modify anything.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

GPUS="0"
LABEL_FRACTION="16"
SPLIT="valid"
MAX_BATCHES="0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) GPUS="$2"; shift 2 ;;
        --label-fraction) LABEL_FRACTION="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --max-batches) MAX_BATCHES="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash scripts/diagnose_pix2seq.sh [--gpus 0] [--label-fraction 16] [--split valid] [--max-batches 0]"
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

OUTPUT_DIR="$REPO_ROOT/baseline/exps/pix2seq/scratch"
BASE_CONFIG="../configs/base/pix2seq/scratch.yaml"
BENCH_CONFIG="../configs/bench/ludb/1over${LABEL_FRACTION}.yaml"

export CUDA_VISIBLE_DEVICES="$GPUS"

# Persist the report next to the run's published results for the record.
REPORT_PATH="$REPO_ROOT/baseline/results/pix2seq_scratch_ludb_1over${LABEL_FRACTION}/diagnostics_${SPLIT}.txt"

cd semi-seg-ecg/src
python diagnose_pix2seq.py \
    -f "$BASE_CONFIG" \
    -o "$BENCH_CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --split "$SPLIT" \
    --max_batches "$MAX_BATCHES" \
    --out_report "$REPORT_PATH"
