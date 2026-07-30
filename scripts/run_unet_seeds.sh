#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

GPUS="0"
SEEDS="0 1 2"
LABEL_FRACTION="16"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --label-fraction)
            LABEL_FRACTION="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: bash scripts/run_unet_seeds.sh [options]"
            echo ""
            echo "Options:"
            echo "  --gpus IDS                 GPU indices (default: 0)"
            echo "  --seeds \"0 1 2\"            Seeds to run (default: 0 1 2)"
            echo "  --label-fraction N         LUDB 1/N split: 16|8|4|2 (default: 16)"
            echo ""
            echo "Multi-seed supervised ResNet-18 + U-Net on LUDB."
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

case "$LABEL_FRACTION" in
    16|8|4|2) ;;
    *)
        echo "Invalid --label-fraction: $LABEL_FRACTION (expected 16|8|4|2)"
        exit 1
        ;;
esac

OUTPUT_DIR="$REPO_ROOT/baseline/exps/resnet18/scratch_unet"
BASE_CONFIG="../configs/base/resnet18/scratch_unet.yaml"

echo "=== U-Net multi-seed run (LUDB 1/${LABEL_FRACTION}) ==="
echo "Repo root:  $REPO_ROOT"
echo "GPU(s):     $GPUS"
echo "Seeds:      $SEEDS"
echo "Fraction:   1/${LABEL_FRACTION}"
echo ""

mkdir -p "$REPO_ROOT/baseline/exps"

if [[ -z "${CONDA_DEFAULT_ENV:-}" ]] || [[ "$CONDA_DEFAULT_ENV" != "semi_seg_ecg" ]]; then
    echo "Warning: conda env 'semi_seg_ecg' is not active."
    echo "Run: conda activate semi_seg_ecg"
fi

INDEX_DIR="semi-seg-ecg/index/ludb"
LABELED_CSV="${INDEX_DIR}/LUDB_train_labeled_1over${LABEL_FRACTION}.csv"
if [[ ! -f "$LABELED_CSV" ]]; then
    echo "Missing labeled CSV: $LABELED_CSV"
    exit 1
fi

for SEED in $SEEDS; do
    BENCH_CONFIG="../configs/bench/ludb/1over${LABEL_FRACTION}_unet_seed${SEED}.yaml"
    RUN_SUBDIR="ludb/1over${LABEL_FRACTION}_unet_seed${SEED}"
    if [[ ! -f "semi-seg-ecg/configs/bench/ludb/1over${LABEL_FRACTION}_unet_seed${SEED}.yaml" ]]; then
        echo "Missing bench config for fraction=${LABEL_FRACTION} seed=${SEED}"
        exit 1
    fi

    echo ""
    echo "################  SEED ${SEED}  ################"

    cd "$REPO_ROOT/semi-seg-ecg"

    echo "--- Train (seed ${SEED}) ---"
    bash scripts/train.sh \
        -f "$BASE_CONFIG" \
        -o "$BENCH_CONFIG" \
        --output_dir "$OUTPUT_DIR" \
        --gpus "$GPUS"

    echo "--- Test (seed ${SEED}) ---"
    bash scripts/test.sh \
        -f "$BASE_CONFIG" \
        -o "$BENCH_CONFIG" \
        --output_dir "$OUTPUT_DIR" \
        --gpu "$GPUS"

    cd "$REPO_ROOT"
    echo "--- Publish (seed ${SEED}) ---"
    python baseline/plot_results.py \
        --run-dir "baseline/exps/resnet18/scratch_unet/${RUN_SUBDIR}" \
        --publish
done

echo ""
echo "=== Multi-seed summary (1/${LABEL_FRACTION}) ==="
python - "$SEEDS" "$LABEL_FRACTION" <<'PY'
import json
import sys
from pathlib import Path

repo = Path.cwd()
seeds = sys.argv[1].split()
frac = sys.argv[2]
vals = []
for s in seeds:
    p = repo / f"baseline/results/resnet18_scratch_unet_ludb_1over{frac}_unet_seed{s}/summary.json"
    if not p.exists():
        print(f"  seed {s}: MISSING {p}")
        continue
    d = json.loads(p.read_text())
    t = d.get("test_mean_iou")
    vals.append(t)
    print(f"  seed {s}: test MeanIoU = {t:.4f}  (best valid {d.get('best_valid_mean_iou'):.4f} @ {d.get('best_valid_mean_iou_epoch')})")

if len(vals) >= 2:
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    std = var ** 0.5
    print(f"\n  n={n}  mean={mean:.4f}  std={std:.4f}  min={min(vals):.4f}  max={max(vals):.4f}")
PY

echo ""
echo "U-Net multi-seed run complete."
