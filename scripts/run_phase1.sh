#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

GPUS="0"
LABEL_FRACTION="16"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --label-fraction)
            LABEL_FRACTION="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: bash scripts/run_phase1.sh [--gpus 0] [--label-fraction 16]"
            echo "Run inside tmux on gpu2 for long training jobs."
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

RUN_DIR="baseline/exps/resnet18/scratch/ludb/1over${LABEL_FRACTION}"
OUTPUT_DIR="../baseline/exps/resnet18/scratch"
BENCH_CONFIG="configs/bench/ludb/1over${LABEL_FRACTION}.yaml"

echo "=== Phase 1 baseline training ==="
echo "Repo root:    $REPO_ROOT"
echo "GPU(s):       $GPUS"
echo "Label frac:   1/${LABEL_FRACTION}"
echo "Output dir:   ${RUN_DIR}/"
echo ""
echo "Tip: run inside tmux — tmux new -s phase1"
echo ""

mkdir -p "$REPO_ROOT/baseline/exps"

if [[ -z "${CONDA_DEFAULT_ENV:-}" ]] || [[ "$CONDA_DEFAULT_ENV" != "semi_seg_ecg" ]]; then
    echo "Warning: conda env 'semi_seg_ecg' is not active."
    echo "Run: conda activate semi_seg_ecg"
fi

cd semi-seg-ecg

echo "--- Step 1/3: Train (supervised ResNet-18) ---"
bash scripts/train.sh \
    -f configs/base/resnet18/scratch.yaml \
    -o "$BENCH_CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --gpus "$GPUS"

echo ""
echo "--- Step 2/3: Test (best MeanIoU checkpoint) ---"
bash scripts/test.sh \
    -f configs/base/resnet18/scratch.yaml \
    -o "$BENCH_CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --gpu "$GPUS"

cd "$REPO_ROOT"

echo ""
echo "--- Step 3/3: Plot training curves ---"
python baseline/plot_results.py --run-dir "$RUN_DIR"

echo ""
echo "Phase 1 run complete."
echo "Artifacts: $RUN_DIR/"
