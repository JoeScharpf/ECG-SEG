#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

GPUS="0"
LABEL_FRACTION="16"
BENCH_CONFIG=""
RUN_SUBDIR=""

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
        --bench-config)
            BENCH_CONFIG="$2"
            shift 2
            ;;
        --run-subdir)
            RUN_SUBDIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: bash scripts/run_pix2seq.sh [--gpus 0] [--label-fraction 16] \\"
            echo "         [--bench-config ../configs/bench/ludb/<file>.yaml] [--run-subdir ludb/<name>]"
            echo "Run Pix2Seq multi-class model on LUDB (same splits as Phase 1 ResNet)."
            echo "--bench-config / --run-subdir let you run experiment variants (e.g. hi-res)"
            echo "into a separate exp dir. --run-subdir must match exp_name in the bench config."
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Default to the standard 1/N bench config unless an override was passed.
if [[ -z "$BENCH_CONFIG" ]]; then
    BENCH_CONFIG="../configs/bench/ludb/1over${LABEL_FRACTION}.yaml"
fi
# RUN_SUBDIR must match exp_name in the bench config (train() joins output_dir/exp_name).
if [[ -z "$RUN_SUBDIR" ]]; then
    RUN_SUBDIR="ludb/1over${LABEL_FRACTION}"
fi
# Flatten the subdir (ludb/1over16_hires -> ludb_1over16_hires) for the results folder.
RESULTS_NAME="pix2seq_scratch_$(echo "$RUN_SUBDIR" | tr '/' '_')"

RUN_DIR="baseline/exps/pix2seq/scratch/${RUN_SUBDIR}"
RESULTS_DIR="baseline/results/${RESULTS_NAME}"
OUTPUT_DIR="$REPO_ROOT/baseline/exps/pix2seq/scratch"
BASE_CONFIG="../configs/base/pix2seq/scratch.yaml"

echo "=== Pix2Seq Phase 2 training ==="
echo "Repo root:    $REPO_ROOT"
echo "GPU(s):       $GPUS"
echo "Label frac:   1/${LABEL_FRACTION}"
echo "Output dir:   ${RUN_DIR}/"
echo ""
echo "Tip: run inside tmux — tmux new -s pix2seq"
echo ""

mkdir -p "$REPO_ROOT/baseline/exps"

if [[ -z "${CONDA_DEFAULT_ENV:-}" ]] || [[ "$CONDA_DEFAULT_ENV" != "semi_seg_ecg" ]]; then
    echo "Warning: conda env 'semi_seg_ecg' is not active."
    echo "Run: conda activate semi_seg_ecg"
fi

cd semi-seg-ecg

echo "--- Step 1/3: Train (Pix2Seq, ResNet-18 encoder) ---"
bash scripts/train.sh \
    -f "$BASE_CONFIG" \
    -o "$BENCH_CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --gpus "$GPUS"

echo ""
echo "--- Step 2/3: Test (best MeanIoU checkpoint) ---"
bash scripts/test.sh \
    -f "$BASE_CONFIG" \
    -o "$BENCH_CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --gpu "$GPUS"

cd "$REPO_ROOT"

echo ""
echo "--- Step 3/3: Plot training curves + publish to baseline/results/ ---"
python baseline/plot_results.py --run-dir "$RUN_DIR" --publish

echo ""
echo "Pix2Seq run complete."
echo "Full artifacts (gitignored): $RUN_DIR/"
echo "Git-tracked summary:         $RESULTS_DIR/"
