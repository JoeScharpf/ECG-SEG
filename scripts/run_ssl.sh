#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

GPUS="0"
LABEL_FRACTION="16"
METHOD="mean_teacher"
HEAD="unet"
SEED="0"
SMOKE="0"

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
        --method)
            METHOD="$2"
            shift 2
            ;;
        --head)
            HEAD="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --smoke)
            SMOKE="1"
            shift 1
            ;;
        -h|--help)
            echo "Usage: bash scripts/run_ssl.sh [options]"
            echo ""
            echo "Options:"
            echo "  --method {mean_teacher|fixmatch|mean_teacher_boundary}"
            echo "                                  SSL algorithm (default: mean_teacher)"
            echo "  --head {fcn|unet}                   Decode head (default: unet)"
            echo "  --gpus IDS                          GPU indices (default: 0)"
            echo "  --label-fraction N                  LUDB 1/N split (default: 16)"
            echo "  --seed N                            RNG seed (default: 0)"
            echo "  --smoke                             Train 1 epoch only (debug)"
            echo ""
            echo "Examples:"
            echo "  bash scripts/run_ssl.sh --method mean_teacher --head unet --gpus 0"
            echo "  bash scripts/run_ssl.sh --method fixmatch --head fcn --seed 1"
            echo "  bash scripts/run_ssl.sh --method mean_teacher_boundary --head unet --gpus 0"
            echo "  bash scripts/run_ssl.sh --method mean_teacher --head unet --smoke"
            echo ""
            echo "Run inside tmux on gpu2 for full 100-epoch jobs."
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

case "$METHOD" in
    mean_teacher|fixmatch|mean_teacher_boundary) ;;
    *)
        echo "Invalid --method: $METHOD (expected mean_teacher|fixmatch|mean_teacher_boundary)"
        exit 1
        ;;
esac

case "$HEAD" in
    fcn|unet) ;;
    *)
        echo "Invalid --head: $HEAD (expected fcn|unet)"
        exit 1
        ;;
esac

if [[ "$METHOD" == "mean_teacher_boundary" && "$HEAD" != "unet" ]]; then
    echo "mean_teacher_boundary currently ships a U-Net config only (--head unet)"
    exit 1
fi

if [[ "$HEAD" == "unet" ]]; then
    CONFIG_STEM="${METHOD}_unet"
else
    CONFIG_STEM="${METHOD}"
fi

BASE_CONFIG="../configs/base/resnet18/${CONFIG_STEM}.yaml"
OUTPUT_DIR="$REPO_ROOT/baseline/exps/resnet18/${CONFIG_STEM}"

if [[ "$SMOKE" == "1" ]]; then
    if [[ "$LABEL_FRACTION" != "16" || "$SEED" != "0" ]]; then
        echo "Smoke mode only supports --label-fraction 16 --seed 0"
        exit 1
    fi
    BENCH_CONFIG="../configs/bench/ludb/1over16_ssl_smoke.yaml"
    RUN_SUBDIR="ludb/1over16_smoke"
elif [[ "$SEED" == "0" && "$LABEL_FRACTION" == "16" ]]; then
    BENCH_CONFIG="../configs/bench/ludb/1over${LABEL_FRACTION}.yaml"
    RUN_SUBDIR="ludb/1over${LABEL_FRACTION}"
elif [[ "$LABEL_FRACTION" == "16" && ( "$SEED" == "1" || "$SEED" == "2" ) ]]; then
    BENCH_CONFIG="../configs/bench/ludb/1over16_ssl_seed${SEED}.yaml"
    RUN_SUBDIR="ludb/1over16_seed${SEED}"
else
    echo "Unsupported combo: label-fraction=${LABEL_FRACTION} seed=${SEED}"
    echo "Supported: fraction 16 with seeds 0/1/2 (seed 0 uses 1over16.yaml)."
    exit 1
fi

RUN_DIR="baseline/exps/resnet18/${CONFIG_STEM}/${RUN_SUBDIR}"
RESULTS_DIR="baseline/results/resnet18_${CONFIG_STEM}_ludb_1over${LABEL_FRACTION}"
if [[ "$SEED" != "0" ]]; then
    RESULTS_DIR="${RESULTS_DIR}_seed${SEED}"
fi

echo "=== SSL training (${METHOD} + ${HEAD}) ==="
echo "Repo root:    $REPO_ROOT"
echo "GPU(s):       $GPUS"
echo "Method:       $METHOD"
echo "Head:         $HEAD"
echo "Label frac:   1/${LABEL_FRACTION}"
echo "Seed:         $SEED"
echo "Base config:  $BASE_CONFIG"
echo "Bench config: $BENCH_CONFIG"
echo "Output dir:   ${RUN_DIR}/"
echo "Smoke:        $SMOKE"
echo ""
echo "Tip: run inside tmux — tmux new -s ssl"
echo ""

mkdir -p "$REPO_ROOT/baseline/exps"

if [[ -z "${CONDA_DEFAULT_ENV:-}" ]] || [[ "$CONDA_DEFAULT_ENV" != "semi_seg_ecg" ]]; then
    echo "Warning: conda env 'semi_seg_ecg' is not active."
    echo "Run: conda activate semi_seg_ecg"
fi

if [[ ! -f "semi-seg-ecg/configs/base/resnet18/${CONFIG_STEM}.yaml" ]]; then
    echo "Missing base config: semi-seg-ecg/configs/base/resnet18/${CONFIG_STEM}.yaml"
    exit 1
fi

INDEX_DIR="semi-seg-ecg/index/ludb"
for csv in LUDB_train_labeled_1over${LABEL_FRACTION}.csv LUDB_train_unlabeled.csv LUDB_valid.csv LUDB_test.csv; do
    if [[ ! -f "${INDEX_DIR}/${csv}" ]]; then
        echo "Missing index CSV: ${INDEX_DIR}/${csv}"
        echo "Run: bash scripts/setup_phase1.sh  (or ensure index/ludb is populated on gpu2)"
        exit 1
    fi
done

LABELED_N=$(wc -l < "${INDEX_DIR}/LUDB_train_labeled_1over${LABEL_FRACTION}.csv" | tr -d ' ')
UNLAB_N=$(wc -l < "${INDEX_DIR}/LUDB_train_unlabeled.csv" | tr -d ' ')
# subtract header
LABELED_N=$((LABELED_N - 1))
UNLAB_N=$((UNLAB_N - 1))
echo "Split check: labeled=${LABELED_N}  unlabeled=${UNLAB_N}"
echo ""

cd semi-seg-ecg

if [[ "$SMOKE" == "1" ]]; then
    echo "Smoke mode: 1 epoch via $BENCH_CONFIG"
fi

echo "--- Step 1/3: Train (${METHOD} + ${HEAD}) ---"
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
if [[ -d "$RUN_DIR" ]]; then
    python baseline/plot_results.py --run-dir "$RUN_DIR" --publish
else
    echo "Warning: run dir not found: $RUN_DIR (skipping plot publish)"
fi

echo ""
echo "SSL run complete."
echo "Full artifacts (gitignored): $RUN_DIR/"
echo "Git-tracked summary:         $RESULTS_DIR/ (if publish succeeded)"
