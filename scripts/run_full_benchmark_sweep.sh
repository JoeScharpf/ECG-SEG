#!/usr/bin/env bash
# Launch full boundary-MT benchmark sweep; skip cells with published summary.json.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

GPUS=(4 5 6 7)
DRY_RUN=0
DATASETS=(ludb qtdb isp zhejiang cross_domain)
FRACTIONS=(16 8 4 2)
SEEDS=(0 1 2)
MAX_PARALLEL="${#GPUS[@]}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            IFS=',' read -r -a GPUS <<< "$2"
            MAX_PARALLEL="${#GPUS[@]}"
            shift 2
            ;;
        --datasets)
            IFS=',' read -r -a DATASETS <<< "$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift 1
            ;;
        -h|--help)
            echo "Usage: bash scripts/run_full_benchmark_sweep.sh [--gpus 4,5,6,7] [--datasets ludb,qtdb,...] [--dry-run]"
            echo "Skips cells whose baseline/results/.../summary.json already exists."
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

results_summary_for() {
    local dataset="$1" frac="$2" seed="$3"
  python3 - "$REPO_ROOT" "$dataset" "$frac" "$seed" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, "baseline")
from benchmark_paths import cell_paths

repo, dataset, frac, seed = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
frac_i = None if dataset == "cross_domain" else int(frac)
p = cell_paths(repo, dataset, frac_i, seed, smoke=False)
print(Path(p["results_dir"]) / "summary.json")
PY
}

job_cmd() {
    local dataset="$1" frac="$2" seed="$3" gpu="$4"
    if [[ "$dataset" == "cross_domain" ]]; then
        echo "bash scripts/run_benchmark.sh --dataset cross_domain --seed ${seed} --gpus ${gpu}"
    else
        echo "bash scripts/run_benchmark.sh --dataset ${dataset} --label-fraction ${frac} --seed ${seed} --gpus ${gpu}"
    fi
}

JOBS=()

for dataset in "${DATASETS[@]}"; do
    if [[ "$dataset" == "cross_domain" ]]; then
        fracs=(0)
    else
        fracs=("${FRACTIONS[@]}")
    fi
    for frac in "${fracs[@]}"; do
        for seed in "${SEEDS[@]}"; do
            if [[ "$dataset" == "cross_domain" ]]; then
                summary="$(results_summary_for "$dataset" 0 "$seed")"
            else
                summary="$(results_summary_for "$dataset" "$frac" "$seed")"
            fi
            if [[ -f "$summary" ]]; then
                echo "SKIP existing: $summary"
                continue
            fi
            if [[ "$dataset" == "cross_domain" ]]; then
                JOBS+=("${dataset}|0|${seed}")
            else
                JOBS+=("${dataset}|${frac}|${seed}")
            fi
        done
    done
done

echo "=== Full boundary-MT benchmark sweep ==="
echo "Pending jobs: ${#JOBS[@]}"
echo "GPUs:         ${GPUS[*]}"
echo "Dry run:      $DRY_RUN"
echo ""

if [[ ${#JOBS[@]} -eq 0 ]]; then
    echo "Nothing to run."
    exit 0
fi

if [[ "$DRY_RUN" == "1" ]]; then
    idx=0
    for job in "${JOBS[@]}"; do
        IFS='|' read -r dataset frac seed <<< "$job"
        gpu="${GPUS[$((idx % MAX_PARALLEL))]}"
        echo "[dry] gpu=${gpu} $(job_cmd "$dataset" "$frac" "$seed" "$gpu")"
        idx=$((idx + 1))
    done
    exit 0
fi

pending=("${JOBS[@]}")
declare -A SESSION_JOB=()

launch_one() {
    local job="$1" gpu="$2"
    IFS='|' read -r dataset frac seed <<< "$job"
    local sess="bench_${dataset}_1over${frac}_s${seed}"
  sess="${sess//cross_domain_1over0/cross_merged}"
    local cmd log
    cmd="$(job_cmd "$dataset" "$frac" "$seed" "$gpu")"
    log="bench_${dataset}_1over${frac}_seed${seed}.log"
    tmux has-session -t "$sess" 2>/dev/null && tmux kill-session -t "$sess"
    tmux new-session -d -s "$sess" \
        "cd '$REPO_ROOT'; \
         source ~/miniconda3/etc/profile.d/conda.sh; \
         conda activate semi_seg_ecg; \
         $cmd 2>&1 | tee '$log'; \
         echo DONE > '${log}.done'"
    SESSION_JOB["$sess"]="$job|$gpu|$log"
    echo "LAUNCH $sess gpu=$gpu -> $log"
}

running_count() {
    local n=0 s
    for s in "${!SESSION_JOB[@]}"; do
        if tmux has-session -t "$s" 2>/dev/null; then
            n=$((n + 1))
        fi
    done
    echo "$n"
}

reap_finished() {
    local s jobmeta
    for s in "${!SESSION_JOB[@]}"; do
        if ! tmux has-session -t "$s" 2>/dev/null; then
            jobmeta="${SESSION_JOB[$s]}"
            echo "FINISHED $s ($jobmeta)"
            unset "SESSION_JOB[$s]"
        fi
    done
}

gpu_in_use() {
    local g="$1" s meta
    for s in "${!SESSION_JOB[@]}"; do
        if tmux has-session -t "$s" 2>/dev/null; then
            meta="${SESSION_JOB[$s]}"
            if [[ "$meta" == *"|${g}|"* ]]; then
                return 0
            fi
        fi
    done
    return 1
}

next_free_gpu() {
    local g
    for g in "${GPUS[@]}"; do
        if ! gpu_in_use "$g"; then
            echo "$g"
            return 0
        fi
    done
    echo ""
}

while [[ ${#pending[@]} -gt 0 || $(running_count) -gt 0 ]]; do
    reap_finished
    while [[ ${#pending[@]} -gt 0 && $(running_count) -lt $MAX_PARALLEL ]]; do
        gpu="$(next_free_gpu)"
        if [[ -z "$gpu" ]]; then
            break
        fi
        job="${pending[0]}"
        pending=("${pending[@]:1}")
        launch_one "$job" "$gpu"
    done
    if [[ ${#pending[@]} -eq 0 && $(running_count) -eq 0 ]]; then
        break
    fi
    sleep 60
done

echo ""
echo "Sweep launcher finished. Check bench_*.log and tmux sessions."
echo "Aggregate: python baseline/aggregate_benchmark_table.py"
