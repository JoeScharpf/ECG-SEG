#!/usr/bin/env bash
# Launch LUDB fraction sweep: supervised / MT / boundary-MT × {8,4,2} × seeds {0,1,2}.
# Skips cells whose published summary.json already exists.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

GPUS=(3 4 5 6 7)
FRACTIONS=(8 4 2)
SEEDS=(0 1 2)
DRY_RUN=0
MAX_PARALLEL="${#GPUS[@]}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            IFS=',' read -r -a GPUS <<< "$2"
            MAX_PARALLEL="${#GPUS[@]}"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift 1
            ;;
        -h|--help)
            echo "Usage: bash scripts/run_ludb_fraction_sweep.sh [--gpus 3,4,5,6,7] [--dry-run]"
            echo "Launches 27 new LUDB fraction×seed jobs (skips existing summaries)."
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

results_summary_path() {
    local method="$1" frac="$2" seed="$3"
    case "$method" in
        supervised)
            echo "baseline/results/resnet18_scratch_unet_ludb_1over${frac}_unet_seed${seed}/summary.json"
            ;;
        mean_teacher)
            echo "baseline/results/resnet18_mean_teacher_unet_ludb_1over${frac}_seed${seed}/summary.json"
            ;;
        mean_teacher_boundary)
            echo "baseline/results/resnet18_mean_teacher_boundary_unet_ludb_1over${frac}_seed${seed}/summary.json"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

job_cmd() {
    local method="$1" frac="$2" seed="$3" gpu="$4"
    case "$method" in
        supervised)
            echo "bash scripts/run_unet_seeds.sh --gpus ${gpu} --seeds \"${seed}\" --label-fraction ${frac}"
            ;;
        mean_teacher)
            echo "bash scripts/run_ssl.sh --method mean_teacher --head unet --gpus ${gpu} --label-fraction ${frac} --seed ${seed}"
            ;;
        mean_teacher_boundary)
            echo "bash scripts/run_ssl.sh --method mean_teacher_boundary --head unet --gpus ${gpu} --label-fraction ${frac} --seed ${seed}"
            ;;
    esac
}

METHODS=(supervised mean_teacher mean_teacher_boundary)
JOBS=()

for frac in "${FRACTIONS[@]}"; do
    for method in "${METHODS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            summary="$(results_summary_path "$method" "$frac" "$seed")"
            if [[ -f "$summary" ]]; then
                echo "SKIP existing: $summary"
                continue
            fi
            JOBS+=("${method}|${frac}|${seed}")
        done
    done
done

echo "=== LUDB fraction sweep ==="
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
        IFS='|' read -r method frac seed <<< "$job"
        gpu="${GPUS[$((idx % MAX_PARALLEL))]}"
        echo "[dry] gpu=${gpu} $(job_cmd "$method" "$frac" "$seed" "$gpu")"
        idx=$((idx + 1))
    done
    exit 0
fi

# Simple queue: keep up to MAX_PARALLEL tmux jobs; fill as they finish.
pending=("${JOBS[@]}")
declare -A SESSION_JOB=()

launch_one() {
    local job="$1" gpu="$2"
    IFS='|' read -r method frac seed <<< "$job"
    local sess="sweep_${method}_1over${frac}_s${seed}"
    sess="${sess//_mean_teacher_boundary_/bmt_}"
    sess="${sess//_mean_teacher_/mt_}"
    sess="${sess//_supervised_/sup_}"
    # shorten further
    sess=$(echo "$sess" | sed 's/mean_teacher_boundary/bmt/;s/mean_teacher/mt/;s/supervised/sup/')
    local cmd
    cmd="$(job_cmd "$method" "$frac" "$seed" "$gpu")"
    local log="sweep_${method}_1over${frac}_seed${seed}.log"
    tmux has-session -t "$sess" 2>/dev/null && tmux kill-session -t "$sess"
    tmux new-session -d -s "$sess" "cd '$REPO_ROOT'; source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate semi_seg_ecg; $cmd 2>&1 | tee '$log'; echo DONE > '${log}.done'"
    SESSION_JOB["$sess"]="$job|$gpu|$log"
    echo "LAUNCH $sess gpu=$gpu -> $log"
}

running_count() {
    local n=0
    local s
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

idx=0
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
echo "Sweep launcher finished queueing/waiting. Check sweep_*.log and tmux sessions."
