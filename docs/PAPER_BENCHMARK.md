# Paper benchmark protocol

Full SemiSegECG evaluation for **ResNet-18 + U-Net + boundary-aware Mean Teacher** (our locked recipe). This document defines the run matrix, artifacts, and how results feed the paper.

## Locked recipe

| Knob | Value |
|------|--------|
| Base config | `semi-seg-ecg/configs/base/resnet18/mean_teacher_boundary_unet.yaml` |
| Algorithm | `mean_teacher_boundary` |
| Backbone | ResNet-18 (scratch), `num_leads: 1` |
| Decoder | `UNetHead` |
| Checkpoint | Best validation MeanIoU |
| Training | 100 epochs, AdamW lr 0.001 (SemiSegECG default) |
| Seeds | 0, 1, 2 |

**Out of scope for this sweep:** FCN baselines, plain Mean Teacher, FixMatch, Lovász, onset/offset heatmap head, supervised-only (LUDB supervised numbers exist separately in `ludb_fraction_benchmark/`).

## Run matrix

| Dataset | Bench overlays | Fractions | Seeds | Cells |
|---------|----------------|-----------|-------|------:|
| LUDB | `configs/bench/ludb/` | 1/16, 1/8, 1/4, 1/2 | 0–2 | 12 |
| QTDB | `configs/bench/qtdb/` | 1/16–1/2 | 0–2 | 12 |
| ISP | `configs/bench/isp/` | 1/16–1/2 | 0–2 | 12 |
| Zhejiang | `configs/bench/zhejiang/` | 1/16–1/2 | 0–2 | 12 |
| Cross-domain | `configs/bench/cross_domain/merged.yaml` | merged | 0–2 | 3 |

**Total:** 51 cells. LUDB boundary-MT cells are mostly complete; the sweep skips any cell whose `summary.json` already exists.

## One-time data setup (gpu2)

```bash
cd ~/ECG-SEG
bash scripts/setup_all_benchmark_data.sh
```

Downloads LUDB, QTDB, ISP, Zhejiang, PTB-XL waveforms and all index splits per [semi-seg-ecg/README.md](../semi-seg-ecg/README.md).

## Run a single cell

```bash
conda activate semi_seg_ecg
bash scripts/run_benchmark.sh --dataset qtdb --label-fraction 16 --seed 0 --gpus 4
```

Smoke test (1 epoch):

```bash
bash scripts/run_benchmark.sh --dataset qtdb --label-fraction 16 --seed 0 --gpus 4 --smoke
```

Each cell runs: **train → test → publish → boundary metrics**.

## Full sweep

```bash
# Preview pending jobs
bash scripts/run_full_benchmark_sweep.sh --dry-run

# Launch on idle GPUs (tmux workers)
bash scripts/run_full_benchmark_sweep.sh --gpus 4,5,6,7
```

## Published artifacts (per cell)

| Path | Contents |
|------|----------|
| `baseline/results/resnet18_mean_teacher_boundary_unet_{dataset}_{suffix}/` | `summary.json`, `test_metrics.csv`, `training_curves.png`, `boundary_metrics.json` |
| `baseline/exps/resnet18/mean_teacher_boundary_unet/{dataset}/{suffix}/` | Checkpoints, `test_outputs.npy`, logs (gitignored on gpu2) |

**Path naming:** LUDB 1/16 seed 0 uses legacy `..._ludb_1over16` (no `_seed0`). All other cells use `..._{dataset}_1over{N}_seed{S}` or `..._cross_domain_merged_seed{S}`.

## Aggregate master table

```bash
python baseline/aggregate_benchmark_table.py
```

Outputs under `baseline/results/paper_benchmark/`:

- `summary_table.csv` — per-seed rows
- `summary_table.md` — mean±std pivot by dataset
- `summary.json` — nested aggregate
- `completion_checklist.md` — done/pending per cell

## Paper framing

- **Claim:** decoder choice and boundary-aware SSL under the SemiSegECG protocol — not novelty of U-Net for ECG per se.
- **Selection rule:** checkpoint and any tolerance hyperparameters chosen on **validation** only; test is reported once per cell.
- **LUDB reference (boundary-MT, 3-seed):** 1/16 0.8454±0.0005, 1/2 0.8591±0.0016 (already in repo; do not re-run unless missing).

## Sync results to local repo

```bash
rsync -avz joe@safeai-gpu2.lan.local.cmu.edu:~/ECG-SEG/baseline/results/ \
  ./baseline/results/
python baseline/aggregate_benchmark_table.py
```
