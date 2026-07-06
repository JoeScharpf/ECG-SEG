# Phase 1: Supervised Baseline on gpu2

Run the official SemiSegECG supervised baseline: **ResNet-18 + FCN**, trained from scratch on LUDB with 1/16 labels (`configs/base/resnet18/scratch.yaml`). Only the labeled train split is used; unlabeled data is for Phase 2 SSL methods.

Outputs go to `baseline/exps/`.

## Prerequisites

- SSH access to gpu3 → gpu2
- Git clone of [ECG-SEG](https://github.com/JoeScharpf/ECG-SEG)

## Steps

### 1. Connect and start tmux

```bash
ssh joe@safeai-gpu3.andrew.cmu.edu
ssh joe@safeai-gpu2.lan.local.cmu.edu
tmux new -s phase1
```

### 2. Clone and set up

```bash
git clone https://github.com/JoeScharpf/ECG-SEG.git
cd ECG-SEG
bash scripts/setup_phase1.sh
conda activate semi_seg_ecg
```

### 3. Pick a free GPU

```bash
nvidia-smi
```

### 4. Run training + test + plots

```bash
bash scripts/run_phase1.sh --gpus 0
```

Optional: `--label-fraction 8` (choices: 2, 4, 8, 16).

Detach tmux: `Ctrl+B`, then `D`. Reattach: `tmux attach -t phase1`.

### 5. Verify outputs

Check `baseline/exps/resnet18/scratch/ludb/1over16/`:

| File | Description |
|------|-------------|
| `best-MeanIoU.pth` | Best validation checkpoint |
| `log.txt` | Per-epoch JSON metrics |
| `test_metrics.csv` | Test set metrics (quick summary) |
| `test_outputs.npy` | Per-sample predictions (for paper-style eval) |
| `test_labels.npy` | Ground-truth labels (for paper-style eval) |
| `training_curves.png` | Loss and mIoU charts |

Re-run plots only:

```bash
python baseline/plot_results.py --run-dir baseline/exps/resnet18/scratch/ludb/1over16
```

Optional TensorBoard:

```bash
tensorboard --logdir baseline/exps/resnet18/scratch/ludb/1over16
```

### 6. Update run notes

Fill in [`baseline/README.md`](../baseline/README.md) with date, GPU, best mIoU, test mIoU.

## Comparing to published results

SemiSegECG paper tables use mIoU from [`semi-seg-ecg/notebooks/perf_eval.ipynb`](../semi-seg-ecg/notebooks/perf_eval.ipynb), not `test_metrics.csv` directly. After testing, open that notebook and point it at your run directory's `test_outputs.npy` and `test_labels.npy` for an apples-to-apples comparison.

## What gets printed

- **During training:** epoch loss, validation MeanIoU (from semi-seg-ecg)
- **After `plot_results.py`:** best valid loss, best MeanIoU, test MeanIoU, chart path

## Troubleshooting

| Problem | Fix |
|---------|-----|
| PyTorch / CUDA error | Ask lab how others install PyTorch on gpu2; try cu113 wheels from requirements.txt |
| OOM | Reduce `batch_size` in `semi-seg-ecg/configs/base/resnet18/scratch.yaml` (16 → 8) |
| Disk full | Only download LUDB; do not duplicate datasets |
| Missing data | `python data/data.py --download` |
| `Checkpoint not found` for test | Training must finish at least one epoch and save `best-MeanIoU.pth` |

## Compute

Roughly 30 minutes to a few hours on one A6000 for 100 epochs. A smoke test (several epochs with checkpoints) is enough to confirm the pipeline works.

## Success criteria

- Training runs without crash
- `log.txt`, `test_metrics.csv`, and `training_curves.png` exist
- `plot_results.py` prints a metric summary
