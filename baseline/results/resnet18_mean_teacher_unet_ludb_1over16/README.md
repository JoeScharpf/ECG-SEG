# LUDB 1/16 — ResNet-18 + U-Net + Mean Teacher (seed 0)

Semi-supervised Mean Teacher on the same ResNet-18 + U-Net head as the
supervised U-Net baseline. Uses the 84 labeled + 1427 unlabeled LUDB 1/16
splits (`ema_decay: 0.99`). Seed 0 of the 3-seed aggregate
(`../resnet18_mean_teacher_unet_ludb_1over16_3seed/`: **0.8397 ± 0.0014**, gate PASS).

Config: `configs/base/resnet18/mean_teacher_unet.yaml` +
`configs/bench/ludb/1over16.yaml`.

## Result

| Model | Test MeanIoU |
|-------|-------------:|
| ResNet-18 + FCN supervised | 0.6661 |
| ResNet-18 + U-Net supervised (3-seed mean) | 0.8178 ± 0.0024 |
| SemiSegECG Table 2 best (ViT-Tiny + FCN + MT) | 0.736 |
| **ResNet-18 + U-Net + Mean Teacher (seed 0)** | **0.8409** |

Best valid MeanIoU **0.8350** @ epoch 44 (see `summary.json`).

Seed-0 lift vs supervised U-Net mean: **+0.023** (~2.3 points). Treat as
preliminary until 3-seed SSL variance is in.

## Boundary metrics (Phase 2)

From `test_outputs.npy` / `test_labels.npy` via `baseline/eval_boundaries.py`
(match IoU ≥ 0.1, fs = 250 Hz):

| Metric | Value |
|--------|------:|
| Pixel MeanIoU (4-class) | 0.847 |
| Match rate vs GT segments | 98.3% |
| P onset / offset MAE | 9.4 / 9.1 ms |
| QRS onset / offset MAE | 6.7 / 7.1 ms |
| T onset / offset MAE | 16.8 / 13.9 ms |
| QRS / PR / QT interval MAE | 10.6 / 12.0 / 16.9 ms |

See `boundary_metrics.json`. T-wave boundaries dominate error; interiors look
stronger than edges (supports a Phase 2.5 teacher diagnostic before boundary-aware MT).

## Phase 2.5 teacher diagnostic (labeled val)

EMA teacher on labeled validation (`band=4`, 16 ms @ 250 Hz):

| Region | Accuracy |
|--------|--------:|
| Interior | 0.965 |
| Onset band | 0.784 |
| Offset band | 0.810 |
| Far background | 0.981 |
| **Boundary gap (interior − onset)** | **0.181** |

`justifies_boundary_ssl: true` — see `teacher_boundary_diag.json`.

## Files

| File | Description |
|------|-------------|
| `training_curves.png` | Loss and validation MeanIoU |
| `test_metrics.csv` | Test set metrics |
| `summary.json` | Best valid / test mIoU and epochs |
| `boundary_metrics.json` | Per-class IoU, onset/offset MAE, intervals |
| `teacher_boundary_diag.json` | Val-only interior vs boundary accuracy |

## Reproduce

```bash
# on gpu2, inside tmux, conda env semi_seg_ecg
bash scripts/run_ssl.sh --method mean_teacher --head unet --gpus 0
```
