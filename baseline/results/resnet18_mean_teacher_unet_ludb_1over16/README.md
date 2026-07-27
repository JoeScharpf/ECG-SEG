# LUDB 1/16 — ResNet-18 + U-Net + Mean Teacher (seed 0)

Semi-supervised Mean Teacher on the same ResNet-18 + U-Net head as the
supervised U-Net baseline. Uses the 84 labeled + 1427 unlabeled LUDB 1/16
splits (`ema_decay: 0.99`). Seed 0 only — replicate on seeds 1–2 before the
decision gate.

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

## Files

| File | Description |
|------|-------------|
| `training_curves.png` | Loss and validation MeanIoU |
| `test_metrics.csv` | Test set metrics |
| `summary.json` | Best valid / test mIoU and epochs |

## Reproduce

```bash
# on gpu2, inside tmux, conda env semi_seg_ecg
bash scripts/run_ssl.sh --method mean_teacher --head unet --gpus 0
```
