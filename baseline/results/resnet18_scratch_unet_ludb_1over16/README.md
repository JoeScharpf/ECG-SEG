# LUDB 1/16 — Supervised ResNet-18 + U-Net decode head (scratch)

Tracked summary artifacts for GitHub (checkpoints stay in gitignored `baseline/exps/`).

Same ResNet-18 encoder and LUDB 1/16 splits as `resnet18_scratch_ludb_1over16`,
but the single-scale `FCNHead` is replaced with a multi-scale `UNetHead` that fuses
all four encoder feature maps via skip connections. Plain CrossEntropy (no loss
change) so the decode head is the only variable vs. the FCN baseline.

## Result

| Model | Test MeanIoU |
|-------|--------------|
| ResNet-18 + FCNHead (baseline) | ~0.666 |
| ResNet-18 + UNetHead (this run) | **0.8197** |

Best valid MeanIoU 0.8171 @ epoch 77 (see `summary.json`).

## Files

| File | Description |
|------|-------------|
| `training_curves.png` | Loss and validation MeanIoU |
| `test_metrics.csv` | Test set metrics |
| `summary.json` | Best valid / test mIoU and epochs |

## Reproduce

```bash
# on gpu2, inside tmux, conda env semi_seg_ecg
bash scripts/run_unet.sh --gpus 0 --label-fraction 16
```

Then copy `baseline/results/resnet18_scratch_unet_ludb_1over16/` to your Mac and commit.
