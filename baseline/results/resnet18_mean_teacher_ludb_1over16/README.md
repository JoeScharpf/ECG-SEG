# LUDB 1/16 — ResNet-18 + FCN + Mean Teacher (seed 0)

Semi-supervised Mean Teacher with the paper-style ResNet-18 + FCN head on the
same LUDB 1/16 labeled/unlabeled splits (`ema_decay: 0.99`). Seed 0 only.

Config: `configs/base/resnet18/mean_teacher.yaml` +
`configs/bench/ludb/1over16.yaml`.

## Result

| Model | Test MeanIoU |
|-------|-------------:|
| ResNet-18 + FCN supervised | 0.6661 |
| **ResNet-18 + FCN + Mean Teacher (seed 0)** | **0.7036** |
| ResNet-18 + U-Net supervised (3-seed mean) | 0.8178 ± 0.0024 |
| ResNet-18 + U-Net + Mean Teacher (seed 0) | 0.8409 |

Lift vs FCN supervised: **+0.038**. Absolute SOTA still lags U-Net supervised;
SSL helps the weak decoder but does not close the decoder gap.

Best valid MeanIoU **0.7031** @ epoch 54 (see `summary.json`).

## Files

| File | Description |
|------|-------------|
| `training_curves.png` | Loss and validation MeanIoU |
| `test_metrics.csv` | Test set metrics |
| `summary.json` | Best valid / test mIoU and epochs |

## Reproduce

```bash
bash scripts/run_ssl.sh --method mean_teacher --head fcn --gpus 0
```
