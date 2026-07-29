# LUDB 1/16 — ResNet-18 + U-Net + FixMatch (seed 0)

Semi-supervised FixMatch with ResNet-18 + U-Net on LUDB 1/16
(`conf_thresh: 0.80`, `per_class: false`). Seed 0 only.

Config: `configs/base/resnet18/fixmatch_unet.yaml` +
`configs/bench/ludb/1over16.yaml`.

## Result

| Model | Test MeanIoU |
|-------|-------------:|
| ResNet-18 + U-Net supervised (3-seed mean) | 0.8178 ± 0.0024 |
| ResNet-18 + U-Net + Mean Teacher (3-seed) | 0.8397 ± 0.0014 |
| **ResNet-18 + U-Net + FixMatch (seed 0)** | **0.8375** |

Lift vs supervised U-Net mean: **+0.020**. On this seed, FixMatch is within
~0.002 of Mean Teacher (slightly lower).

Best valid MeanIoU **0.8302** @ epoch 31 (see `summary.json`).

## Files

| File | Description |
|------|-------------|
| `training_curves.png` | Loss and validation MeanIoU |
| `test_metrics.csv` | Test set metrics |
| `summary.json` | Best valid / test mIoU and epochs |

## Reproduce

```bash
bash scripts/run_ssl.sh --method fixmatch --head unet --gpus 0
```
