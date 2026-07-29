# LUDB 1/16 — ResNet-18 + FCN + FixMatch (seed 0)

Semi-supervised FixMatch with the paper-style ResNet-18 + FCN head on the same
LUDB 1/16 labeled/unlabeled splits (`conf_thresh: 0.80`, `per_class: false`).
Seed 0 only.

Config: `configs/base/resnet18/fixmatch.yaml` +
`configs/bench/ludb/1over16.yaml`.

## Result

| Model | Test MeanIoU |
|-------|-------------:|
| ResNet-18 + FCN supervised | 0.6661 |
| ResNet-18 + FCN + Mean Teacher (seed 0) | 0.7036 |
| **ResNet-18 + FCN + FixMatch (seed 0)** | **0.6977** |
| ResNet-18 + U-Net supervised (3-seed mean) | 0.8178 ± 0.0024 |

Lift vs FCN supervised: **+0.032**. Slightly below FCN + Mean Teacher on this
seed; both SSL methods help FCN modestly.

Best valid MeanIoU **0.6969** @ epoch 41 (see `summary.json`).

## Files

| File | Description |
|------|-------------|
| `training_curves.png` | Loss and validation MeanIoU |
| `test_metrics.csv` | Test set metrics |
| `summary.json` | Best valid / test mIoU and epochs |

## Reproduce

```bash
bash scripts/run_ssl.sh --method fixmatch --head fcn --gpus 0
```
