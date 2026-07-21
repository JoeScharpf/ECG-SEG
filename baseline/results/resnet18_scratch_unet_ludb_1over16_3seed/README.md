# LUDB 1/16 — ResNet-18 + U-Net (supervised), 3-seed variance

Robustness check for the headline U-Net result: three independent seeds, same
config (`configs/base/resnet18/scratch_unet.yaml` + per-seed
`configs/bench/ludb/1over16_unet_seed{0,1,2}.yaml`), same LUDB 1/16 split.

## Results (test MeanIoU)

| Seed | Test MeanIoU | Best valid MeanIoU (epoch) |
|------|--------------|----------------------------|
| 0    | 0.8204       | 0.8147 (@71)               |
| 1    | 0.8156       | 0.8118 (@90)               |
| 2    | 0.8175       | 0.8129 (@71)               |
| **Mean ± std** | **0.8178 ± 0.0024** | min 0.8156, max 0.8204 |

## Context

| Reference | Test MeanIoU |
|-----------|--------------|
| ResNet-18 + FCN baseline (this repo, reproduces paper Scratch 67.3) | 0.6661 |
| **ResNet-18 + U-Net, supervised (this work, 3-seed mean)** | **0.8178 ± 0.0024** |
| SemiSegECG Table 2, best LUDB 1/16 (ViT-Tiny + FCN + Mean Teacher) | 0.736 |

The 3-seed std (0.0024) is far smaller than the margin over the baseline
(+0.15) and over the paper's best LUDB 1/16 entry (+0.082), so the result is
robust to seed variation rather than a lucky run.

Per-seed folders (with individual `training_curves.png`, `summary.json`,
`test_metrics.csv`) are in the sibling
`resnet18_scratch_unet_ludb_1over16_unet_seed{0,1,2}/` directories.

**Visual examples:** [`../resnet18_scratch_unet_ludb_1over16/visual_examples/`](../resnet18_scratch_unet_ludb_1over16/visual_examples/) — ECG + GT vs prediction overlays for the test set.

## Reproduce

```bash
# on gpu2, inside tmux, conda env semi_seg_ecg
bash scripts/run_unet_seeds.sh --gpus 0 --seeds "0 1 2"
```
