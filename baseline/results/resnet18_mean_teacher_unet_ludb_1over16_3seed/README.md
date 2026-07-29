# LUDB 1/16 — ResNet-18 + U-Net + Mean Teacher (3-seed)

Replication of Mean Teacher + U-Net on seeds 0/1/2 with locked protocol
(`ema_decay: 0.99`, same steps/augs/val cadence as supervised U-Net).

## Result

| Seed | Test MeanIoU |
|------|-------------:|
| 0 | 0.8409 |
| 1 | 0.8401 |
| 2 | 0.8381 |
| **Mean ± std** | **0.8397 ± 0.0014** |

| Comparison | Test MeanIoU |
|------------|-------------:|
| ResNet-18 + U-Net supervised (3-seed) | 0.8178 ± 0.0024 |
| **ResNet-18 + U-Net + Mean Teacher (3-seed)** | **0.8397 ± 0.0014** |
| ResNet-18 + U-Net + FixMatch (seed 0) | 0.8375 |
| SemiSegECG Table 2 best (ViT-Tiny + FCN + MT) | 0.736 |

**Decision gate: PASS.** Stable SSL lift of **+0.0219** (~2.2 points) over
supervised U-Net. Proceed to Phase 2.5 (teacher boundary diagnostic on labeled
val) before implementing boundary-aware Mean Teacher.

## Per-seed artifacts

- `../resnet18_mean_teacher_unet_ludb_1over16/` (seed 0)
- `../resnet18_mean_teacher_unet_ludb_1over16_seed1/`
- `../resnet18_mean_teacher_unet_ludb_1over16_seed2/`
