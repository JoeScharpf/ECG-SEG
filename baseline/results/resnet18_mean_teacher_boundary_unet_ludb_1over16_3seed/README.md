# LUDB 1/16 — ResNet-18 + U-Net + Boundary-aware Mean Teacher (3-seed)

Soft boundary-aware Mean Teacher on the same ResNet-18 + U-Net head as plain
Mean Teacher. Seeds 0/1/2 with locked protocol.

## Result

| Seed | Test MeanIoU |
|------|-------------:|
| 0 | 0.8449 |
| 1 | 0.8454 |
| 2 | 0.8458 |
| **Mean ± std** | **0.8454 ± 0.0005** |

| Comparison | Test MeanIoU |
|------------|-------------:|
| U-Net supervised (3-seed) | 0.8178 ± 0.0024 |
| U-Net + Mean Teacher (3-seed) | 0.8397 ± 0.0014 |
| **U-Net + Boundary-aware MT (3-seed)** | **0.8454 ± 0.0005** |

Lift vs plain MT: **+0.0057**. Boundary/interval MAEs (esp. T onset/offset and
QT) improve consistently; fragmentation rate rises slightly (see per-seed
`boundary_metrics.json`).

## Per-seed artifacts

- `../resnet18_mean_teacher_boundary_unet_ludb_1over16/` (seed 0)
- `../resnet18_mean_teacher_boundary_unet_ludb_1over16_seed1/`
- `../resnet18_mean_teacher_boundary_unet_ludb_1over16_seed2/`
