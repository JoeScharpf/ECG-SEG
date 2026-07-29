# LUDB 1/16 — ResNet-18 + U-Net + Boundary-aware Mean Teacher (seed 0)

Soft boundary-aware consistency on the same ResNet-18 + U-Net head as the
standard Mean Teacher baseline.

## Overall segmentation

| Model | Test MeanIoU |
|---|---:|
| U-Net + Mean Teacher (seed 0) | 0.8409 |
| **U-Net + Boundary-aware MT (seed 0)** | **0.8449** |

## Boundary metrics (Phase 2 evaluator; match IoU >= 0.1)

Compared against `resnet18_mean_teacher_unet_ludb_1over16/boundary_metrics.json`:

| Metric | MT | Boundary-aware MT | Delta |
|---|---:|---:|---:|
| Pixel MeanIoU (4-class) | 0.8469 | 0.8509 | +0.0040 |
| Match rate vs GT segments | 0.9832 | 0.9829 | -0.0003 |
| Fragmentation rate | 0.0306 | 0.0372 | +0.0066 |
| T onset MAE (ms) | 16.784 | 15.164 | **-1.621** |
| T offset MAE (ms) | 13.898 | 12.702 | **-1.196** |
| QRS onset MAE (ms) | 6.662 | 6.754 | +0.092 |
| QRS offset MAE (ms) | 7.148 | 6.996 | **-0.152** |
| P onset / offset MAE (ms) | 9.422 / 9.052 | 8.886 / 9.443 | -0.537 / +0.391 |
| QRS duration MAE (ms) | 10.595 | 10.639 | +0.044 |
| PR MAE (ms) | 11.952 | 11.622 | **-0.330** |
| QT MAE (ms) | 16.913 | 15.799 | **-1.114** |

## Files

- `summary.json`, `test_metrics.csv`, `training_curves.png`
- `boundary_metrics.json`: computed from `test_outputs.npy` / `test_labels.npy`

