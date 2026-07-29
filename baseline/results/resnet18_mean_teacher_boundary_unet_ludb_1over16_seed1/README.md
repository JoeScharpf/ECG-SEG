# LUDB 1/16 — ResNet-18 + U-Net + Boundary-aware Mean Teacher (seed 1)

See the 3-seed aggregate:
`../resnet18_mean_teacher_boundary_unet_ludb_1over16_3seed/README.md`.

## Result

| Model | Test MeanIoU |
|-------|-------------:|
| **Boundary-aware MT (seed 1)** | **0.8454** |

Best valid MeanIoU **0.8373** @ epoch 48. Boundary metrics in
`boundary_metrics.json`.

## Reproduce

```bash
bash scripts/run_ssl.sh --method mean_teacher_boundary --head unet --seed 1 --gpus 0
```
