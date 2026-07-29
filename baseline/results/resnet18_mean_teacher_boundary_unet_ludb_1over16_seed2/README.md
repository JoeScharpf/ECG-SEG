# LUDB 1/16 — ResNet-18 + U-Net + Boundary-aware Mean Teacher (seed 2)

See the 3-seed aggregate:
`../resnet18_mean_teacher_boundary_unet_ludb_1over16_3seed/README.md`.

## Result

| Model | Test MeanIoU |
|-------|-------------:|
| **Boundary-aware MT (seed 2)** | **0.8458** |

Best valid MeanIoU **0.8380** @ epoch 46. Boundary metrics in
`boundary_metrics.json`.

## Reproduce

```bash
bash scripts/run_ssl.sh --method mean_teacher_boundary --head unet --seed 2 --gpus 0
```
