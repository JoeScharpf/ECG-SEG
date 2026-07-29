# LUDB 1/16 — ResNet-18 + U-Net + Mean Teacher (seed 2)

Replication of seed-0 Mean Teacher + U-Net. See the 3-seed aggregate README:
`../resnet18_mean_teacher_unet_ludb_1over16_3seed/README.md`.

## Result

| Model | Test MeanIoU |
|-------|-------------:|
| **ResNet-18 + U-Net + Mean Teacher (seed 2)** | **0.8381** |

Best valid MeanIoU **0.8327** @ epoch 54.

## Reproduce

```bash
bash scripts/run_ssl.sh --method mean_teacher --head unet --seed 2 --gpus 0
```
