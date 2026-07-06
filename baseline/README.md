# Phase 1 Baseline (Supervised ResNet-18)

Supervised baseline on LUDB with 1/16 labeled data.

## Run (on gpu2)

```bash
cd ~/ECG-SEG
bash scripts/run_phase1.sh --gpus 0
```

## Fill in after a run

| Field | Value |
|-------|-------|
| Date | |
| GPU | |
| Label fraction | 1/16 |
| Best valid MeanIoU | |
| Best epoch | |
| Test MeanIoU | (from `test_metrics.csv`) |
| Output directory | `baseline/exps/resnet18/scratch/ludb/1over16/` |
| Training chart | `baseline/exps/.../training_curves.png` |

Checkpoints (`.pth`) and charts stay on the server under `baseline/exps/` (gitignored).
