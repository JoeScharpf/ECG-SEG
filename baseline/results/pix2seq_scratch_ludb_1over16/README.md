# Pix2Seq — LUDB 1/16 (multi-class)

Tracked summary artifacts for the Pix2Seq Phase 2 run.

Populate after training on gpu2:

```bash
python baseline/plot_results.py \
  --run-dir baseline/exps/pix2seq/scratch/ludb/1over16 \
  --publish
```

Compare test MeanIoU to ResNet Phase 1: **0.6661** (paper Scratch 67.3%).
