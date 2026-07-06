# LUDB 1/16 — Supervised ResNet-18 scratch

Tracked summary artifacts for GitHub (checkpoints stay in gitignored `baseline/exps/`).

Expected files after `plot_results.py --publish`:

| File | Description |
|------|-------------|
| `training_curves.png` | Loss and validation MeanIoU |
| `test_metrics.csv` | Test set metrics |
| `summary.json` | Best valid / test mIoU and epochs |

Populate from a gpu2 run:

```bash
python baseline/plot_results.py \
  --run-dir baseline/exps/resnet18/scratch/ludb/1over16 \
  --publish
```

Then copy `baseline/results/` to your Mac and commit.
