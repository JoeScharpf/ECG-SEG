# U-Net visual examples (LUDB test set)

Inference overlays from the supervised ResNet-18 + U-Net model (test MeanIoU **0.8197**).

Each figure has three panels:
1. **ECG** — standardized waveform
2. **Ground truth** — P (blue), QRS (red), T (green)
3. **U-Net prediction** — model output on the same recording

| File | Case | Sample mIoU | Recording |
|------|------|-------------|-----------|
| `example_best_idx226.png` | Best on test set | 0.957 | patient 172, lead II |
| `example_good_p90_idx162.png` | 90th percentile | 0.924 | patient 145, aVF |
| `example_good_p75_idx8.png` | 75th percentile | 0.908 | patient 102, V6 |
| `example_median_idx259.png` | Median | 0.869 | patient 177, aVL |
| `example_worst_idx87.png` | Hardest case | 0.254 | patient 116, V3 |

**Meeting tip:** lead with `example_best` or `example_good_p90` to show sharp P/QRS/T boundaries; mention `example_worst` as an honest failure case (often noisy lead or missing P).

Regenerate:

```bash
python baseline/plot_ecg_predictions.py \\
  --run-dir baseline/exps/resnet18/scratch_unet/ludb/1over16 \\
  --publish
```

Requires `test_outputs.npy` and `test_labels.npy` in the run directory (from `test.sh`).
