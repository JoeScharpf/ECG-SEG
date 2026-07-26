# Pix2Seq visual examples (LUDB test set, hi-res)

Inference overlays from the hi-res Pix2Seq run (`num_bins=1000`, soft coord loss,
sinusoidal memory PE; test MeanIoU **0.2634**).

Each figure has three panels:
1. **ECG** — standardized waveform
2. **Ground truth** — P (blue), QRS (red), T (green)
3. **Pix2Seq prediction** — rasterized segment tokens on the same recording

| File | Case | Sample mIoU | Recording |
|------|------|-------------|-----------|
| `example_best_idx340.png` | Best on test set | 0.442 | patient 36, V5 |
| `example_good_p90_idx178.png` | 90th percentile | 0.358 | patient 151, II |
| `example_good_p75_idx185.png` | 75th percentile | 0.313 | patient 151, V6 |
| `example_median_idx115.png` | Median | 0.266 | patient 132, aVL |
| `example_worst_idx95.png` | Hardest case | 0.115 | patient 130, III |

**What the failure looks like:** even the best case (0.44) has wrong segment widths
and missing P/T waves; the median/worst cases show runaway-wide segments and
hallucinated labels on flat signal. Boundaries land near waves but not tightly
enough for high IoU on narrow P/QRS.

Regenerate:

```bash
python baseline/plot_ecg_predictions.py \
  --run-dir baseline/exps/pix2seq/scratch/ludb/1over16_hires \
  --pred-label "Pix2Seq prediction" \
  --publish
```

Requires `test_outputs.npy` and `test_labels.npy` in the run directory.
