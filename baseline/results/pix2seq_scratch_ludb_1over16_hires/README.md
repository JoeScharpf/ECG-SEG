# Pix2Seq — LUDB 1/16 hi-res (multi-class)

Hi-res Pix2Seq experiment: `num_bins=1000`, `max_segments=40`, sinusoidal memory PE,
soft coordinate loss. Test MeanIoU **0.2634** (vs original 0.227 / ResNet FCN 0.666 /
U-Net 0.820).

## Files

| File | Description |
|------|-------------|
| `test_metrics.csv` | Test MeanIoU + per-class IoU |
| `visual_examples/` | Best / median / worst prediction overlays |

## Reproduce figures

```bash
python baseline/plot_ecg_predictions.py \
  --run-dir baseline/exps/pix2seq/scratch/ludb/1over16_hires \
  --pred-label "Pix2Seq prediction" \
  --publish
```
