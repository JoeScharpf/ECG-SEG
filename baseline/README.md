# Phase 1 Baseline (Supervised ResNet-18)

Supervised baseline on LUDB with 1/16 labeled data.

**Model:** ResNet-18 (1D) + FCN decode head (`algorithm: base`), trained from scratch (`mode: scratch`) on the labeled train split only. Unlabeled data in the bench config is unused here — SSL methods use it in Phase 2.

Configs: `semi-seg-ecg/configs/base/resnet18/scratch.yaml` + `configs/bench/ludb/1over16.yaml`.

## Run (on gpu2)

```bash
cd ~/ECG-SEG
bash scripts/setup_phase1.sh
conda activate semi_seg_ecg
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
| Test MeanIoU | (from `test_metrics.csv`; see note below) |
| Output directory | `baseline/exps/resnet18/scratch/ludb/1over16/` |
| Training chart | `baseline/exps/.../training_curves.png` |

Checkpoints (`.pth`) and charts stay on the server under `baseline/exps/` (gitignored).

## Comparing to the paper

Published SemiSegECG mIoU numbers are computed with [`semi-seg-ecg/notebooks/perf_eval.ipynb`](../semi-seg-ecg/notebooks/perf_eval.ipynb) from `test_outputs.npy` and `test_labels.npy` in the run directory. `test_metrics.csv` is convenient for quick checks but may differ slightly from the paper tables.
