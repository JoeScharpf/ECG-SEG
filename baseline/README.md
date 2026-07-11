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

`run_phase1.sh` writes full artifacts to gitignored `baseline/exps/` and copies plots + metrics to **`baseline/results/`** for GitHub.

## Results (LUDB 1/16)

| Field | Value |
|-------|-------|
| Date | 2026-07-06 |
| GPU | gpu2, NVIDIA RTX A6000 |
| Label fraction | 1/16 |
| Best valid MeanIoU | 0.6672 |
| Best epoch | 83 |
| Test MeanIoU | 0.6661 |
| Paper baseline (Scratch) | 67.3% |
| Training chart | [`baseline/results/resnet18_scratch_ludb_1over16/training_curves.png`](results/resnet18_scratch_ludb_1over16/training_curves.png) |

Checkpoints (`.pth`) and `.npy` files stay on the server under `baseline/exps/` (gitignored).

## Publish results to the repo

After a run on gpu2, publish from existing exps (if you ran train/test manually):

```bash
python baseline/plot_results.py \
  --run-dir baseline/exps/resnet18/scratch/ludb/1over16 \
  --publish
```

Copy `baseline/results/` to your Mac, then commit and push:

```bash
scp -J joe@safeai-gpu3.andrew.cmu.edu \
  -r joe@safeai-gpu2.lan.local.cmu.edu:~/ECG-SEG/baseline/results/ \
  ~/Desktop/safe/baseline/
```

## Comparing to the paper

Published SemiSegECG mIoU numbers are computed with [`semi-seg-ecg/notebooks/perf_eval.ipynb`](../semi-seg-ecg/notebooks/perf_eval.ipynb) from `test_outputs.npy` and `test_labels.npy` in the run directory. `test_metrics.csv` is convenient for quick checks but may differ slightly from the paper tables.

## Phase 2: Pix2Seq

See [`docs/PHASE2.md`](../docs/PHASE2.md). Same LUDB 1/16 splits and MeanIoU; ResNet-18 encoder + autoregressive segment decoder.

```bash
bash scripts/run_pix2seq.sh --gpus 0
```
