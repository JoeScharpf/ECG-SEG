# ECG-SEG

Research workspace for semi-supervised ECG delineation (segmentation), built on the [SemiSegECG](https://github.com/vuno/semi-seg-ecg) benchmark.

## Structure

```
ECG-SEG/
├── baseline/       # Phase 1/2 outputs, plot_results.py, published results
├── data/           # Data download, inspection, and split audit scripts
├── docs/           # Phase guides (PHASE1.md, PHASE2.md)
├── scripts/        # setup_phase1.sh, run_phase1.sh, run_pix2seq.sh
├── semi-seg-ecg/   # SemiSegECG benchmark (training configs, models, scripts)
└── README.md
```

## Quick start

```bash
# Install dependencies
pip install numpy pandas gdown

# Download LUDB data and inspect
python data/data.py --download

# Audit official train/val/test splits for leakage
python data/audit.py
```

See [data/README.md](data/README.md) for details on data layout, LUDB stats, and split audit results.

## Phase 1 baseline (gpu2)

Supervised **ResNet-18 + FCN** on LUDB (1/16 labels), trained from scratch on labeled data only — matches upstream `configs/base/resnet18/scratch.yaml`. See [docs/PHASE1.md](docs/PHASE1.md).

```bash
bash scripts/setup_phase1.sh
conda activate semi_seg_ecg
bash scripts/run_phase1.sh --gpus 0
```

Outputs: `baseline/exps/...` (full run, gitignored) and `baseline/results/...` (plots + metrics for GitHub).

To compare with published SemiSegECG tables, recompute mIoU via [`semi-seg-ecg/notebooks/perf_eval.ipynb`](semi-seg-ecg/notebooks/perf_eval.ipynb) from `test_outputs.npy` and `test_labels.npy` (not `test_metrics.csv` alone).

## Phase 2 Pix2Seq (gpu2)

Multi-class **Pix2Seq-style** model (ResNet-18 encoder + Transformer decoder) on the same LUDB 1/16 splits. See [docs/PHASE2.md](docs/PHASE2.md).

```bash
conda activate semi_seg_ecg
bash scripts/run_pix2seq.sh --gpus 0
```

Compare against Phase 1 ResNet test mIoU **0.6661** (paper 67.3%).

## Training (semi-supervised)

Training uses the `semi-seg-ecg` benchmark. Set up a conda environment per [semi-seg-ecg/README.md](semi-seg-ecg/README.md), then run on a GPU server:

```bash
cd semi-seg-ecg
bash scripts/train.sh \
  -f configs/base/resnet18/fixmatch.yaml \
  -o configs/bench/ludb/1over16.yaml \
  --gpus 0
```

## References

- SemiSegECG paper: [CIKM 2025](https://dl.acm.org/doi/10.1145/3746252.3760790)
- Upstream benchmark: https://github.com/vuno/semi-seg-ecg
