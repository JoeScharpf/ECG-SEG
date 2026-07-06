# ECG-SEG

Research workspace for semi-supervised ECG delineation (segmentation), built on the [SemiSegECG](https://github.com/vuno/semi-seg-ecg) benchmark.

## Structure

```
ECG-SEG/
├── data/           # Data download, inspection, and split audit scripts
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

## Training

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
