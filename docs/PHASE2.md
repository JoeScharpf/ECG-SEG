# Phase 2: Pix2Seq (multi-class) on LUDB

Pix2Seq-style ECG delineation: **ResNet-18 1D encoder + Transformer decoder** that emits quantized segment tokens `(class, start, end)`, then rasterizes to a dense multi-class mask for MeanIoU.

This matches the SemiSegECG **multi-class** setup (same as Phase 1 ResNet). Multi-label is out of scope here.

## Compare to Phase 1

| | Phase 1 ResNet | Phase 2 Pix2Seq |
|--|----------------|-----------------|
| Encoder | ResNet-18 1D | ResNet-18 1D (same) |
| Head | FCN (per-timestep) | Autoregressive segment tokens |
| Data | LUDB 1/16 | LUDB 1/16 (same splits) |
| Metric | MeanIoU | MeanIoU (rasterized masks) |
| Baseline to beat | — | test mIoU **0.6661** (paper 67.3%) |

## Run on gpu2

```bash
cd ~/ECG-SEG
git pull
conda activate semi_seg_ecg
tmux new -s pix2seq
bash scripts/run_pix2seq.sh --gpus 0
```

Requires Phase 1 setup already done (`bash scripts/setup_phase1.sh`).

## Outputs

| Path | Contents |
|------|----------|
| `baseline/exps/pix2seq/scratch/ludb/1over16/` | Checkpoints, `log.txt`, `test_metrics.csv`, `.npy` (gitignored) |
| `baseline/results/pix2seq_scratch_ludb_1over16/` | Published chart + metrics for GitHub |

Publish only:

```bash
python baseline/plot_results.py \
  --run-dir baseline/exps/pix2seq/scratch/ludb/1over16 \
  --publish
```

## Design notes

- **Multi-class only** — non-overlapping P/QRS/T segments (advisor-aligned with SemiSegECG)
- Token vocab: BOS/EOS/PAD + class tokens + coordinate bins (`num_bins=250`)
- `num_classes: 4` is set in `configs/base/pix2seq/scratch.yaml` (and must match `metric.num_classes`); out-of-range mask ids raise instead of silent clamping
- **Train loss:** token cross-entropy with teacher forcing on the **token** sequence only (no teacher-forced `seg_logits`)
- **Eval / MeanIoU:** real autoregressive `generate()` → rasterize to a dense mask → MeanIoU (same metric path as Phase 1). Do not use teacher-forced token argmax for reported IoU — that would be conditioned on ground-truth prefixes and inflate scores vs test-time decoding
- Fixed-length LUDB inputs (`signal_length: 2500`); no encoder memory padding mask
- `generate(max_len=...)` only limits decode *steps*; returned sequences are always padded to `max_seq_len` for batch decode

## Tokenizer smoke test (local)

```bash
python tests/test_pix2seq_tokenizer.py
```
