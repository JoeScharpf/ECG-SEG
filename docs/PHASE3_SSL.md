# Phase 3: Decoder design × SSL (LUDB)

Controlled study: same ResNet-18 encoder, **FCN vs U-Net** decode heads, under
supervised / Mean Teacher / FixMatch. Then (only if justified) soft
boundary-aware Mean Teacher.

## Research question

How does decoder **design and capacity** change the benefit of semi-supervised
learning for ECG delineation, and can ECG-specific boundary consistency still
improve a strong multiscale model?

Do **not** frame this as “SemiSegECG used a weak decoder.” Frame as: existing
ECG SSL benchmarks primarily evaluate lightweight heads; we study how decoder
design (multiscale skips + progressive upsampling + capacity) changes SSL value.

## Matched training budget (fairness)

Across FCN vs U-Net and supervised vs SSL, keep identical:

| Knob | Value (LUDB 1/16 configs) |
|------|---------------------------|
| Epochs | 100 |
| Batch size | 16 |
| Optimizer | AdamW, lr 0.001, wd 0.05, warmup 10 |
| Weak augs | `random_resize_crop` |
| Strong augs (SSL) | RandAugment block (same ops/level) |
| MT `ema_decay` | 0.99 |
| FixMatch `conf_thresh` | 0.80 |
| Checkpoint | `best-MeanIoU` on **validation** |
| Val / test cadence | framework default (each epoch val; test after train) |
| `metric.per_class` | `false` (scalar MeanIoU for checkpointing; per-class IoU from `test_*.npy` in Phase 2) |

Also record: trainable param count, train time/epoch, peak GPU memory.

## Run matrix (LUDB 1/16)

| Decoder | Supervised | Mean Teacher | FixMatch |
|---------|----------:|-------------:|---------:|
| FCN | Done (~0.666) | `run_ssl.sh --method mean_teacher --head fcn` | `... --method fixmatch --head fcn` |
| U-Net | Done (~0.818) | `... --method mean_teacher --head unet` (**first**) | `... --method fixmatch --head unet` |

Seed 0 = debug. Replicate **U-Net + Mean Teacher** on seeds 1–2 before the
decision gate. Call an SSL gain convincing only across seeds (supervised U-Net
std ≈ 0.002).

## Commands (gpu2, tmux)

```bash
cd ~/ECG-SEG   # or your clone path
git pull
conda activate semi_seg_ecg
tmux new -s ssl

# Smoke (1 epoch)
bash scripts/run_ssl.sh --method mean_teacher --head unet --gpus 0 --smoke

# Full seed 0
bash scripts/run_ssl.sh --method mean_teacher --head unet --gpus 0
bash scripts/run_ssl.sh --method fixmatch --head unet --gpus 0
bash scripts/run_ssl.sh --method mean_teacher --head fcn --gpus 0
bash scripts/run_ssl.sh --method fixmatch --head fcn --gpus 0

# U-Net + MT seeds 1–2
bash scripts/run_ssl.sh --method mean_teacher --head unet --seed 1 --gpus 0
bash scripts/run_ssl.sh --method mean_teacher --head unet --seed 2 --gpus 0
```

## Outputs

| Path | Contents |
|------|----------|
| `baseline/exps/resnet18/mean_teacher_unet/ludb/1over16/` | Checkpoints, log, test npy (gitignored) |
| `baseline/exps/resnet18/fixmatch_unet/ludb/1over16/` | Same for FixMatch + U-Net |
| `baseline/exps/resnet18/mean_teacher/ludb/1over16/` | FCN + Mean Teacher |
| `baseline/exps/resnet18/fixmatch/ludb/1over16/` | FCN + FixMatch |
| `baseline/results/resnet18_*_ludb_1over16/` | Published curves + metrics |

## Phase 2 — Boundary / interval metrics

After a run has `test_outputs.npy` / `test_labels.npy`:

```bash
python baseline/eval_boundaries.py \
  --run-dir baseline/exps/resnet18/mean_teacher_unet/ludb/1over16 \
  --out baseline/results/resnet18_mean_teacher_unet_ludb_1over16/boundary_metrics.json
```

Matching rules (`--match-iou`, `--beat-window`) should be chosen on **validation**
and locked before final test claims.

## Phase 2.5 — Teacher boundary diagnostic (labeled val only)

```bash
cd semi-seg-ecg
python diagnose_teacher_boundaries.py \
  --config_path configs/base/resnet18/mean_teacher_unet.yaml \
  --override_config_path configs/bench/ludb/1over16.yaml \
  --checkpoint ../baseline/exps/resnet18/mean_teacher_unet/ludb/1over16/best-MeanIoU.pth \
  --band 4 \
  --out ../baseline/results/resnet18_mean_teacher_unet_ludb_1over16/teacher_boundary_diag.json
```

Proceed to boundary-aware MT only if interior accuracy clearly exceeds
onset/offset-band accuracy on validation.

## Phase 3 — Soft boundary-aware Mean Teacher

Implemented as `algorithms/mean_teacher_boundary.py` +
`configs/base/resnet18/mean_teacher_boundary_unet.yaml`.

Loss: `L_sup + λ_region L_u_region + λ_boundary L_u_boundary`, with soft edge
weights `0.5 Σ_c |p_t − p_{t−1}|` dilated by `boundary_band` (default 4).
Interiors use hard conf gating (`conf_thresh: 0.80`); boundary bands use soft
confidence (no hard drop). Globally low-confidence unlabeled clips are rejected
via `sample_conf_thresh`.

```bash
bash scripts/run_ssl.sh --method mean_teacher_boundary --head unet --gpus 0
bash scripts/run_ssl.sh --method mean_teacher_boundary --head unet --smoke
```

Default hyps (tune on labeled val before test claims): `boundary_band: 4`,
`lambda_region: 1.0`, `lambda_boundary: 1.0`, `conf_thresh: 0.80`,
`sample_conf_thresh: 0.50`.

## Hygiene (locked before test claims)

1. Teacher-boundary diagnostic runs on the **labeled validation set** only.
2. Boundary-matching rules, tolerance bands, and loss hyperparameters are chosen
   on validation and **locked** before final test evaluation.
3. Matched optimizer steps, labeled/unlabeled batch counts, augmentations,
   validation frequency, and checkpoint policy (table above).

## Decision gate

- **Outcome A:** U-Net + SSL beats supervised U-Net across seeds → boundary
  diagnostic → soft boundary-aware MT.
- **Outcome B:** no mIoU gain → verify seeds + another fraction; check whether
  interval/onset metrics still move; then decide empirical vs method paper.

Do not expand into small-U-Net sweeps, new backbones, or extra SSL algorithms
until this grid + gate are done.
