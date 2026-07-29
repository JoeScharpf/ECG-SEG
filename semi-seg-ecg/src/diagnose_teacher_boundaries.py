"""
Phase 2.5: teacher pseudo-label accuracy near boundaries vs interiors.

Runs on the **labeled validation set only** (not test, not unlabeled).
Requires a trained Mean Teacher checkpoint (student or teacher weights).

Regions (relative to GT segments):
  - interior: samples more than --band samples inside a wave
  - onset_band / offset_band: +/- --band samples around GT onset/offset
  - far_background: background farther than --band from any wave

Usage (from semi-seg-ecg/, after conda activate):
  cd semi-seg-ecg
  python -m src.diagnose_teacher_boundaries \\
    --config_path configs/base/resnet18/mean_teacher_unet.yaml \\
    --override_config_path configs/bench/ludb/1over16.yaml \\
    --checkpoint ../baseline/exps/resnet18/mean_teacher_unet/ludb/1over16/best-MeanIoU.pth \\
    --band 4 \\
    --out ../baseline/results/resnet18_mean_teacher_unet_ludb_1over16/teacher_boundary_diag.json

Prefer the EMA teacher weights if the checkpoint stores model_ema; otherwise
uses the student (best-MeanIoU) as a proxy diagnostic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from mergedeep import merge

# Allow running as script from repo or as module from semi-seg-ecg/src
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from algorithms.base import init_model_from_cfg  # noqa: E402
from utils.semi_dataset import build_seg_dataset, get_dataloader  # noqa: E402


def load_config(config_path: Path, override_path: Path | None) -> dict:
    with open(config_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    if override_path is not None:
        with open(override_path) as f:
            override = yaml.load(f, Loader=yaml.FullLoader)
        cfg = merge({}, cfg, override)
    return cfg


def region_masks(gt: np.ndarray, band: int) -> dict:
    """gt: (T,) class ids. Returns boolean masks for diagnostic regions."""
    t = len(gt)
    interior = np.zeros(t, dtype=bool)
    onset = np.zeros(t, dtype=bool)
    offset = np.zeros(t, dtype=bool)
    near_wave = np.zeros(t, dtype=bool)

    i = 0
    while i < t:
        c = int(gt[i])
        if c == 0:
            i += 1
            continue
        start = i
        while i < t and int(gt[i]) == c:
            i += 1
        end = i - 1
        # near any wave (for far-background)
        lo = max(0, start - band)
        hi = min(t, end + band + 1)
        near_wave[lo:hi] = True
        # onset / offset bands
        o0 = max(0, start - band)
        o1 = min(t, start + band + 1)
        onset[o0:o1] = True
        f0 = max(0, end - band)
        f1 = min(t, end + band + 1)
        offset[f0:f1] = True
        # interior: strictly inside by band
        if end - start + 1 > 2 * band:
            interior[start + band : end - band + 1] = True

    far_bg = (gt == 0) & (~near_wave)
    return {
        "interior": interior,
        "onset_band": onset,
        "offset_band": offset,
        "far_background": far_bg,
    }


def accuracy(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict:
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "acc": None}
    correct = int(((pred == gt) & mask).sum())
    return {"n": n, "acc": correct / n}


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Teacher boundary diagnostic on val")
    parser.add_argument("--config_path", type=Path, required=True)
    parser.add_argument("--override_config_path", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--band", type=int, default=4, help="+/- samples around boundary")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = load_config(args.config_path, args.override_config_path)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = init_model_from_cfg(cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    # Prefer EMA teacher if present
    state = None
    if isinstance(ckpt, dict):
        for key in ("model_ema", "teacher", "model"):
            if key in ckpt:
                state = ckpt[key]
                used = key
                break
        if state is None and "state_dict" in ckpt:
            state = ckpt["state_dict"]
            used = "state_dict"
        if state is None:
            # raw state dict
            state = ckpt
            used = "raw"
    else:
        state = ckpt
        used = "raw"
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint via '{used}' (missing={len(missing)}, unexpected={len(unexpected)})")

    model.to(device)
    model.eval()

    # Validation uses labels (never test / unlabeled for this diagnostic).
    val_set = build_seg_dataset(cfg["dataset"], split="valid")
    loader = get_dataloader(
        val_set,
        is_distributed=False,
        mode="valid",  # non-train -> SequentialSampler (same as mean_teacher.py)
        **cfg["dataloader"],
    )

    totals = {
        k: {"correct": 0, "n": 0}
        for k in ("interior", "onset_band", "offset_band", "far_background", "all")
    }

    for batch in loader:
        ecg = batch["ecg"].to(device)
        labels = batch["target"]
        if labels.dim() == 3:
            # one-hot (B,C,T) or (B,1,T)
            if labels.size(1) > 1:
                gt = labels.argmax(dim=1).cpu().numpy()
            else:
                gt = labels.squeeze(1).cpu().numpy().astype(np.int64)
        else:
            gt = labels.cpu().numpy().astype(np.int64)

        out = model(ecg, return_loss=False)
        logits = out["seg_logits"]
        # upsample to label length if needed
        if logits.shape[-1] != gt.shape[-1]:
            logits = torch.nn.functional.interpolate(
                logits, size=gt.shape[-1], mode="linear", align_corners=False
            )
        pred = logits.argmax(dim=1).cpu().numpy()

        for b in range(pred.shape[0]):
            regions = region_masks(gt[b], args.band)
            regions["all"] = np.ones_like(gt[b], dtype=bool)
            for name, mask in regions.items():
                n = int(mask.sum())
                if n == 0:
                    continue
                totals[name]["n"] += n
                totals[name]["correct"] += int(((pred[b] == gt[b]) & mask).sum())

    report = {
        "checkpoint": str(args.checkpoint),
        "split": "valid",
        "band_samples": args.band,
        "band_ms": args.band * 1000.0 / 250.0,
        "regions": {},
    }
    for name, d in totals.items():
        report["regions"][name] = {
            "n": d["n"],
            "acc": (d["correct"] / d["n"]) if d["n"] else None,
        }

    # Decision hint
    onset_acc = report["regions"]["onset_band"]["acc"]
    interior_acc = report["regions"]["interior"]["acc"]
    if onset_acc is not None and interior_acc is not None:
        report["boundary_gap"] = float(interior_acc - onset_acc)
        report["justifies_boundary_ssl"] = report["boundary_gap"] > 0.05
    else:
        report["boundary_gap"] = None
        report["justifies_boundary_ssl"] = None

    text = json.dumps(report, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
