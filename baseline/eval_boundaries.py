"""
Phase 2 boundary / interval metrics for ECG delineation.

Reads test_outputs.npy + test_labels.npy from a run directory and reports:
  - overall MeanIoU
  - per-class IoU (bg / P / QRS / T)
  - onset and offset MAE (samples and ms @ 250 Hz) per wave class
  - QRS duration / PR / QT interval MAE where matched pairs exist
  - fragmented-segment rate

Matching protocol (locked conceptually on validation before test claims):
  1. Extract contiguous non-background segments from GT and pred masks.
  2. Match within each class independently by greedy max temporal IoU.
  3. A pair is accepted only if temporal IoU >= --match-iou (default 0.1).
  4. Unmatched GT -> miss (counts toward detection recall); unmatched pred -> FP.
  5. Onset/offset errors use matched pairs only.
  6. Interval metrics: QRS duration = offset-onset; PR = QRS_onset - P_onset;
     QT = T_offset - QRS_onset, using nearest matched same-beat pairing by
     QRS onset proximity within --beat-window samples (default 250 = 1 s).

Tolerance bands for future boundary-aware SSL should be chosen on labeled
validation and locked before test evaluation (see docs/PHASE3_SSL.md).

Usage (from repo root):
  python baseline/eval_boundaries.py \\
      --run-dir baseline/exps/resnet18/mean_teacher_unet/ludb/1over16 \\
      --out baseline/results/resnet18_mean_teacher_unet_ludb_1over16/boundary_metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CLASS_NAMES = {0: "background", 1: "P", 2: "QRS", 3: "T"}
FS_HZ = 250.0
WAVE_CLASSES = (1, 2, 3)

Segment = Tuple[int, int, int]  # class, start, end_inclusive


def mask_to_segments(mask: np.ndarray) -> List[Segment]:
    mask = np.asarray(mask).astype(np.int64).reshape(-1)
    segments: List[Segment] = []
    t = 0
    n = len(mask)
    while t < n:
        c = int(mask[t])
        if c == 0:
            t += 1
            continue
        start = t
        while t < n and int(mask[t]) == c:
            t += 1
        end = t - 1
        if c in WAVE_CLASSES:
            segments.append((c, start, end))
    return segments


def temporal_iou(a: Segment, b: Segment) -> float:
    _, as_, ae = a
    _, bs, be = b
    inter = max(0, min(ae, be) - max(as_, bs) + 1)
    union = (ae - as_ + 1) + (be - bs + 1) - inter
    return float(inter / union) if union > 0 else 0.0


def match_segments(
    gt_segs: Sequence[Segment],
    pred_segs: Sequence[Segment],
    class_id: int,
    min_iou: float,
) -> Tuple[List[Tuple[Segment, Segment]], List[Segment], List[Segment]]:
    """Greedy max-IoU matching within one class."""
    gt = [s for s in gt_segs if s[0] == class_id]
    pred = [s for s in pred_segs if s[0] == class_id]
    pairs: List[Tuple[Segment, Segment]] = []
    used_g = set()
    used_p = set()
    candidates = []
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            iou = temporal_iou(g, p)
            if iou >= min_iou:
                candidates.append((iou, i, j))
    candidates.sort(reverse=True)
    for _, i, j in candidates:
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        pairs.append((gt[i], pred[j]))
    unmatched_gt = [g for i, g in enumerate(gt) if i not in used_g]
    unmatched_pred = [p for j, p in enumerate(pred) if j not in used_p]
    return pairs, unmatched_gt, unmatched_pred


def per_class_iou(pred: np.ndarray, gt: np.ndarray, num_classes: int = 4) -> Dict[str, float]:
    out = {}
    for c in range(num_classes):
        inter = np.sum((pred == c) & (gt == c))
        union = np.sum((pred == c) | (gt == c))
        out[CLASS_NAMES[c]] = float(inter / union) if union > 0 else float("nan")
    vals = [v for v in out.values() if not np.isnan(v)]
    out["mean"] = float(np.mean(vals)) if vals else float("nan")
    return out


def samples_to_ms(samples: float) -> float:
    return float(samples) * 1000.0 / FS_HZ


def evaluate_run(
    pred: np.ndarray,
    gt: np.ndarray,
    match_iou: float = 0.1,
    beat_window: int = 250,
) -> dict:
    """pred/gt: (N, T) int class ids."""
    assert pred.shape == gt.shape
    n, t = pred.shape

    # Pixel IoU over whole test set
    iou = per_class_iou(pred.ravel(), gt.ravel())

    onset_errs: Dict[int, List[float]] = {c: [] for c in WAVE_CLASSES}
    offset_errs: Dict[int, List[float]] = {c: [] for c in WAVE_CLASSES}
    qrs_dur_errs: List[float] = []
    pr_errs: List[float] = []
    qt_errs: List[float] = []
    n_gt_segs = 0
    n_pred_segs = 0
    n_matched = 0
    n_fragmented_pred = 0  # pred segments with no GT match (proxy) + multi-fragment heuristic

    for i in range(n):
        gt_segs = mask_to_segments(gt[i])
        pred_segs = mask_to_segments(pred[i])
        n_gt_segs += len(gt_segs)
        n_pred_segs += len(pred_segs)

        # Fragmentation proxy: more pred segments than GT for same class
        for c in WAVE_CLASSES:
            n_g = sum(1 for s in gt_segs if s[0] == c)
            n_p = sum(1 for s in pred_segs if s[0] == c)
            if n_g > 0 and n_p > n_g:
                n_fragmented_pred += n_p - n_g

        matched_by_class: Dict[int, List[Tuple[Segment, Segment]]] = {}
        for c in WAVE_CLASSES:
            pairs, _, _ = match_segments(gt_segs, pred_segs, c, match_iou)
            matched_by_class[c] = pairs
            n_matched += len(pairs)
            for g, p in pairs:
                onset_errs[c].append(abs(p[1] - g[1]))
                offset_errs[c].append(abs(p[2] - g[2]))
                if c == 2:  # QRS duration
                    qrs_dur_errs.append(abs((p[2] - p[1]) - (g[2] - g[1])))

        # PR / QT using matched P/QRS/T near the same QRS
        qrs_pairs = matched_by_class[2]
        p_pairs = matched_by_class[1]
        t_pairs = matched_by_class[3]
        for g_q, p_q in qrs_pairs:
            # nearest matched P by GT onset proximity
            best_p = None
            best_d = beat_window + 1
            for g_p, p_p in p_pairs:
                d = abs(g_q[1] - g_p[1])
                if d < best_d:
                    best_d = d
                    best_p = (g_p, p_p)
            if best_p is not None:
                g_p, p_p = best_p
                pr_gt = g_q[1] - g_p[1]
                pr_pr = p_q[1] - p_p[1]
                pr_errs.append(abs(pr_pr - pr_gt))

            best_t = None
            best_d = beat_window + 1
            for g_t, p_t in t_pairs:
                d = abs(g_t[1] - g_q[1])
                if d < best_d:
                    best_d = d
                    best_t = (g_t, p_t)
            if best_t is not None:
                g_t, p_t = best_t
                qt_gt = g_t[2] - g_q[1]
                qt_pr = p_t[2] - p_q[1]
                qt_errs.append(abs(qt_pr - qt_gt))

    def mae_block(errs: List[float]) -> dict:
        if not errs:
            return {"n": 0, "mae_samples": None, "mae_ms": None}
        m = float(np.mean(errs))
        return {"n": len(errs), "mae_samples": m, "mae_ms": samples_to_ms(m)}

    boundary = {}
    for c in WAVE_CLASSES:
        boundary[CLASS_NAMES[c]] = {
            "onset": mae_block(onset_errs[c]),
            "offset": mae_block(offset_errs[c]),
        }

    return {
        "n_recordings": n,
        "signal_length": t,
        "fs_hz": FS_HZ,
        "match_iou_threshold": match_iou,
        "beat_window_samples": beat_window,
        "pixel_iou": iou,
        "detection": {
            "n_gt_segments": n_gt_segs,
            "n_pred_segments": n_pred_segs,
            "n_matched_pairs": n_matched,
            "match_rate_vs_gt": float(n_matched / n_gt_segs) if n_gt_segs else None,
            "extra_pred_fragments": n_fragmented_pred,
            "fragmentation_rate": float(n_fragmented_pred / max(n_pred_segs, 1)),
        },
        "boundary_mae": boundary,
        "intervals": {
            "qrs_duration": mae_block(qrs_dur_errs),
            "pr": mae_block(pr_errs),
            "qt": mae_block(qt_errs),
        },
    }


def load_pred_gt(run_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    outputs = np.load(run_dir / "test_outputs.npy")
    labels = np.load(run_dir / "test_labels.npy")
    # outputs: (N,C,T) probs or logits; labels: (N,C,T) one-hot or (N,T)
    if outputs.ndim == 3:
        pred = outputs.argmax(axis=1)
    elif outputs.ndim == 2:
        pred = outputs.astype(np.int64)
    else:
        raise ValueError(f"Unexpected test_outputs shape {outputs.shape}")
    if labels.ndim == 3:
        gt = labels.argmax(axis=1)
    elif labels.ndim == 2:
        gt = labels.astype(np.int64)
    else:
        raise ValueError(f"Unexpected test_labels shape {labels.shape}")
    if pred.shape != gt.shape:
        raise ValueError(f"pred {pred.shape} != gt {gt.shape}")
    return pred, gt


def main() -> None:
    parser = argparse.ArgumentParser(description="ECG boundary / interval metrics")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory with test_outputs.npy and test_labels.npy",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON output path",
    )
    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.1,
        help="Min temporal IoU to accept a GT-pred segment pair (lock on val)",
    )
    parser.add_argument(
        "--beat-window",
        type=int,
        default=250,
        help="Max samples between P/T and QRS for PR/QT pairing",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    pred, gt = load_pred_gt(run_dir)
    report = evaluate_run(pred, gt, match_iou=args.match_iou, beat_window=args.beat_window)
    report["run_dir"] = str(run_dir)

    text = json.dumps(report, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
