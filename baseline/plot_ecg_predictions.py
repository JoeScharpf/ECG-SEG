"""
Plot ECG delineation examples: waveform + ground truth vs U-Net prediction.

Usage (from repo root):
    python baseline/plot_ecg_predictions.py \\
        --run-dir baseline/exps/resnet18/scratch_unet/ludb/1over16 \\
        --publish

Requires test_outputs.npy and test_labels.npy in --run-dir (produced by test.sh).
Waveforms are loaded from semi-seg-ecg/data/ludb with the same preprocessing as eval.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
ECG_DIR = REPO_ROOT / "semi-seg-ecg" / "data" / "ludb" / "ecg"
INDEX_CSV = REPO_ROOT / "semi-seg-ecg" / "index" / "ludb" / "LUDB_test.csv"
TARGET_LENGTH = 2500

# Class id -> display name and color (colorblind-friendly).
CLASS_NAMES = {0: "Background", 1: "P", 2: "QRS", 3: "T"}
CLASS_COLORS = {
    0: "#E8E8E8",
    1: "#4477AA",
    2: "#EE6677",
    3: "#228833",
}


def load_waveform(fname: str) -> np.ndarray:
    """Load and preprocess a test waveform to match eval length (2500)."""
    with open(ECG_DIR / fname, "rb") as f:
        x = pickle.load(f)
    x = np.asarray(x, dtype=np.float64).ravel()
    if len(x) != TARGET_LENGTH:
        src_t = np.linspace(0, 1, len(x))
        dst_t = np.linspace(0, 1, TARGET_LENGTH)
        x = np.interp(dst_t, src_t, x)
    x = (x - x.mean()) / (x.std() + 1e-8)
    return x


def per_sample_mean_iou(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Mean IoU over classes with non-empty union, per sample."""
    n = pred.shape[0]
    ious = np.zeros(n, dtype=np.float64)
    for i in range(n):
        pi, gi = pred[i], gt[i]
        cls_ious = []
        for c in range(4):
            inter = np.sum((pi == c) & (gi == c))
            union = np.sum((pi == c) | (gi == c))
            if union > 0:
                cls_ious.append(inter / union)
        ious[i] = float(np.mean(cls_ious)) if cls_ious else 0.0
    return ious


def _shade_classes(ax, t: np.ndarray, classes: np.ndarray, alpha: float = 0.45):
    """Draw semi-transparent class bands along the time axis."""
    if len(classes) == 0:
        return
    start = 0
    cur = int(classes[0])
    for i in range(1, len(classes)):
        if int(classes[i]) != cur:
            if cur != 0:
                ax.axvspan(t[start], t[i - 1], color=CLASS_COLORS[cur], alpha=alpha, lw=0)
            start = i
            cur = int(classes[i])
    if cur != 0:
        ax.axvspan(t[start], t[-1], color=CLASS_COLORS[cur], alpha=alpha, lw=0)


def plot_example(
    ecg: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    title: str,
    sample_iou: float,
    out_path: Path,
) -> None:
    t = np.arange(len(ecg))
    fig, axes = plt.subplots(3, 1, figsize=(14, 5), sharex=True, height_ratios=[1.2, 1, 1])

    axes[0].plot(t, ecg, color="black", linewidth=0.6)
    axes[0].set_ylabel("ECG")
    axes[0].set_title(f"{title}  |  sample mIoU = {sample_iou:.3f}")
    axes[0].grid(True, alpha=0.25)

    for ax, labels, label in zip(axes[1:], [gt, pred], ["Ground truth", "U-Net prediction"]):
        _shade_classes(ax, t, labels)
        ax.plot(t, ecg, color="black", linewidth=0.4, alpha=0.35)
        ax.set_ylabel(label)
        ax.set_ylim(ecg.min() - 0.3, ecg.max() + 0.3)
        ax.grid(True, alpha=0.25)

    patches = [
        mpatches.Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c])
        for c in (1, 2, 3)
    ]
    axes[0].legend(handles=patches, loc="upper right", ncol=3, fontsize=9)

    axes[-1].set_xlabel("Sample index (2500 @ 250 Hz ≈ 10 s)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def pick_indices(ious: np.ndarray, k_extra: int = 2) -> list[tuple[str, int]]:
    """Return (label, index) pairs for a diverse set of examples."""
    order = np.argsort(ious)
    n = len(ious)
    picks = [
        ("best", int(order[-1])),
        ("median", int(order[n // 2])),
        ("worst", int(order[0])),
    ]
    # Two additional high-quality examples (75th and 90th percentile).
    for pct, tag in [(75, "good_p75"), (90, "good_p90")]:
        idx = int(order[int(n * pct / 100)])
        if idx not in {p[1] for p in picks}:
            picks.append((tag, idx))
    return picks


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot U-Net ECG delineation examples")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=REPO_ROOT / "baseline/exps/resnet18/scratch_unet/ludb/1over16",
        help="Directory with test_outputs.npy and test_labels.npy",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to save PNGs (default: <run-dir>/visual_examples)",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Also copy figures to baseline/results/.../visual_examples/",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else run_dir / "visual_examples"
    outputs_path = run_dir / "test_outputs.npy"
    labels_path = run_dir / "test_labels.npy"
    if not outputs_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"Missing test_outputs.npy or test_labels.npy in {run_dir}\n"
            "Run test.sh on gpu2 first, then pull the .npy files."
        )

    probs = np.load(outputs_path)
    labels_oh = np.load(labels_path)
    pred = probs.argmax(axis=1)
    gt = labels_oh.argmax(axis=1)
    ious = per_sample_mean_iou(pred, gt)
    test_df = pd.read_csv(INDEX_CSV)
    if len(test_df) != len(pred):
        raise ValueError(
            f"Test CSV rows {len(test_df)} != predictions {len(pred)}"
        )

    picks = pick_indices(ious)

    manifest = []
    for tag, idx in picks:
        row = test_df.iloc[idx]
        fname = row["waveform"]
        ecg = load_waveform(fname)
        patient_id = row["ID"]
        title = f"{fname}  (patient {patient_id})"
        png_name = f"example_{tag}_idx{idx}.png"
        plot_example(
            ecg, gt[idx], pred[idx], title, float(ious[idx]), out_dir / png_name
        )
        manifest.append({
            "tag": tag,
            "index": idx,
            "waveform": fname,
            "patient_id": int(patient_id),
            "sample_mean_iou": round(float(ious[idx]), 4),
            "file": png_name,
        })
        print(f"  saved {png_name}  (mIoU={ious[idx]:.3f})")

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"\nFigures saved to: {out_dir}/")

    if args.publish:
        # Map exps path -> results folder name (same logic as plot_results.py).
        parts = run_dir.relative_to(REPO_ROOT).parts
        if len(parts) >= 6 and parts[0] == "baseline" and parts[1] == "exps":
            results_name = f"{parts[2]}_{parts[3]}_{parts[4]}_{parts[5]}"
        else:
            results_name = run_dir.name
        publish_dir = REPO_ROOT / "baseline" / "results" / results_name / "visual_examples"
        publish_dir.mkdir(parents=True, exist_ok=True)
        for item in out_dir.iterdir():
            shutil.copy2(item, publish_dir / item.name)
        print(f"Published to: {publish_dir}/")


if __name__ == "__main__":
    main()
