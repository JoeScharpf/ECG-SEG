"""
Plot training curves from SemiSegECG baseline run log.txt.

Usage (from repo root):
    python baseline/plot_results.py
    python baseline/plot_results.py --run-dir baseline/exps/resnet18/scratch/ludb/1over16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_DIR = REPO_ROOT / "baseline/exps/resnet18/scratch/ludb/1over16"


def load_log(log_path: Path) -> pd.DataFrame:
    if not log_path.exists():
        raise FileNotFoundError(
            f"Missing log file: {log_path}\n"
            "Run training first: bash scripts/run_phase1.sh"
        )
    rows = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No epochs found in {log_path}")
    return pd.DataFrame(rows)


def _best_at_epoch(df: pd.DataFrame, col: str, *, higher_is_better: bool) -> tuple[float, int]:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in log.txt. Available: {list(df.columns)}")
    if higher_is_better:
        idx = df[col].idxmax()
    else:
        idx = df[col].idxmin()
    row = df.loc[idx]
    return float(row[col]), int(row["epoch"])


def plot_training_curves(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = df["epoch"]

    if "train_loss" in df.columns and "valid_loss" in df.columns:
        axes[0].plot(epochs, df["train_loss"], label="train loss")
        axes[0].plot(epochs, df["valid_loss"], label="valid loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training and validation loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].set_title("Loss columns not found in log.txt")
        axes[0].axis("off")

    if "MeanIoU" in df.columns:
        axes[1].plot(epochs, df["MeanIoU"], label="valid MeanIoU", color="C2")
        best_miou, best_epoch = _best_at_epoch(df, "MeanIoU", higher_is_better=True)
        axes[1].axvline(best_epoch, color="gray", linestyle="--", alpha=0.7)
        axes[1].scatter([best_epoch], [best_miou], color="C3", zorder=5, label=f"best @ {best_epoch}")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MeanIoU")
        axes[1].set_title("Validation MeanIoU")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].set_title("MeanIoU not found in log.txt")
        axes[1].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_summary(df: pd.DataFrame, run_dir: Path, chart_path: Path) -> None:
    print("\n=== Baseline training summary ===")
    print(f"  Run directory:   {run_dir}")
    print(f"  Epochs logged:   {len(df)}")

    if "valid_loss" in df.columns:
        best_loss, loss_epoch = _best_at_epoch(df, "valid_loss", higher_is_better=False)
        print(f"  Best valid loss: {best_loss:.4f} @ epoch {loss_epoch}")

    if "MeanIoU" in df.columns:
        best_miou, miou_epoch = _best_at_epoch(df, "MeanIoU", higher_is_better=True)
        print(f"  Best MeanIoU:    {best_miou:.4f} @ epoch {miou_epoch}")

    test_csv = run_dir / "test_metrics.csv"
    if test_csv.exists():
        test_df = pd.read_csv(test_csv)
        if "MeanIoU" in test_df.columns:
            print(f"  Test MeanIoU:    {test_df['MeanIoU'].iloc[0]:.4f}")

    print(f"  Chart saved:     {chart_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot baseline training curves from log.txt")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Directory containing log.txt",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    log_path = run_dir / "log.txt"
    chart_path = run_dir / "training_curves.png"

    df = load_log(log_path)
    plot_training_curves(df, chart_path)
    print_summary(df, run_dir, chart_path)


if __name__ == "__main__":
    main()
