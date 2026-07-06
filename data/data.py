"""
Load and inspect SemiSegECG benchmark data (LUDB example).

Usage:
    python data/data.py                  # inspect only (expects data already downloaded)
    python data/data.py --download       # download LUDB first, then inspect
    python data/data.py --label-fraction 16

Dependencies:
    pip install numpy pandas gdown
"""

from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent / "semi-seg-ecg"
DATA_ROOT = REPO_ROOT / "data"
INDEX_ROOT = REPO_ROOT / "index"

LABEL_NAMES = {0: "background", 1: "P", 2: "QRS", 3: "T"}

LUDB_DOWNLOADS = {
    "data_id": "1qPAEmilpbSfCArhfDDKl1Vrqn4j89ZWK",
    "index_id": "1vWSola1ySAt5XI8jMoG6ZPAwcFn8OAjP",
    "data_zip": "ludb.zip",
    "index_zip": "ludb.zip",
}


def _run(cmd: list[str]) -> None:
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _ensure_gdown() -> None:
    try:
        import gdown  # noqa: F401
    except ImportError:
        print("Installing gdown...")
        _run([sys.executable, "-m", "pip", "install", "gdown"])


def download_ludb() -> None:
    """Download and unzip LUDB data + index files."""
    print("\n=== Downloading LUDB ===")
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    INDEX_ROOT.mkdir(parents=True, exist_ok=True)

    _ensure_gdown()

    data_zip = DATA_ROOT / LUDB_DOWNLOADS["data_zip"]
    index_zip = INDEX_ROOT / LUDB_DOWNLOADS["index_zip"]

    if not data_zip.exists():
        print("Downloading LUDB waveforms + labels...")
        _run(["gdown", LUDB_DOWNLOADS["data_id"], "-O", str(data_zip)])
    else:
        print(f"Data zip already exists: {data_zip}")

    if not index_zip.exists():
        print("Downloading LUDB index splits...")
        _run(["gdown", LUDB_DOWNLOADS["index_id"], "-O", str(index_zip)])
    else:
        print(f"Index zip already exists: {index_zip}")

    ludb_data_ecg = DATA_ROOT / "ludb" / "ecg"
    ludb_data_label = DATA_ROOT / "ludb" / "label"
    ludb_index_dir = INDEX_ROOT / "ludb"

    if not ludb_data_ecg.exists() or not ludb_data_label.exists():
        print("Unzipping data...")
        with zipfile.ZipFile(data_zip, "r") as zf:
            zf.extractall(DATA_ROOT)
    else:
        print("Data already extracted.")

    if not ludb_index_dir.exists():
        print("Unzipping index...")
        with zipfile.ZipFile(index_zip, "r") as zf:
            zf.extractall(INDEX_ROOT)
    else:
        print("Index already extracted.")

    print("LUDB download complete.")


def load_index(index_dir: Path, csv_name: str) -> pd.DataFrame:
    path = index_dir / csv_name
    if not path.exists():
        raise FileNotFoundError(f"Missing index file: {path}")
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".pkl":
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported index format: {path}")


def load_pkl(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        arr = pickle.load(f)
    return np.asarray(arr)


def inspect_sample(ecg_path: Path, label_path: Path | None = None) -> dict:
    ecg = load_pkl(ecg_path)
    stats = {
        "ecg_shape": ecg.shape,
        "ecg_dtype": str(ecg.dtype),
        "ecg_min": float(ecg.min()),
        "ecg_max": float(ecg.max()),
        "ecg_mean": float(ecg.mean()),
        "ecg_std": float(ecg.std()),
    }
    if label_path is not None:
        label = load_pkl(label_path)
        unique, counts = np.unique(label, return_counts=True)
        stats["label_shape"] = label.shape
        stats["label_dtype"] = str(label.dtype)
        stats["signal_length"] = len(ecg)
        stats["class_counts"] = {
            LABEL_NAMES.get(int(u), str(u)): int(c)
            for u, c in zip(unique, counts)
        }
        if ecg.shape != label.shape:
            stats["length_mismatch"] = True
    return stats


def _resolve_columns(df: pd.DataFrame) -> tuple[str, str | None]:
    waveform_col = "waveform" if "waveform" in df.columns else df.columns[0]
    label_col = "label" if "label" in df.columns else None
    if label_col is None and len(df.columns) > 1:
        label_col = df.columns[1]
    return waveform_col, label_col


def print_dataset_summary(
    dataset: str,
    label_fraction: int,
    ecg_dir: Path,
    label_dir: Path,
    index_dir: Path,
) -> None:
    print(f"\n=== Dataset: {dataset.upper()} (1/{label_fraction} labels) ===")
    print(f"ECG dir:   {ecg_dir}")
    print(f"Label dir: {label_dir}")
    print(f"Index dir: {index_dir}")

    missing = []
    for name, path in [("ECG", ecg_dir), ("Label", label_dir), ("Index", index_dir)]:
        if path.exists():
            print(f"  {name} directory found.")
        else:
            print(f"  {name} directory missing: {path}")
            missing.append(name)

    if missing:
        print("\nCannot inspect — run with --download first.")
        return

    splits = {
        "train_labeled": f"{dataset.upper()}_train_labeled_1over{label_fraction}.csv",
        "train_unlabeled": f"{dataset.upper()}_train_unlabeled.csv",
        "valid": f"{dataset.upper()}_valid.csv",
        "test": f"{dataset.upper()}_test.csv",
    }

    print("\n--- Split sizes ---")
    index_dfs: dict[str, pd.DataFrame] = {}
    for split_name, csv_name in splits.items():
        df = load_index(index_dir, csv_name)
        index_dfs[split_name] = df
        print(f"  {split_name:16s}: {len(df):5d} samples  ({csv_name})")
        print(f"    columns: {list(df.columns)}")

    ecg_files = list(ecg_dir.glob("*.pkl"))
    label_files = list(label_dir.glob("*.pkl"))
    print("\n--- On-disk file counts ---")
    print(f"  ECG .pkl files:   {len(ecg_files)}")
    print(f"  Label .pkl files: {len(label_files)}")

    print("\n--- Sample inspection (first train_labeled row) ---")
    row = index_dfs["train_labeled"].iloc[0]
    waveform_col, label_col = _resolve_columns(index_dfs["train_labeled"])
    ecg_path = ecg_dir / row[waveform_col]
    label_path = label_dir / row[label_col]
    print(f"  waveform: {row[waveform_col]}")
    print(f"  label:    {row[label_col]}")

    stats = inspect_sample(ecg_path, label_path)
    print(f"  ECG shape:    {stats['ecg_shape']}, dtype={stats['ecg_dtype']}")
    print(f"  ECG range:    [{stats['ecg_min']:.4f}, {stats['ecg_max']:.4f}]")
    print(f"  ECG mean±std: {stats['ecg_mean']:.4f} ± {stats['ecg_std']:.4f}")
    print(f"  Label shape:  {stats['label_shape']}, dtype={stats['label_dtype']}")
    print("  Class counts:")
    for cls, count in stats["class_counts"].items():
        pct = 100.0 * count / stats["signal_length"]
        print(f"    {cls:12s}: {count:5d} samples ({pct:.1f}%)")

    print("\n--- Unlabeled sample (first train_unlabeled row) ---")
    row_u = index_dfs["train_unlabeled"].iloc[0]
    ecg_path_u = ecg_dir / row_u[waveform_col]
    stats_u = inspect_sample(ecg_path_u, label_path=None)
    print(f"  waveform: {row_u[waveform_col]}")
    print(f"  ECG shape: {stats_u['ecg_shape']}, dtype={stats_u['ecg_dtype']}")
    print("  (no label loaded for unlabeled split)")

    print("\nData load inspection complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and inspect SemiSegECG data")
    parser.add_argument("--download", action="store_true", help="Download LUDB before inspecting")
    parser.add_argument("--dataset", default="ludb", choices=["ludb"])
    parser.add_argument("--label-fraction", type=int, default=16, choices=[2, 4, 8, 16])
    args = parser.parse_args()

    if args.download:
        download_ludb()

    ecg_dir = DATA_ROOT / args.dataset / "ecg"
    label_dir = DATA_ROOT / args.dataset / "label"
    index_dir = INDEX_ROOT / args.dataset

    print_dataset_summary(
        dataset=args.dataset,
        label_fraction=args.label_fraction,
        ecg_dir=ecg_dir,
        label_dir=label_dir,
        index_dir=index_dir,
    )


if __name__ == "__main__":
    main()
