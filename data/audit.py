"""
Audit official SemiSegECG LUDB benchmark splits for leakage and summary stats.

Validates existing index CSVs — does not create new splits.

Usage (run from project root):
    cd ~/Desktop/safe
    python data/audit.py
    python data/audit.py --label-fraction 8

Dependencies:
    pip install pandas
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Import sibling data.py (works when run as: python data/audit.py from project root)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import INDEX_ROOT, load_index  # noqa: E402

SPLIT_ORDER = ("train_labeled", "train_unlabeled", "valid", "test")


def split_filenames(dataset: str, label_fraction: int) -> dict[str, str]:
    prefix = dataset.upper()
    return {
        "train_labeled": f"{prefix}_train_labeled_1over{label_fraction}.csv",
        "train_unlabeled": f"{prefix}_train_unlabeled.csv",
        "valid": f"{prefix}_valid.csv",
        "test": f"{prefix}_test.csv",
    }


def load_all_splits(
    index_dir: Path,
    dataset: str,
    label_fraction: int,
) -> dict[str, pd.DataFrame]:
    if not index_dir.exists():
        raise FileNotFoundError(
            f"Index directory not found: {index_dir}\n"
            "Run: python data/data.py --download"
        )
    filenames = split_filenames(dataset, label_fraction)
    return {
        name: load_index(index_dir, fname)
        for name, fname in filenames.items()
    }


def _get_waveforms(df: pd.DataFrame) -> set[str]:
    col = "waveform" if "waveform" in df.columns else df.columns[0]
    return set(df[col].astype(str))


def _get_patient_ids(df: pd.DataFrame) -> set:
    if "ID" not in df.columns:
        raise KeyError("Index CSV missing 'ID' column for patient-level leakage checks")
    return set(df["ID"])


def _overlap_count(a: set, b: set) -> int:
    return len(a & b)


def print_split_summary(splits: dict[str, pd.DataFrame]) -> int:
    total_rows = sum(len(df) for df in splits.values())
    all_waveforms: set[str] = set()

    print("\n--- Split summary ---")
    for name in SPLIT_ORDER:
        df = splits[name]
        waveforms = _get_waveforms(df)
        patients = _get_patient_ids(df)
        all_waveforms |= waveforms
        pct = 100.0 * len(df) / total_rows if total_rows else 0.0
        print(
            f"  {name:16s}: {len(df):5d} rows | "
            f"{len(waveforms):5d} waveforms | "
            f"{len(patients):4d} patients | {pct:5.1f}%"
        )

    train_rows = len(splits["train_labeled"]) + len(splits["train_unlabeled"])
    eval_rows = len(splits["valid"]) + len(splits["test"])
    train_pct = 100.0 * train_rows / total_rows if total_rows else 0.0
    eval_pct = 100.0 * eval_rows / total_rows if total_rows else 0.0

    print(f"\n  Training pool (labeled + unlabeled): {train_rows:5d} rows ({train_pct:.1f}%)")
    print(f"  Eval pool (valid + test):            {eval_rows:5d} rows ({eval_pct:.1f}%)")
    print(f"  TOTAL (split rows, may overlap):     {total_rows:5d} rows")
    print(f"  Unique waveforms across all splits:  {len(all_waveforms):5d}")

    return total_rows


def _print_check(label: str, overlap: int, *, critical: bool) -> bool:
    if critical:
        status = "PASS" if overlap == 0 else "FAIL"
        print(f"  {label}: {overlap}  {status}")
        return overlap == 0
    status = "INFO"
    print(f"  {label}: {overlap}  {status}")
    return True


def run_leakage_checks(splits: dict[str, pd.DataFrame]) -> bool:
    print("\n--- Leakage checks (patient ID) ---")
    ids = {name: _get_patient_ids(df) for name, df in splits.items()}
    all_pass = True

    critical_id_pairs = [
        ("train_labeled", "valid"),
        ("train_labeled", "test"),
        ("train_unlabeled", "valid"),
        ("train_unlabeled", "test"),
        ("valid", "test"),
    ]
    for a, b in critical_id_pairs:
        overlap = _overlap_count(ids[a], ids[b])
        ok = _print_check(f"{a} ∩ {b} (patient ID)", overlap, critical=True)
        all_pass = all_pass and ok

    overlap = _overlap_count(ids["train_labeled"], ids["train_unlabeled"])
    _print_check("train_labeled ∩ train_unlabeled (patient ID)", overlap, critical=False)

    print("\n--- Leakage checks (waveform file) ---")
    wfs = {name: _get_waveforms(df) for name, df in splits.items()}

    critical_wf_pairs = [
        ("train_labeled", "valid"),
        ("train_labeled", "test"),
        ("train_unlabeled", "valid"),
        ("train_unlabeled", "test"),
        ("valid", "test"),
    ]
    for a, b in critical_wf_pairs:
        overlap = _overlap_count(wfs[a], wfs[b])
        ok = _print_check(f"{a} ∩ {b} (waveform)", overlap, critical=True)
        all_pass = all_pass and ok

    overlap = _overlap_count(wfs["train_labeled"], wfs["train_unlabeled"])
    _print_check("train_labeled ∩ train_unlabeled (waveform)", overlap, critical=False)

    print()
    if all_pass:
        print("AUDIT PASSED — no leakage between train and eval splits.")
    else:
        print("AUDIT FAILED — see FAIL lines above.")

    return all_pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit official SemiSegECG LUDB benchmark splits"
    )
    parser.add_argument("--dataset", default="ludb", choices=["ludb"])
    parser.add_argument("--label-fraction", type=int, default=16, choices=[2, 4, 8, 16])
    args = parser.parse_args()

    index_dir = INDEX_ROOT / args.dataset
    print(f"=== LUDB Split Audit (1/{args.label_fraction} labels) ===")
    print(f"Index dir: {index_dir}")

    try:
        splits = load_all_splits(index_dir, args.dataset, args.label_fraction)
    except FileNotFoundError as exc:
        print(f"\nError: {exc}")
        return 1

    print_split_summary(splits)
    passed = run_leakage_checks(splits)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
