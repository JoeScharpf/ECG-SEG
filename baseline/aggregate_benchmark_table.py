#!/usr/bin/env python3
"""Aggregate boundary-MT benchmark cells into paper tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from benchmark_paths import FRACTIONS, SEEDS, cell_paths, iter_sweep_cells


def load_summary(path: Path) -> dict | None:
    p = path / "summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_boundary(path: Path) -> dict | None:
    p = path / "boundary_metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def mean_std(vals: list[float]) -> tuple[float | None, float | None]:
    if not vals:
        return None, None
    n = len(vals)
    m = sum(vals) / n
    if n == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, math.sqrt(var)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: baseline/results/paper_benchmark",
    )
    args = ap.parse_args()
    repo = args.repo
    out_dir = args.out_dir or (repo / "baseline" / "results" / "paper_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    nested: dict = {}

    for dataset, frac, seed in iter_sweep_cells():
        paths = cell_paths(repo, dataset, frac, seed, smoke=False)
        results_dir = Path(paths["results_dir"])
        s = load_summary(results_dir)
        if s is None:
            continue
        b = load_boundary(results_dir)
        iou = s.get("test_mean_iou")
        frac_label = "merged" if dataset == "cross_domain" else f"1/{frac}"
        row = {
            "dataset": dataset,
            "fraction": frac_label,
            "fraction_n": frac if frac is not None else 0,
            "seed": seed,
            "test_mean_iou": iou,
            "results_dir": str(results_dir.relative_to(repo)),
            "t_onset_mae_ms": None,
            "t_offset_mae_ms": None,
            "qt_mae_ms": None,
        }
        if b is not None:
            row["t_onset_mae_ms"] = b.get("boundary_mae", {}).get("T", {}).get(
                "onset", {}
            ).get("mae_ms")
            row["t_offset_mae_ms"] = b.get("boundary_mae", {}).get("T", {}).get(
                "offset", {}
            ).get("mae_ms")
            row["qt_mae_ms"] = b.get("intervals", {}).get("qt", {}).get("mae_ms")
        rows.append(row)

        nested.setdefault(dataset, {})
        key = frac_label
        nested[dataset].setdefault(
            key,
            {"seeds": {}, "n": 0, "test_mean_iou_mean": None, "test_mean_iou_std": None},
        )
        nested[dataset][key]["seeds"][seed] = iou

    for dataset in nested:
        for key in nested[dataset]:
            vals = [
                v
                for v in nested[dataset][key]["seeds"].values()
                if v is not None
            ]
            m, sd = mean_std(vals)
            nested[dataset][key]["n"] = len(vals)
            nested[dataset][key]["complete"] = len(vals) == len(SEEDS)
            nested[dataset][key]["test_mean_iou_mean"] = (
                None if m is None else round(m, 4)
            )
            nested[dataset][key]["test_mean_iou_std"] = (
                None if sd is None else round(sd, 4)
            )

    csv_path = out_dir / "summary_table.csv"
    fields = [
        "dataset",
        "fraction",
        "fraction_n",
        "seed",
        "test_mean_iou",
        "t_onset_mae_ms",
        "t_offset_mae_ms",
        "qt_mae_ms",
        "results_dir",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["dataset"], x["fraction_n"], x["seed"])):
            w.writerow(r)

    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(nested, indent=2) + "\n")

    def fmt_cell(dataset: str, col: str) -> str:
        cell = nested.get(dataset, {}).get(col, {})
        if cell.get("n", 0) == 0:
            return "—"
        mean = cell["test_mean_iou_mean"]
        std = cell["test_mean_iou_std"]
        if not cell.get("complete"):
            return f"{mean:.4f}±{std:.4f}* ({cell['n']}/3)"
        return f"{mean:.4f}±{std:.4f}"

    in_domain = ["ludb", "qtdb", "isp", "zhejiang"]
    frac_cols = [f"1/{f}" for f in FRACTIONS]
    labels = {
        "ludb": "LUDB",
        "qtdb": "QTDB",
        "isp": "ISP",
        "zhejiang": "Zhejiang",
    }
    cross = nested.get("cross_domain", {}).get("merged", {})
    cross_n = cross.get("n", 0)
    cross_complete = bool(cross.get("complete"))
    if cross_n == 0:
        cross_line = "Cross-domain (merged): —"
    elif not cross_complete:
        cross_line = (
            f"Cross-domain (merged): "
            f"{cross['test_mean_iou_mean']:.4f}±{cross['test_mean_iou_std']:.4f}* "
            f"({cross_n}/3)"
        )
    else:
        cross_line = (
            f"Cross-domain (merged): "
            f"{cross['test_mean_iou_mean']:.4f}±{cross['test_mean_iou_std']:.4f}"
        )

    lines = [
        "# Paper benchmark — boundary-aware Mean Teacher (ResNet-18 + U-Net)",
        "",
        "Test MeanIoU, mean ± std over seeds 0/1/2.",
        "",
        "| Dataset | 1/16 | 1/8 | 1/4 | 1/2 |",
        "|---------|-----:|----:|----:|----:|",
    ]
    for dataset in in_domain:
        cells = [fmt_cell(dataset, col) for col in frac_cols]
        lines.append(f"| {labels[dataset]} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            cross_line,
            "",
            "Regenerate:",
            "",
            "```bash",
            "python baseline/aggregate_benchmark_table.py",
            "```",
            "",
        ]
    )
    md_path = out_dir / "summary_table.md"
    md_path.write_text("\n".join(lines))

    all_cells = list(iter_sweep_cells())
    all_done = sum(
        1
        for dataset, frac, seed in all_cells
        if (
            Path(cell_paths(repo, dataset, frac, seed, smoke=False)["results_dir"])
            / "summary.json"
        ).exists()
    )
    in_domain_cells = [c for c in all_cells if c[0] != "cross_domain"]
    in_domain_done = sum(
        1
        for dataset, frac, seed in in_domain_cells
        if (
            Path(cell_paths(repo, dataset, frac, seed, smoke=False)["results_dir"])
            / "summary.json"
        ).exists()
    )
    cross_cells = [c for c in all_cells if c[0] == "cross_domain"]
    cross_done = sum(
        1
        for dataset, frac, seed in cross_cells
        if (
            Path(cell_paths(repo, dataset, frac, seed, smoke=False)["results_dir"])
            / "summary.json"
        ).exists()
    )
    checklist_path = out_dir / "completion_checklist.md"
    checklist_lines = [
        "# Benchmark completion checklist",
        "",
        f"Total completed: **{all_done}/{len(all_cells)}**",
        f"In-domain: **{in_domain_done}/{len(in_domain_cells)}**",
        f"Cross-domain: **{cross_done}/{len(cross_cells)}**"
        + (" (complete)" if cross_done == len(cross_cells) else ""),
        "",
        "| Dataset | Fraction | Seed | Status |",
        "|---------|----------|------|--------|",
    ]
    for dataset, frac, seed in all_cells:
        paths = cell_paths(repo, dataset, frac, seed, smoke=False)
        ok = (Path(paths["results_dir"]) / "summary.json").exists()
        frac_s = "merged" if dataset == "cross_domain" else f"1/{frac}"
        status = "done" if ok else "pending"
        checklist_lines.append(f"| {dataset} | {frac_s} | {seed} | {status} |")
    checklist_path.write_text("\n".join(checklist_lines) + "\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {checklist_path}")
    print(f"Total completed: {all_done}/{len(all_cells)}")
    print(f"In-domain: {in_domain_done}/{len(in_domain_cells)}")
    print(f"Cross-domain: {cross_done}/{len(cross_cells)}")


if __name__ == "__main__":
    main()
