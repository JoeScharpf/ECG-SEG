#!/usr/bin/env python3
"""Aggregate LUDB fraction×method×seed results into CSV/MD/JSON tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def cell_dirs(repo: Path, method: str, frac: int, seed: int) -> list[Path]:
    """Preferred results dirs in priority order (first existing wins)."""
    base = repo / "baseline" / "results"
    if method == "supervised":
        return [
            base / f"resnet18_scratch_unet_ludb_1over{frac}_unet_seed{seed}",
        ]
    stem = {
        "mt": "mean_teacher_unet",
        "boundary_mt": "mean_teacher_boundary_unet",
    }[method]
    # Legacy 1/16 seed0 has no _seed0 suffix
    if frac == 16 and seed == 0:
        return [
            base / f"resnet18_{stem}_ludb_1over16",
            base / f"resnet18_{stem}_ludb_1over16_seed0",
        ]
    return [base / f"resnet18_{stem}_ludb_1over{frac}_seed{seed}"]


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
        help="Default: baseline/results/ludb_fraction_benchmark",
    )
    args = ap.parse_args()
    repo = args.repo
    out_dir = args.out_dir or (repo / "baseline" / "results" / "ludb_fraction_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = ["supervised", "mt", "boundary_mt"]
    fracs = [16, 8, 4, 2]
    seeds = [0, 1, 2]

    rows = []
    nested: dict = {}

    for method in methods:
        nested[method] = {}
        for frac in fracs:
            seed_ious = {}
            seed_rows = []
            for seed in seeds:
                found = None
                for d in cell_dirs(repo, method, frac, seed):
                    s = load_summary(d)
                    if s is not None:
                        found = (d, s)
                        break
                if found is None:
                    continue
                d, s = found
                b = load_boundary(d)
                iou = s.get("test_mean_iou")
                seed_ious[seed] = iou
                row = {
                    "method": method,
                    "fraction": f"1/{frac}",
                    "fraction_n": frac,
                    "seed": seed,
                    "test_mean_iou": iou,
                    "results_dir": str(d.relative_to(repo)),
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
                seed_rows.append(row)

            vals = [v for v in seed_ious.values() if v is not None]
            m, sd = mean_std(vals)
            nested[method][f"1/{frac}"] = {
                "seeds": seed_ious,
                "n": len(vals),
                "test_mean_iou_mean": None if m is None else round(m, 4),
                "test_mean_iou_std": None if sd is None else round(sd, 4),
                "complete": len(vals) == 3,
            }

    # CSV
    csv_path = out_dir / "summary_table.csv"
    fields = [
        "method",
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
        for r in rows:
            w.writerow(r)

    # JSON
    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(nested, indent=2) + "\n")

    # Markdown pivot
    labels = {
        "supervised": "U-Net supervised",
        "mt": "U-Net + Mean Teacher",
        "boundary_mt": "U-Net + Boundary-aware MT",
    }
    lines = [
        "# LUDB label-fraction benchmark",
        "",
        "Mean ± std test MeanIoU over seeds 0/1/2.",
        "",
        "| Method | 1/16 | 1/8 | 1/4 | 1/2 |",
        "|--------|-----:|----:|----:|----:|",
    ]
    for method in methods:
        cells = []
        for frac in fracs:
            cell = nested[method][f"1/{frac}"]
            if cell["n"] == 0:
                cells.append("—")
            elif not cell["complete"]:
                cells.append(
                    f"{cell['test_mean_iou_mean']:.4f}±{cell['test_mean_iou_std']:.4f}* ({cell['n']}/3)"
                )
            else:
                cells.append(
                    f"{cell['test_mean_iou_mean']:.4f}±{cell['test_mean_iou_std']:.4f}"
                )
        lines.append(f"| {labels[method]} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "\\* incomplete seeds. Regenerate with:",
            "",
            "```bash",
            "python baseline/aggregate_ludb_fraction_table.py",
            "```",
            "",
        ]
    )
    md_path = out_dir / "summary_table.md"
    md_path.write_text("\n".join(lines))

    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# LUDB fraction benchmark tables",
                "",
                "Generated by `baseline/aggregate_ludb_fraction_table.py`.",
                "",
                "- `summary_table.csv` — per-seed rows",
                "- `summary_table.md` — mean±std MeanIoU pivot",
                "- `summary.json` — nested machine-readable aggregate",
                "",
                "See `summary_table.md` for the current table.",
                "",
            ]
        )
    )

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")
    print(md_path.read_text())


if __name__ == "__main__":
    main()
