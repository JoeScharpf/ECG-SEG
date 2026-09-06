"""Path helpers for multi-dataset boundary-MT benchmark runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

CONFIG_STEM = "mean_teacher_boundary_unet"
IN_DOMAIN = ("ludb", "qtdb", "isp", "zhejiang")
FRACTIONS = (16, 8, 4, 2)
SEEDS = (0, 1, 2)


def exp_suffix(dataset: str, fraction: int | None, seed: int, smoke: bool) -> str:
    if dataset == "cross_domain":
        return "merged_smoke" if smoke else f"merged_seed{seed}"
    if smoke:
        return f"1over{fraction}_smoke"
    # LUDB legacy: 1/16 seed 0 without _seed0
    if dataset == "ludb" and fraction == 16 and seed == 0:
        return "1over16"
    return f"1over{fraction}_seed{seed}"


def results_dir_name(dataset: str, fraction: int | None, seed: int, smoke: bool) -> str:
    suffix = exp_suffix(dataset, fraction, seed, smoke)
    return f"resnet18_{CONFIG_STEM}_{dataset}_{suffix}"


def run_subdir(dataset: str, fraction: int | None, seed: int, smoke: bool) -> str:
    suffix = exp_suffix(dataset, fraction, seed, smoke)
    return f"{dataset}/{suffix}"


def bench_yaml_path(
    repo: Path, dataset: str, fraction: int | None, seed: int, smoke: bool
) -> Path:
    root = repo / "semi-seg-ecg" / "configs" / "bench"
    if dataset == "cross_domain":
        return root / "cross_domain" / "merged.yaml"
    if dataset == "ludb" and smoke:
        return root / "ludb" / "1over16_ssl_smoke.yaml"
    if dataset == "ludb" and fraction == 16 and seed == 0 and not smoke:
        return root / "ludb" / "1over16.yaml"
    if dataset == "ludb" and not smoke:
        return root / "ludb" / f"1over{fraction}_ssl_seed{seed}.yaml"
    return root / dataset / f"1over{fraction}.yaml"


def bench_yaml_rel(bench_path: Path, repo: Path) -> str:
  semi = repo / "semi-seg-ecg"
  rel = bench_path.relative_to(semi)
  return f"../{rel.as_posix()}"


def cell_paths(
    repo: Path,
    dataset: str,
    fraction: int | None = 16,
    seed: int = 0,
    smoke: bool = False,
) -> dict[str, str]:
    if dataset == "cross_domain":
        fraction = None
    elif fraction is None:
        raise ValueError("fraction required for in-domain datasets")

    sub = run_subdir(dataset, fraction, seed, smoke)
    run_dir = repo / "baseline" / "exps" / "resnet18" / CONFIG_STEM / sub
    results_dir = repo / "baseline" / "results" / results_dir_name(
        dataset, fraction, seed, smoke
    )
    bench_path = bench_yaml_path(repo, dataset, fraction, seed, smoke)
    bench_rel = bench_yaml_rel(bench_path, repo)

    exp_name = sub
    return {
        "dataset": dataset,
        "fraction": fraction,
        "seed": seed,
        "smoke": smoke,
        "config_stem": CONFIG_STEM,
        "exp_name": exp_name,
        "run_subdir": sub,
        "run_dir": str(run_dir),
        "results_dir": str(results_dir),
        "bench_yaml_rel": bench_rel,
        "bench_path": str(bench_path),
    }


def write_override(
    bench_path: Path,
    out_path: Path,
    seed: int,
    exp_name: str,
    smoke: bool,
) -> None:
    with bench_path.open(encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}
    cfg["seed"] = seed
    cfg["exp_name"] = exp_name
    if smoke:
        train = dict(cfg.get("train") or {})
        train["epochs"] = 1
        train["warmup_epochs"] = 0
        cfg["train"] = train
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)


def needs_override(dataset: str, fraction: int | None, seed: int, smoke: bool) -> bool:
    if dataset == "ludb":
        return False
    return True


def index_files_from_bench(bench_path: Path, semi_root: Path) -> list[Path]:
    with bench_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    ds = cfg.get("dataset") or {}
    index_dir = Path(ds.get("index_dir", ""))
    if not index_dir.is_absolute():
        index_dir = (semi_root / index_dir).resolve()
    names = [
        ds.get("train_labeled_csv"),
        ds.get("train_unlabeled_csv"),
        ds.get("valid_csv"),
        ds.get("test_csv"),
    ]
    files = []
    for name in names:
        if name:
            files.append(index_dir / name)
    return files


def iter_sweep_cells(
    datasets: tuple[str, ...] = ("ludb", "qtdb", "isp", "zhejiang", "cross_domain"),
) -> list[tuple[str, int | None, int]]:
    cells: list[tuple[str, int | None, int]] = []
    for dataset in datasets:
        if dataset == "cross_domain":
            for seed in SEEDS:
                cells.append((dataset, None, seed))
        else:
            for frac in FRACTIONS:
                for seed in SEEDS:
                    cells.append((dataset, frac, seed))
    return cells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--label-fraction", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--write-override", type=Path, default=None)
    args = ap.parse_args()

    frac = None if args.dataset == "cross_domain" else args.label_fraction
    paths = cell_paths(args.repo, args.dataset, frac, args.seed, args.smoke)

    if args.write_override:
        bench = Path(paths["bench_path"])
        write_override(
            bench,
            args.write_override,
            args.seed,
            paths["exp_name"],
            args.smoke,
        )
        paths["override_path"] = str(args.write_override)

    if args.json:
        print(json.dumps(paths, indent=2))
    else:
        for k, v in paths.items():
            if v is not None:
                print(f"{k.upper()}={v}")


if __name__ == "__main__":
    main()
