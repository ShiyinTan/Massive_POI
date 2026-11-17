#!/usr/bin/env python3
"""Run the STHGCN ablation suite over multiple seeds; write mean±std summaries.

Does not set CUDA_VISIBLE_DEVICES (honors the environment). Omitting --gpu leaves
run.py on the yaml default. Paper single-seed: --runs 1 --seeds 54607333.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import os.path as osp
import random
import subprocess
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Allow `python3 scripts/run_ablations.py` from the model root.
_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))

# Import ablation_cli by path so we do not pull utils/__init__.py (PyG / torch_sparse).
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "ablation_cli", osp.join(_ROOT, "utils", "ablation_cli.py")
)
_ablation_cli = _ilu.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_ablation_cli)
ablation_run_name = _ablation_cli.ablation_run_name

CITIES = ("beijing", "tokyo", "shanghai", "melbourne")
METRICS = ("Recall@1", "Recall@5", "Recall@10", "Recall@20", "MRR")
PAPER_SEED = 54607333

# One flag away from paper defaults (suite order; paper first).
ABLATION_SUITE: List[Tuple[str, Dict[str, str]]] = [
    ("paper", {}),
    ("no_user", {"use_user_embed": "off"}),
    ("edges_ci2traj", {"hypergraph_edges": "ci2traj"}),
    ("edges_traj2traj", {"hypergraph_edges": "traj2traj"}),
    ("edges_none", {"hypergraph_edges": "none"}),
    ("scope_train", {"graph_scope": "train"}),
    ("prefix_drop", {"unseen_prefix_poi": "drop"}),
]
SUITE_ORDER = [name for name, _ in ABLATION_SUITE]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run STHGCN ablations repeatedly and report mean±std."
    )
    p.add_argument(
        "--city",
        action="append",
        choices=list(CITIES),
        default=None,
        help="City to run (repeatable). Default: beijing.",
    )
    p.add_argument("--runs", type=int, default=5, help="Repeats per ablation. Default: 5.")
    p.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Passed through to run.py. Default: omit (use yaml run_args.gpu).",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Seeds for the repeats. Must match --runs if set. "
        f"If omitted, draw --runs distinct seeds. Paper: --runs 1 --seeds {PAPER_SEED}.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Redo even when results/summary.csv already has city+run_name+seed.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned run.py commands without executing them.",
    )
    return p


class _FlagNS:
    """Minimal namespace so ablation_run_name() works for suite flag dicts."""

    def __init__(self, flags: Dict[str, str]):
        self.use_user_embed = flags.get("use_user_embed", "on")
        self.hypergraph_edges = flags.get("hypergraph_edges", "both")
        self.graph_scope = flags.get("graph_scope", "full")
        self.unseen_prefix_poi = flags.get("unseen_prefix_poi", "unk")


def draw_seeds(n: int, rng: Optional[random.Random] = None) -> List[int]:
    rng = rng or random.Random()
    seeds: List[int] = []
    while len(seeds) < n:
        s = rng.randint(0, 100_000_000)
        if s not in seeds:
            seeds.append(s)
    return seeds


def resolve_seeds(runs: int, seeds: Optional[Sequence[int]]) -> List[int]:
    if seeds is None:
        out = draw_seeds(runs)
        print(f"[ablation-runner] drew seeds={out}")
        return out
    if len(seeds) != runs:
        raise SystemExit(
            f"--seeds has {len(seeds)} value(s) but --runs={runs}; they must match."
        )
    out = list(seeds)
    if len(set(out)) != len(out):
        raise SystemExit("--seeds must be distinct.")
    print(f"[ablation-runner] using seeds={out}")
    return out


def summary_csv_path(root: str) -> str:
    return osp.join(root, "results", "summary.csv")


def load_summary_rows(path: str) -> List[dict]:
    if not osp.exists(path) or osp.getsize(path) == 0:
        return []
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def completed_keys(rows: Iterable[dict]) -> set:
    """Set of (city, run_name, seed_str) already present in summary.csv."""
    keys = set()
    for r in rows:
        city = (r.get("city") or "").strip()
        run_name = (r.get("run_name") or "").strip()
        seed = (r.get("seed") or "").strip()
        if city and run_name and seed != "":
            keys.add((city, run_name, seed))
    return keys


def build_run_cmd(
    city: str,
    flags: Dict[str, str],
    seed: int,
    gpu: Optional[int],
) -> List[str]:
    cmd = [
        sys.executable,
        "run.py",
        "-f",
        f"std_conf/{city}.yml",
        "--seed",
        str(seed),
    ]
    if gpu is not None:
        cmd.extend(["--gpu", str(gpu)])
    for k, v in flags.items():
        cmd.extend([f"--{k}", v])
    return cmd


def run_suite(
    cities: Sequence[str],
    seeds: Sequence[int],
    gpu: Optional[int],
    force: bool,
    dry_run: bool,
    root: str,
) -> None:
    os.chdir(root)
    summary_path = summary_csv_path(root)
    done = completed_keys(load_summary_rows(summary_path))

    for city in cities:
        for run_name, flags in ABLATION_SUITE:
            # Sanity: suite token must match ablation_run_name
            assert ablation_run_name(_FlagNS(flags)) == run_name, (flags, run_name)
            for seed in seeds:
                key = (city, run_name, str(seed))
                if not force and key in done:
                    print(f"[skip] {city}/{run_name}/seed={seed} (already in summary.csv)")
                    continue
                cmd = build_run_cmd(city, flags, seed, gpu)
                print(f"[run] {' '.join(cmd)}")
                if dry_run:
                    continue
                # Honor existing CUDA_VISIBLE_DEVICES; do not set/override it.
                env = os.environ.copy()
                proc = subprocess.run(cmd, cwd=root, env=env)
                if proc.returncode != 0:
                    raise SystemExit(
                        f"run.py failed (exit {proc.returncode}) for "
                        f"city={city} run_name={run_name} seed={seed}"
                    )
                done.add(key)


def _to_float(val) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def mean_std(values: List[float]) -> Tuple[float, Optional[float]]:
    n = len(values)
    if n == 0:
        raise ValueError("empty")
    mu = sum(values) / n
    if n == 1:
        return mu, None
    var = sum((x - mu) ** 2 for x in values) / (n - 1)  # sample std, ddof=1
    return mu, math.sqrt(var)


def aggregate_rows(
    rows: Sequence[dict],
    cities: Sequence[str],
) -> List[dict]:
    """One aggregated row per city+run_name (suite order, paper first)."""
    buckets: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    city_set = set(cities)
    suite_set = set(SUITE_ORDER)
    for r in rows:
        city = (r.get("city") or "").strip()
        run_name = (r.get("run_name") or "").strip()
        if city not in city_set or run_name not in suite_set:
            continue
        if any(_to_float(r.get(m)) is None for m in METRICS):
            continue
        buckets[(city, run_name)].append(r)

    out: List[dict] = []
    for city in cities:
        for run_name in SUITE_ORDER:
            group = buckets.get((city, run_name), [])
            if not group:
                continue
            row: dict = {"city": city, "run_name": run_name, "n": len(group)}
            for m in METRICS:
                vals = [_to_float(r[m]) for r in group]
                vals = [v for v in vals if v is not None]
                mu, sd = mean_std(vals)
                row[f"{m}_mean"] = mu
                row[f"{m}_std"] = "" if sd is None else sd
            out.append(row)
    return out


ABLATION_SUMMARY_HEADER = (
    ["city", "run_name", "n"]
    + [f"{m}_mean" for m in METRICS]
    + [f"{m}_std" for m in METRICS]
)


def write_ablation_summary_csv(path: str, rows: Sequence[dict]) -> None:
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ABLATION_SUMMARY_HEADER)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in ABLATION_SUMMARY_HEADER})


def _fmt_mean_std(mean, std) -> str:
    mean_f = float(mean)
    if std is None or std == "":
        return f"{mean_f:.4f}"
    return f"{mean_f:.4f}±{float(std):.4f}"


def render_ablation_summary_md(rows: Sequence[dict], cities: Sequence[str]) -> str:
    lines: List[str] = ["# STHGCN ablation summary", ""]
    by_city: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_city[r["city"]].append(r)

    for city in cities:
        city_rows = by_city.get(city, [])
        if not city_rows:
            continue
        lines.append(f"## {city}")
        lines.append("")
        header = ["run_name", "n"] + list(METRICS)
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for r in city_rows:
            cells = [r["run_name"], str(r["n"])]
            for m in METRICS:
                cells.append(_fmt_mean_std(r[f"{m}_mean"], r[f"{m}_std"]))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_summaries(root: str, cities: Sequence[str]) -> str:
    rows = load_summary_rows(summary_csv_path(root))
    agg = aggregate_rows(rows, cities)
    csv_path = osp.join(root, "results", "ablation_summary.csv")
    md_path = osp.join(root, "results", "ablation_summary.md")
    write_ablation_summary_csv(csv_path, agg)
    md = render_ablation_summary_md(agg, cities)
    os.makedirs(osp.dirname(md_path), exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[ablation-runner] wrote {csv_path}")
    print(f"[ablation-runner] wrote {md_path}")
    return md


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cities = args.city or ["beijing"]
    seeds = resolve_seeds(args.runs, args.seeds)
    root = _ROOT

    run_suite(
        cities=cities,
        seeds=seeds,
        gpu=args.gpu,
        force=args.force,
        dry_run=args.dry_run,
        root=root,
    )
    md = write_summaries(root, cities)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
