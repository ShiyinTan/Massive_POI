#!/usr/bin/env python3
"""Unit-style checks for run_ablations city choices + summary merge (no training)."""
from __future__ import annotations

import csv
import importlib.util as ilu
import os.path as osp
import tempfile


_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
_SCRIPT = osp.join(_ROOT, "scripts", "run_ablations.py")


def _load():
    spec = ilu.spec_from_file_location("run_ablations_check", _SCRIPT)
    mod = ilu.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _write_summary(path: str, rows: list) -> None:
    header = [
        "time",
        "city",
        "run_name",
        "seed",
        "use_user_embed",
        "hypergraph_edges",
        "graph_scope",
        "unseen_prefix_poi",
        "late_sequence",
        "gpu",
        "Recall@1",
        "Recall@5",
        "Recall@10",
        "Recall@20",
        "MRR",
        "log_dir",
    ]
    os_makedirs = __import__("os").makedirs
    os_makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def _write_ablation_csv(path: str, rows: list, header: list) -> None:
    __import__("os").makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def main() -> int:
    m = _load()

    # 1) argparse accepts bandung (and all 15)
    p = m.build_parser()
    args = p.parse_args(["--city", "bandung", "--runs", "1", "--dry-run"])
    assert args.city == ["bandung"], args.city
    for city in m.CITIES:
        a = p.parse_args(["--city", city, "--runs", "1"])
        assert city in a.city
    assert len(m.CITIES) == 15

    # late_sequence suite tokens present
    names = [n for n, _ in m.ABLATION_SUITE]
    for tok in ("late_poi", "late_category", "late_time", "late_location"):
        assert tok in names, names

    # yaml hard-fail names the missing file
    try:
        m.require_city_yamls(_ROOT, ["not_a_real_city"])
        raise AssertionError("expected SystemExit for missing yaml")
    except SystemExit as exc:
        msg = str(exc)
        assert "conf/std_conf/not_a_real_city.yml" in msg, msg

    # existing cities resolve
    m.require_city_yamls(_ROOT, ["bandung", "beijing"])

    # 2) merge: old tokyo row survives when rebuilding for beijing only
    with tempfile.TemporaryDirectory() as tmp:
        results = osp.join(tmp, "results")
        __import__("os").makedirs(results)

        # Pretend prior ablation_summary had tokyo/paper
        old_tokyo = {
            "city": "tokyo",
            "run_name": "paper",
            "n": "2",
            "Recall@1_mean": "0.4",
            "Recall@5_mean": "0.5",
            "Recall@10_mean": "0.6",
            "Recall@20_mean": "0.7",
            "MRR_mean": "0.45",
            "Recall@1_std": "0.01",
            "Recall@5_std": "0.01",
            "Recall@10_std": "0.01",
            "Recall@20_std": "0.01",
            "MRR_std": "0.01",
        }
        abl_csv = osp.join(results, "ablation_summary.csv")
        _write_ablation_csv(abl_csv, [old_tokyo], m.ABLATION_SUMMARY_HEADER)

        # summary.csv now has a new beijing row (and no tokyo — tokyo only in ablation_summary)
        _write_summary(
            osp.join(results, "summary.csv"),
            [
                {
                    "time": "t1",
                    "city": "beijing",
                    "run_name": "paper",
                    "seed": "1",
                    "Recall@1": "0.5",
                    "Recall@5": "0.6",
                    "Recall@10": "0.7",
                    "Recall@20": "0.8",
                    "MRR": "0.55",
                }
            ],
        )

        m.write_summaries(tmp, cities=["beijing"], quiet=True)

        with open(abl_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        cities_present = {r["city"] for r in rows}
        assert "tokyo" in cities_present, rows
        assert "beijing" in cities_present, rows
        tokyo_row = next(r for r in rows if r["city"] == "tokyo")
        assert tokyo_row["run_name"] == "paper"
        assert tokyo_row["Recall@1_mean"] == "0.4"
        beijing_row = next(r for r in rows if r["city"] == "beijing")
        assert beijing_row["run_name"] == "paper"
        assert float(beijing_row["Recall@1_mean"]) == 0.5

        md = open(osp.join(results, "ablation_summary.md"), encoding="utf-8").read()
        assert "## tokyo" in md and "## beijing" in md

    print(
        "ABLATION_RUNNER_CITIES_MERGE_OK: bandung accepted; "
        "15 cities; late_* in suite; tokyo row kept after beijing rebuild."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
