#!/usr/bin/env python3
"""Bias analysis: STHGCN top-1 vs MostPop / UserMostPop / Markov1 on official HF split."""
from __future__ import annotations

import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

STHGCN_ROOT = Path("/tmp/work/Massive-STEPS/Spatio-Temporal-Hypergraph-Model")
HF_ROOT = Path("/tmp/massive-steps-data")
OUT_DIR = Path("/cursor/stores/bc-db8784de-62ed-49a8-a30b-dc3d37dc0422/docs/sthgcn-bias")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CITIES = {
    "beijing": ("Beijing", 0.453),
    "tokyo": ("Tokyo", 0.250),
    "shanghai": ("Shanghai", 0.193),
    "melbourne": ("Melbourne", 0.168),
}

PAPER_ACC = {k: v[1] for k, v in CITIES.items()}


def find_latest_log(city: str) -> Path | None:
    log_root = STHGCN_ROOT / "log"
    if not log_root.exists():
        return None
    cands = sorted(
        [p for p in log_root.glob(f"*/{city}") if (p / "test_preds.pt").exists()],
        key=lambda p: p.parent.name,
    )
    return cands[-1] if cands else None


def load_splits(city_key: str, city_hf: str):
    base = HF_ROOT / city_hf
    train = pd.read_csv(base / f"{city_key}_checkins_train.csv")
    test = pd.read_csv(base / f"{city_key}_checkins_test.csv")
    return train, test


def fit_baselines(train: pd.DataFrame):
    """Fit MostPop, UserMostPop, Markov1 on train check-ins (venue_id space)."""
    mostpop = [vid for vid, _ in Counter(train["venue_id"]).most_common()]

    user_counts: dict[int, Counter] = defaultdict(Counter)
    for u, v in zip(train["user_id"], train["venue_id"]):
        user_counts[int(u)][int(v)] += 1
    usermost: dict[int, list[int]] = {}
    for u, c in user_counts.items():
        ranked = [vid for vid, _ in c.most_common()]
        # pad with global MostPop
        seen = set(ranked)
        usermost[u] = ranked + [v for v in mostpop if v not in seen]

    # Markov1: consecutive check-ins within trail_id
    markov: dict[int, Counter] = defaultdict(Counter)
    for _, g in train.groupby("trail_id", sort=False):
        g = g.sort_values("timestamp")
        venues = g["venue_id"].tolist()
        for a, b in zip(venues[:-1], venues[1:]):
            markov[int(a)][int(b)] += 1
    markov_top: dict[int, list[int]] = {}
    for prev, c in markov.items():
        ranked = [vid for vid, _ in c.most_common()]
        seen = set(ranked)
        markov_top[prev] = ranked + [v for v in mostpop if v not in seen]

    return {
        "mostpop": mostpop,
        "usermost": usermost,
        "markov": markov_top,
        "fallback": mostpop,
    }


def baseline_top1(baselines, user_id: int, prev_venue: int | None):
    mp = baselines["mostpop"][0]
    ump_list = baselines["usermost"].get(int(user_id), baselines["fallback"])
    ump = ump_list[0]
    if prev_venue is not None and int(prev_venue) in baselines["markov"]:
        mk = baselines["markov"][int(prev_venue)][0]
    else:
        mk = mp
    return mp, ump, mk


def prev_venue_for_test_rows(test_sample: pd.DataFrame, sample_all: pd.DataFrame) -> list[int | None]:
    """Previous venue (encoded PoiId) = second-to-last check-in in the same trail.

    STHGCN test keeps only the last check-in per trail as the target, so Markov1
    conditions on the immediate prior venue in that trail (not the query traj id,
    which can point at the full same trail and would incorrectly yield the target).
    """
    traj_groups = {}
    for traj_id, g in sample_all.groupby("pseudo_session_trajectory_id", sort=False):
        g = g.sort_values("UTCTimeOffset")
        traj_groups[traj_id] = g["PoiId"].tolist()

    prevs = []
    for _, row in test_sample.iterrows():
        tid = row["pseudo_session_trajectory_id"]
        seq = traj_groups.get(tid, [])
        if len(seq) >= 2:
            prevs.append(int(seq[-2]))
        else:
            prevs.append(None)
    return prevs


def decode_poi(le, encoded: int) -> int | None:
    classes = list(le.classes_)
    if encoded < 0 or encoded >= len(classes):
        return None
    return int(classes[encoded])


def decode_user(le, encoded: int) -> int | None:
    classes = list(le.classes_)
    if encoded < 0 or encoded >= len(classes):
        return None
    return int(classes[encoded])


def analyze_city(city_key: str, city_hf: str, paper_acc: float) -> dict:
    log_dir = find_latest_log(city_key)
    if log_dir is None:
        return {"city": city_hf, "status": "missing"}

    pre = STHGCN_ROOT / "data" / city_key / "preprocessed"
    test_sample = pd.read_csv(pre / "test_sample.csv")
    sample_all = pd.read_csv(pre / "sample.csv")
    with open(pre / "label_encoding.pkl", "rb") as f:
        le_pack = pickle.load(f)
    poi_le, _, user_le = le_pack[0], le_pack[1], le_pack[2]

    preds_payload = torch.load(log_dir / "test_preds.pt", map_location="cpu", weights_only=False)
    pred = np.asarray(preds_payload["pred"])  # [N, K] ranked encoded poi ids
    label = np.asarray(preds_payload["label"]).reshape(-1).astype(int)
    top1 = pred[:, 0].astype(int)
    assert len(top1) == len(test_sample), (len(top1), len(test_sample))

    metrics = json.loads((log_dir / "test_metrics.json").read_text())
    acc1 = float(metrics["hparam/Recall@1"])
    acc5 = float(metrics["hparam/Recall@5"])
    acc10 = float(metrics["hparam/Recall@10"])

    train_raw, _ = load_splits(city_key, city_hf)
    baselines = fit_baselines(train_raw)

    # Map encoded -> raw for targets and top1
    target_raw = [decode_poi(poi_le, int(x)) for x in label]
    top1_raw = [decode_poi(poi_le, int(x)) for x in top1]
    user_raw = [decode_user(user_le, int(u)) for u in test_sample["UserId"].tolist()]

    # Previous venue: encoded then decode
    prev_enc = prev_venue_for_test_rows(test_sample, sample_all)
    prev_raw = [decode_poi(poi_le, p) if p is not None else None for p in prev_enc]

    rows = []
    for i in range(len(test_sample)):
        u = user_raw[i]
        prev = prev_raw[i]
        mp, ump, mk = baseline_top1(baselines, u, prev)
        t = target_raw[i]
        p = top1_raw[i]
        rows.append(
            {
                "idx": i,
                "user_id": u,
                "target": t,
                "sthgcn_top1": p,
                "mostpop_top1": mp,
                "usermostpop_top1": ump,
                "markov1_top1": mk,
                "hit@1": int(p == t),
                "eq_mostpop": int(p == mp),
                "eq_usermostpop": int(p == ump),
                "eq_markov1": int(p == mk),
                "target_is_mostpop": int(t == mp),
                "target_is_ump": int(t == ump),
                "target_is_markov1": int(t == mk),
                "mp_hit": int(mp == t),
                "ump_hit": int(ump == t),
                "mk_hit": int(mk == t),
            }
        )
    df = pd.DataFrame(rows)
    n = len(df)

    def cond_acc(mask):
        m = df.loc[mask]
        return float(m["hit@1"].mean()) if len(m) else float("nan")

    summary = {
        "city": city_hf,
        "city_key": city_key,
        "n_test_sthgcn": n,
        "sthgcn_acc1": acc1,
        "sthgcn_acc5": acc5,
        "sthgcn_acc10": acc10,
        "paper_acc1": paper_acc,
        "delta_vs_paper": acc1 - paper_acc,
        "mostpop_acc1_on_sthgcn_test": float(df["mp_hit"].mean()),
        "usermostpop_acc1_on_sthgcn_test": float(df["ump_hit"].mean()),
        "markov1_acc1_on_sthgcn_test": float(df["mk_hit"].mean()),
        "frac_top1_eq_mostpop": float(df["eq_mostpop"].mean()),
        "frac_top1_eq_usermostpop": float(df["eq_usermostpop"].mean()),
        "frac_top1_eq_markov1": float(df["eq_markov1"].mean()),
        "frac_top1_eq_any_of_three": float(
            ((df["eq_mostpop"] | df["eq_usermostpop"] | df["eq_markov1"]) > 0).mean()
        ),
        "frac_target_is_mostpop": float(df["target_is_mostpop"].mean()),
        "frac_target_is_ump": float(df["target_is_ump"].mean()),
        "frac_target_is_markov1": float(df["target_is_markov1"].mean()),
        "acc_when_target_is_mostpop": cond_acc(df["target_is_mostpop"] == 1),
        "acc_when_target_not_mostpop": cond_acc(df["target_is_mostpop"] == 0),
        "acc_when_target_is_ump": cond_acc(df["target_is_ump"] == 1),
        "acc_when_target_not_ump": cond_acc(df["target_is_ump"] == 0),
        "acc_when_target_is_markov1": cond_acc(df["target_is_markov1"] == 1),
        "acc_when_target_not_markov1": cond_acc(df["target_is_markov1"] == 0),
        "acc_when_pred_eq_ump": cond_acc(df["eq_usermostpop"] == 1),
        "acc_when_pred_ne_ump": cond_acc(df["eq_usermostpop"] == 0),
        "acc_when_pred_eq_markov1": cond_acc(df["eq_markov1"] == 1),
        "acc_when_pred_ne_markov1": cond_acc(df["eq_markov1"] == 0),
        "jaccard_top1_ump_vs_markov": float(
            ((df["eq_usermostpop"] == 1) & (df["eq_markov1"] == 1)).sum()
            / max(1, ((df["eq_usermostpop"] == 1) | (df["eq_markov1"] == 1)).sum())
        ),
        "log_dir": str(log_dir),
    }

    # pairwise overlap among baseline top1s themselves
    summary["baseline_ump_eq_mp"] = float((df["usermostpop_top1"] == df["mostpop_top1"]).mean())
    summary["baseline_mk_eq_mp"] = float((df["markov1_top1"] == df["mostpop_top1"]).mean())
    summary["baseline_mk_eq_ump"] = float((df["markov1_top1"] == df["usermostpop_top1"]).mean())

    df.to_csv(OUT_DIR / f"{city_key}_per_sample_bias.csv", index=False)
    return summary


def make_figures(summaries: list[dict]):
    cities = [s["city"] for s in summaries]
    x = np.arange(len(cities))
    w = 0.2

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 1.5 * w, [s["paper_acc1"] for s in summaries], w, label="STHGCN paper Acc@1", color="#4C78A8")
    ax.bar(x - 0.5 * w, [s["sthgcn_acc1"] for s in summaries], w, label="STHGCN measured Acc@1", color="#F58518")
    ax.bar(x + 0.5 * w, [s["usermostpop_acc1_on_sthgcn_test"] for s in summaries], w, label="UserMostPop Acc@1*", color="#54A24B")
    ax.bar(x + 1.5 * w, [s["markov1_acc1_on_sthgcn_test"] for s in summaries], w, label="Markov1 Acc@1*", color="#E45756")
    ax.set_xticks(x)
    ax.set_xticklabels(cities)
    ax.set_ylabel("Acc@1")
    ax.set_title("STHGCN vs paper and popularity baselines (* on STHGCN filtered test)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, max(0.55, max(s["sthgcn_acc1"] for s in summaries) + 0.08))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "acc_vs_paper_baselines.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w, [s["frac_top1_eq_mostpop"] for s in summaries], w, label="= MostPop", color="#4C78A8")
    ax.bar(x, [s["frac_top1_eq_usermostpop"] for s in summaries], w, label="= UserMostPop", color="#54A24B")
    ax.bar(x + w, [s["frac_top1_eq_markov1"] for s in summaries], w, label="= Markov1", color="#E45756")
    ax.set_xticks(x)
    ax.set_xticklabels(cities)
    ax.set_ylabel("Fraction of test samples")
    ax.set_title("How often STHGCN top-1 equals a popularity baseline top-1")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "top1_overlap_baselines.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w, [s["acc_when_target_is_ump"] for s in summaries], w, label="Acc | target=UMP", color="#54A24B")
    ax.bar(x, [s["acc_when_target_not_ump"] for s in summaries], w, label="Acc | target≠UMP", color="#B2DF8A")
    ax.bar(x + w, [s["acc_when_target_is_markov1"] for s in summaries], w, label="Acc | target=Markov1", color="#E45756")
    # also not markov as hatch? keep 3 bars + add 4th
    ax.bar(x + 2 * w, [s["acc_when_target_not_markov1"] for s in summaries], w, label="Acc | target≠Markov1", color="#FBB4AE")
    ax.set_xticks(x)
    ax.set_xticklabels(cities)
    ax.set_ylabel("STHGCN Acc@1")
    ax.set_title("Conditional Acc@1 when target is / isn't baseline top-1")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "conditional_acc.png", dpi=150)
    plt.close(fig)


def main():
    summaries = []
    for city_key, (city_hf, paper) in CITIES.items():
        print("analyzing", city_hf)
        s = analyze_city(city_key, city_hf, paper)
        print(json.dumps({k: s[k] for k in s if k != "log_dir"}, indent=2))
        if s.get("status") != "missing":
            summaries.append(s)

    if summaries:
        pd.DataFrame(summaries).to_csv(OUT_DIR / "bias_summary.csv", index=False)
        make_figures(summaries)
        print("wrote", OUT_DIR)


if __name__ == "__main__":
    main()
