"""Split helpers on top of Massive-STEPS official train/val/test CSVs.

Primary protocol: use the published ``*_checkins_{train,validation,test}.csv``
files (trail-level, user-stratified) — same as Flashback/GETNext.

Within the official **train** period only, we optionally carve an estimator
holdout for Specific MI / SURD so probability heads are not evaluated on the
same samples they were just fitted on.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .preprocessing import NextPOISample


def carve_estimator_holdout(
    train_samples: Sequence[NextPOISample],
    holdout_ratio: float = 0.2,
) -> Tuple[List[NextPOISample], List[NextPOISample]]:
    """只在 official-train 里再切一块：前面训模型，后面估 SURD（避免同一批又训又估）。"""
    if not train_samples:
        return [], []
    # 按目标时间排序，取较晚的一段当 estimation holdout
    ordered = sorted(train_samples, key=lambda s: (s.target_timestamp, s.sample_id))
    n = len(ordered)
    n_est = max(1, int(round(n * holdout_ratio))) if n >= 5 else max(1, n // 5)
    n_est = min(n_est, n - 1) if n > 1 else 0
    model_train = list(ordered[: n - n_est])  # 真正更新推荐模型的样本
    estimation = list(ordered[n - n_est :])  # 只用来算 Specific MI / SURD
    return model_train, estimation


def assert_official_split_disjoint_trails(
    frames: Dict[str, "pd.DataFrame"],  # noqa: F821
) -> None:
    """Sanity-check Massive-STEPS trail disjointness across official splits."""
    sets = {k: set(df["trail_id"].unique()) for k, df in frames.items()}
    assert sets["train"].isdisjoint(sets["val"]), "train/val trail overlap"
    assert sets["train"].isdisjoint(sets["test"]), "train/test trail overlap"
    assert sets["val"].isdisjoint(sets["test"]), "val/test trail overlap"