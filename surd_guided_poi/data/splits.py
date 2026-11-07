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
    """从 official-train 再切两份（不碰 val/test）。

    返回 ``(model_train, estimation)``：
      - model_train：训推荐 / 拟合子集头（Phase B + Phase A 的 fit）
      - estimation：只在 Phase A 算 Specific MI / SURD 权重（holdout）

    目的：避免「同一批样本既训概率头又拿来估 MI」。调用处见
    ``scripts/train.py``、``scripts/estimate_surd.py``。
    """
    if not train_samples:
        return [], []
    # 按目标时间排序，取较晚 holdout_ratio 一段当 estimation
    ordered = sorted(train_samples, key=lambda s: (s.target_timestamp, s.sample_id))
    n = len(ordered)
    n_est = max(1, int(round(n * holdout_ratio))) if n >= 5 else max(1, n // 5)
    n_est = min(n_est, n - 1) if n > 1 else 0
    model_train = list(ordered[: n - n_est])  # → train_loader / estimator_fit_loader
    estimation = list(ordered[n - n_est :])  # → estimation_loader（只估 SURD）
    return model_train, estimation


def assert_official_split_disjoint_trails(
    frames: Dict[str, "pd.DataFrame"],  # noqa: F821
) -> None:
    """Sanity-check Massive-STEPS trail disjointness across official splits."""
    sets = {k: set(df["trail_id"].unique()) for k, df in frames.items()}
    assert sets["train"].isdisjoint(sets["val"]), "train/val trail overlap"
    assert sets["train"].isdisjoint(sets["test"]), "train/test trail overlap"
    assert sets["val"].isdisjoint(sets["test"]), "val/test trail overlap"