"""Massive-STEPS official split path helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from surd_guided_poi.data.massive_steps import (
    load_official_split_frames,
    resolve_official_splits,
    split_filename,
)
from surd_guided_poi.data.preprocessing import build_samples_from_split, build_vocab_from_frames


BEIJING_DIR = Path("/tmp/massive-steps-data/Beijing")


def test_split_filename_convention():
    assert split_filename("beijing", "train") == "beijing_checkins_train.csv"
    assert split_filename("beijing", "val") == "beijing_checkins_validation.csv"
    assert split_filename("new_york", "test") == "new_york_checkins_test.csv"


def test_resolve_and_build_from_local_beijing():
    if not (BEIJING_DIR / "beijing_checkins_train.csv").is_file():
        return  # skip if HF cache not present in this environment
    paths = resolve_official_splits("beijing", data_dir=BEIJING_DIR)
    assert paths["train"].is_file()
    frames = load_official_split_frames("beijing", data_dir=BEIJING_DIR)
    vocab = build_vocab_from_frames(frames)
    train = build_samples_from_split(frames["train"], vocab, "train", use_all_prefixes=True)
    assert vocab.num_pois > 0
    assert len(train) > 0
    assert all(s.split == "train" for s in train)
    # Flashback-style unix seconds present
    assert train[0].history_unix
    assert isinstance(frames["train"], pd.DataFrame)