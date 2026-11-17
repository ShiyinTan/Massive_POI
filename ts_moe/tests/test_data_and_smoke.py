"""Data symlink, history windows, and a one-epoch smoke train."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from ts_moe.dataset.build import frames_to_examples
from ts_moe.dataset.load import DEFAULT_DATA_DIR
from ts_moe.dataset.synthetic import make_synthetic_frames
from ts_moe.inference.predict import predict_history_frame
from ts_moe.model.ts_moe import TSMoE
from ts_moe.run import estimate_active_prior, resolve_device
from ts_moe.training.config import load_config
from ts_moe.training.loop import Trainer


def test_data_symlink_points_at_massive_steps():
    link = DEFAULT_DATA_DIR
    assert link.is_symlink()
    assert os.readlink(link) == "../Massive-STEPS/data"
    assert link.resolve() == Path(__file__).resolve().parents[2] / "Massive-STEPS" / "data"
    assert (link / "beijing" / "beijing_checkins.json").is_file()


def test_history_does_not_include_the_target_checkin():
    frame = pd.DataFrame(
        {
            "trail_id": ["t"] * 3,
            "user_id": ["U0"] * 3,
            "venue_id": [10, 20, 30],
            "latitude": [1.0, 2.0, 3.0],
            "longitude": [1.0, 2.0, 3.0],
            "venue_category": ["a", "b", "c"],
            "timestamp": ["2020-01-01 08:00:00", "2020-01-01 09:00:00", "2020-01-01 10:00:00"],
        }
    )
    vocab, splits = frames_to_examples({"train": frame, "val": frame, "test": frame}, max_history=10, min_history=1, n_folds=2, seed=0)
    examples = splits["train"]
    assert [ex.target for ex in examples] == [vocab.poi_to_idx["20"], vocab.poi_to_idx["30"]]
    assert examples[0].poi == [vocab.poi_to_idx["10"]]
    assert examples[1].poi == [vocab.poi_to_idx["10"], vocab.poi_to_idx["20"]]
    assert vocab.poi_to_idx["30"] not in examples[0].poi


def test_smoke_train_one_epoch(tmp_path):
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml")
    config["output_dir"] = str(tmp_path)
    config["train"]["epochs"] = 1
    frames = make_synthetic_frames(
        n_users=config["data"]["n_users"],
        n_venues=config["data"]["n_venues"],
        length=config["data"]["length"],
        seed=config["seed"],
    )
    vocab, splits = frames_to_examples(
        frames,
        max_history=config["data"]["max_history"],
        min_history=config["data"]["min_history"],
        n_folds=config["surd"]["n_folds"],
        seed=config["seed"],
    )
    device = resolve_device("cpu")
    fold_priors, full_prior, names = estimate_active_prior(
        config, vocab, splits["train"], "synthetic", device, reestimate=True
    )
    assert fold_priors.shape[-1] == 21
    assert abs(full_prior.sum(-1) - 1).max() < 1e-5
    model = TSMoE(vocab, config)
    trainer = Trainer(model, vocab, config, fold_priors, full_prior, splits["train"], splits["val"], device)
    result = trainer.fit(tmp_path / "model.pt", expert_names=names)
    assert result["history"][0]["loss"] > 0
    assert (tmp_path / "model.pt").is_file()
    assert "val_recall@10" in result["history"][0]
    trail_id = frames["train"]["trail_id"].iloc[0]
    history = frames["train"][frames["train"]["trail_id"] == trail_id]
    ranked = predict_history_frame(tmp_path / "model.pt", history, top_k=3, device=device)
    assert len(ranked["recommendations"]) == 3
    assert set(ranked["recommendations"][0]["routing"]) == set(names)
