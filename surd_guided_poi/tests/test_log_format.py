"""Training log formatter helpers."""

from __future__ import annotations

from surd_guided_poi.training.log_format import (
    epoch_title,
    format_metrics,
    format_train_losses,
    log_block,
)


def test_format_metrics_orders_recall_then_ndcg():
    text = format_metrics(
        {"ndcg@10": 0.2, "recall@5": 0.1, "recall@1": 0.05, "ndcg@1": 0.04}
    )
    assert text.index("R@1") < text.index("R@5")
    assert text.index("R@5") < text.index("N@1")
    assert "R@1=0.0500" in text


def test_format_train_losses_and_block():
    losses = format_train_losses({"loss": 1.0, "rec": 0.8, "u": 0.1, "s": 0.1})
    assert losses.startswith("loss=1.0000")
    assert "U=0.1000" in losses
    block = log_block(epoch_title("epoch", 3, 40), ["train  " + losses])
    assert "epoch  03/40" in block
    assert "────" in block


def test_evaluate_parser_accepts_device():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
    # Smoke: ensure the script source declares --device (avoids importing full CLI).
    src = path.read_text(encoding="utf-8")
    assert "--device" in src
    assert 'cfg.get("device")' in src
