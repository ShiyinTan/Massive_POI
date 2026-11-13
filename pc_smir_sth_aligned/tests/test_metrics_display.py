from __future__ import annotations

from pc_smir_sth_aligned.evaluation.metrics import format_metrics_table


def test_format_metrics_table_aligned_and_readable():
    val = {
        "n": 58,
        "loss": 2.34567,
        "acc@1": 0.12,
        "recall@1": 0.12,
        "acc@5": 0.34,
        "recall@5": 0.34,
        "ndcg@1": 0.12,
        "ndcg@5": 0.25,
        "mrr": 0.2,
        "route_entropy": 1.2345,
    }
    test = {
        "n": 115,
        "loss": 2.5,
        "acc@1": 0.1,
        "recall@1": 0.1,
        "acc@5": 0.3,
        "recall@5": 0.3,
        "ndcg@1": 0.1,
        "ndcg@5": 0.22,
        "mrr": 0.18,
        "route_entropy": 1.3,
    }
    text = format_metrics_table({"val": val, "test": test}, epoch=3, train_loss=1.2345)
    assert "Epoch 3" in text
    assert "Acc@1" in text
    assert "Acc@5" in text
    assert "NDCG@1" in text
    assert "val" in text and "test" in text
    assert "0.1200" in text
    lines = [ln for ln in text.splitlines() if ln.strip()]
    header = next(ln for ln in lines if ln.startswith("split"))
    val_line = next(ln for ln in lines if ln.startswith("val"))
    assert len(header.split()) == len(val_line.split())
