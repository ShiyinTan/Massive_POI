from __future__ import annotations

import torch

from pc_smir_sth_aligned.evaluation.metrics import rank_metrics_from_scores


def test_acc_equals_recall():
    scores = torch.tensor([[3.0, 1.0, 0.0], [0.1, 2.0, 0.0]], dtype=torch.float32)
    target = torch.tensor([0, 1], dtype=torch.long)
    m = rank_metrics_from_scores(scores, target, ks=(1, 5))
    assert m["acc@1"] == m["recall@1"] == 1.0
    assert m["acc@5"] == m["recall@5"] == 1.0
