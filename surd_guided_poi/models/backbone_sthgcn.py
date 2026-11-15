"""Full hypergraph STHGCN backbone (Massive-STEPS Spatio-Temporal-Hypergraph-Model).

``model.backbone: sthgcn`` uses PyG NeighborSampler batches — **not** ``NextPOIBatch``.
See ``data/sthgcn_hypergraph.py`` and ``training/sthgcn_runner.py``.

The former local Transformer lived here; it is deprecated as ``backbone_sthgcn_seq``
(``model.backbone: sthgcn_seq`` only).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from ..data.dataset import NextPOIBatch
from ..data.sthgcn_hypergraph import STHGCN_PYG_IMPORT_ERROR, require_sthgcn_deps


class HypergraphSTHGCNBackbone(nn.Module):
    """Thin wrapper around upstream ``model.sthgcn.STHGCN``.

    Construct after ``prepare_sthgcn_pipeline`` (dataset_args must be filled).
    """

    def __init__(self, upstream_model: nn.Module):
        super().__init__()
        require_sthgcn_deps()
        self.core = upstream_model

    def forward_sampler(
        self,
        input_data: dict,
        label: Optional[torch.Tensor] = None,
        mode: str = "train",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(logits [B,K], loss)`` — same contract as upstream ``STHGCN.forward``."""
        return self.core(input_data, label=label, mode=mode)

    def forward(self, batch: NextPOIBatch, h: Any = None):
        raise NotImplementedError(
            "Full hypergraph STHGCN does not accept NextPOIBatch. "
            "Use training/sthgcn_runner.py (NeighborSampler path). "
            f"Deprecated NextPOIBatch adapter: model.backbone=sthgcn_seq. "
            f"({STHGCN_PYG_IMPORT_ERROR})"
        )

    @property
    def location_embedding(self) -> nn.Embedding:
        return self.core.checkin_embedding_layer.poi_embedding


# Public name for factory / docs
STHGCNBackbone = HypergraphSTHGCNBackbone
