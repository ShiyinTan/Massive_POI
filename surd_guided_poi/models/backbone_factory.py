"""Factory for configurable next-POI backbones.

``model.backbone`` ∈ ``flashback | graph_flashback | sthgcn``.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from scipy.sparse import coo_matrix, csr_matrix

from ..data.preprocessing import NextPOISample
from .backbone_flashback import FlashbackBackbone
from .backbone_graph_flashback import GraphFlashbackBackbone
from .backbone_sthgcn import STHGCNBackbone
from .graph_builders import resolve_graph_flashback_graphs

SUPPORTED_BACKBONES = ("flashback", "graph_flashback", "sthgcn")


def build_backbone(
    name: str,
    *,
    num_pois: int,
    num_users: int,
    hidden_dim: int = 64,
    rnn_type: str = "gru",
    lambda_t: float = 0.1,
    lambda_s: float = 100.0,
    dropout: float = 0.1,
    train_samples: Optional[Iterable[NextPOISample]] = None,
    transition_graph: Optional[coo_matrix] = None,
    interact_graph: Optional[csr_matrix] = None,
    transition_pkl: Optional[str] = None,
    interact_pkl: Optional[str] = None,
    lambda_loc: float = 1.0,
    lambda_user: float = 1.0,
    use_weight: bool = False,
    use_graph_user: bool = False,
    use_spatial_graph: bool = False,
    sthgcn_num_layers: int = 2,
    sthgcn_num_heads: int = 4,
    sthgcn_ff_dim: Optional[int] = None,
    **_: Any,
):
    key = str(name).strip().lower()
    if key not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unknown model.backbone={name!r}; expected one of {SUPPORTED_BACKBONES}"
        )

    if key == "flashback":
        return FlashbackBackbone(
            num_pois=num_pois,
            num_users=num_users,
            hidden_dim=hidden_dim,
            rnn_type=rnn_type,
            lambda_t=lambda_t,
            lambda_s=lambda_s,
        )

    if key == "graph_flashback":
        if transition_graph is None or interact_graph is None:
            transition_graph, interact_graph = resolve_graph_flashback_graphs(
                num_pois=num_pois,
                num_users=num_users,
                train_samples=train_samples,
                transition_pkl=transition_pkl,
                interact_pkl=interact_pkl,
            )
        return GraphFlashbackBackbone(
            num_pois=num_pois,
            num_users=num_users,
            transition_graph=transition_graph,
            interact_graph=interact_graph,
            hidden_dim=hidden_dim,
            rnn_type=rnn_type,
            lambda_t=lambda_t,
            lambda_s=lambda_s,
            lambda_loc=lambda_loc,
            lambda_user=lambda_user,
            use_weight=use_weight,
            use_graph_user=use_graph_user,
            use_spatial_graph=use_spatial_graph,
        )

    # sthgcn
    return STHGCNBackbone(
        num_pois=num_pois,
        num_users=num_users,
        hidden_dim=hidden_dim,
        num_layers=int(sthgcn_num_layers),
        num_heads=int(sthgcn_num_heads),
        ff_dim=int(sthgcn_ff_dim if sthgcn_ff_dim is not None else max(128, 2 * hidden_dim)),
        dropout=dropout,
    )
