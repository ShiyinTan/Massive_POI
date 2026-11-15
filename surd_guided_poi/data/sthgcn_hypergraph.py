"""Bridge official Massive-STEPS CSVs → full hypergraph STHGCN pipeline.

Wraps ``Massive-STEPS/Spatio-Temporal-Hypergraph-Model`` preprocess +
``LBSNDataset`` + ``NeighborSampler``. Requires PyG stack
(``requirements-sthgcn.txt``). Does **not** convert ``NextPOIBatch``.
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .massive_steps import STHGCN_DIR, resolve_official_splits

STHGCN_PYG_IMPORT_ERROR = (
    "Full hypergraph STHGCN needs PyTorch Geometric + torch_sparse/scatter. "
    "Install deps from surd_guided_poi/requirements-sthgcn.txt "
    "(match your torch/CUDA build; see Massive-STEPS/Spatio-Temporal-Hypergraph-Model/requirements.txt). "
    "Then re-run with model.backbone=sthgcn."
)

_STHGCN_COLLIDING = (
    "utils",
    "layer",
    "dataset",
    "model",
    "preprocess",
    "metric",
)


def require_sthgcn_deps() -> None:
    """Fail fast if the hypergraph stack is missing."""
    missing = []
    for mod in ("torch_geometric", "torch_sparse", "torch_scatter"):
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise ImportError(
            f"Missing {missing}. {STHGCN_PYG_IMPORT_ERROR}"
        )


def fixed_sthgcn_root() -> str:
    """Absolute Massive-STEPS STHGCN package root (cwd-independent)."""
    return str(STHGCN_DIR)


@contextmanager
def _sthgcn_cwd():
    """``chdir`` into the STHGCN package so cwd-based ``get_root_dir`` succeeds."""
    prev = os.getcwd()
    os.chdir(STHGCN_DIR)
    try:
        yield
    finally:
        os.chdir(prev)


@contextmanager
def _sthgcn_import_context():
    """Put STHGCN package first on sys.path; isolate bare module names."""
    saved = {}
    for key in list(sys.modules):
        root = key.split(".", 1)[0]
        if root in _STHGCN_COLLIDING or key in _STHGCN_COLLIDING:
            saved[key] = sys.modules.pop(key)

    path = str(STHGCN_DIR)
    inserted = path not in sys.path
    if inserted:
        sys.path.insert(0, path)
    from .massive_steps import FLASHBACK_DIR, GRAPH_FLASHBACK_DIR

    removed_paths = []
    for p in (str(FLASHBACK_DIR), str(GRAPH_FLASHBACK_DIR)):
        if p in sys.path:
            sys.path.remove(p)
            removed_paths.append(p)
    try:
        yield
    finally:
        if inserted and path in sys.path:
            sys.path.remove(path)
        for p in reversed(removed_paths):
            if p not in sys.path:
                sys.path.insert(0, p)
        for name, mod in saved.items():
            sys.modules[name] = mod


def _install_fixed_get_root_dir() -> None:
    """Rebind ``get_root_dir`` in every loaded upstream module (+ class attrs).

    Upstream ``utils.sys_util.get_root_dir`` parses ``os.getcwd()`` for
    ``Spatio-Temporal-Hypergraph-Model``. ``preprocess.file_reader.FileReaderBase``
    also caches ``root_path = get_root_dir()`` at class body time.
    """
    import utils.sys_util as sys_util_mod  # type: ignore

    sys_util_mod.get_root_dir = fixed_sthgcn_root
    for name, mod in list(sys.modules.items()):
        root = name.split(".", 1)[0]
        if root not in _STHGCN_COLLIDING and name not in _STHGCN_COLLIDING:
            continue
        if hasattr(mod, "get_root_dir"):
            setattr(mod, "get_root_dir", fixed_sthgcn_root)
        # Class-level cache set during import under chdir — refresh to absolute path.
        frb = getattr(mod, "FileReaderBase", None)
        if frb is not None and hasattr(frb, "root_path"):
            frb.root_path = fixed_sthgcn_root()


def import_sthgcn_stack() -> Dict[str, Any]:
    """Import upstream STHGCN modules under isolation; cache on first call."""
    cache_key = "surd_guided_poi._vendor.sthgcn_stack"
    if cache_key in sys.modules:
        return sys.modules[cache_key]  # type: ignore[return-value]

    require_sthgcn_deps()
    with _sthgcn_import_context():
        # Patch BEFORE importing preprocess: FileReaderBase.root_path = get_root_dir()
        # at class body. Also chdir so any unpatched cwd lookup still finds the package.
        with _sthgcn_cwd():
            import utils.sys_util as sys_util_mod  # type: ignore

            sys_util_mod.get_root_dir = fixed_sthgcn_root

            import utils as sthgcn_utils  # type: ignore

            sthgcn_utils.get_root_dir = fixed_sthgcn_root  # type: ignore[attr-defined]

            import preprocess as sthgcn_preprocess  # type: ignore
            import dataset as sthgcn_dataset  # type: ignore
            import layer as sthgcn_layer  # type: ignore
            import model as sthgcn_model  # type: ignore
            import utils.conf_util as conf_util_mod  # type: ignore

            _install_fixed_get_root_dir()

        stack = {
            "utils": sthgcn_utils,
            "preprocess": sthgcn_preprocess,
            "dataset": sthgcn_dataset,
            "layer": sthgcn_layer,
            "model": sthgcn_model,
            "Cfg": conf_util_mod.Cfg,
            "get_root_dir": fixed_sthgcn_root,
        }
    # Keep stack modules alive under vendor names; leave bare names cleared.
    for bare in _STHGCN_COLLIDING:
        sys.modules.pop(bare, None)
    sys.modules[cache_key] = stack  # type: ignore[assignment]
    return stack


def materialize_official_csvs(
    city: str,
    *,
    data_dir: Optional[str | Path] = None,
    cache_dir: Optional[str | Path] = None,
    hf_dataset_id: Optional[str] = None,
) -> Path:
    """Copy/symlink official splits into ``STHGCN_DIR/data/{city}/``."""
    city_key = city.lower().replace(" ", "_").replace("-", "_")
    dest = STHGCN_DIR / "data" / city_key
    dest.mkdir(parents=True, exist_ok=True)
    paths = resolve_official_splits(
        city_key,
        data_dir=data_dir,
        cache_dir=cache_dir,
        hf_dataset_id=hf_dataset_id,
    )
    # Upstream expects {city}_checkins_{train,validation,test}.csv
    mapping = {
        "train": f"{city_key}_checkins_train.csv",
        "val": f"{city_key}_checkins_validation.csv",
        "test": f"{city_key}_checkins_test.csv",
    }
    for split, fname in mapping.items():
        src = Path(paths[split])
        dst = dest / fname
        if dst.exists() or dst.is_symlink():
            continue
        try:
            os.symlink(src.resolve(), dst)
        except OSError:
            shutil.copy2(src, dst)
    return dest


def load_sthgcn_cfg(
    conf_relpath: str = "std_conf/beijing.yml",
    *,
    city: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
):
    """Load upstream ``Cfg`` from ``conf/{conf_relpath}`` with optional overrides."""
    stack = import_sthgcn_stack()
    Cfg = stack["Cfg"]
    # Cfg joins get_root_dir()/conf/file_name
    cfg = Cfg(conf_relpath)
    if city:
        cfg.dataset_args.dataset_name = city.lower().replace(" ", "_").replace("-", "_")
    sizes = cfg.model_args.sizes
    if isinstance(sizes, str):
        cfg.model_args.sizes = [int(i) for i in sizes.split("-")]
    cfg.model_args.model_name = "sthgcn"
    if overrides:
        for group_name, group_vals in overrides.items():
            group = getattr(cfg, group_name, None)
            if group is None or not isinstance(group_vals, dict):
                continue
            for k, v in group_vals.items():
                setattr(group, k, v)
    return cfg


def prepare_sthgcn_pipeline(
    *,
    city: str,
    data_dir: Optional[str | Path] = None,
    cache_dir: Optional[str | Path] = None,
    hf_dataset_id: Optional[str] = None,
    conf_relpath: str = "std_conf/beijing.yml",
    device: str = "cpu",
    run_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Any, Any, Any, Any]:
    """Materialize CSVs → preprocess → LBSNDataset → NeighborSamplers → STHGCN.

    Returns ``(cfg, model, sampler_train, sampler_validate, sampler_test)``.
    """
    stack = import_sthgcn_stack()
    materialize_official_csvs(
        city, data_dir=data_dir, cache_dir=cache_dir, hf_dataset_id=hf_dataset_id
    )
    overrides: Dict[str, Any] = {"run_args": {"gpu": -1 if device == "cpu" else 0}}
    if run_overrides:
        overrides.setdefault("run_args", {}).update(run_overrides)
    cfg = load_sthgcn_cfg(conf_relpath, city=city, overrides=overrides)
    cfg.run_args.device = device

    # Preprocess / LBSNDataset call get_root_dir(); stay under package cwd and
    # keep the absolute-path patch installed (import may have cleared bare names).
    with _sthgcn_cwd():
        _install_fixed_get_root_dir()
        stack["preprocess"].preprocess(cfg)
        lbsn_dataset = stack["dataset"].LBSNDataset(cfg)

    cfg.dataset_args.spatial_slots = lbsn_dataset.spatial_slots
    cfg.dataset_args.num_user = lbsn_dataset.num_user
    cfg.dataset_args.num_poi = lbsn_dataset.num_poi
    cfg.dataset_args.num_category = lbsn_dataset.num_category
    cfg.dataset_args.padding_poi_id = lbsn_dataset.padding_poi_id
    cfg.dataset_args.padding_user_id = lbsn_dataset.padding_user_id
    cfg.dataset_args.padding_poi_category = lbsn_dataset.padding_poi_category
    cfg.dataset_args.padding_hour_id = lbsn_dataset.padding_hour_id
    cfg.dataset_args.padding_weekday_id = lbsn_dataset.padding_weekday_id

    sizes = cfg.model_args.sizes
    NeighborSampler = stack["layer"].NeighborSampler
    common = dict(
        x=lbsn_dataset.x,
        edge_index=lbsn_dataset.edge_index,
        edge_attr=lbsn_dataset.edge_attr,
        intra_jaccard_threshold=cfg.model_args.intra_jaccard_threshold,
        inter_jaccard_threshold=cfg.model_args.inter_jaccard_threshold,
        edge_t=lbsn_dataset.edge_t,
        edge_delta_t=lbsn_dataset.edge_delta_t,
        edge_type=lbsn_dataset.edge_type,
        sizes=sizes,
        edge_delta_s=lbsn_dataset.edge_delta_s,
        num_workers=0,
        pin_memory=False,
    )
    sampler_train = NeighborSampler(
        **common,
        sample_idx=lbsn_dataset.sample_idx_train,
        node_idx=lbsn_dataset.node_idx_train,
        max_time=lbsn_dataset.max_time_train,
        label=lbsn_dataset.label_train,
        batch_size=cfg.run_args.batch_size,
        shuffle=True,
    )
    sampler_validate = NeighborSampler(
        **common,
        sample_idx=lbsn_dataset.sample_idx_valid,
        node_idx=lbsn_dataset.node_idx_valid,
        max_time=lbsn_dataset.max_time_valid,
        label=lbsn_dataset.label_valid,
        batch_size=cfg.run_args.eval_batch_size,
        shuffle=False,
    )
    sampler_test = NeighborSampler(
        **common,
        sample_idx=lbsn_dataset.sample_idx_test,
        node_idx=lbsn_dataset.node_idx_test,
        max_time=lbsn_dataset.max_time_test,
        label=lbsn_dataset.label_test,
        batch_size=cfg.run_args.eval_batch_size,
        shuffle=False,
    )

    model = stack["model"].STHGCN(cfg)
    return cfg, model, sampler_train, sampler_validate, sampler_test


def sampler_batch_to_input(batch, device) -> Tuple[dict, Any]:
    """Convert NeighborSampler batch → STHGCN ``forward`` dict + labels."""
    import torch

    split_index = int(torch.max(batch.adjs_t[1].storage.row()).item())
    batch = batch.to(device)
    input_data = {
        "x": batch.x,
        "edge_index": batch.adjs_t,
        "edge_attr": batch.edge_attrs,
        "split_index": split_index,
        "delta_ts": batch.edge_delta_ts,
        "delta_ss": batch.edge_delta_ss,
        "edge_type": batch.edge_types,
    }
    label = batch.y[:, 0]
    return input_data, label
