"""Baseline-only train/eval for full hypergraph STHGCN (SURD off).

Mirrors ``Massive-STEPS/Spatio-Temporal-Hypergraph-Model/run.py`` training loop
and ``utils.pipeline_util.test_step`` metrics, driven from surd_guided_poi configs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from tqdm.auto import tqdm

from ..data.sthgcn_hypergraph import prepare_sthgcn_pipeline, sampler_batch_to_input
from .checkpoint import write_json
from .log_format import format_metrics, log_block


@torch.no_grad()
def evaluate_sampler(model, sampler, device, ks=(1, 5, 10), mode="test") -> Dict[str, float]:
    model.eval()
    pred_list, label_list, losses = [], [], []
    for row in tqdm(sampler, desc=f"eval:{mode}", leave=False):
        input_data, label = sampler_batch_to_input(row, device)
        out, loss = model(input_data, label=label, mode=mode)
        losses.append(float(loss.detach().cpu()))
        ranking = torch.sort(out, descending=True)[1]
        pred_list.append(ranking.cpu())
        label_list.append(label.view(-1, 1).cpu())
    pred = torch.cat(pred_list, dim=0)
    lab = torch.cat(label_list, dim=0)
    hits = pred == lab
    out_m: Dict[str, float] = {"loss": float(np.mean(losses)) if losses else 0.0}
    for k in ks:
        hit_k = hits[:, :k].any(dim=1).float()
        out_m[f"recall@{k}"] = float(hit_k.mean().item())
        out_m[f"acc@{k}"] = out_m[f"recall@{k}"]
        discounts = 1.0 / torch.log2(torch.arange(1, k + 1, dtype=torch.float32) + 1.0)
        dcg = (hits[:, :k].float() * discounts.unsqueeze(0)).sum(dim=1)
        out_m[f"ndcg@{k}"] = float(dcg.mean().item())
    first = hits.float().argmax(dim=1)
    has = hits.any(dim=1)
    rr = torch.zeros(hits.size(0))
    rr[has] = 1.0 / (first[has].float() + 1.0)
    out_m["mrr"] = float(rr.mean().item())
    return out_m


def run_sthgcn_baseline(
    cfg: Dict[str, Any],
    *,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, float]:
    """Train + test full hypergraph STHGCN; SURD is not supported on this path."""
    train_cfg = cfg.get("train") or {}
    if bool(train_cfg.get("use_surd", False)):
        raise ValueError(
            "Full hypergraph STHGCN path does not support SURD yet "
            "(NeighborSampler batches ≠ NextPOIBatch). Set train.use_surd: false."
        )

    data_cfg = cfg["data"]
    city = data_cfg["city"]
    model_cfg = cfg.get("model") or {}
    conf_relpath = str(model_cfg.get("sthgcn_conf", "std_conf/beijing.yml"))
    cache_dir = Path(data_cfg.get("cache_dir") or "outputs/data_cache")
    if not cache_dir.is_absolute():
        # resolve relative to surd_guided_poi/
        pkg = Path(__file__).resolve().parents[1]
        cache_dir = pkg / cache_dir

    run_overrides = {
        "epoch": int(train_cfg.get("total_epochs", 20)),
        "batch_size": int(train_cfg.get("batch_size", 16)),
        "learning_rate": float(train_cfg.get("lr", 1e-4)),
        "eval_batch_size": int(train_cfg.get("eval_batch_size", 64)),
        "valid_steps": int(train_cfg.get("valid_steps", 200)),
        "max_steps": int(train_cfg.get("max_steps", 100000)),
        "seed": int(cfg.get("seed", 42)),
    }
    # Optional smaller neighbor sizes for CPU smoke
    sizes_override = model_cfg.get("sthgcn_sizes")
    prep_overrides: Dict[str, Any] = dict(run_overrides)
    if sizes_override:
        # applied after Cfg load via model_args
        pass

    print(
        log_block(
            "STHGCN hypergraph",
            [
                f"city={city}  conf={conf_relpath}  device={device}",
                "SURD=off  NeighborSampler path (not NextPOIBatch)",
                "deps: requirements-sthgcn.txt / Massive-STEPS Spatio-Temporal-Hypergraph-Model",
            ],
        )
    )

    sthgcn_cfg, model, sampler_train, sampler_val, sampler_test = prepare_sthgcn_pipeline(
        city=city,
        data_dir=data_cfg.get("data_dir"),
        cache_dir=cache_dir / city,
        hf_dataset_id=data_cfg.get("hf_dataset_id"),
        conf_relpath=conf_relpath,
        device=str(device),
        run_overrides=run_overrides,
    )
    if sizes_override:
        if isinstance(sizes_override, str):
            sthgcn_cfg.model_args.sizes = [int(x) for x in sizes_override.split("-")]
        else:
            sthgcn_cfg.model_args.sizes = list(sizes_override)

    model = model.to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(run_overrides["learning_rate"]),
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "sthgcn_data_summary.json",
        {
            "backbone": "sthgcn",
            "path": "full_hypergraph",
            "city": city,
            "conf": conf_relpath,
            "num_poi": int(sthgcn_cfg.dataset_args.num_poi),
            "num_user": int(sthgcn_cfg.dataset_args.num_user),
            "sizes": list(sthgcn_cfg.model_args.sizes),
            "n_train": len(sampler_train),
            "n_val": len(sampler_val),
            "n_test": len(sampler_test),
        },
    )

    best_metric = 0.0
    best_path = output_dir / "best.pt"
    history = []
    global_step = 0
    valid_every = int(run_overrides["valid_steps"])
    max_steps = int(run_overrides["max_steps"])
    epochs = int(run_overrides["epoch"])
    ks = tuple(cfg.get("eval", {}).get("ks") or [1, 5, 10])

    for eph in range(epochs):
        if global_step >= max_steps:
            break
        model.train()
        epoch_losses = []
        for data in tqdm(sampler_train, desc=f"epoch {eph+1}/{epochs}"):
            input_data, label = sampler_batch_to_input(data, device)
            _, loss = model(input_data, label=label, mode="train")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            global_step += 1
            if global_step % valid_every == 0:
                val_m = evaluate_sampler(model, sampler_val, device, ks=ks, mode="validate")
                print(log_block(f"val step {global_step}", [format_metrics(val_m)]))
                if val_m.get("recall@1", 0.0) >= best_metric:
                    best_metric = val_m["recall@1"]
                    torch.save({"model_state_dict": model.state_dict(), "step": global_step}, best_path)
            if global_step >= max_steps:
                break

        val_m = evaluate_sampler(model, sampler_val, device, ks=ks, mode="validate")
        test_m = evaluate_sampler(model, sampler_test, device, ks=ks, mode="test")
        row = {
            "epoch": eph + 1,
            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else None,
            "val": val_m,
            "test": test_m,
        }
        history.append(row)
        print(
            log_block(
                f"epoch {eph+1}/{epochs}",
                [
                    f"train_loss={row['train_loss']:.4f}" if row["train_loss"] is not None else "train_loss=n/a",
                    f"val   {format_metrics(val_m)}",
                    f"test  {format_metrics(test_m)}",
                ],
            )
        )
        if val_m.get("recall@1", 0.0) >= best_metric:
            best_metric = val_m["recall@1"]
            torch.save({"model_state_dict": model.state_dict(), "step": global_step}, best_path)

    if best_path.is_file():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate_sampler(model, sampler_test, device, ks=ks, mode="test")
    write_json(output_dir / "history.json", history)
    write_json(output_dir / "test_metrics.json", test_metrics)
    print(log_block("TEST (hypergraph STHGCN)", [format_metrics(test_metrics), f"→ {output_dir}"]))
    return test_metrics
