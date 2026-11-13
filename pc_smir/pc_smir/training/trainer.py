from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pc_smir.evaluation.metrics import format_metrics_table, merge_metric_dicts, rank_metrics_from_scores
from pc_smir.losses.ranking import bpr_loss, full_softmax_ce, routing_entropy


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        cfg: dict[str, Any],
        out_dir: Path,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric = -1.0
        self.best_path = self.out_dir / "best.pt"
        self.history: list[dict] = []

    def _move(self, batch: dict) -> dict:
        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

    def _loss(self, scores, routes, batch):
        obj = self.cfg.get("loss", "ce")
        if obj == "bpr":
            loss = bpr_loss(scores, batch["target_index"], batch.get("candidate_mask"))
        else:
            loss = full_softmax_ce(scores, batch["target_index"], temperature=float(self.cfg.get("temperature", 1.0)))
        ent_coef = float(self.cfg.get("entropy_coef", 0.0))
        if ent_coef != 0.0:
            # Positive coef increases entropy (anti-collapse); negative encourages concentration.
            loss = loss + ent_coef * (-routing_entropy(routes, batch.get("candidate_mask")))
        return loss

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        total = 0.0
        n = 0
        for batch in tqdm(loader, desc="train", leave=False):
            batch = self._move(batch)
            scores, routes = self.model.score(batch)
            loss = self._loss(scores, routes, batch)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.get("grad_clip", 5.0)))
            self.optimizer.step()
            total += float(loss.item()) * batch["target_index"].size(0)
            n += batch["target_index"].size(0)
        return {"loss": total / max(n, 1)}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, ks=(1, 5, 10, 20), desc: str = "eval") -> dict[str, float]:
        self.model.eval()
        parts = []
        total_loss = 0.0
        n = 0
        route_ent = 0.0
        for batch in tqdm(loader, desc=desc, leave=False):
            batch = self._move(batch)
            scores, routes = self.model.score(batch)
            loss = full_softmax_ce(scores, batch["target_index"])
            total_loss += float(loss.item()) * batch["target_index"].size(0)
            n += batch["target_index"].size(0)
            parts.append(
                rank_metrics_from_scores(scores, batch["target_index"], batch.get("candidate_mask"), ks=ks)
            )
            route_ent += float(routing_entropy(routes, batch.get("candidate_mask")).item()) * batch["target_index"].size(0)
        metrics = merge_metric_dicts(parts)
        metrics["loss"] = total_loss / max(n, 1)
        metrics["route_entropy"] = route_ent / max(n, 1)
        return metrics

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int,
        select_metric: str = "recall@10",
        test_loader=None,
    ):
        patience = int(self.cfg.get("patience", 5))
        bad = 0
        for epoch in range(1, epochs + 1):
            tr = self.train_epoch(train_loader)
            va = self.evaluate(val_loader, desc="val")
            te = self.evaluate(test_loader, desc="test") if test_loader is not None else None
            row = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in tr.items()},
                **{f"val_{k}": v for k, v in va.items()},
            }
            if te is not None:
                row.update({f"test_{k}": v for k, v in te.items()})
            self.history.append(row)

            splits = {"val": va}
            if te is not None:
                splits["test"] = te
            print(
                format_metrics_table(
                    splits,
                    epoch=epoch,
                    train_loss=float(tr.get("loss", 0.0)),
                )
            )

            score = float(va.get(select_metric, 0.0))
            if score > self.best_metric:
                self.best_metric = score
                bad = 0
                payload = {
                    "model": self.model.state_dict(),
                    "cfg": self.cfg,
                    "epoch": epoch,
                    "val": va,
                }
                if te is not None:
                    payload["test"] = te
                torch.save(payload, self.best_path)
            else:
                bad += 1
                if bad >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        with open(self.out_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        return self.best_path
