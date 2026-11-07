"""Alternating Phase A (MI/SURD) and Phase B (recommendation) training."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..evaluation.ranking_metrics import evaluate_ranking
from ..information.calibration import summarize_calibration
from ..information.specific_mi import (
    collect_subset_probs,
    enforce_subset_monotonicity,
    estimate_specific_mi,
)
from ..information.surd import aggregate_global_surd, assign_surd_from_specific_mi
from ..information.weights import build_poi_specific_weights, ema_update_weights, weights_to_tensors
from ..losses.recommendation_loss import recommendation_loss
from ..losses.surd_auxiliary_loss import synergy_auxiliary_loss, unique_auxiliary_loss
from ..models.subset_estimators import SOURCE_NAMES, SUBSET_NAMES, subset_key
from .checkpoint import save_checkpoint, write_json


class AlternatingSURDTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        estimator_fit_loader: DataLoader,
        estimation_loader: DataLoader,
        poi_prior: np.ndarray,
        config: Dict[str, Any],
        device: torch.device,
        output_dir: str | Path,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.estimator_fit_loader = estimator_fit_loader
        self.estimation_loader = estimation_loader
        self.poi_prior = np.asarray(poi_prior, dtype=np.float64)
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        train_cfg = config["train"]
        self.optimizer = torch.optim.Adam(
            self._rec_parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"],
        )
        self.weights: Optional[Dict[str, Any]] = None
        self.weight_tensors: Optional[Dict[str, Any]] = None
        self.best_val = -1.0
        self.history = []

    def _rec_parameters(self):
        # Encoders + recommendation head; subset heads updated in Phase A separately.
        params = list(self.model.history_encoder.parameters())
        params += list(self.model.temporal_encoder.parameters())
        params += list(self.model.spatial_encoder.parameters())
        params += list(self.model.recommendation.parameters())
        return params

    def _subset_parameters(self):
        return list(self.model.subset_estimators.parameters())

    def warmup(self) -> None:
        epochs = int(self.config["train"]["warmup_epochs"])
        for epoch in range(epochs):
            self.model.train()
            # Freeze subset heads during warmup (recommendation only).
            for p in self._subset_parameters():
                p.requires_grad_(False)
            total = 0.0
            n = 0
            for batch in tqdm(self.train_loader, desc=f"warmup {epoch+1}/{epochs}", leave=False):
                batch = batch.to(self.device)
                out = self.model(batch, compute_subsets=False)
                loss = recommendation_loss(out["rec_logits"], batch.target_poi_id)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._rec_parameters(), self.config["train"]["grad_clip"]
                )
                self.optimizer.step()
                total += float(loss.item())
                n += 1
            metrics = evaluate_ranking(
                self.model, self.val_loader, self.device, ks=self.config["eval"]["ks"]
            )
            print(f"[warmup {epoch+1}] loss={total/max(n,1):.4f} val={metrics}")

    def fit_subset_estimators(self) -> None:
        """Phase A part 1: fine-tune subset heads with encoders optionally frozen."""
        epochs = int(self.config["train"]["subset_fit_epochs"])
        lr = float(self.config["train"]["subset_fit_lr"])
        # Freeze encoders while fitting probability heads on current representations.
        for p in self._rec_parameters():
            p.requires_grad_(False)
        for p in self._subset_parameters():
            p.requires_grad_(True)
        opt = torch.optim.Adam(self._subset_parameters(), lr=lr)
        self.model.train()
        for epoch in range(epochs):
            total = 0.0
            n = 0
            for batch in self.estimator_fit_loader:
                batch = batch.to(self.device)
                with torch.no_grad():
                    hs = self.model.encode(batch)
                # Allow grads through subset heads only.
                hs = [h.detach() for h in hs]
                logits = self.model.subset_estimators(hs)
                loss = sum(F.cross_entropy(logits[k], batch.target_poi_id) for k in logits) / len(
                    logits
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.item())
                n += 1
            print(f"[subset-fit {epoch+1}/{epochs}] loss={total/max(n,1):.4f}")
        for p in self._rec_parameters():
            p.requires_grad_(True)
        for p in self._subset_parameters():
            p.requires_grad_(False)

    def refresh_surd(self, tag: str) -> Dict[str, Any]:
        """Phase A: fit heads, estimate specific MI on holdout, assign SURD, build weights."""
        self.fit_subset_estimators()
        probs, labels = collect_subset_probs(self.model, self.estimation_loader, self.device)
        calib = summarize_calibration(probs, labels)
        specific_mi, counts = estimate_specific_mi(
            probs,
            labels,
            self.poi_prior,
            eps=float(self.config["surd"]["eps"]),
            units=self.config["surd"]["units"],
        )
        specific_mi = enforce_subset_monotonicity(specific_mi)
        poi_surd = assign_surd_from_specific_mi(specific_mi)
        global_surd = aggregate_global_surd(poi_surd, self.poi_prior)
        new_weights = build_poi_specific_weights(
            poi_surd=poi_surd,
            poi_sample_counts=counts,
            poi_prior=self.poi_prior,
            min_poi_samples=int(self.config["surd"]["min_poi_samples"]),
            shrink_kappa=float(self.config["surd"]["shrink_kappa"]),
            eps=float(self.config["surd"]["eps"]),
        )
        self.weights = ema_update_weights(
            self.weights, new_weights, beta=float(self.config["train"]["ema_beta"])
        )
        self.weight_tensors = weights_to_tensors(
            self.weights, num_pois=self.model.num_pois, device=self.device
        )
        self._dump_surd_artifacts(tag, specific_mi, counts, poi_surd, global_surd, calib)
        print(
            f"[SURD {tag}] fallback_frac={self.weights['fallback_fraction']:.3f} "
            f"global_U={global_surd['unique']}"
        )
        return self.weights

    def _dump_surd_artifacts(self, tag, specific_mi, counts, poi_surd, global_surd, calib):
        out = self.output_dir / "surd" / tag
        out.mkdir(parents=True, exist_ok=True)
        # specific_mi.csv
        keys = [subset_key(s) for s in SUBSET_NAMES]
        with (out / "specific_mi.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["poi_id", "sample_count"] + keys)
            K = len(counts)
            for y in range(K):
                w.writerow([y, int(counts[y])] + [float(specific_mi[k][y]) for k in keys])
        with (out / "poi_specific_surd.csv").open("w", newline="", encoding="utf-8") as f:
            u_keys = list(SOURCE_NAMES)
            r_keys = [subset_key(s) for s in SUBSET_NAMES if len(s) >= 2]
            s_keys = r_keys
            header = (
                ["poi_id", "sample_count"]
                + [f"U_{k}" for k in u_keys]
                + [f"R_{k}" for k in r_keys]
                + [f"S_{k}" for k in s_keys]
            )
            w = csv.writer(f)
            w.writerow(header)
            for y in range(len(counts)):
                terms = poi_surd.get(y, {"unique": {}, "redundant": {}, "synergistic": {}})
                row = [y, int(counts[y])]
                row += [terms["unique"].get(k, 0.0) for k in u_keys]
                row += [terms["redundant"].get(k, 0.0) for k in r_keys]
                row += [terms["synergistic"].get(k, 0.0) for k in s_keys]
                w.writerow(row)
        write_json(out / "global_surd.json", global_surd)
        with (out / "poi_sample_counts.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["poi_id", "sample_count"])
            for y, c in enumerate(counts):
                w.writerow([y, int(c)])
        write_json(out / "estimator_calibration.json", calib)
        # Detached weights snapshot
        write_json(
            out / "weights_meta.json",
            {
                "fallback_fraction": self.weights["fallback_fraction"],
                "global_unique_weights": self.weights["global_unique_weights"],
                "global_synergy_weights": self.weights["global_synergy_weights"],
                "note": "Predictive information structure (observational); not causal effects.",
            },
        )

    def train_recommendation_epoch(self, epoch: int) -> Dict[str, float]:
        """Phase B: update encoders/rec head with fixed detached SURD weights."""
        self.model.train()
        for p in self._subset_parameters():
            p.requires_grad_(False)
        for p in self._rec_parameters():
            p.requires_grad_(True)

        lambda_u = float(self.config["train"]["lambda_u"])
        lambda_s = float(self.config["train"]["lambda_s"])
        totals = {"loss": 0.0, "rec": 0.0, "u": 0.0, "s": 0.0}
        n = 0
        for batch in tqdm(self.train_loader, desc=f"epoch {epoch}", leave=False):
            batch = batch.to(self.device)
            out = self.model(batch, compute_subsets=True)
            loss_rec = recommendation_loss(out["rec_logits"], batch.target_poi_id)
            if self.weight_tensors is None:
                loss_u = batch.target_poi_id.new_zeros(())
                loss_s = batch.target_poi_id.new_zeros(())
            else:
                # Weights are detached tensors — no backprop through SURD estimation.
                loss_u = unique_auxiliary_loss(
                    out["single_source_logits"],
                    batch.target_poi_id,
                    self.weight_tensors["unique"],
                    source_order=self.weight_tensors["unique_keys"],
                )
                loss_s = synergy_auxiliary_loss(
                    out["multi_source_logits"],
                    batch.target_poi_id,
                    self.weight_tensors["synergy"],
                    subset_keys=self.weight_tensors["synergy_keys"],
                )
            loss = loss_rec + lambda_u * loss_u + lambda_s * loss_s
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._rec_parameters(), self.config["train"]["grad_clip"])
            self.optimizer.step()
            totals["loss"] += float(loss.item())
            totals["rec"] += float(loss_rec.item())
            totals["u"] += float(loss_u.item()) if torch.is_tensor(loss_u) else float(loss_u)
            totals["s"] += float(loss_s.item()) if torch.is_tensor(loss_s) else float(loss_s)
            n += 1
        return {k: v / max(n, 1) for k, v in totals.items()}

    def fit(self) -> Dict[str, Any]:
        self.warmup()
        self.refresh_surd(tag="epoch_0000_init")

        total_epochs = int(self.config["train"]["total_epochs"])
        refresh_every = int(self.config["train"]["surd_refresh_interval"])
        for epoch in range(1, total_epochs + 1):
            train_stats = self.train_recommendation_epoch(epoch)
            val_metrics = evaluate_ranking(
                self.model, self.val_loader, self.device, ks=self.config["eval"]["ks"]
            )
            record = {"epoch": epoch, "train": train_stats, "val": val_metrics}
            self.history.append(record)
            print(f"[epoch {epoch}] train={train_stats} val={val_metrics}")

            score = val_metrics.get("recall@5", val_metrics.get("recall@1", 0.0))
            if score >= self.best_val:
                self.best_val = score
                save_checkpoint(
                    self.output_dir / "best.pt",
                    self.model,
                    self.optimizer,
                    meta={"epoch": epoch, "val": val_metrics},
                )

            if epoch % refresh_every == 0 and epoch < total_epochs:
                self.refresh_surd(tag=f"epoch_{epoch:04d}")

        save_checkpoint(
            self.output_dir / "last.pt",
            self.model,
            self.optimizer,
            meta={"history": self.history, "best_val": self.best_val},
        )
        write_json(self.output_dir / "history.json", self.history)
        return {"best_val": self.best_val, "history": self.history}