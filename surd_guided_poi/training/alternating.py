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
from .log_format import (
    epoch_title,
    format_global_u,
    format_metrics,
    format_train_losses,
    log_block,
)


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
        test_loader: Optional[DataLoader] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader  # carve 的 model_train：推荐主训
        self.val_loader = val_loader
        self.test_loader = test_loader  # 可选：每 epoch 打 test（不参与选 best）
        # carve 的 model_train：Phase A 拟合子集头（与估 MI 的样本分开）
        self.estimator_fit_loader = estimator_fit_loader
        # carve 的 estimation holdout：Phase A 只在这里算 Specific MI / SURD
        self.estimation_loader = estimation_loader
        # P(Y=y)=N_y/N（概率）；fallback 用的整数 N_y 来自估计集 counts，不是这个向量
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
        milestones = list(train_cfg.get("lr_milestones") or [])
        self.scheduler = (
            torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=milestones,
                gamma=float(train_cfg.get("lr_gamma", 0.2)),
            )
            if milestones
            else None
        )
        self.weights: Optional[Dict[str, Any]] = None
        self.weight_tensors: Optional[Dict[str, Any]] = None
        self.best_val = -1.0
        self.history = []

    def _use_surd(self) -> bool:
        train_cfg = self.config.get("train", {})
        if "use_surd" in train_cfg:
            return bool(train_cfg["use_surd"])
        return float(train_cfg.get("lambda_u", 0.0)) > 0 or float(train_cfg.get("lambda_s", 0.0)) > 0

    def _eval_test_every_epoch(self) -> bool:
        return bool(self.config.get("eval", {}).get("eval_test_every_epoch", False)) and (
            self.test_loader is not None
        )

    def _best_metric_key(self) -> str:
        return str(self.config.get("eval", {}).get("best_metric", "recall@1"))

    def _ranking_on(self, loader: DataLoader) -> Dict[str, float]:
        return evaluate_ranking(
            self.model, loader, self.device, ks=self.config["eval"]["ks"]
        )

    def _rec_parameters(self):
        """Flashback backbone + SURD side encoders (not subset heads)."""
        params = list(self.model.backbone.parameters())
        # Side encoders may share Flashback POI embedding; dedupe by id.
        seen = {id(p) for p in params}
        for module in (
            self.model.history_encoder,
            self.model.temporal_encoder,
            self.model.spatial_encoder,
        ):
            for p in module.parameters():
                if id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))
        return params

    def _subset_parameters(self):
        return list(self.model.subset_estimators.parameters())

    def warmup(self) -> None:
        """先只训推荐（Flashback），子集头先冻住。"""
        epochs = int(self.config["train"]["warmup_epochs"])
        if epochs <= 0:
            return
        for epoch in range(epochs):
            self.model.train()
            for p in self._subset_parameters():
                p.requires_grad_(False)  # warm-up 不训 SURD 头
            total = 0.0
            n = 0
            for batch in tqdm(
                self.train_loader,
                desc=epoch_title("warmup", epoch + 1, epochs),
                leave=False,
            ):
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
            lines = [
                f"train  loss={total / max(n, 1):.4f}",
                f"val    {format_metrics(self._ranking_on(self.val_loader))}",
            ]
            if self._eval_test_every_epoch():
                lines.append(f"test   {format_metrics(self._ranking_on(self.test_loader))}")
            print(log_block(epoch_title("warmup", epoch + 1, epochs), lines))

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
            print(
                f"  subset-fit  {epoch + 1:02d}/{epochs:02d}  "
                f"loss={total / max(n, 1):.4f}"
            )
        for p in self._rec_parameters():
            p.requires_grad_(True)
        for p in self._subset_parameters():
            p.requires_grad_(False)

    def refresh_surd(self, tag: str) -> Dict[str, Any]:
        """Phase A：估 Specific MI → SURD → 建成训练用的 U/S 权重（不反传）。"""
        self.fit_subset_estimators()
        # 估计 holdout 上收集 7 子集 P(Y|Q_A)[N,K] + labels[N] → 再估 Specific MI
        probs, labels = collect_subset_probs(self.model, self.estimation_loader, self.device)
        calib = summarize_calibration(probs, labels)
        # prior 进 MI 分母；返回的 counts 是估计集上的整数 N_y（给 fallback）
        specific_mi, counts = estimate_specific_mi(
            probs,
            labels,
            self.poi_prior,
            eps=float(self.config["surd"]["eps"]),
            units=self.config["surd"]["units"],
        )
        specific_mi = enforce_subset_monotonicity(specific_mi)
        poi_surd = assign_surd_from_specific_mi(specific_mi)
        global_surd = aggregate_global_surd(poi_surd, self.poi_prior)  # 按 P(y) 加权
        new_weights = build_poi_specific_weights(
            poi_surd=poi_surd,
            poi_sample_counts=counts,  # 整数 N_y：够不够样本、往全局收缩多少
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
            log_block(
                f"SURD  {tag}",
                [
                    f"fallback_frac  {self.weights['fallback_fraction']:.3f}",
                    f"global_U       {format_global_u(global_surd['unique'])}",
                ],
            )
        )
        return self.weights

    def _dump_surd_artifacts(self, tag, specific_mi, counts, poi_surd, global_surd, calib):
        out = self.output_dir / "surd" / tag
        out.mkdir(parents=True, exist_ok=True)
        # specific_mi.csv
        keys = [subset_key(s) for s in SUBSET_NAMES]  # CSV 列名：Q1, Q1_Q2, ...
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
        """Phase B：固定 SURD 权重，继续训推荐；总损失 = L_rec + λ_U L_U + λ_S L_S。"""
        self.model.train()
        for p in self._subset_parameters():
            p.requires_grad_(False)  # 这阶段不更新子集头
        for p in self._rec_parameters():
            p.requires_grad_(True)

        use_surd = self._use_surd()
        lambda_u = float(self.config["train"]["lambda_u"]) if use_surd else 0.0
        lambda_s = float(self.config["train"]["lambda_s"]) if use_surd else 0.0
        totals = {"loss": 0.0, "rec": 0.0, "u": 0.0, "s": 0.0}
        n = 0
        for batch in tqdm(
            self.train_loader,
            desc=epoch_title("epoch", epoch, int(self.config["train"]["total_epochs"])),
            leave=False,
        ):
            batch = batch.to(self.device)
            out = self.model(batch, compute_subsets=use_surd and self.weight_tensors is not None)
            loss_rec = recommendation_loss(out["rec_logits"], batch.target_poi_id)
            if (not use_surd) or self.weight_tensors is None:
                loss_u = batch.target_poi_id.new_zeros(())
                loss_s = batch.target_poi_id.new_zeros(())
            else:
                # 用真实标签 y 查表加权；权重已 detach，梯度不流回 SURD 估计
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
        # 流程：warm-up →（可选）首次估 SURD → 循环训推荐；可选重估 SURD
        total_epochs = int(self.config["train"]["total_epochs"])
        refresh_every = int(self.config["train"]["surd_refresh_interval"])
        use_surd = self._use_surd()
        best_key = self._best_metric_key()
        print(
            log_block(
                "train schedule",
                [
                    f"warmup_epochs={self.config['train']['warmup_epochs']}  "
                    f"total_epochs={total_epochs}  "
                    f"use_surd={use_surd}  "
                    f"surd_refresh_every={refresh_every if use_surd else 'n/a'}  "
                    f"subset_fit_epochs={self.config['train']['subset_fit_epochs']}",
                    f"best_metric={best_key}  "
                    f"eval_every_epoch  val=yes  "
                    f"test={'yes' if self._eval_test_every_epoch() else 'no'}  "
                    f"lr_milestones={list(self.config['train'].get('lr_milestones') or [])}",
                ],
            )
        )
        self.warmup()
        if use_surd:
            self.refresh_surd(tag="epoch_0000_init")
        else:
            print("  SURD disabled — backbone-only recommendation (lambda_u/s ignored).")

        for epoch in range(1, total_epochs + 1):
            train_stats = self.train_recommendation_epoch(epoch)
            # 每个 epoch 必跑 val；test 由 eval.eval_test_every_epoch 控制（不参与选 best）
            val_metrics = self._ranking_on(self.val_loader)
            test_metrics = (
                self._ranking_on(self.test_loader) if self._eval_test_every_epoch() else None
            )
            record: Dict[str, Any] = {"epoch": epoch, "train": train_stats, "val": val_metrics}
            if test_metrics is not None:
                record["test"] = test_metrics
            self.history.append(record)

            score = float(
                val_metrics.get(
                    best_key,
                    val_metrics.get("acc@1", val_metrics.get("recall@1", 0.0)),
                )
            )
            is_best = score >= self.best_val
            lines = [
                f"train  {format_train_losses(train_stats)}",
                f"val    {format_metrics(val_metrics)}",
            ]
            if test_metrics is not None:
                lines.append(f"test   {format_metrics(test_metrics)}")
            if is_best:
                lines.append(f"★ new best (val {best_key}={score:.4f})")
            print(log_block(epoch_title("epoch", epoch, total_epochs), lines))

            if is_best:
                self.best_val = score
                save_checkpoint(
                    self.output_dir / "best.pt",
                    self.model,
                    self.optimizer,
                    meta={"epoch": epoch, "val": val_metrics, "test": test_metrics},
                )

            if use_surd and epoch % refresh_every == 0 and epoch < total_epochs:
                self.refresh_surd(tag=f"epoch_{epoch:04d}")

            if self.scheduler is not None:
                self.scheduler.step()

        save_checkpoint(
            self.output_dir / "last.pt",
            self.model,
            self.optimizer,
            meta={"history": self.history, "best_val": self.best_val},
        )
        write_json(self.output_dir / "history.json", self.history)
        print(
            log_block(
                "training finished",
                [f"best_val_{best_key}={float(self.best_val):.4f}  output={self.output_dir}"],
            )
        )
        return {"best_val": self.best_val, "history": self.history}