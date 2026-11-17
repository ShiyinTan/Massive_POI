"""Joint training of encoders, experts, and the router.

The offline SURD table is a constant. Training samples in fold ``k`` read
``fold_priors[k]`` at the candidate id, never at the label alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..dataset.build import Example, Vocab, make_collate
from ..evaluation.metrics import evaluate_loader
from ..model.ts_moe import TSMoE
from .losses import total_loss


class ExampleDataset(Dataset):
    def __init__(self, examples: list[Example]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Example:
        return self.examples[index]


def sample_candidates(targets: torch.Tensor, n_pois: int, n_neg: int) -> torch.Tensor:
    """Put the true next POI in column 0 and draw negatives from the other ids."""
    real = n_pois - 1
    if real <= 1:
        return targets.unsqueeze(1)
    n_neg = min(int(n_neg), real - 1)
    if n_neg >= real - 1:
        rows = []
        others_pool = list(range(1, n_pois))
        for target in targets.tolist():
            rows.append([target] + [poi for poi in others_pool if poi != target])
        return torch.tensor(rows, device=targets.device, dtype=torch.long)
    neg = torch.randint(1, n_pois, (targets.size(0), n_neg), device=targets.device)
    for _ in range(8):
        clash = neg == targets.unsqueeze(1)
        if not bool(clash.any()):
            break
        replacement = torch.randint(1, n_pois, neg.shape, device=targets.device)
        neg = torch.where(clash, replacement, neg)
    return torch.cat([targets.unsqueeze(1), neg], dim=1)


def _to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


def save_checkpoint(
    path: Path,
    model: TSMoE,
    vocab: Vocab,
    config: dict,
    prior_full: np.ndarray,
    expert_names: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "vocab": vocab.state_dict(),
            "config": config,
            "prior_full": prior_full,
            "expert_names": expert_names,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # older torch without weights_only
        return torch.load(path, map_location=device)


def build_model_from_checkpoint(payload: dict, device: torch.device) -> tuple[TSMoE, Vocab, torch.Tensor]:
    vocab = Vocab.from_state_dict(payload["vocab"])
    model = TSMoE(vocab, payload["config"]).to(device)
    model.load_state_dict(payload["model"])
    prior = torch.tensor(np.asarray(payload["prior_full"]), dtype=torch.float32, device=device)
    return model, vocab, prior


class Trainer:
    def __init__(
        self,
        model: TSMoE,
        vocab: Vocab,
        config: dict,
        fold_priors: np.ndarray,
        full_prior: np.ndarray,
        train_examples: list[Example],
        val_examples: list[Example] | None,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.vocab = vocab
        self.config = config
        self.device = device
        self.fold_priors = torch.tensor(fold_priors, dtype=torch.float32, device=device)
        self.full_prior = torch.tensor(full_prior, dtype=torch.float32, device=device)
        train_cfg = config["train"]
        self.optim = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_cfg["lr"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
        collate = make_collate(vocab, int(config["model"]["max_len"]))
        self.loader = DataLoader(
            ExampleDataset(train_examples),
            batch_size=int(train_cfg["batch_size"]),
            shuffle=True,
            collate_fn=collate,
        )
        self.val_loader = None
        if val_examples:
            self.val_loader = DataLoader(
                ExampleDataset(val_examples),
                batch_size=int(train_cfg["batch_size"]),
                shuffle=False,
                collate_fn=collate,
            )
        self.train_targets = np.array([ex.target for ex in train_examples], dtype=np.int64)

    def step(self, batch: dict) -> dict[str, float]:
        batch = _to_device(batch, self.device)
        candidates = sample_candidates(
            batch["target"],
            self.model.n_pois,
            int(self.config["train"]["num_negatives"]),
        )
        # Fold k reads the prior estimated without fold k. Column 0 of
        # ``candidates`` is the positive, but every column looks up its own row.
        prior = self.fold_priors[batch["fold"].unsqueeze(1), candidates]
        out = self.model(batch, candidates, prior)
        losses = total_loss(
            out["scores"],
            out["pi"],
            prior,
            out["shared_loss"],
            mode=self.model.routing_mode,
            lambda_route=float(self.config["loss"]["lambda_route"]),
            lambda_balance=float(self.config["loss"]["lambda_balance"]),
            lambda_shared=float(self.config["loss"]["lambda_shared"]),
        )
        self.optim.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config["train"]["grad_clip"]))
        self.optim.step()
        return {key: float(value.detach().cpu()) for key, value in losses.items()}

    def fit(self, checkpoint_path: Path | None = None, expert_names: list[str] | None = None) -> dict:
        epochs = int(self.config["train"]["epochs"])
        history = []
        best_score = -1.0
        best_state = None
        for epoch in range(1, epochs + 1):
            self.model.train()
            totals = {"loss": 0.0, "ce": 0.0, "route": 0.0, "balance": 0.0, "shared": 0.0}
            steps = 0
            for batch in self.loader:
                stats = self.step(batch)
                for key in totals:
                    totals[key] += stats[key]
                steps += 1
            row = {key: value / max(steps, 1) for key, value in totals.items()}
            row["epoch"] = epoch
            if self.val_loader is not None:
                metrics = evaluate_loader(
                    self.model,
                    self.val_loader,
                    self.full_prior,
                    self.train_targets,
                    self.config,
                )
                row["val_recall@10"] = metrics["recall"]["10"]
                row["val_ndcg@10"] = metrics["ndcg"]["10"]
                if row["val_recall@10"] >= best_score:
                    best_score = row["val_recall@10"]
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            history.append(row)
            printable = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items())
            print(printable, flush=True)

        if best_state is not None:
            self.model.load_state_dict(best_state)
        if checkpoint_path is not None:
            save_checkpoint(
                checkpoint_path,
                self.model,
                self.vocab,
                self.config,
                self.full_prior.detach().cpu().numpy(),
                expert_names or [],
            )
        return {"history": history, "best_val_recall@10": best_score}
