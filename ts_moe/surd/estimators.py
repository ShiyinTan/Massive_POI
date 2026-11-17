"""Conditional estimators for ``P(Y=y | Q_A)``.

Both estimators implement the spec's specific-MI identity

    i(y; Q_A) = E[ log P(y | Q_A) - log P(y) | Y = y ]

on an estimation slice that excludes the cross-fit fold being scored.

* ``discrete`` — Dirichlet-style smoothing toward the marginal. Default for
  smoke runs. States are the last-step summary of each source: user id,
  last POI, last hour, spatial bin of the last coordinate. Joint states of
  a subset are counted sparsely. Plug-in counts and the MI average use the
  same estimation slice (standard empirical specific MI).
* ``neural`` — lightweight conditional network (spec §4.2). Four mean-pooled
  embeddings, one head per non-empty subset, scores ``h_A · e_y``. The head
  is fit on an inner trail hold-in; specific MI is read on the hold-out
  trails so the probabilities are not the training likelihood of the same
  rows. Full-vocabulary log-softmax (chunked) is used for the MI readout
  even when training used sampled softmax.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..dataset.build import Example, collate_examples
from .atoms import ALL_SUBSETS, SOURCES, subset_key
from .specific_mi import aggregate_specific_mi, marginal_prior, observation_counts


class SpatialGrid:
    """Equal-width bins on the estimation slice. Degenerate axes stay bin 0."""

    def __init__(self, lat: np.ndarray, lon: np.ndarray, n_bins: int = 8):
        self.n_bins = int(n_bins)
        self.lat_min = float(np.min(lat)) if len(lat) else 0.0
        self.lat_max = float(np.max(lat)) if len(lat) else 1.0
        self.lon_min = float(np.min(lon)) if len(lon) else 0.0
        self.lon_max = float(np.max(lon)) if len(lon) else 1.0

    def _bin(self, values: np.ndarray, lo: float, hi: float) -> np.ndarray:
        if hi <= lo:
            return np.zeros(len(values), dtype=np.int64)
        scaled = np.clip((values - lo) / (hi - lo), 0.0, 0.999999)
        return np.floor(scaled * self.n_bins).astype(np.int64)

    def assign(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        return self._bin(lat, self.lat_min, self.lat_max) * self.n_bins + self._bin(
            lon, self.lon_min, self.lon_max
        )


def _split_fit_eval(examples: list[Example], eval_fraction: float, seed: int) -> tuple[list[Example], list[Example]]:
    trails = sorted({ex.trail_id for ex in examples})
    if len(trails) < 2 or eval_fraction <= 0:
        return examples, examples
    rng = np.random.default_rng(seed)
    order = list(trails)
    rng.shuffle(order)
    n_eval = max(1, int(round(len(order) * eval_fraction)))
    n_eval = min(n_eval, len(order) - 1)
    eval_ids = set(order[:n_eval])
    fit = [ex for ex in examples if ex.trail_id not in eval_ids]
    ev = [ex for ex in examples if ex.trail_id in eval_ids]
    if not fit or not ev:
        return examples, examples
    return fit, ev


def _source_matrix(examples: list[Example], grid: SpatialGrid) -> tuple[np.ndarray, np.ndarray]:
    user = np.array([ex.user for ex in examples], dtype=np.int64)
    poi = np.array([ex.poi[-1] for ex in examples], dtype=np.int64)
    hour = np.array([ex.hour[-1] for ex in examples], dtype=np.int64)
    loc = grid.assign(
        np.array([ex.lat[-1] for ex in examples], dtype=np.float64),
        np.array([ex.lon[-1] for ex in examples], dtype=np.float64),
    )
    labels = np.array([ex.target for ex in examples], dtype=np.int64)
    return np.stack([user, poi, hour, loc], axis=1), labels


def _state_ids(rows: np.ndarray) -> np.ndarray:
    """Factorize integer rows into dense state ids."""
    if len(rows) == 0:
        return np.zeros(0, dtype=np.int64)
    contiguous = np.ascontiguousarray(rows)
    width = contiguous.dtype.itemsize * contiguous.shape[1]
    keys = contiguous.view(np.dtype((np.void, width))).ravel()
    _, inverse = np.unique(keys, return_inverse=True)
    return inverse.astype(np.int64)


def estimate_specific_mi_discrete(
    examples: list[Example],
    n_pois: int,
    *,
    alpha: float = 1.0,
    n_bins: int = 8,
    eps: float = 1e-12,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Plug-in specific MI for every non-empty source subset."""
    if not examples:
        zeros = {subset_key(s): np.zeros(n_pois) for s in ALL_SUBSETS}
        return zeros, np.zeros(n_pois)

    lat = np.array([ex.lat[-1] for ex in examples], dtype=np.float64)
    lon = np.array([ex.lon[-1] for ex in examples], dtype=np.float64)
    grid = SpatialGrid(lat, lon, n_bins=n_bins)
    features, labels = _source_matrix(examples, grid)
    prior = marginal_prior(labels, n_pois)
    log_prior = np.log(np.clip(prior, eps, None))
    counts = observation_counts(labels, n_pois)
    col = {name: i for i, name in enumerate(SOURCES)}
    specific: dict[str, np.ndarray] = {}

    for subset in ALL_SUBSETS:
        idxs = [col[name] for name in subset]
        state = _state_ids(features[:, idxs])
        n_states = int(state.max()) + 1 if len(state) else 1
        state_count = np.bincount(state, minlength=n_states).astype(np.float64)
        joint: dict[tuple[int, int], int] = {}
        for sid, lab in zip(state.tolist(), labels.tolist()):
            joint[(sid, lab)] = joint.get((sid, lab), 0) + 1
        log_ratio = np.empty(len(labels), dtype=np.float64)
        for i, (sid, lab) in enumerate(zip(state.tolist(), labels.tolist())):
            observed = joint.get((sid, int(lab)), 0)
            total = state_count[sid]
            prob = (observed + alpha * prior[lab]) / (total + alpha)
            prob = float(np.clip(prob, eps, 1.0))
            log_ratio[i] = np.log(prob) - log_prior[lab]
        specific[subset_key(subset)] = aggregate_specific_mi(log_ratio, labels, n_pois)
    return specific, counts


class SubsetMINet(nn.Module):
    """Mean-pool each source, then a small head per subset scores POI embeddings."""

    def __init__(self, n_users: int, n_pois: int, d_model: int = 32):
        super().__init__()
        self.n_pois = n_pois
        self.user_emb = nn.Embedding(n_users, d_model)
        self.poi_emb = nn.Embedding(n_pois, d_model, padding_idx=0)
        self.hour_emb = nn.Embedding(24, d_model)
        self.dow_emb = nn.Embedding(7, d_model)
        self.loc_proj = nn.Linear(2, d_model)
        self.heads = nn.ModuleDict()
        for subset in ALL_SUBSETS:
            key = subset_key(subset)
            width = d_model * len(subset)
            self.heads[key] = nn.Sequential(
                nn.Linear(width, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        self.out_emb = nn.Embedding(n_pois, d_model, padding_idx=0)

    def encode_sources(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pad = batch["pad"]
        valid = (~pad).unsqueeze(-1).to(batch["loc"].dtype)
        denom = valid.sum(dim=1).clamp(min=1.0)
        user = self.user_emb(batch["user"]).unsqueeze(1).expand(-1, batch["poi"].size(1), -1)
        poi = self.poi_emb(batch["poi"])
        time = self.hour_emb(batch["hour"]) + self.dow_emb(batch["dow"])
        loc = self.loc_proj(batch["loc"])
        pooled = {}
        for name, seq in (("U", user), ("P", poi), ("T", time), ("L", loc)):
            pooled[name] = (seq * valid).sum(dim=1) / denom
        return pooled

    def subset_hidden(self, pooled: dict[str, torch.Tensor], key: str) -> torch.Tensor:
        names = key.split("_")
        return self.heads[key](torch.cat([pooled[name] for name in names], dim=-1))

    def log_prob_true(self, hidden: torch.Tensor, targets: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
        """Chunked log-softmax of ``h · e_y`` at the true class. Class 0 is masked."""
        weight = self.out_emb.weight
        running_max = None
        running_sum = None
        for start in range(0, self.n_pois, chunk):
            logits = hidden @ weight[start : start + chunk].T
            if start == 0:
                logits = logits.clone()
                logits[:, 0] = torch.finfo(logits.dtype).min
            block_max = logits.max(dim=-1).values
            if running_max is None:
                running_max = block_max
                running_sum = torch.exp(logits - block_max.unsqueeze(-1)).sum(dim=-1)
            else:
                new_max = torch.maximum(running_max, block_max)
                running_sum = running_sum * torch.exp(running_max - new_max) + torch.exp(
                    logits - new_max.unsqueeze(-1)
                ).sum(dim=-1)
                running_max = new_max
        log_z = running_max + torch.log(running_sum.clamp_min(1e-12))
        true_logit = (hidden * weight[targets]).sum(dim=-1)
        return true_logit - log_z


def _coord_stats(examples: list[Example]) -> tuple[float, float, float, float]:
    lat = np.array([v for ex in examples for v in ex.lat], dtype=np.float64)
    lon = np.array([v for ex in examples for v in ex.lon], dtype=np.float64)
    lat_mean = float(lat.mean()) if len(lat) else 0.0
    lon_mean = float(lon.mean()) if len(lon) else 0.0
    lat_std = float(lat.std()) if len(lat) and float(lat.std()) > 1e-6 else 1.0
    lon_std = float(lon.std()) if len(lon) and float(lon.std()) > 1e-6 else 1.0
    return lat_mean, lat_std, lon_mean, lon_std


def _batch_examples(
    examples: list[Example],
    max_len: int,
    device: torch.device,
    stats: tuple[float, float, float, float],
) -> dict[str, torch.Tensor]:
    lat_mean, lat_std, lon_mean, lon_std = stats
    batch = collate_examples(
        examples,
        max_len=max_len,
        lat_mean=lat_mean,
        lat_std=lat_std,
        lon_mean=lon_mean,
        lon_std=lon_std,
    )
    return {k: v.to(device) for k, v in batch.items()}


def _iter_minibatches(examples: list[Example], batch_size: int, rng: np.random.Generator):
    order = np.arange(len(examples))
    rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        chosen = order[start : start + batch_size]
        yield [examples[int(i)] for i in chosen]


def estimate_specific_mi_neural(
    examples: list[Example],
    n_pois: int,
    n_users: int,
    *,
    d_model: int = 32,
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-3,
    neg_samples: int = 64,
    full_softmax_limit: int = 4096,
    eval_fraction: float = 0.2,
    max_len: int = 50,
    seed: int = 0,
    device: torch.device | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Fit subset heads on a trail hold-in and read specific MI on the hold-out."""
    device = device or torch.device("cpu")
    if not examples:
        zeros = {subset_key(s): np.zeros(n_pois) for s in ALL_SUBSETS}
        return zeros, np.zeros(n_pois)

    fit, ev = _split_fit_eval(examples, eval_fraction, seed)
    stats = _coord_stats(fit + ev)
    torch.manual_seed(seed)
    model = SubsetMINet(n_users=n_users, n_pois=n_pois, d_model=d_model).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    keys = [subset_key(s) for s in ALL_SUBSETS]
    use_full = n_pois <= full_softmax_limit

    model.train()
    for _ in range(epochs):
        for minibatch in _iter_minibatches(fit, batch_size, rng):
            batch = _batch_examples(minibatch, max_len, device, stats)
            pooled = model.encode_sources(batch)
            loss = torch.zeros((), device=device)
            targets = batch["target"]
            for key in keys:
                hidden = model.subset_hidden(pooled, key)
                if use_full:
                    weight = model.out_emb.weight
                    logits = hidden @ weight.T
                    logits = logits.clone()
                    logits[:, 0] = torch.finfo(logits.dtype).min
                    loss = loss + F.cross_entropy(logits, targets)
                else:
                    neg = torch.randint(1, n_pois, (targets.size(0), neg_samples), device=device)
                    weight = model.out_emb.weight
                    pos = (hidden * weight[targets]).sum(-1, keepdim=True)
                    neg_scores = torch.einsum("bd,bsd->bs", hidden, weight[neg])
                    logits = torch.cat([pos, neg_scores], dim=-1)
                    gold = torch.zeros(targets.size(0), dtype=torch.long, device=device)
                    loss = loss + F.cross_entropy(logits, gold)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

    model.eval()
    prior = marginal_prior(np.array([ex.target for ex in fit], dtype=np.int64), n_pois)
    log_prior = np.log(np.clip(prior, 1e-12, None))
    collected: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    label_parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(ev), batch_size):
            minibatch = ev[start : start + batch_size]
            batch = _batch_examples(minibatch, max_len, device, stats)
            pooled = model.encode_sources(batch)
            labels = batch["target"].detach().cpu().numpy()
            label_parts.append(labels)
            log_prior_t = torch.tensor(log_prior, device=device, dtype=torch.float32)
            for key in keys:
                hidden = model.subset_hidden(pooled, key)
                log_p = model.log_prob_true(hidden, batch["target"])
                ratio = (log_p - log_prior_t[batch["target"]]).detach().cpu().numpy()
                collected[key].append(ratio)
    labels = np.concatenate(label_parts) if label_parts else np.zeros(0, dtype=np.int64)
    specific = {
        key: aggregate_specific_mi(np.concatenate(parts), labels, n_pois)
        for key, parts in collected.items()
    }
    return specific, observation_counts(labels, n_pois)

