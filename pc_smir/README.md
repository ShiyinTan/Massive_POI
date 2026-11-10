# PC-SMIR — POI-Conditioned, SURD-Guided Multi-order Interaction Routing Network

Self-contained next-POI recommendation package implementing the project spec:

**POI-conditioned, SURD-guided Multi-order Interaction Routing Network (PC-SMIR)**.

This folder does **not** depend on `surd_guided_poi/` or other experiment packages. The only
external data contract is **Massive-STEPS official trail splits**
(`{city}_checkins_{train,validation,test}.csv`), matching
`Massive-STEPS/src/create_next_poi_dataset.py`.

## Layout

```
pc_smir/
  configs/           # YAML configs + ablations
  data/              # Official Massive-STEPS CSVs (downloaded)
  pc_smir/
    data/            # Split loading, vocab, dataset
    features/        # Q1–Q4 derived variables
    models/          # Encoders, experts, router, PCSMIR
    surd/            # mmi_pid_discrete_v1 estimator + profiles
    training/        # Trainer
    evaluation/      # Recall@K / NDCG@K / MRR
    pipeline.py      # prepare → loaders → model
  scripts/           # prepare_data / train / evaluate / run_pipeline
  tests/
```

## Data splits (Massive-STEPS official)

Splits are **trail-level**, user-stratified, `random_state=42` (upstream). This package
never re-splits randomly. Loader: `pc_smir.data.splits.load_official_splits`.

Prediction examples: within each trail, history = all but last check-in; target = last POI.
**Prediction timestamp** = last observed (history) check-in time — not the target timestamp.

## Derived variables / preprocessing

| Source | Derived inputs |
|--------|----------------|
| Q1 Personal | Historical POI/category IDs, log time gaps; KMeans preference state for SURD |
| Q2 Temporal | Cyclical hour/DoW, weekend flag; hour×day-type bins for SURD |
| Q3 Spatial | Normalized coords, spatial cells (train-fitted resolution), displacements; KMeans spatial state |
| Q4 Stable user | Prefix visit count, distinct POIs, category entropy, median move, activity radius, train-only popularity affinity |
| Candidate | ID embedding, distance from last known location, train-only popularity, category, coords |
| SURD profile | Per-POI unique/synergy atoms (`mmi_pid_discrete_v1`), bootstrap SE, shrinkage, robust scaling, temporal cross-fit |

Vocabularies, bin/cluster fits, popularity, and SURD profiles use **training data only**.

## Install

```bash
cd pc_smir
pip install -r requirements.txt
```

## Run (Beijing smoke)

```bash
# 1) Download official splits (if needed) + build examples/SURD artifacts
python scripts/prepare_data.py --config configs/smoke.yaml --force

# 2) Train + evaluate once on test
python scripts/train.py --config configs/smoke.yaml

# Or one-shot:
python scripts/run_pipeline.py configs/smoke.yaml
```

Fuller Beijing run:

```bash
python scripts/train.py --config configs/base.yaml configs/dataset_beijing.yaml configs/model_pcsmir.yaml
```

Ablations:

```bash
python scripts/train.py --config configs/base.yaml configs/dataset_beijing.yaml configs/ablation_uniform.yaml
python scripts/train.py --config configs/base.yaml configs/dataset_beijing.yaml configs/ablation_no_surd.yaml
```

Evaluate a checkpoint:

```bash
python scripts/evaluate.py --config configs/smoke.yaml --checkpoint outputs/beijing_smoke/best.pt --split test
```

## Tests

```bash
cd pc_smir
python -m pytest tests -q
```

## SURD estimator

Named estimator: **`mmi_pid_discrete_v1`** (MMI-style discrete PID with Dirichlet smoothing).

- Unique / synergy / redundancy atoms are estimated offline per training POI.
- **Redundancy is never routed as synergy** (diagnostic storage only by default).
- Training profiles are **temporally cross-fitted**; val/test use full-train profiles.
- Synthetic unit tests cover unique / XOR-synergy / duplicate-redundancy patterns.

## Notes

- Default training uses **sampled softmax** (`train_neg_samples`) for speed; validation/test
  score the full training catalog.
- OOV / unseen test POIs map to an OOV id and are excluded from ranking targets by default.
