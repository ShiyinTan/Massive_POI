# SURD-Guided Next-POI Recommendation

First-version pipeline that **reuses Massive-STEPS official data splits** and
configurable recommendation backbones from Massive-STEPS
(`flashback` | `graph_flashback` | `sthgcn`), then wraps **POI-specific Specific MI
+ SURD guidance weights** for auxiliary training (predictive information structure
on observational data — not causal claims).

- Beginner Chinese walkthrough: [`CODE_GUIDE.md`](CODE_GUIDE.md)
- Spec: [`docs/SURD_Guided_Next_POI_Recommendation_Implementation_Spec.md`](docs/SURD_Guided_Next_POI_Recommendation_Implementation_Spec.md)

## What is aligned with Massive-STEPS

| Piece | Source of truth in this repo |
|------|------------------------------|
| Split files | `{city}_checkins_{train,validation,test}.csv` (same as Flashback / GETNext) |
| Split construction | `Massive-STEPS/src/create_next_poi_dataset.py` (trail-level, user-stratified) |
| Coord fallback | Missing venue lat/lon ← city center (`Flashback/dataloader.py`) |
| Recommendation backbones | `model.backbone`: `flashback` → `Massive-STEPS/Flashback`; `graph_flashback` → `Massive-STEPS/Graph-Flashback`; `sthgcn` → SequentialTransformer-style adapter (full hypergraph STHGCN needs PyG / upstream `run.py`) |

## Quick start

```bash
cd surd_guided_poi
pip install -r requirements.txt

# Official Beijing splits (local dir or HuggingFace)
python scripts/train.py \
  --config configs/base.yaml configs/dataset_beijing.yaml configs/model_surd.yaml \
  --data_dir /path/to/Beijing   # contains beijing_checkins_{train,validation,test}.csv

# Switch backbone (examples)
python scripts/train.py \
  --config configs/base.yaml configs/dataset_beijing.yaml configs/model_flashback_aligned.yaml \
  --data_dir /path/to/Beijing
python scripts/train.py \
  --config configs/base.yaml configs/dataset_beijing.yaml configs/model_graph_flashback.yaml \
  --data_dir /path/to/Beijing
python scripts/train.py \
  --config configs/base.yaml configs/dataset_beijing.yaml configs/model_sthgcn.yaml \
  --data_dir /path/to/Beijing

python scripts/evaluate.py --checkpoint outputs/beijing/best.pt --data_dir /path/to/Beijing
```

Tests (repo root):

```bash
python -m pytest surd_guided_poi/tests -q
```

## Design snapshot (v1, aligned)

| Item | Choice |
|------|--------|
| Data | Massive-STEPS official train/val/test CSVs |
| Backbone | `model.backbone`: flashback / graph_flashback / sthgcn (shared last-step logits `[B,K]`) |
| SURD sources | Q1 history / Q2 time / Q3 spatial side encoders |
| SURD role | Detached U/S weights for auxiliary CE; no backprop through SURD |
| Inference | Selected backbone only (no SURD lookup) |
