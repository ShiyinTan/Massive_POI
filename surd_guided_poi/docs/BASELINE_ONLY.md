# Baseline-only runs (SURD off)

Goal: match Massive-STEPS Acc@1 protocol for Flashback / Graph-Flashback
**without** SURD auxiliary loss (`train.use_surd: false`, `λ_U=λ_S=0`).

For **STHGCN**, run the original Massive-STEPS package directly
(`Massive-STEPS/Spatio-Temporal-Hypergraph-Model/`); it is not wrapped in
`surd_guided_poi`.

## Shared protocol (Flashback / Graph-Flashback)

| Axis | Setting |
|------|---------|
| SURD | off |
| History | same-split trail prefix before target |
| Train samples | all prefixes |
| Eval samples | **last POI only** |
| Cold user/POI | **keep** (`drop_unseen_user_poi: false`) |
| Batch type | `NextPOIBatch` |

`Acc@1 ≡ recall@1` (single-label next-POI).

## Commands

From `surd_guided_poi/`:

```bash
DATA=outputs/data_cache/beijing   # or HF / Massive-STEPS city CSV dir

# Flashback — Massive-STEPS Beijing Acc@1 target 0.278
python scripts/train.py \
  --config configs/base.yaml configs/dataset_beijing.yaml configs/model_flashback_aligned.yaml \
  --data_dir "$DATA" --output_dir outputs/beijing_flashback_baseline

# Graph-Flashback — target 0.304 (needs KGE pkls for full parity)
python scripts/train.py \
  --config configs/base.yaml configs/dataset_beijing.yaml configs/model_graph_flashback.yaml \
  --data_dir "$DATA" --output_dir outputs/beijing_graph_flashback_baseline
```

## Turn SURD back on

Flashback / Graph-Flashback: `configs/model_surd.yaml` (or `train.use_surd: true`).
