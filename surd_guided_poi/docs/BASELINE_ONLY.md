# Baseline-only runs (SURD off)

Goal: match Massive-STEPS Acc@1 protocol for Flashback / Graph-Flashback / STHGCN
**without** SURD auxiliary loss (`train.use_surd: false`, `λ_U=λ_S=0`).

## Shared protocol

| Axis | Flashback / Graph-Flashback | STHGCN (this package) |
|------|-----------------------------|------------------------|
| SURD | off | off |
| History | same-split trail prefix before target | same |
| Train samples | all prefixes | all prefixes |
| Eval samples | **last POI only** (`eval.use_all_prefixes: false`) | last-only |
| Cold user/POI | **keep** (`drop_unseen_user_poi: false`) | **drop** (`true`, STHGCN `remove_unseen_user_poi`) |

`Acc@1 ≡ recall@1` (single-label next-POI).

## Commands

From `surd_guided_poi/` (or repo root with `python surd_guided_poi/scripts/...`):

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
# Optional: set model.transition_pkl / model.interact_pkl to
#   Massive-STEPS/Graph-Flashback/POI_graph/beijing_scheme2_transe_*.pkl
# Without pkls, count-graphs from train are used (Acc may be lower than 0.304).

# STHGCN sequence adapter — cold filter on; Acc ≠ full-hypergraph 0.453
python scripts/train.py \
  --config configs/base.yaml configs/dataset_beijing.yaml configs/model_sthgcn.yaml \
  --data_dir "$DATA" --output_dir outputs/beijing_sthgcn_seq_baseline
```

Evaluate a checkpoint (same model YAML so cold-filter / backbone match):

```bash
python scripts/evaluate.py \
  --config configs/base.yaml configs/dataset_beijing.yaml configs/model_flashback_aligned.yaml \
  --checkpoint outputs/beijing_flashback_baseline/best.pt \
  --data_dir "$DATA"
```

## Turn SURD back on

Use `configs/model_surd.yaml` (or set `train.use_surd: true`, `lambda_u` / `lambda_s` > 0).
Keep the same `model.backbone` and eval flags.

## Full hypergraph STHGCN

Published Beijing Acc@1 **0.453** is the PyG hypergraph model. Run:

```bash
cd Massive-STEPS/Spatio-Temporal-Hypergraph-Model
# see README / run.py + conf/std_conf/beijing.yml
```

This package’s `backbone: sthgcn` is a SequentialTransformer-style adapter for
`NextPOIBatch`, with STHGCN’s **cold-drop + last-only** Acc population.
