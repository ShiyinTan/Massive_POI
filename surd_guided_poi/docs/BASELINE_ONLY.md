# Baseline-only runs (SURD off)

Goal: match Massive-STEPS Acc@1 protocol for Flashback / Graph-Flashback / STHGCN
**without** SURD auxiliary loss (`train.use_surd: false`, `λ_U=λ_S=0`).

## Shared protocol

| Axis | Flashback / Graph-Flashback | STHGCN (full hypergraph) |
|------|-----------------------------|---------------------------|
| SURD | off | off (required) |
| History | same-split trail prefix before target | upstream time-filtered traj checkins |
| Train samples | all prefixes | non-first checkins (upstream) |
| Eval samples | **last POI only** | upstream `only_keep_last` |
| Cold user/POI | **keep** | **drop** (`remove_unseen_user_poi` in preprocess) |
| Batch type | `NextPOIBatch` | PyG NeighborSampler |

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

# Full hypergraph STHGCN — target 0.453 (needs requirements-sthgcn.txt / PyG)
pip install -r requirements-sthgcn.txt   # + torch-sparse/scatter wheels
python scripts/train.py \
  --config configs/base.yaml configs/dataset_beijing.yaml configs/model_sthgcn.yaml \
  --data_dir "$DATA" --output_dir outputs/beijing_sthgcn_hypergraph
```

Details: [`STHGCN.md`](STHGCN.md).

## Deprecated

`model.backbone: sthgcn_seq` — local NextPOIBatch Transformer; **not** Massive-STEPS STHGCN.
Do not use for Acc reporting.

## Turn SURD back on

Flashback / Graph-Flashback: `configs/model_surd.yaml` (or `train.use_surd: true`).
Hypergraph STHGCN: SURD not supported yet.
