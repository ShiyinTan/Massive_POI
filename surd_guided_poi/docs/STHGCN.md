# Full hypergraph STHGCN (`model.backbone: sthgcn`)

This is the **Massive-STEPS** STHGCN (PyG NeighborSampler + hypergraph convolutions),
not a sequence Transformer on `NextPOIBatch`.

## Dependencies

Install extras (match your torch / CUDA build):

```bash
cd surd_guided_poi
pip install -r requirements-sthgcn.txt
# Then install torch-scatter / torch-sparse / torch-cluster from the PyG wheel index
# for your torch version — see Massive-STEPS/Spatio-Temporal-Hypergraph-Model/requirements.txt
```

Fail-fast: importing / training with `backbone=sthgcn` raises `ImportError` if
`torch_geometric` / `torch_sparse` / `torch_scatter` are missing.

## Run (SURD off)

```bash
python scripts/train.py \
  --config configs/base.yaml configs/dataset_beijing.yaml configs/model_sthgcn.yaml \
  --data_dir /path/to/Beijing \
  --output_dir outputs/beijing_sthgcn_hypergraph
```

Flow:

1. Materialize official CSVs into `Massive-STEPS/Spatio-Temporal-Hypergraph-Model/data/{city}/`
2. Upstream `preprocess` (cold user/POI drop, last-only val/test, hypergraph `.pt`)
3. `LBSNDataset` + `NeighborSampler` + `model.sthgcn.STHGCN`
4. Train/eval via `training/sthgcn_runner.py` (not `AlternatingSURDTrainer`)

Config: `model.sthgcn_conf: std_conf/beijing.yml` (upstream). Optional
`model.sthgcn_sizes: "50-30"` for smaller neighbor hops on CPU.

## Deprecated: `sthgcn_seq`

`model.backbone: sthgcn_seq` is a **local** NextPOIBatch Transformer kept only for
ablations. It is **not** STHGCN and must not be reported as a Massive-STEPS baseline.

## SURD

Not supported on the hypergraph path yet (batch type mismatch). Keep
`train.use_surd: false`.
