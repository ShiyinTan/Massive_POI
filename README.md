# Massive_POI

Analysis and experiment notes on [Massive-STEPS](https://github.com/cruiseresearchgroup/Massive-STEPS).

## Docs

- [Massive-STEPS analysis](docs/massive-steps-analysis/) — dataset + baseline Acc tables
- [STHGCN bias analysis](docs/sthgcn-bias-analysis.md) — CPU Acc vs paper; popularity overlap
- [STHGCN ablation flags](docs/sthgcn-ablation.md) — CLI switches for user embed / hypergraph / graph scope / unseen prefix

## Code

- [`Spatio-Temporal-Hypergraph-Model/`](Spatio-Temporal-Hypergraph-Model/) — Massive-STEPS STHGCN fork with ablation argparse (see `docs/sthgcn-ablation.md`); `data` → `../Massive-STEPS/data`
