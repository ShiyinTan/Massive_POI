# Massive-STEPS dataset analysis

Analysis notes, figures, and companion CSVs for the [Massive-STEPS](https://github.com/cruiseresearchgroup/Massive-STEPS) POI check-in dataset, developed for Massive_POI.

## Reports

| Report | Description |
|--------|-------------|
| [massive-steps-dataset-analysis.md](massive-steps-dataset-analysis.md) | Dataset profile, cited city scale, measured distributions |
| [massive-steps-research-angles.md](massive-steps-research-angles.md) | Research questions, baselines, spatial/OD figures |
| [massive-steps-top-venue-topics.md](massive-steps-top-venue-topics.md) | Cheap baselines, weekend metrics, split-stability topics |
| [massive-steps-temporal-baselines.md](massive-steps-temporal-baselines.md) | Temporal baselines and trail-duration analysis |
| [massive-steps-map-overlays.md](massive-steps-map-overlays.md) | Basemap flow overlays (Tokyo / Melbourne / Beijing / Shanghai) |
| [massive-steps-minimal-maps.md](massive-steps-minimal-maps.md) | Minimal grey basemap bubble and flow maps |

## Artifact directories

- `massive-steps-figures/` — dataset-profile plots
- `massive-steps-research-figures/` — research evidence and city OD / density figures
- `massive-steps-split-stability/` — Acc@1 by split protocol
- `massive-steps-temporal-baselines/` — temporal Acc@1 and trail duration
- `massive-steps-map-overlays/` — street-basemap flow overlays + CSVs
- `massive-steps-minimal-maps/` — minimal basemap bubbles/flows + CSVs

Companion CSVs for the research reports live alongside the markdown in this directory (`massive-steps-research-*.csv`, `massive-steps-measured-summary.csv`, `massive-steps-cited-city-table.csv`).
