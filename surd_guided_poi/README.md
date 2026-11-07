# SURD-Guided Next-POI Recommendation

First-version pipeline for **concrete POI-ID** next-check-in recommendation with
POI-specific Specific Mutual Information and SURD **guidance weights**
(predictive information structure on observational data — not causal claims).

- Beginner-friendly Chinese walkthrough: [`CODE_GUIDE.md`](CODE_GUIDE.md)
- Full implementation spec: [`docs/SURD_Guided_Next_POI_Recommendation_Implementation_Spec.md`](docs/SURD_Guided_Next_POI_Recommendation_Implementation_Spec.md)

## Quick start

```bash
cd surd_guided_poi
pip install -r requirements.txt

# Train on Beijing (downloads Massive-STEPS CSV from HuggingFace if needed)
python scripts/train.py \
  --config configs/base.yaml configs/dataset_beijing.yaml configs/model_surd.yaml

# Evaluate
python scripts/evaluate.py --checkpoint outputs/beijing/best.pt
```

Local CSV override:

```bash
python scripts/train.py \
  --config configs/base.yaml configs/dataset_beijing.yaml configs/model_surd.yaml \
  --checkins_csv /path/to/beijing_checkins.csv
```

## Tests

From the repository root:

```bash
python -m pytest surd_guided_poi/tests -q
```

## Design snapshot (v1)

| Item | Choice |
|------|--------|
| Sources | Q1 history, Q2 time, Q3 spatial (no Q4 in SURD yet) |
| Split | Temporal by target timestamp (train → val → test) |
| SURD role | Periodically refreshed, detached U/S weights for auxiliary CE |
| Inference | Unified recommendation head only |
| Setting | Transductive POI vocabulary |

Data adapters read Massive-STEPS check-in CSVs already used in this repo /
published on HuggingFace (`CRUISEResearchGroup/Massive-STEPS-*`).
