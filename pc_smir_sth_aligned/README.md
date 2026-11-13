# PC-SMIR STH-aligned (`pc_smir_sth_aligned/`)

Uplift package aimed at closing the Beijing Acc@1 gap vs Massive-STEPS **STHGCN (0.453)**.

This folder is a **sibling** of `pc_smir/`. It does **not** modify `pc_smir/`.

Diagnosis (store doc): see project `docs/sthgen-gap-analysis.md` — the gap is mostly missing user–POI personalization and thinner training, **not** Acc@1 vs Recall@1 or sampled eval candidates.

## What changed vs `pc_smir/`

| Lever | Change |
|-------|--------|
| Metrics | `acc@k` ≡ `recall@k`; early-stop / select on **`recall@1`** |
| Train candidates | Default **full train-catalog CE** (`train_neg_samples: null`) |
| Train examples | **`expand_train_prefixes: true`** → all intermediate next-POI steps (~644 Beijing) |
| Eval | Still **last-of-trail**, full catalog, OOV targets excluded (~88 Beijing) |
| Model | Explicit **user × POI** + **last-POI → next** scores fused with SURD experts; `d=128` |

## Install

```bash
cd pc_smir_sth_aligned
pip install -r requirements.txt
```

Beijing CSVs: symlink `data/beijing` → `../pc_smir/data/beijing` if already downloaded, or set `data.download: true`.

## Run (Beijing)

### Smoke

```bash
cd pc_smir_sth_aligned
export CUDA_VISIBLE_DEVICES=0   # optional

python scripts/prepare_data.py --config configs/smoke.yaml --force
python scripts/train.py --config configs/smoke.yaml
```

### Full Beijing (target protocol)

```bash
cd pc_smir_sth_aligned
export CUDA_VISIBLE_DEVICES=0

python scripts/prepare_data.py \
  --config configs/base.yaml configs/dataset_beijing.yaml configs/model_pcsmir.yaml \
  --force

python scripts/train.py \
  --config configs/base.yaml configs/dataset_beijing.yaml configs/model_pcsmir.yaml
```

One-shot:

```bash
python scripts/run_pipeline.py \
  configs/base.yaml configs/dataset_beijing.yaml configs/model_pcsmir.yaml
```

Evaluate:

```bash
python scripts/evaluate.py \
  --config configs/base.yaml configs/dataset_beijing.yaml configs/model_pcsmir.yaml \
  --checkpoint outputs/beijing/best.pt \
  --split test
```

Personalization-only ablation (no SURD expert mixture in the score):

```bash
python scripts/train.py \
  --config configs/base.yaml configs/dataset_beijing.yaml configs/model_pcsmir.yaml \
             configs/ablation_personalization_only.yaml
```

## Expected metrics

Report **`acc@1` / `recall@1`** from `outputs/beijing/test_metrics.json`.

Comparable protocol to Massive-STEPS STHGCN Beijing cite: last-of-trail, train-seen POI targets, full train POI catalog Softmax ranking.

Success band: **≳ UserMostPop (~0.40 filtered) → toward STHGCN 0.453**.

## Tests

```bash
cd pc_smir_sth_aligned
python -m pytest tests -q
```
