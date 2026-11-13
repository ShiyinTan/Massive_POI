# STHGCN on Massive-STEPS — Acc vs paper & popularity bias

**Source:** [Massive-STEPS](https://github.com/cruiseresearchgroup/Massive-STEPS) · [STHGCN fork](https://github.com/w11wo/Spatio-Temporal-Hypergraph-Model) · [arXiv:2505.11239](https://arxiv.org/abs/2505.11239)  
**Cities (CPU-feasible):** Tokyo · Beijing · Shanghai · Melbourne  
**Data:** full Hugging Face official splits (`train`/`validation`/`test`; `covers_full_csv=True`)  
**Device:** CPU (`gpu: -1`); paper `std_conf/{city}.yml` (seed `54607333`, 20 epochs)  
**Artifacts:** [`sthgcn-bias/`](sthgcn-bias/)

---

## 1. Setup

| item | value |
|------|-------|
| HF repos | `CRUISEResearchGroup/Massive-STEPS-{Tokyo,Beijing,Shanghai,Melbourne}` |
| Code | Massive-STEPS submodule `Spatio-Temporal-Hypergraph-Model@758c50b` |
| Config | `conf/std_conf/{city}.yml` with `gpu: -1` only |
| Metric | Recall@k ≡ Acc@k (full POI candidate set after LabelEncode on train) |
| Eval set | STHGCN preprocessed `test_sample.csv` (official test trails, last check-in only, unseen user/POI removed) |

Provenance: [`sthgcn-bias/hf_download_provenance.csv`](sthgcn-bias/hf_download_provenance.csv).

**Note on spatial features.** The Massive-STEPS STHGCN adapter maps `venue_city_latitude/longitude` → model lat/lon (city centroid), so pairwise distances are near-zero. Temporal / collaborative hypergraph signal carries the model.

**Pandas patch.** Newer pandas broke `haversine` row apply (`x[0]` → KeyError); fixed with `.iloc` for this CPU rerun (same math).

---

## 2. Acc vs paper

| city | n_test (STHGCN) | paper Acc@1 | **measured Acc@1** | Δ | Acc@5 | Acc@10 |
|------|----------------:|------------:|-------------------:|--:|------:|-------:|
| Beijing | 75 | 0.453 | **0.453** | +0.000 | 0.653 | 0.747 |
| Tokyo | 907 | 0.250 | **0.254** | +0.004 | 0.445 | 0.506 |
| Shanghai | 493 | 0.193 | **0.205** | +0.012 | 0.331 | 0.381 |
| Melbourne | 1151 | 0.168 | **0.168** | −0.000 | 0.311 | 0.370 |

CSV: [`sthgcn-bias/acc_and_bias_table.csv`](sthgcn-bias/acc_and_bias_table.csv) · per-city metrics JSON under [`sthgcn-bias/`](sthgcn-bias/).

![Acc vs paper / baselines](sthgcn-bias/acc_vs_paper_baselines.png)

**图注：** Measured STHGCN Acc@1 matches the paper table within ~1pp on all four cities (Shanghai +1.2pp largest gap). On the **same** STHGCN-filtered test, UserMostPop is competitive with STHGCN (Beijing: **tied** at 0.453); Markov1 is weaker but non-trivial; global MostPop is weak outside Beijing.

---

## 3. Bias vs MostPop / UserMostPop / Markov1

Baselines fit on official **train** check-ins (raw `venue_id`); scored on the same STHGCN test rows (LabelEncode inverse). Markov1 conditions on the **second-to-last** venue in the trail (target = last).

### 3.1 How often is STHGCN top-1 a popularity baseline?

| city | = MostPop | = UserMostPop | = Markov1 | = any of three |
|------|----------:|--------------:|----------:|---------------:|
| Beijing | 0.227 | **0.520** | 0.480 | 0.733 |
| Tokyo | 0.079 | **0.333** | 0.238 | 0.492 |
| Shanghai | 0.051 | **0.316** | 0.181 | 0.442 |
| Melbourne | 0.026 | **0.235** | 0.151 | 0.349 |

![Top-1 overlap](sthgcn-bias/top1_overlap_baselines.png)

**读数：** STHGCN top-1 coincides with **UserMostPop** most often (23–52%), then Markov1 (15–48%), rarely global MostPop outside Beijing. On Beijing, **¾** of predictions equal at least one of the three cheap baselines.

### 3.2 Conditional Acc@1 when the target is / isn’t that baseline

| city | Acc \| target=UMP | Acc \| target≠UMP | Acc \| target=Markov1 | Acc \| target≠Markov1 |
|------|------------------:|------------------:|----------------------:|----------------------:|
| Beijing | **0.765** | 0.195 | **0.923** | 0.204 |
| Tokyo | **0.673** | 0.133 | **0.733** | 0.173 |
| Shanghai | **0.759** | 0.099 | **0.729** | 0.148 |
| Melbourne | **0.634** | 0.090 | **0.702** | 0.115 |

![Conditional Acc](sthgcn-bias/conditional_acc.png)

**读数：** When the true next POI **is** the user’s historical mode (UMP) or the 1-step Markov mode, STHGCN Acc@1 is 0.63–0.92. When it is **not**, Acc collapses to ~0.09–0.20. The model’s headline Acc is therefore heavily driven by recovering **repeat / transition-popular** destinations rather than long-tail novelty.

### 3.3 Baseline Acc on the STHGCN test (for context)

| city | MostPop | UserMostPop | Markov1 | STHGCN |
|------|--------:|------------:|--------:|-------:|
| Beijing | 0.213 | **0.453** | 0.347 | **0.453** |
| Tokyo | 0.041 | 0.223 | 0.144 | **0.254** |
| Shanghai | 0.032 | 0.160 | 0.097 | **0.205** |
| Melbourne | 0.016 | 0.142 | 0.090 | **0.168** |

Lift over UMP (Acc_STHGCN − Acc_UMP): Beijing **0.000**, Tokyo **+0.031**, Shanghai **+0.045**, Melbourne **+0.025**.

Per-sample dumps: `sthgcn-bias/{city}_per_sample_bias.csv`.

---

## 4. Test-time context：不是「只用 train」

对照 Massive-STEPS STHGCN 适配器（`preprocess_std` → `sample.csv` → `generate_hypergraph_from_file` → `NeighborSampler`）：

1. **测试样本过滤（相对完整 official test）**  
   - `ignore_first`：每条 trail 的**首个** check-in 标为 `ignore`（无历史）。  
   - `only_keep_last`：val/test 只保留 trail **最后一个** check-in 作为评测样本。  
   - `remove_unseen_user_poi`：若该样本的 **User 或 目标 POI 未出现在 train SplitTag 样本中**，则从 val/test 丢弃。  
   - 结果：评测集更「暖」（北京 HF test trails ~115 → STHGCN n_test **75**）。UserMostPop 若在**完整** official test 上算会更吃亏。

2. **图 / 特征：train+val+test 全量建图（非 train-only）**  
   - `preprocess_std` 把 official `*_train/validation/test.csv` **concat** 成一张表；写出的 `sample.csv` = 全量 check-in（含被标 `ignore` 的中间点）。  
   - **LabelEncode** 的 user/POI/category/hour/weekday：**只在 train 上 fit**；未见 ID → padding。Embedding 表尺寸与监督损失来自 train 样本。  
   - **超图 ci2traj / traj2traj** 由 `sample.csv` 生成 → **包含 val 与 test 的 trail 与 check-in**，不是 train-only，也不是严格「预测时刻之前的因果切窗建图」。  
   - traj2traj 边带时间序约束 `source.end ≤ target.start`；采样时再对 query trail 的 ci2traj 做 `edge_t ≤ max_time`，并对 traj2traj 做 `filter_traj2traj_with_leakage`（按当前可见 check-in 重算 Jaccard）。**目标 check-in 本身**因时间掩码不可见；但**更早的 val/其他 test trail** 只要结束时间早于 query，仍可作邻居。

3. **当前 trail 前缀可见，且前缀事件属于 test 划分**  
   - 测试样本的 query = `query_pseudo_session_trajectory_id`（同 trail 的 shift）→ 查询的是**当前 test trail** 这条超边。  
   - `max_time = last_checkin_epoch_time`（目标前一笔的时间）→ 仅保留 `t ≤ max_time` 的 check-in → **目标之前的 trail 前缀**，不含目标 label。  
   - Massive-STEPS 按 **trail** 划分 train/val/test，故该前缀 check-in **属于 official test 文件**，不是 train。UserMostPop **不会**吃到这些 test 前缀（只统计 train 频率）；Markov1 会用到前一站（与前缀末站对齐）。

4. **为何 Acc 显得「强」、相对 UserMostPop 被抬高**  
   - **评测过滤**去掉冷用户/未见目标 POI → 与「全量 official test 上的 UMP」不可直接比；即便对齐到 STHGCN test，北京仍 **STHGCN Acc@1 = UMP = 0.453**。  
   - **图协作**：邻居可含 train **以及** 时间上更早的 val/其他 test trail（全量建图 + 时间序边），UMP 无此信号。  
   - **Test trail 前缀**作为序列上下文进入超图消息传递；UMP 无当前 trail 上下文。短 trail 上这往往强化「重复/转移热门」行为，与已测到的高 UMP/Markov1 top-1 重合一致。  
   - **结论：** 测试时 **不是**「只用 official train」。监督标签与 ID 词表来自 train；**图与当前 trail 前缀上下文来自全量 sample（含 val/test 事件）**，再加时间掩码避免直接看见目标点。

---

## 5. Interpretation

1. **Reproduction holds on CPU.** Four-city Acc@1 tracks the Massive-STEPS paper table; Beijing/Melbourne essentially exact.
2. **Popularity / loyalty bias is large.** One-third to one-half of STHGCN top-1s are identical to UserMostPop; conditional Acc shows the model is strong mainly when the target *is* UMP or Markov1.
3. **Reported STHGCN gains vs classical neural baselines do not imply gains vs UserMostPop.** On Beijing, STHGCN ≡ UMP Acc@1 on the filtered test; on other cities the absolute lift is only 2.5–4.5pp.
4. **Eval + graph protocol inflate the headline.** Unseen user/POI drop + last-only test + **full-split hypergraph** + **test-trail prefix context** make Acc look stronger than a pure train-fit / no-test-context baseline (UMP). Bias numbers above are on the **STHGCN-filtered** test.

---

## 6. More bias probes (designed, not all run here)

Priority probes for follow-up (same official split + dump of STHGCN top-k):

1. **Rank of UMP / Markov1 in STHGCN’s score list** — mean reciprocal rank of the baseline POI; “how far is the popular choice from top-1 when they disagree?”
2. **Disagreement set analysis** — Acc / category / distance on samples where `STHGCN_top1 ≠ UMP_top1`; is the residual lift geographically or categorically structured?
3. **Cold vs warm users** — stratify by `#train check-ins` or whether UMP is defined from ≥k visits; does STHGCN help cold users or only re-rank warm loyalty?
4. **Context length** — Acc and UMP-overlap vs `ctx_len` (trail length − 1). Short trails (majority) may collapse to UMP-like behavior.
5. **Time-conditioned UMP (UMP_DN / UMP_WW)** — overlap of STHGCN with day/night and work/weekend user modes (from MS-10/11); does the GNN rediscover time bins?
6. **Top-k set overlap** — Jaccard(STHGCN@k, UMP@k) / Markov@k for k∈{5,10}; Acc@k conditional on target∈UMP@k.
7. **Novelty / long-tail** — Acc on targets in bottom-quartile train frequency; fraction of top-1 predictions that are globally rare.
8. **Calibration / confidence** — softmax margin or entropy of STHGCN scores when pred=UMP vs when pred≠UMP (overconfidence on loyalty?).
9. **Ablation dump** — with `do_traj2traj=False` / no ST encoder (existing ablation configs), does UMP-overlap rise? Isolates which component encodes popularity.
10. **Cross-city transfer of bias** — train on city A, test B (if vocab allows via category-only); does bias pattern transfer with culture/scale?

Suggested next run order: (1)+(2)+(3) on the existing `test_preds.pt` dumps (no retrain); then (5) using already-computed UMP_DN tables; then (9) if GPU available.

---

## 7. Repro commands

```bash
# data
hf download CRUISEResearchGroup/Massive-STEPS-$City --repo-type dataset --local-dir data_hf/$City
# link into Spatio-Temporal-Hypergraph-Model/data/$city/*_checkins_*.csv
# set gpu: -1 in conf/std_conf/$city.yml
python run.py -f std_conf/$city.yml   # dumps log/.../test_preds.pt + test_metrics.json
# bias
python sthgcn_bias_analysis.py        # → docs/sthgcn-bias/
```

Analysis script (working copy): store `internal/sthgcn-bias-analysis-script.py`.
