# SURD 引导的下一 POI 推荐：代码理解指南（入门版）

这份指南面向**不一定熟悉深度机器学习**的读者，说明本目录如何：

1. **对齐 Massive-STEPS 官方数据 / 切分 / 预处理约定**  
2. **用可切换的 Massive-STEPS 推荐骨干**（`flashback` / `graph_flashback` / `sthgcn`）  
3. **在骨干之上挂 SURD 指导训练层**

完整数学规格：[`docs/SURD_Guided_Next_POI_Recommendation_Implementation_Spec.md`](docs/SURD_Guided_Next_POI_Recommendation_Implementation_Spec.md)

> SURD 结果是观测 check-in 上的 **predictive information structure（预测信息结构）**，不要直接说成因果效应。

---

## 1. 和上一版（from-scratch）比，改了什么？

| 项目 | 上一版（已废弃） | 当前版（对齐后） |
|------|------------------|------------------|
| 数据切分 | 自定义按时间戳全局切 train/val/test | **Massive-STEPS 官方** `{city}_checkins_{train,validation,test}.csv`（trail 级、按 user 分层，与 Flashback/GETNext 相同） |
| 加载方式 | 自写单 CSV adapter | `data/massive_steps.py` 解析本地 / `Massive-STEPS/data/` / HuggingFace；坐标缺失时用城市中心填充（同 Flashback） |
| 推荐骨干 | 自研 History/Time/Space + concat MLP | **可配置** `model.backbone`：Flashback / Graph-Flashback / STHGCN-seq |
| SURD | 挂在自研 encoder 上 | 仍用 Q1/Q2/Q3 子集头估 Specific MI / SURD；**推荐 logits 来自所选骨干**；Q1 可共享骨干 POI embedding |

---

## 2. 文件夹一眼看懂

```text
surd_guided_poi/
├── CODE_GUIDE.md / README.md
├── docs/          ← 实现规格
├── configs/       ← 城市与 backbone/SURD 超参（含 model_graph_flashback / model_sthgcn）
├── data/
│   ├── massive_steps.py   ← 官方 split 路径 / HF 下载 / Flashback·Graph-Flashback 路径入口
│   ├── preprocessing.py   ← 在官方 split 内构造 next-POI 样本
│   ├── splits.py          ← 仅在 official-train 内再切 MI 估计 holdout
│   └── dataset.py         ← batch（含骨干需要的 unix 时间与坐标）
├── models/
│   ├── backbone_factory.py         ← model.backbone 工厂
│   ├── backbone_flashback.py       ← Massive-STEPS/Flashback
│   ├── backbone_graph_flashback.py ← Massive-STEPS/Graph-Flashback
│   ├── backbone_sthgcn.py          ← STHGCN 序列 Transformer 适配
│   ├── graph_builders.py           ← 计数转移图 / 用户–POI 交互图
│   ├── full_model.py               ← 骨干推荐 + SURD 子集头
│   ├── encoders.py                 ← Q1/Q2/Q3（给 SURD，不是主推荐头）
│   └── subset_estimators.py        ← 7 个非空子集概率头
├── information/   ← Specific MI / SURD assign / 权重 / 校准
├── losses/        ← L_rec + λ_U L_U + λ_S L_S
├── training/      ← 交替 Phase A/B
├── evaluation/
├── scripts/       ← train / evaluate / estimate_surd
└── tests/
```

---

## 3. 数据：用了哪些 Massive-STEPS API / 约定？

### 3.1 官方 split 文件（必须）

与 `Massive-STEPS/Flashback/run_train.sh`、`GETNext/run_train.sh` 一致：

```text
{city}_checkins_train.csv
{city}_checkins_validation.csv
{city}_checkins_test.csv
```

切分来源：`Massive-STEPS/src/create_next_poi_dataset.py`  
（trail 级约 7:1:2，并对 `user_id` stratify）。

解析入口：`data/massive_steps.py`

- `resolve_official_splits(city, ...)`
- `load_official_split_frames(city, ...)`
- 可选：`load_flashback_poi_dataloader(...)` 直接构造 Flashback 的 `PoiDataloader`

CSV 搜索顺序：`data.data_dir` → `Massive-STEPS/data/{city}/` → HF 缓存下载（`CRUISEResearchGroup/Massive-STEPS-*`）。

**骨干代码位置是固定的**（不做路径探测）：

| `model.backbone` | 仓库路径 | 适配器 |
|------------------|----------|--------|
| `flashback` | `Massive-STEPS/Flashback/` | `models/backbone_flashback.py` |
| `graph_flashback` | `Massive-STEPS/Graph-Flashback/` | `models/backbone_graph_flashback.py`（`import_graph_flashback_network()` 避免与 Flashback 的 `network.py` 冲突） |
| `sthgcn` | 序列适配器（见下） | `models/backbone_sthgcn.py` |

Flashback：先 `add_flashback_dir_to_sys_path()`，再 `from network import Flashback`。  
Graph-Flashback：可选 `model.transition_pkl` / `model.interact_pkl`；缺省时用 official-train 样本计数构图（`graph_builders.py`）。  
STHGCN：**完整超图** STHGCN 依赖 `torch_geometric` + NeighborSampler，与本目录 `NextPOIBatch` 不直接兼容；此处挂的是同仓库 **SequentialTransformer** 思路的序列 Transformer 基线。完整超图请跑 `Massive-STEPS/Spatio-Temporal-Hypergraph-Model/run.py`。

### 3.2 预处理约定（对齐 Flashback）

在 `load_official_split_frames` / `build_samples_from_split` 中：

- 字段：`trail_id, user_id, venue_id, timestamp, latitude, longitude, venue_category, ...`
- 缺失坐标：`latitude/longitude ← venue_city_latitude/longitude`（同 `Flashback/dataloader.py`）
- 轨迹：`groupby([user_id, trail_id])`，长度 ≥ 2（由 `min_history_len+1` 决定，默认 2）
- 词表：train+val+test 上 **transductive** 构建 `poi2id` / `user2id`（同 Flashback）
- 时间：为 Flashback 核函数提供 **unix 秒**（`timestamp` → `int64 ns // 10**9`）

**Eval / val / test 的 history 从哪来（无未来泄漏）**

- 每个 split **单独**调用 `build_samples_from_split(frames[split], …)`。
- 一条样本的序列历史 = **该 split CSV 内、同一 trail、目标时刻之前的前缀**；不会把 train+val+test 拼成 history。
- Flashback / Q1–Q3 只吃这条样本自己的 history 张量；`user_features` 也只看 `timestamp < target`（默认同一 split）。
- Transductive 只影响「词表里有没有这个 POI id」，不把别的 split 的打卡塞进序列。

**与 Massive-STEPS Flashback Acc@1 对齐**

- 表中 Beijing **Flashback = 0.278**，**Graph-Flashback = 0.304**（不要把 0.304 当成纯 Flashback）。
- Acc@1 ≡ Recall@1（单标签 next-POI）；评测默认 `eval.use_all_prefixes: false`（**每条 trail 只评最后一站**，同 Flashback `evaluation.py`）。
- 推荐默认配置：`configs/model_flashback_aligned.yaml`（`hidden=10`、`rnn`、`lr=0.01`、`use_surd=false`）。要挂 SURD 时再用 `model_surd.yaml` 并设 `train.use_surd=true`。

**关于「序列长度 < 20」：** 本目录默认路径 **不会** 按用户总打卡数 <20 过滤。  
`configs/base.yaml` 里的 `max_history_len: 50` 只是把每条样本的 **history 截断到最近 50 步**（比上游 Flashback CLI 默认 `sequence_length=20` 更长，仍 Flashback-friendly）；`min_history_len: 1` 只要求每条 trail 至少能切出「1 个历史 + 1 个目标」。  
上游 Flashback CLI（`Massive-STEPS/Flashback/setting.py`）会设 `sequence_length=20` 且 `min_checkins=sequence_length+1`，那是他们自己的训练入口；我们的 `train.py` 走的是 `load_official_split_frames`，可选 helper `load_flashback_poi_dataloader(..., min_checkins=0)` 默认也不开该过滤。

### 3.3 SURD 估计用的 holdout

**不再**做跨全库的自定义时间切分。  
只在 **official train** 样本里再切一块 estimation holdout（`data/splits.py::carve_estimator_holdout`），避免用同一批样本既拟合子集头又估 MI。

---

## 4. 骨干：`model.backbone` 怎么切？

配置项（见 `configs/base.yaml`）：

```yaml
model:
  backbone: flashback   # 或 graph_flashback / sthgcn
```

预制覆盖：`configs/model_flashback_aligned.yaml`、`model_graph_flashback.yaml`、`model_sthgcn.yaml`。

统一接口：各适配器 `forward(batch) → (last_logits [B,K], seq_logits, h_next)`；`full_model` / train / eval 只吃 `last_logits`。工厂：`models/backbone_factory.py`。

| 名字 | 行为摘要 |
|------|----------|
| `flashback` | 官方 Flashback RNN + 时空衰减 |
| `graph_flashback` | 官方 Graph-Flashback（转移图 GCN + 用户–POI 交互）；无 pkl 则用 train 计数图 |
| `sthgcn` | POI/时间 embedding + TransformerEncoder + 线性头（序列基线） |

推理路径：`SURDGuidedPOIModel.recommend` → **只走所选骨干**，不查真实标签权重。

**Q2 时间编码：** 不只看目标时刻。历史每一步用日历特征 + 相对目标的 Δt（`[B,L,10]`），经 GRU 得到 `h_hist`；目标 8 维经 MLP 得到 `h_tgt`；再 concat 融合成 Q2 向量。

**Q3 空间编码：** 对称于 Q2。历史每一步用坐标/缺测/步距/时差/相对位置（`[B,L,6]`），经 GRU 得到 `h_seq`；另有 12 维历史汇总统计经 MLP 得到 `h_sum`；再 concat 融合成 Q3 向量。

---

## 5. SURD 如何“包”在骨干外面？

```text
Massive-STEPS official batch
        │
        ├──────────────────────────────► backbone (flashback|graph_flashback|sthgcn)
        │                                         └──► rec_logits  (= L_rec)
        ├─ Q1 HistoryEncoder (可共享骨干 POI emb)
        ├─ Q2 TemporalEncoder (历史时间序列 + 目标时刻 → 融合)
        └─ Q3 SpatialEncoder  (历史空间序列 + 12 维汇总 → 融合)
                 │
                 └─ 7 个子集概率头 → Specific MI → SURD assign
                         │
                         └─ 冻结的 U/S 权重 → L_U / L_S（辅助监督）
```

总损失（规格第 6.4 节）：

\[
\mathcal L = \mathcal L_{\mathrm{rec}} + \lambda_U \mathcal L_U + \lambda_S \mathcal L_S
\]

- `\mathcal L_rec`：骨干对具体 POI ID 的交叉熵  
- `\mathcal L_U` / `\mathcal L_S`：按真实目标 POI 查表的 SURD 权重，加权子集头 CE  
- SURD 数值估计 **不反传**（权重 `detach`）  
- 组合头 CE **不等于**“纯 synergy”，只是被 synergy 权重调节的辅助损失  

交替训练：`training/alternating.py`（warm-up → Phase A 更新 SURD → Phase B 训骨干+侧 encoder）。

---

## 6. 怎么跑？

```bash
cd surd_guided_poi
pip install -r requirements.txt

# 使用 HF 上的官方 Beijing split（或 --data_dir 指向本地官方 CSV 目录）
python scripts/train.py \
  --config configs/base.yaml configs/dataset_beijing.yaml configs/model_surd.yaml \
  --data_dir /tmp/massive-steps-data/Beijing

# 换骨干：model_flashback_aligned / model_graph_flashback / model_sthgcn
python scripts/train.py \
  --config configs/base.yaml configs/dataset_beijing.yaml configs/model_graph_flashback.yaml \
  --data_dir /tmp/massive-steps-data/Beijing

python scripts/evaluate.py --checkpoint outputs/beijing/best.pt \
  --data_dir /tmp/massive-steps-data/Beijing \
  --config configs/base.yaml configs/dataset_beijing.yaml configs/model_graph_flashback.yaml

# 仓库根目录跑测试
python -m pytest surd_guided_poi/tests -q
```

本地目录里需要有：

```text
beijing_checkins_train.csv
beijing_checkins_validation.csv
beijing_checkins_test.csv
```

---

## 7. 建议阅读顺序

1. `data/massive_steps.py` → 官方 split 从哪来  
2. `models/backbone_factory.py` + 对应 `backbone_*.py` → 骨干本身  
3. `models/full_model.py` → SURD 如何包住骨干  
4. `information/specific_mi.py` → `surd.py` → `weights.py`  
5. `training/alternating.py` → `scripts/train.py`

---

## 8. 第一版范围

**已对齐 / 已实现**

- Massive-STEPS 官方 split + Flashback 坐标/轨迹约定  
- 可切换骨干：`flashback` / `graph_flashback` / `sthgcn`（序列适配）+ SURD U/S 指导  
- 中文指南与规格入库  

**仍延后**

- GETNext；完整超图 STHGCN（PyG）直接吃 `NextPOIBatch`  
- Q4 进入 SURD、candidate-conditioned 推理、R 正则、大规模消融、inductive cold-start  
