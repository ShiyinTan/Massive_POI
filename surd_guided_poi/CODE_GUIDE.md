# SURD 引导的下一 POI 推荐：代码理解指南（入门版）

这份指南面向**不一定熟悉深度机器学习**的读者。目标是让你能顺着文件夹把整条流水线读通：数据从哪来、三个信息源怎么编码、SURD 权重怎么算出、训练时如何用这些权重、推理时为什么不用真实标签。

更完整的数学与实现规格见：

- [`docs/SURD_Guided_Next_POI_Recommendation_Implementation_Spec.md`](docs/SURD_Guided_Next_POI_Recommendation_Implementation_Spec.md)

> **重要表述**：本项目使用的是观测到的 check-in 数据。SURD 结果应理解为 **predictive information structure（预测信息结构）**，不要直接说成“真实因果效应”。

---

## 1. 一句话理解这个项目在做什么

给定用户到某一时刻为止的历史访问、当前时间、空间上下文，模型要预测**下一次会去哪个具体 POI ID**（不是类别）。

训练时，除了普通推荐损失，还会用 SURD 分解得到的「每个目标 POI 更依赖哪些信息源」作为**指导权重**，加强对应分支的辅助学习。  
推理时只用统一推荐头，**不能**偷看真实答案去查权重。

---

## 2. 文件夹长什么样（按阅读顺序）

```text
surd_guided_poi/
├── CODE_GUIDE.md          ← 你正在读的入门指南
├── README.md              ← 安装与运行命令速查
├── requirements.txt
├── docs/                  ← 实现规格（已随 PR 上传）
├── configs/               ← YAML 配置（数据城市场景、模型、训练超参）
├── data/                  ← 读 Massive-STEPS、时间切分、构造样本
├── models/                ← 三个 encoder + 推荐头 + 7 个子集概率头
├── information/           ← Specific MI、SURD 分解、权重、校准
├── losses/                ← 主推荐损失 + U/S 辅助损失
├── training/              ← 交替训练（先估 SURD，再训推荐）
├── evaluation/            ← Recall/NDCG/MRR、长尾、校准指标
├── scripts/               ← train / estimate_surd / evaluate 入口
├── tests/                 ← 单元测试与冒烟测试
└── outputs/               ← 运行产物（默认不入库大文件）
```

可以把它想成一条流水线：

**原始 check-in CSV → 样本 → 三个向量 h1/h2/h3 → 推荐分数 / 子集概率 → Specific MI → SURD → 权重 → 辅助损失 → 更好的推荐模型**

---

## 3. 数据从哪里来？怎么避免“偷看未来”？

### 3.1 数据适配器

入口文件：`data/preprocessing.py`

- `load_checkins(...)`：读取 Massive-STEPS 格式 CSV  
  - 可本地路径（`data.checkins_csv`）  
  - 或从 HuggingFace 下载（`data.hf_dataset_id` + `data.hf_filename`）
- 期望字段包括：`trail_id, user_id, venue_id, timestamp`，以及可选的 `latitude/longitude/venue_category/...`
- `build_samples(...)`：把轨迹变成「预测下一次 POI」的样本  
  - 输入：历史 POI 序列、目标时刻的时间特征、仅由历史算出的空间特征  
  - 标签：`target_poi_id`（具体 POI，不是类别）

### 3.2 时间切分（必须）

入口文件：`data/splits.py`

官方 Massive-STEPS 默认是按 trail + 用户分层随机切分；**本项目第一版按规格改为时间顺序切分**：

1. 按样本的 `target_timestamp` 排序  
2. 前面 70% → train，中间 10% → val，最后 20% → test（比例可在配置改）  
3. 再从 train 时段内切出一块 `estimation` holdout，专门用来估 Specific MI / SURD，降低“用同一批样本又训练概率头又估 MI”的偏差

对应函数：

- `temporal_split_samples`  
- `carve_estimator_holdout`

### 3.3 一条样本里有什么？

见 `data/dataset.py` 的 `NextPOIBatch`：

| 字段 | 含义 | 对应信息源 |
|------|------|------------|
| `history_poi_ids` | 历史访问的 POI | Q1 Personal / History |
| `target_time_features` | 预测时刻的小时/星期等 | Q2 Temporal |
| `spatial_features` | 由历史位置统计得到的空间上下文 | Q3 Spatial |
| `target_poi_id` | 要预测的下一个 POI | 标签 Y |

第一版只用 **3 个信息源**。`user_features` 已算好，留给以后可选的 Q4，但当前 SURD **不**把它算作第 4 源。

---

## 4. 模型：三个编码器 + 一个推荐头 + 七个子集头

入口文件：`models/full_model.py`（总装）

```text
历史序列 ── HistoryEncoder ──► h1 ─┐
时间特征 ── TemporalEncoder ─► h2 ─┼─ concat+MLP ─► 推荐 logits（推理只用这个）
空间特征 ── SpatialEncoder ──► h3 ─┘

h1,h2,h3 还会送进 7 个子集概率头：
  {Q1},{Q2},{Q3},{Q1,Q2},{Q1,Q3},{Q2,Q3},{Q1,Q2,Q3}
```

### 4.1 编码器（`models/encoders.py`）

- **Q1 `HistoryEncoder`**：POI embedding + GRU，输出固定维度向量  
- **Q2 `TemporalEncoder`**：时间特征 MLP  
- **Q3 `SpatialEncoder`**：空间特征 MLP  

三个输出维度相同（默认 64），方便拼接。

### 4.2 推荐头（`models/recommendation.py`）

把 `h1||h2||h3` 融合成 `h_rec`，再映射到全部 POI 上的分数。  
**推理路径**就是 `model.recommend(batch)`，不依赖真实标签。

### 4.3 子集概率头（`models/subset_estimators.py`）

3 个源共有 \(2^3-1=7\) 个非空子集。  
每个头**只能**看到自己被授权的那些 `h_i`，不能偷用其他源——这是后面算 Specific MI / SURD 的前提。

---

## 5. Specific MI 与 SURD：在算什么？

### 5.1 Specific MI（按目标 POI 算）

文件：`information/specific_mi.py`

普通互信息回答的是“整体平均有多少信息”。  
这里要回答的是：**当真实下一站是某个 POI y 时，某个信息源组合 Q_A 提供了多少预测信息？**

对估计集上每个标签为 y 的样本，取：

\[
\log \frac{P_A(Y=y\mid Q_A)}{P(Y=y)}
\]

再对同一 y 平均。先验 \(P(Y=y)\) 用训练集频率 \(N_y/N\)，**不是**均匀的 \(1/K\)。

### 5.2 SURD 分解（正式算法，不是随便相减）

文件：`information/surd.py`

实现对齐官方 SURD 包中的 `assign` 规则（Martínez-Sánchez et al., Nature Communications 2024；代码参考 ALD-Lab/SURD）：

1. 对每个目标 POI y，拿到它在 7 个子集上的 specific MI  
2. 排序、处理并列、按规则把增量分配到：  
   - **Unique (U)**：某个单源独有  
   - **Redundant (R)**：多源重复提供  
   - **Synergistic (S)**：必须联合才出现  
3. 再按 \(P(y)\) 加权得到 **global SURD**

第一版把 R 主要当作**分析量**；训练里主要用 U 和 S 权重。

### 5.3 变成训练可用的权重

文件：`information/weights.py`

对每个 POI y：

- 把非负的 U 项归一化 → `w_i^U(y)`  
- 把非负的 S 项归一化 → `w_A^S(y)`  
- 样本太少的 POI：向全局权重收缩 / fallback，避免长尾噪声权重

这些权重会 `detach`，**不参与反向传播**。SURD 估计本身不回传梯度。

---

## 6. SURD 权重如何指导训练？

损失在 `losses/`：

\[
\mathcal L = \mathcal L_{\mathrm{rec}} + \lambda_U \mathcal L_U + \lambda_S \mathcal L_S
\]

- `\mathcal L_rec`：主推荐交叉熵（`recommendation_loss.py`）  
- `\mathcal L_U`：按真实标签 y 查出的 unique 权重，加权单源分支 CE（`surd_auxiliary_loss.py`）  
- `\mathcal L_S`：按 y 查出的 synergy 权重，加权多源组合分支 CE  

直观例子：

- 若某个 POI 的 unique 主要来自历史偏好（Q1），则该样本会更用力训练 Q1 分支  
- 若另一个 POI 更依赖“时间+空间”的协同，则对应 `Q2_Q3` 组合头权重更高  

请注意：组合头上的交叉熵**不是**“纯 synergy 信息量”，只是被 synergy 权重调节的辅助预测损失。

### 交替流程（`training/alternating.py`）

1. **Warm-up**：先只训推荐（encoder + 推荐头）  
2. **Phase A**：固定/冻结 encoder，微调 7 个子集头 → 在 estimation 集上估 Specific MI → SURD → 更新权重（可 EMA 平滑）  
3. **Phase B**：固定权重，继续训推荐；子集头可冻结  
4. 每隔 `surd_refresh_interval` 个 epoch 再跑一次 Phase A  

产物示例：`outputs/.../surd/epoch_XXXX/`

- `specific_mi.csv`  
- `poi_specific_surd.csv`  
- `global_surd.json`  
- `poi_sample_counts.csv`  
- `estimator_calibration.json`

---

## 7. 怎么运行？

在仓库根目录或本目录下均可；下面假设当前目录是 `surd_guided_poi/`。

### 7.1 安装依赖

```bash
pip install -r requirements.txt
```

### 7.2 训练（默认 Beijing，较小，适合试跑）

```bash
python scripts/train.py \
  --config configs/base.yaml configs/dataset_beijing.yaml configs/model_surd.yaml
```

若本地已有 CSV：

```bash
python scripts/train.py \
  --config configs/base.yaml configs/dataset_beijing.yaml configs/model_surd.yaml \
  --checkins_csv /path/to/beijing_checkins.csv
```

Tokyo：

```bash
python scripts/train.py \
  --config configs/base.yaml configs/dataset_tokyo.yaml configs/model_surd.yaml
```

### 7.3 单独估 SURD / 评测

```bash
python scripts/estimate_surd.py --checkpoint outputs/beijing/best.pt
python scripts/evaluate.py --checkpoint outputs/beijing/best.pt
```

### 7.4 跑测试

在仓库根目录：

```bash
python -m pytest surd_guided_poi/tests -q
```

---

## 8. 建议你怎么读代码（一条最短路径）

1. 打开 `docs/SURD_Guided_Next_POI_Recommendation_Implementation_Spec.md` 第 0–7 节，建立概念  
2. 读 `data/preprocessing.py` + `data/splits.py`：样本与时间切分  
3. 读 `models/full_model.py`：前向时算出了哪些东西  
4. 读 `information/specific_mi.py` → `information/surd.py` → `information/weights.py`  
5. 读 `training/alternating.py`：权重何时更新、损失如何相加  
6. 读 `scripts/train.py`：端到端入口  

如果你只想先确认“推理不用标签”，直接看 `SURDGuidedPOIModel.recommend` 和 `evaluation/ranking_metrics.py`。

---

## 9. 第一版做了什么 / 刻意没做什么

**已实现**

- Massive-STEPS 数据适配 + **时间切分**  
- 3 信息源 encoder、统一推荐头、7 子集概率头  
- POI-specific Specific MI + 官方风格 SURD assign  
- U/S 指导权重、低样本 fallback、EMA、交替训练  
- Recall@K / NDCG@K / MRR、长尾分桶、校准指标输出  
- 中文入门指南 + 规格文档入库  

**刻意延后（规格允许）**

- 第 4 信息源 Q4 进入 SURD  
- 推理期 candidate-conditioned scoring（用候选 POI 的离线权重打分）  
- redundancy 正则 / embedding 正交约束（规格明确不建议直接等价）  
- 完整论文级 baseline 对比与大规模消融  
- Inductive / cold-start POI 设定（第一版为 transductive）

---

## 10. 常见误解速查

| 误解 | 更准确的说法 |
|------|----------------|
| SURD 证明了因果 | 这里是观测数据上的**预测信息结构** |
| 组合头 CE = synergy | CE 只是被 synergy **权重调节**的辅助损失 |
| 推理时按真实 y 查权重 | 第一版推理**只用**统一推荐头 |
| 先验用 1/K | 应用训练集频率 \(N_y/N\) |
| Massive-STEPS 官方随机切分即可 | 本项目第一版必须 **temporal split** |
