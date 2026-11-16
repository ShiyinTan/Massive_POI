# TS-MoE：Target-conditioned SURD-guided Mixture-of-Experts Transformer

下一个具体 POI 的排序模型。对每个候选 POI，用 SURD 把 User / POI / Time / Location 的预测信息拆成 Unique、Redundant、Synergistic，再路由到对应的交互 Expert。

```
四路序列 → 四个独立 Transformer → 候选条件交叉注意力
        → SURD prior + context router → Top-K Experts
        → Base 分数 + MoE 分数
```

## 数据

`data` 是指向 Massive-STEPS 数据目录的相对软链接：

```text
ts_moe/data → ../Massive-STEPS/data
```

官方切分表放在：

```text
data/<city>/<city>_checkins_{train,validation,test}.csv
```

列与 `Massive-STEPS/src/preprocess_std.py` 一致（`trail_id`, `user_id`, `venue_id`, `latitude`, `longitude`, `timestamp`, `venue_category`）。当前仓库里的 `Massive-STEPS/data` 提供各城市 check-in 统计和边界 geojson；轨迹 CSV 需要放到上述路径后才能跑真实城市。不要把数据文件提交进本目录。

## 运行

在仓库根目录：

```bash
python -m ts_moe.scripts.estimate_surd --config ts_moe/configs/base.yaml
python -m ts_moe.scripts.train --config ts_moe/configs/base.yaml
python -m ts_moe.scripts.evaluate --checkpoint ts_moe/outputs/ts_moe_beijing_surd.pt
python -m ts_moe.scripts.infer --checkpoint ts_moe/outputs/ts_moe_beijing_surd.pt --history_csv trail.csv
```

不依赖城市 CSV 的通路检查：

```bash
python -m ts_moe.scripts.train --config ts_moe/configs/smoke.yaml
```

对比实验只改配置，不改结构：

- `--routing-mode surd|plain|random|fixed`：SURD 先验、普通 MoE、随机路由、固定均匀路由
- `--expert-variant 4|10|21`：TS-MoE-4（仅 Unique）、TS-MoE-10（Unique + 成对 Redundancy）、TS-MoE-21
- `model.expert_groups`：例如 `[synergy]`、`[unique, synergy]`，对应去掉某一组 Expert 的消融

初始超参（`configs/base.yaml`）：隐藏维 128、4 个独立 Transformer、21 个 Expert、Top-4、损失为推荐交叉熵 + 路由 KL + 与 SURD 先验一致的负载均衡。`loss.lambda_shared` 默认 0（冗余 Expert 的重构头仍在，第一版不把它加进损失）。

## 实现时定下的读法

规格没有写成唯一公式的地方，按下面的方式和方程对齐：

1. SURD 原子使用 Martínez-Sánchez 等 Nature Communications 2024 / ALD-Lab `surd.assign` 的 specific-MI 增量记账，而不是 15 个子集互信息的加减。每个目标 POI 单独留原子，不乘 `P(Y=y)`。
2. 冗余原子可能是高阶集合。21 个 Expert 里的冗余槽只有 6 个成对 Expert，因此把集合 `A` 的冗余质量均分到 `A` 内部的二元组上，总冗余质量守恒。
3. 负原子在路由前用 `max(0, ·)` 去掉。全零原子视为「无 Expert 偏好」（均匀分布），再按 `c_y = n_y / (n_y + κ)` 与全局 prior 混合。`n_y` 是进入 specific-MI 平均的样本数。
4. 路由 KL 和负载均衡都在「该样本实际打分的全部候选」上平均，每个候选只用自己的 prior，不用真实标签那一行去路由其他候选。
5. 固定路由是当前 Expert 集合上的均匀权重（不走 Top-K）。随机路由是正态 logits 的 softmax，并停止梯度。只有 `surd` 模式加路由 KL；`plain` 模式的负载目标是均匀分布。
6. 训练用 sampled softmax。推理默认两阶段：POI 编码器池化向量点积召回 Top-M，再对这 M 个候选做完整的交叉注意力和 Top-K Expert。`eval.candidate_strategy: full` 时对词表里每个 POI 走完整模型。
7. 离线 prior 按 trail 做 cross-fitting。第 k 折的样本只读取其余折估计的表；验证和测试读取全训练集估计的表。神经网络估计器（默认）在折外再留一块 trail 做 specific MI 的读取；离散平滑估计器是同一公式的插件式可选项（smoke 配置使用它）。
8. 动态 SURD（规格 §11.1）不在这一版里。
