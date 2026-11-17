# STHGCN 消融开关（Massive-STEPS 适配）

代码路径：Massive_POI 仓库内 `Spatio-Temporal-Hypergraph-Model/`（基于 w11wo fork，CPU 可跑）。  
**省略全部消融 flag = 论文 / 先前 bias 报告的默认行为。**

GPU：`conf/std_conf/$city.yml` 里 `run_args.gpu`（`0`=CUDA0）；CLI **`--gpu -1`** 覆盖为 CPU（与此前成功复现一致）。seed / epochs 仍来自 yaml（`54607333` / `20`）。

## Flag 一览

| CLI | 默认 | 作用 |
|-----|------|------|
| `--use_user_embed {on,off}` | `on` | `off`：`CheckinEmbedding` 将 user lookup **置零**（仍占 concat 位，但不贡献用户向量）。 |
| `--hypergraph_edges {both,ci2traj,traj2traj,none}` | `both` | 控制前向消息传递：`both`=ci2traj+traj2traj；`ci2traj`=仅时间掩码后的 check-in→traj；`traj2traj`=跳过 ci2traj 层；`none`=不做超图卷积，对 check-in embed 均值池化后线性打分。 |
| `--graph_scope {full,train}` | `full` | `full`：超图来自 `sample.csv`=train∪val∪test（论文）。`train`：图内保留 train trail + 当前 eval query trail 的 check-in；**traj2traj 邻居仅限 train trail**（val/test 不作邻居）。preprocess 缓存目录会变为 `preprocessed_scope-train_prefix-*`。 |
| `--unseen_prefix_poi {unk,drop}` | `unk` | `unk`：前缀未见 POI → `padding_poi_id`（论文）。`drop`：若当前 trail **前缀**任一点 POI 不在 train，**丢弃整条 eval 样本**（目标/用户未见过滤仍走 `remove_unseen_user_poi`）。 |
| `--gpu INT` | yaml | 覆盖 `run_args.gpu`；CPU 用 `-1`。 |

## 命令（CPU · 四城 · 勿省略 `--gpu -1`）

在 `Spatio-Temporal-Hypergraph-Model/` 下执行（需先按 README 链好 HF `data/$city/*_checkins_*.csv`）。

```bash
cd Spatio-Temporal-Hypergraph-Model
city=beijing   # tokyo | shanghai | melbourne

# 论文默认（与省略消融 flag 等价）
python3 run.py -f std_conf/${city}.yml --gpu -1

# 1) 关掉 user embedding
python3 run.py -f std_conf/${city}.yml --gpu -1 --use_user_embed off

# 2) 超图边
python3 run.py -f std_conf/${city}.yml --gpu -1 --hypergraph_edges both
python3 run.py -f std_conf/${city}.yml --gpu -1 --hypergraph_edges ci2traj
python3 run.py -f std_conf/${city}.yml --gpu -1 --hypergraph_edges traj2traj
python3 run.py -f std_conf/${city}.yml --gpu -1 --hypergraph_edges none

# 3) 图范围
python3 run.py -f std_conf/${city}.yml --gpu -1 --graph_scope full
python3 run.py -f std_conf/${city}.yml --gpu -1 --graph_scope train

# 4) 未见前缀 POI
python3 run.py -f std_conf/${city}.yml --gpu -1 --unseen_prefix_poi unk
python3 run.py -f std_conf/${city}.yml --gpu -1 --unseen_prefix_poi drop
```

验证 CLI：`python3 utils/ablation_cli.py -h`（不加载 PyG）。

实现入口：`utils/ablation_cli.py` → `run.py` → `CheckinEmbedding` / `STHGCN.forward` / `preprocess_main` / `generate_hypergraph`（`neighbor_traj_ids`）。
