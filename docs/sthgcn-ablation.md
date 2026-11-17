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

## 输出目录与 run_name

全部 flag 为默认 → **`run_name=paper`**。否则只拼**非默认**项，多项用 `__` 连接：

| 非默认设置 | token |
|------------|-------|
| `--use_user_embed off` | `no_user` |
| `--hypergraph_edges ci2traj` | `edges_ci2traj` |
| `--hypergraph_edges traj2traj` | `edges_traj2traj` |
| `--hypergraph_edges none` | `edges_none` |
| `--graph_scope train` | `scope_train` |
| `--unseen_prefix_poi drop` | `prefix_drop` |

布局（log / tensorboard / checkpoint 都在同一 run 目录下）：

```text
results/<city>/<run_name>/<YYYYMMDD_HHMMSS>/
  log/            # train.log
  tensorboard/    # TB events
  checkpoints/    # checkpoint.pt
```

例：

- 论文默认：`results/beijing/paper/20261117_193012/`
- 关 user：`results/beijing/no_user/20261117_194500/`
- 组合：`results/tokyo/no_user__scope_train/20261117_200000/`

每次 test 结束追加一行到 **`results/summary.csv`**（无则写表头）：

```csv
time,city,run_name,use_user_embed,hypergraph_edges,graph_scope,unseen_prefix_poi,gpu,Recall@1,Recall@5,Recall@10,Recall@20,MRR,log_dir
20261117_193012,beijing,paper,on,both,full,unk,-1,0.453,0.653,0.747,0.813,0.541,results/beijing/paper/20261117_193012
20261117_194500,beijing,no_user,off,both,full,unk,-1,0.410,0.600,0.700,0.780,0.500,results/beijing/no_user/20261117_194500
```

同 city+run_name 不同时间戳各占一行，不覆盖。

## 命令（CPU · 四城 · 勿省略 `--gpu -1`）

`Spatio-Temporal-Hypergraph-Model/data` → `../Massive-STEPS/data`（相对 symlink）；把 HF 的 `{city}_checkins_{train,validation,test}.csv` 放到 `Massive-STEPS/data/<city>/`（该目录下 `*.csv` 已 gitignore，勿提交）。

在 `Spatio-Temporal-Hypergraph-Model/` 下执行：

```bash
cd Spatio-Temporal-Hypergraph-Model
city=beijing   # tokyo | shanghai | melbourne

# 论文默认 → results/<city>/paper/<ts>/
python3 run.py -f std_conf/${city}.yml --gpu -1

# 1) 关掉 user embedding → results/<city>/no_user/<ts>/
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

实现入口：`utils/ablation_cli.py`（`ablation_run_name`）→ `run.py` → `CheckinEmbedding` / `STHGCN.forward` / `preprocess_main` / `generate_hypergraph`（`neighbor_traj_ids`）。
