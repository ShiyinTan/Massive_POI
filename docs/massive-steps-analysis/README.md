# Massive-STEPS 分析（精简）

面向 Massive_POI 的 Massive-STEPS 数据集分析交付物。

## 入口

**主报告（中文）：** [massive-steps-analysis.md](massive-steps-analysis.md)

覆盖：15 城 **完整 HF** 规模/分布、图轴术语表、研究角度证据、official·local 廉价基线（MostPop / Nearest / Markov1 / CatMarkov / 时间桶 / UserMostPop·UMP_DN·UMP_WW；长尾过滤）、轨迹时间跨度、四城极简灰底地图、优先课题。协议：**official only**；prediction time = target check-in 本地时；主表不含 user-disjoint。

## 目录

| 路径 | 内容 |
|------|------|
| `massive-steps-analysis.md` | 合并后的唯一报告 |
| `massive-steps-cited-city-table.csv` | 15 城规模表（full HF） |
| `massive-steps-measured-summary.csv` | 15 城实测分布摘要 |
| `figures/` | 规模与分布、证据快照、基线、UMP 时段、时长图 |
| `data/` | 研究证据、基线、时长、过滤、`hf_download_provenance.csv`、`bin_definition.json` |
| `maps/` | 极简灰底气泡 / 流向图及格子·OD CSV（Tokyo / Melbourne / Beijing / Shanghai） |
