# Massive-STEPS 分析（精简）

面向 Massive_POI 的 Massive-STEPS 数据集分析交付物。

## 入口

**主报告（中文）：** [massive-steps-analysis.md](massive-steps-analysis.md)

覆盖：数据集概览 / 字段 / 分布、**图轴术语表**、研究角度与廉价基线（含 **UTC-as-stored vs 城市本地时**、**UserMostPop_DayNight / Weekday / WorkWeekend**、长尾用户过滤与八模型 Acc 总表；**完整 HF city CSV**、时间桶 **official split only**，预测时刻 = 被预测 check-in 的 timestamp）、切分稳定性、周末指标、轨迹时间跨度、极简灰底地图、优先课题；关键图附 2–4 句图注。

时间 / UMP 桶主表评测城（full HF）：Tokyo · Melbourne · Beijing · Shanghai · Sydney · Palembang · Moscow · New-York · Sao-Paulo（见 `data/hf_download_provenance.csv`）。

## 目录

| 路径 | 内容 |
|------|------|
| `massive-steps-analysis.md` | 合并后的唯一报告 |
| `massive-steps-cited-city-table.csv` | 15 城规模表 |
| `massive-steps-measured-summary.csv` | 10 城实测分布摘要 |
| `figures/` | 规模与分布、UTC–local 时段、证据快照、基线、UMP 时段、切分、时长图 |
| `data/` | 研究证据、基线、切分稳定性、时间条件（含 `utc_local_*`、`usermostpop_time_bins*`、`acc*_full_comparison_*`、`longtail_filter_meta.csv`、`hf_download_provenance.csv`）CSV |
| `maps/` | 极简灰底气泡 / 流向图及格子·OD CSV（Tokyo / Melbourne / Beijing / Shanghai） |

已移除：彩色街道路网叠加图、早期无底图 desire-line / hexbin / choropleth 等被极简地图替代的视觉稿，以及被合并报告取代的多份旧 markdown。
