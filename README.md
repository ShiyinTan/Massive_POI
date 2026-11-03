# Massive_POI

上游 [Massive-STEPS](https://github.com/cruiseresearchgroup/Massive-STEPS) 代码在 [`Massive-STEPS/`](Massive-STEPS/)。快照对应提交 [`233aaee03230ec830e18051abfca778e2a39c4b4`](https://github.com/cruiseresearchgroup/Massive-STEPS/commit/233aaee03230ec830e18051abfca778e2a39c4b4)（2026-06-02，*Added Claude Haiku 4.5 Zero-shot Next POI results*）。

数据集分析文档在 [`docs/massive-steps-analysis/`](docs/massive-steps-analysis/)。

## 代码布局

`Massive-STEPS/` 是上游工作区的原样拷贝，子模块已按锁定提交展开成普通文件，不再是 git submodule。主仓库为 Apache-2.0，见 [`Massive-STEPS/LICENSE`](Massive-STEPS/LICENSE)。各模型目录保留自己的许可证文件。

| 目录 | 上游 | 提交 |
|------|------|------|
| `Massive-STEPS/` | [cruiseresearchgroup/Massive-STEPS](https://github.com/cruiseresearchgroup/Massive-STEPS) | `233aaee03230ec830e18051abfca778e2a39c4b4` |
| `GETNext/` | [w11wo/GETNext](https://github.com/w11wo/GETNext) | `51a9cc06172811fa61ba72643028c30ff61193a6` |
| `Spatio-Temporal-Hypergraph-Model/` | [w11wo/Spatio-Temporal-Hypergraph-Model](https://github.com/w11wo/Spatio-Temporal-Hypergraph-Model) | `758c50b37e5409000049c84bd2acf03c62a65fad` |
| `Bigscity-LibCity/` | [w11wo/Bigscity-LibCity](https://github.com/w11wo/Bigscity-LibCity) | `e9248ffe5d22fe7f730e47cda15ac779fb03b032` |
| `UniMove/` | [w11wo/UniMove](https://github.com/w11wo/UniMove) | `248290e452d056cd1a55e43e61c3ee0afea57a32` |
| `Flashback/` | [w11wo/Flashback](https://github.com/w11wo/Flashback) | `7127132f093e7fdb6f1df2cc8bcdd594d4a91817` |
| `Graph-Flashback/` | [w11wo/Graph-Flashback](https://github.com/w11wo/Graph-Flashback) | `7c8aa326710abd74bc4543554c68a7e382e00e4f` |
| `Graph-Flashback/KGE-Graph-Flashback/` 与 `LoTNext/KGE-Graph-Flashback/` | [w11wo/KGE-Graph-Flashback](https://github.com/w11wo/KGE-Graph-Flashback) | `af1e8026547552d33a7bed635ab467baedc92d62` |
| `LoTNext/` | [w11wo/LoTNext](https://github.com/w11wo/LoTNext) | `72c1829263fbbc084bd7edc35a51e32d272c95ce` |

数据集构建与 LLM 评测入口在 `Massive-STEPS/src/`（`preprocess_std.py`、`create_next_poi_dataset.py`、`run_next_poi_llm.py` 等）。使用方式见 [`Massive-STEPS/README.md`](Massive-STEPS/README.md)。

未纳入两份论文 PDF：`Spatio-Temporal-Hypergraph-Model/SIGIR2023_STHGCN.pdf`、`Flashback/ijcai20.pdf`。其余文件与上述提交的工作区一致。
