# SURD-Guided Next POI Recommendation：完整模型与实现方案

> **文档用途**：作为 AI coding agent（如 ChatGPT、Claude Code、Cursor
> 等）的实现规格说明。\
> **任务**：基于用户历史
> check-in、时间和空间上下文，预测用户下一次访问的具体 POI，并利用
> POI-specific Specific Mutual Information（specific MI）与 SURD
> 分解，指导推荐模型学习不同 POI 所需的信息来源。\
> **重要边界**：本方案使用 observational check-in
> 数据。除非额外建立并验证干预/因果识别假设，否则应将 SURD 结果称为
> *predictive information structure / information
> contribution*，不要将其直接解释为真实因果效应。

------------------------------------------------------------------------

## 0. 给 AI 编程工具的总指令

请按照本文档实现模型，不要自行改变任务定义、信息论量的含义或数据划分方式。

实现时遵循以下原则：

1.  预测目标是**具体 POI ID**，不是 POI 类别。
2.  每条训练样本对应一个预测时刻：输入为截至时刻 (t)
    的用户历史与上下文，标签为下一次访问的 POI $y=POI_{u,t+1}$。
3.  每个信息源分别编码；所有 encoder 表示可被推荐网络和 SURD
    概率估计器共享。
4.  Specific MI 必须按目标 POI 状态 (y) 计算；SURD
    也先在每个目标状态 (y) 上分解，再按目标分布加权得到 global SURD。
5.  SURD
    数值估计与离散分配过程不参与反向传播。它们作为定期更新的指导权重（guidance
    weights）。
6.  训练阶段可以用真实标签 (y_n) 查询该 POI
    对应的权重；推理阶段不能使用真实标签。
7.  第一版优先实现稳定、可复现的方案：POI-specific SURD
    用于辅助训练，推理仍使用统一的推荐头。
8.  不要把某个组合输入的交叉熵称为"纯 synergy loss"。它是由 SURD synergy
    权重调节的组合分支辅助预测损失。
9.  所有概率估计都应考虑校准；specific MI 不能只依赖分类准确率。
10. 先实现 3 个信息源的完整流程，再扩展到 4 个信息源。对 (m)
    个源枚举全部非空子集的成本是 (2^m-1)。

------------------------------------------------------------------------

# 1. 任务定义

## 1.1 Next POI Recommendation

给定用户 (u) 在时刻 (t) 之前的历史 check-in 序列：

$$
\mathcal{H}_u^t = \{(v_1,\tau_1),(v_2,\tau_2),\ldots,(v_t,\tau_t)\}
$$

其中：

-   (v_i)：第 (i) 次访问的 POI ID；
-   $\tau_i$：访问时间；
-   每个 POI (v_i) 有坐标 (loc(v_i)=(lat_i,lon_i))；
-   可选地有 POI category、用户属性等元数据。

模型预测下一次访问的具体 POI：

$$
Y_n = \mathrm{POI}_{u,t+1} \in \mathcal{P}
$$

其中 $\mathcal{P}$ 是 POI 集合，$K=|\mathcal{P}|$。

模型输出：

$$
\hat{\mathbf{p}}_n = \big[\hat{P}(Y=1\mid x_n),\ldots,\hat{P}(Y=K\mid x_n)\big]
$$

推荐结果是概率最高的 Top-K POI。

## 1.2 数据划分

必须采用时间顺序划分，避免未来信息泄漏：

-   Train：较早时间段；
-   Validation：其后的时间段；
-   Test：最后时间段。

所有输入特征只能由预测时刻 (t) 及之前的信息构建。

需要明确并固定以下评估协议：

-   **Transductive setting**：训练阶段可见测试期间可能出现的 POI
    ID，但不可见测试标签；
-   **Inductive/unseen POI setting**：测试时可能出现训练阶段未见
    POI。若要支持该设置，必须有 POI 内容/坐标等 side information，并设计
    cold-start POI 编码机制。

第一版可先实现 transductive setting。不要在实验中混淆两种协议。

------------------------------------------------------------------------

# 2. 信息源定义（Q variables）

第一版建议使用 3 个核心信息源，降低组合数量与实现复杂度。

  -------------------------------------------------------------------------------------------------------------
  变量              名称              输入内容                                   直观含义
  ----------------- ----------------- ------------------------------------------ ------------------------------
  (Q_1)             Personal          用户历史 POI 序列及其可选 category         用户通常喜欢去哪里
                    Preference /                                                 
                    History                                                      

  (Q_2)             Temporal Context  当前预测时间、小时、星期、周期性时间特征   用户在什么时间出行

  (Q_3)             Spatial Context   最近访问位置、历史位置序列、候选 POI       用户当前在哪里、哪些地方可达
                                      相对距离等可用上下文                       
  -------------------------------------------------------------------------------------------------------------

可选的第四个变量：

  -------------------------------------------------------------------------------------------------------
  变量                    名称                    输入内容
  ----------------------- ----------------------- -------------------------------------------------------
  (Q_4)                   User Characteristics    用户静态/动态统计特征，如历史活跃度、历史访问多样性等

  -------------------------------------------------------------------------------------------------------

注意：

-   不要求不同 (Q_i) 的原始输入具有相同长度或结构。
-   每个 (Q_i) 经各自 encoder 转成固定维度表示 (h_i)。
-   SURD
    的输入是每个样本对应的信息源表示/随机变量，不要求原始数据张量形状相同。
-   不要把目标 POI (Y) 或目标访问之后的信息输入任何 encoder。
-   对空间信息要严格保证候选 POI 特征的使用符合推理协议。候选 POI
    的静态坐标可以使用，但不能使用未来访问轨迹信息。

## 2.1 变量编码

每个信息源对应一个 encoder：

$$
h_i=E_{\theta_i}(Q_i),
\qquad h_i\in\mathbb{R}^d
$$
建议：

-   History encoder：POI embedding + positional/time embedding +
    GRU/Transformer。
-   Temporal encoder：小时、星期、周期时间的 sin/cos 编码 + MLP。
-   Spatial encoder：最近位置、历史访问位置统计、距离/方向特征 + MLP
    或轻量序列 encoder。
-   User encoder（可选）：用户统计特征 + MLP。

第一版可令各 encoder 输出相同维度 (d)，便于融合。对缺失特征使用明确的
mask 或可学习 missing embedding，不要随意填入可能带来偏差的常数。

------------------------------------------------------------------------

# 3. 推荐网络（Recommendation Network）

## 3.1 融合表示

将所有信息源表示融合：

$$
h_{\mathrm{rec}} =
\mathrm{Fusion}(h_1,h_2,\ldots,h_m)
$$
第一版可使用 concat + MLP：

$$
h_{\mathrm{rec}} = \mathrm{MLP}_{\psi}\big([h_1 \Vert h_2 \Vert \cdots \Vert h_m]\big)
$$

后续可替换为 attention/gating，但应先建立简单 baseline。

## 3.2 POI 分类头

$$
z_{\mathrm{rec}}=W_{\mathrm{rec}}h_{\mathrm{rec}}+b_{\mathrm{rec}}
$$

$$
\hat{P}_{\mathrm{rec}}(Y=y\mid Q_1,\ldots,Q_m)
=\mathrm{Softmax}(z_{\mathrm{rec}})_y
$$

其中输出维度为训练 POI vocabulary 的大小 (K)。

## 3.3 推荐损失

$$
\mathcal{L}_{\mathrm{rec}}
=-\frac{1}{N}\sum_{n=1}^{N}
\log \hat{P}_{\mathrm{rec}}(Y=y_n\mid Q^{(n)})
$$

可以在后续实验中加入 class-balanced loss、按频率采样或其他 long-tail
方法，但第一版应先保留标准 cross-entropy，以便分离 SURD 的实际贡献。

------------------------------------------------------------------------

# 4. Specific Mutual Information（Specific MI）

## 4.1 为什么需要 Specific MI

普通互信息 (I(Y;Q_A)) 汇总了整个目标变量 (Y) 的信息量。它回答：

> 信息源组合 (Q_A) 平均而言，对预测所有 POI 有多少信息？

但 Next POI Recommendation 是一个多类别任务。不同 POI
可能依赖不同的信息来源：

-   某个 POI 主要由个人偏好预测；
-   某个 POI 主要由时间规律预测；
-   某个 POI 需要偏好与空间上下文共同判断。

因此需要计算目标 POI 状态特定的信息量，即 specific information。

## 4.2 子集条件概率模型

对于任意非空信息源子集
$A \subseteq \{1,\ldots,m\}$，建立概率估计器：

$$
\hat{P}_A(Y=y\mid Q_A)
$$

例如 (m=3) 时，需要考虑：

$$
\{Q_1\},\{Q_2\},\{Q_3\},\{Q_1,Q_2\},\{Q_1,Q_3\},\{Q_2,Q_3\},\{Q_1,Q_2,Q_3\}
$$

总共 (2^3-1=7) 个非空子集。

每个子集估计器只能接收它对应的 (h_i)，不能访问其他信息源。

可使用轻量 MLP：

$$
\hat{P}_A(Y\mid Q_A)
=\mathrm{Softmax}\big(g_{\phi_A}([h_i]_{i\in A})\big)
$$

### 共享与独立的建议

-   所有子集 head 共享底层信息源 encoder $E_{\theta_i}$。
-   不同子集有各自的轻量预测 head。
-   推荐 head 与这些 head 共享 encoder
    表示，但输出层和融合结构可以独立。
-   子集 head 不应通过旁路访问未授权的信息源。

## 4.3 目标 POI 的先验概率

定义训练集目标 POI 的总体频率：

$$
\hat{P}(Y=y)=\frac{N_y}{N}
$$

其中：

-   (N_y)：训练样本中目标 POI 为 (y) 的数量；
-   (N)：训练样本总数。

不要默认使用 (1/K)。总体先验和均匀先验是不同的统计量。

## 4.4 POI-specific Specific MI

对目标 POI (y) 和信息源组合 (Q_A)，定义：

$$
\tilde{i}(y;Q_A) = \mathbb{E}_{q_A \mid Y=y}
\left[
\log
\frac{P(Y=y\mid Q_A=q_A)}{P(Y=y)}
\right]
$$

用训练/校准样本估计：

$$
\widehat{\tilde{i}}(y;Q_A) = \frac{1}{N_y}
\sum_{n:Y_n=y} \log
\frac{\hat{P}_A(Y=y\mid Q_A^{(n)})}{\hat{P}(Y=y)}
$$

注意：

1.  对每个目标 POI (y)，只平均标签为 (y) 的样本。
2.  对每个样本，取概率模型对真实目标类别 (y) 输出的概率。
3.  分子是
    $\hat{P}_A(Y=y\mid Q_A^{(n)})$，分母是总体先验
    $\hat{P}(Y=y)$。
4.  可以使用自然对数（nats）或 log2（bits），但整个实验必须统一。
5.  单个样本的 log-ratio 可以为负；不要把单样本值强制截断为正。
6.  这里的 $\tilde{i}(y;Q_A$) 是 specific
    information，不应随意写成 (I(Y=y;Q_A))。

## 4.5 数值稳定性

实现时：

-   对预测概率进行 clamp，例如
    $p\leftarrow\max(p,\epsilon$)；
-   对先验概率同样进行 clamp；
-   推荐使用 float64 计算 log-ratio 和聚合；
-   记录每个 POI 的样本数 (N_y)；
-   对低样本 POI 标记估计不可靠，不要把 1--2
    个样本产生的极端估计直接当成强监督。

## 4.6 概率校准

Specific MI 依赖概率而不只是分类结果。需要检查：

-   NLL；
-   Brier score；
-   Expected Calibration Error（ECE）；
-   reliability diagram（可选）。

可以在独立 calibration/validation split 上做 temperature
scaling。不要用测试集拟合校准参数。

------------------------------------------------------------------------

# 5. SURD：对每个目标 POI 进行 R/U/S 分解

## 5.1 核心概念

SURD 将多个输入信息源对目标变量的信息贡献分成：

-   **Redundant information (R)**：多个信息源重复提供的预测信息；
-   **Unique information (U)**：某一个信息源独有的预测信息；
-   **Synergistic information
    (S)**：只有联合多个信息源时才体现出来的信息。

在本任务中，使用术语：

-   POI-specific predictive redundancy；
-   POI-specific unique predictive information；
-   POI-specific synergistic predictive information。

由于数据是 observational
check-in，不应仅凭这些分解声称发现了真实因果效应。

## 5.2 Specific SURD，而不是只计算 Global SURD

对每个目标 POI (y)，先基于所有子集的 specific information：

$$
\widehat{\tilde{i}}(y;Q_A)
$$

执行 SURD 分解，得到：

$$
\Delta\tilde{i}_A^R(y),\qquad
\Delta\tilde{i}_i^U(y),\qquad
\Delta\tilde{i}_A^S(y)
$$
其中：

-   $A$ 表示一个包含两个或更多变量的子集时，可对应冗余或协同项；
-   $i$ 表示单个信息源；
-   实际 R/U/S 分配必须遵循 SURD 原论文定义的排序、增量和分配规则。

**重要实现约束：不能用任意几个互信息相减，随意定义 R/U/S。**
例如，不能直接把 (I(Y;Q_1,Q_2)-I(Y;Q_1)) 称为 synergy。请按 SURD
原论文的正式算法实现，并将分解模块独立封装、单元测试。

## 5.3 从 Specific SURD 得到 Global SURD

对所有 POI 状态按目标先验加权：

$$
\Delta I_A^R =
\sum_{y\in \mathcal{P}}
P(Y=y)\Delta\tilde{i}_A^R(y)
$$

$$
\Delta I_i^U =
\sum_{y\in \mathcal{P}}
P(Y=y)\Delta\tilde{i}_i^U(y)
$$

$$
\Delta I_A^S =
\sum_{y\in \mathcal{P}}
P(Y=y)\Delta\tilde{i}_A^S(y)
$$

所以 SURD 并非只考虑全局。正确流程是：

1.  先逐个 target state / POI 计算 specific information；
2.  对每个 POI 做 specific SURD；
3.  再按 (P(Y=y)) 加权，得到 global SURD。

Global SURD 回答总体平均信息结构；POI-specific SURD 回答不同目标 POI
的信息结构是否不同。

## 5.4 需要实现的输出

建议每次 SURD 更新后保存：

``` text
surd/
  epoch_XXXX/
    specific_mi.csv
    poi_specific_surd.csv
    global_surd.json
    poi_sample_counts.csv
    estimator_calibration.json
```

`poi_specific_surd.csv` 至少包含：

``` text
poi_id
sample_count
U_Q1
U_Q2
U_Q3
R_Q1_Q2
R_Q1_Q3
R_Q2_Q3
S_Q1_Q2
S_Q1_Q3
S_Q2_Q3
S_Q1_Q2_Q3
```

具体列应与所实现的 SURD
算法输出结构一致；不要为了填满表格而创造算法中不存在的项。

------------------------------------------------------------------------

# 6. SURD 如何指导推荐模型

## 6.1 两个层次的指导

建议区分：

### A. Global SURD：宏观指导

Global SURD 汇总所有 POI 的平均信息结构，可用于：

-   分析哪类信息源总体更重要；
-   判断信息源之间的冗余是否明显；
-   判断模型是否需要更强的多源交互建模；
-   设计消融实验和模型结构。

第一版不一定要将 global R/U/S 全部转成训练 loss。

### B. POI-specific SURD：目标 POI 自适应辅助监督

对每个目标 POI (y)，其 SURD 结果产生权重：

$$
w_i^U(y) = \frac{\max(\Delta\tilde{i}_i^U(y),0)}{\sum_j\max(\Delta\tilde{i}_j^U(y),0)+\epsilon}
$$

$$
w_A^S(y) = \frac{\max(\Delta\tilde{i}_A^S(y),0)}{\sum_{B:|B|\ge 2}\max(\Delta\tilde{i}_B^S(y),0)+\epsilon}
$$

可选冗余权重：

$$
w_A^R(y) = \frac{\max(\Delta\tilde{i}_A^R(y),0)}{\sum_B\max(\Delta\tilde{i}_B^R(y),0)+\epsilon}
$$

说明：

-   以上是用于模型指导的非负归一化权重，不是 SURD 原始定义本身。
-   归一化前保留原始 signed values，便于分析。
-   若某一类贡献总和接近 0，则对应权重应设为全零或采用明确的
    fallback，不要产生 NaN。
-   对低样本 POI，使用收缩/平滑，见第 8 节。

## 6.2 Unique auxiliary loss

每个单变量分支输出：

$$
\hat{P}_i(Y\mid Q_i)
$$

训练样本 (n) 的真实目标 POI 是 (y_n)。使用该 POI 对应的 unique weight：

$$
\mathcal{L}_U = -\frac{1}{N} \sum_{n=1}^{N}
\sum_i w_i^U(y_n)
\log \hat{P}_i(Y=y_n\mid Q_i^{(n)})
$$

解释：

-   如果 POI (y_n) 的 unique information 主要来自 (Q_1)，该样本对 (Q_1)
    分支的辅助监督权重就更高。
-   如果另一个 POI 的 unique information 主要来自
    (Q_3)，其样本就会更多地训练 (Q_3) 分支。

## 6.3 Synergy-weighted combination auxiliary loss

对多变量子集 (A) 建立组合预测 head：

$$
\hat{P}_A(Y\mid Q_A)
$$

定义：

$$
\mathcal{L}_S = -\frac{1}{N} \sum_{n=1}^{N}
\sum_{A:|A|\ge 2} w_A^S(y_n)
\log \hat{P}_A(Y=y_n\mid Q_A^{(n)})
$$

解释：

-   若目标 POI
    的偏好与空间信息之间有较强协同贡献，对应组合分支获得更高权重。
-   组合 head 的 CE 衡量的是该组合整体预测目标的能力，**并不是纯 synergy
    信息量**。
-   SURD synergy 权重只是对组合辅助监督的指导，不代表 CE
    本身已经被分解成纯 R/U/S。

## 6.4 总训练目标

第一版建议：

$$
\boxed{
\mathcal{L}
=
\mathcal{L}_{\mathrm{rec}}
+
\lambda_U\mathcal{L}_U
+
\lambda_S\mathcal{L}_S
}
$$

其中：

-   $\mathcal{L}_{\mathrm{rec}}$：主推荐损失；
-   $\mathcal{L}_U$：POI-specific unique-weighted auxiliary
    loss；
-   $\mathcal{L}_S$：POI-specific synergy-weighted auxiliary
    loss；
-   $\lambda_U,\lambda_S$：验证集调节的超参数。

初版不强制加入 redundancy loss。原因是预测信息冗余并不意味着 embedding
应该正交；直接对不同 embedding
加正交约束，不能保证移除的是冗余预测信息。

## 6.5 可选的 redundancy 使用方式

第一版将 R 作为分析量即可。后续若要使用，可以研究：

-   对冗余较高的信息源组合减少重复辅助分支；
-   对高度冗余的输入进行参数共享或压缩；
-   使用冗余贡献分析指导模型简化。

不要直接使用：

$$
\|h_i^{\top} h_j\|^2
$$

并声称它等价于 SURD redundancy。两者不是同一个量。

------------------------------------------------------------------------

# 7. 训练流程：共享 Encoder + 交替估计

## 7.1 模块结构

``` text
Raw inputs
  ├── Q1 History ── Encoder E1 ── h1 ─┬── Recommendation Fusion
  ├── Q2 Time ───── Encoder E2 ── h2 ─┤        └── Recommendation Head
  └── Q3 Spatial ── Encoder E3 ── h3 ─┘

h1, h2, h3
  ├── Q1 probability head
  ├── Q2 probability head
  ├── Q3 probability head
  ├── Q1+Q2 probability head
  ├── Q1+Q3 probability head
  ├── Q2+Q3 probability head
  └── Q1+Q2+Q3 probability head

Probability estimates
  └── Specific MI per POI
        └── Specific SURD per POI
              ├── Global SURD aggregation
              └── POI-specific U/S weights
                    └── Auxiliary recommendation losses
```

## 7.2 交替优化的动机

共享 encoder 会随着推荐训练而变化，因而各信息源的条件概率估计和 specific
MI 也会变化。若始终使用早期估计的 SURD 权重，指导信号可能逐渐过时。

采用交替流程：

### Phase A：MI estimator / SURD update

1.  固定共享 encoder（或从推荐网络当前 checkpoint 读取 encoder）。
2.  使用当前 (h_i) 训练/微调各子集概率 head。
3.  在独立 calibration/estimation split 上获得每个子集对各 POI
    的条件概率。
4.  计算 (\widehat{\tilde{i}}(y;Q_A))。
5.  按正式 SURD 算法对每个 POI 做分解。
6.  生成 POI-specific weights 和 global SURD。
7.  将权重保存，并在下一阶段 stop-gradient / detach。

### Phase B：Recommendation update

1.  固定当前 SURD weights。
2.  更新共享 encoder 和 recommendation head。
3.  同时使用
    $\mathcal{L}_{\mathrm{rec}}$、$\mathcal{L}_U$、$\mathcal{L}_S$。
4.  概率 heads 可冻结，避免同一轮中权重不断变化。
5.  经过若干 epoch 后重新执行 Phase A。

### 推荐更新频率

第一版可以：

-   先训练推荐模型 warm-up 若干 epoch；
-   每隔 (T) 个 epoch 更新一次 MI estimators 和 SURD；
-   两次 SURD 更新之间固定权重；
-   使用 EMA 平滑权重：

$$
\bar{w}^{(k)} = \beta\bar{w}^{(k-1)} + (1-\beta)w^{(k)}
$$

其中 $k$ 是 SURD 更新轮次，$\beta \in [0,1)$。

具体 warm-up、$T$、$\beta$ 作为超参数，通过验证集确定，不应在代码中写死为理论常数。

## 7.3 防止 MI 高估与信息泄漏

非常重要：

- 不要用一个概率模型在它训练过的同一批样本上直接估计 specific MI，然后把结果当作无偏估计。
- 推荐使用独立 estimation/calibration split，或 cross-fitting。
- 训练、验证、测试的时间顺序必须保持。
- 测试集不能参与 MI estimator 训练、SURD 权重计算或超参数选择。
- 若使用 validation 数据估计 SURD，要清楚说明其用途，并避免把 validation label 作为训练监督反复拟合。更稳妥的做法是从训练时间段内再切分出 estimator-calibration subset。
- 每轮 encoder 更新后，旧 probability head 可能不再校准；需重新微调/校准后再估计 Specific MI。

---

# 8. Long-tail POI 与低样本估计

## 8.1 问题

对于低频 POI $y$，样本数 $N_y$ 很小：

$$
\widehat{\tilde{i}}(y;Q_A)
=
\frac{1}{N_y}\sum_{n:Y_n=y}\cdots
$$

方差可能很大。因此不能因为某个 tail POI 的估计值特别高，就认定该信息源真的非常重要。

## 8.2 建议策略

第一版至少实现：

1. 保存每个 POI 的样本数；
2. 设置最小样本阈值；
3. 对低样本 POI 使用 global 或相近频率组的 SURD 权重作为 fallback；
4. 在日志中标记 fallback 的 POI 比例；
5. 报告不同 POI frequency buckets 的估计稳定性。

可进一步使用层次化收缩：

$$
\tilde{w}(y)
=
\rho_y w(y)
+
(1-\rho_y)w_{\mathrm{group}(y)}
$$

例如：

$$
\rho_y=\frac{N_y}{N_y+\kappa}
$$

其中 $w_{\mathrm{group}(y)}$ 可以是同频率 bucket 的平均权重，$\kappa$ 控制收缩程度。

注意：该公式是模型指导权重的稳定化策略，不是 SURD 原始理论公式。

## 8.3 不同聚合目标

要区分三种统计目标：

**Population-weighted global SURD**

$$
\Delta I_{\mathrm{pop}}
=
\sum_yP(y)\Delta\tilde{i}(y)
$$

代表真实总体目标分布下的平均信息结构。

**POI-balanced summary**

$$
\Delta I_{\mathrm{balanced}}
=
\frac{1}{|\mathcal{P}|}
\sum_y\Delta\tilde{i}(y)
$$

每个 POI 权重相同，适合分析不同 POI 的平均结构，但它不等于总体互信息。

**Tail-specific summary**

$$
\Delta I_{\mathrm{tail}}
=
\frac{1}{|\mathcal{P}_{\mathrm{tail}}|}
\sum_{y\in \mathcal{P}_{\mathrm{tail}}}
\Delta\tilde{i}(y)
$$

用于分析 tail POI 的信息结构。必须同时报告 tail POI 的样本覆盖率和不确定性。

---

# 9. 推理阶段（Inference）

推理阶段没有真实目标 POI $y$，因此不能根据真实标签查询 $w(y)$ 再做预测。

第一版采用最稳妥方案：

1. 输入用户历史、时间和空间信息；
2. 计算 $h_1,\ldots,h_m$；
3. 经过统一 recommendation fusion/head；
4. 输出所有 POI 的概率；
5. 返回 Top-K。

即：

$$
\hat{y}
=
\arg\max_y
\hat{P}_{\mathrm{rec}}(Y=y\mid Q_1,\ldots,Q_m)
$$

POI-specific SURD 在第一版中只用于**训练期的辅助监督**，不依赖推理时的真实标签。

## 9.1 可选扩展：Candidate-conditioned scoring

后续可以研究候选 POI 条件化的打分：

$$
s(y\mid Q)
=
g(h_1,\ldots,h_m; w(y))
$$

其中 $w(y)$ 是训练阶段离线估计的候选 POI-specific 权重。推理时对每个候选 POI 使用其离线权重，而不是使用真实标签。

但这会改变模型结构和推理成本，需作为独立扩展实验。不要把它与第一版训练期辅助监督混为一谈。

---

# 10. 计算复杂度与可扩展性

## 10.1 子集数量

$m$ 个信息源时，所有非空子集数量：

$$
2^m-1
$$

| 信息源数 $m$ | 非空子集数 |
|---:|---:|
| 3 | 7 |
| 4 | 15 |
| 5 | 31 |
| 6 | 63 |
| 8 | 255 |

组合数随变量数指数增长。

## 10.2 降低开销

建议按以下顺序扩展：

1. 先使用 3 个核心信息源；
2. 仅在必要时加入第四个信息源；
3. 共享各子集 head 的底层 MLP 参数或使用 set encoder；
4. 对组合子集限制最大阶数，例如先估计单变量与二阶组合；
5. 只对研究假设相关的组合进行分解，但必须说明此时得到的是受限子集下的分解，不能冒称完整变量集的完整 SURD；
6. SURD 不是每个 batch 计算，而是每隔若干 epoch 更新一次；
7. 对每个 POI 的估计可批量向量化，避免 Python 层逐 POI 循环计算概率。

## 10.3 计算成本报告

论文/实验报告中应记录：

- probability heads 参数量；
- 单次 MI estimator update 时间；
- 单次 Specific MI 计算时间；
- 单次 SURD decomposition 时间；
- 每轮额外显存；
- 总训练时间相对 baseline 的增长比例；
- 推理 latency（第一版推理只用 recommendation head，额外推理成本应很低）。

---

# 11. 推荐代码工程结构

建议按模块实现，不要把所有逻辑写在一个训练脚本中。

```text
project/
├── configs/
│   ├── base.yaml
│   ├── dataset.yaml
│   └── model_surd.yaml
├── data/
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── splits.py
│   └── feature_builder.py
├── models/
│   ├── encoders.py
│   ├── recommendation.py
│   ├── subset_estimators.py
│   └── full_model.py
├── information/
│   ├── specific_mi.py
│   ├── surd.py
│   ├── weights.py
│   └── calibration.py
├── losses/
│   ├── recommendation_loss.py
│   └── surd_auxiliary_loss.py
├── training/
│   ├── trainer.py
│   ├── alternating.py
│   └── checkpoint.py
├── evaluation/
│   ├── ranking_metrics.py
│   ├── calibration_metrics.py
│   └── long_tail_metrics.py
├── scripts/
│   ├── train.py
│   ├── estimate_surd.py
│   └── evaluate.py
└── outputs/
```

---

# 12. 核心模块接口建议

以下是接口设计建议，AI coding agent 可以据此实现。具体类名可调整，但功能边界应保持。

## 12.1 Dataset

每个样本至少返回：

```python
{
    "history_poi_ids": ...,
    "history_timestamps": ...,
    "target_time_features": ...,
    "spatial_features": ...,
    "user_features": ...,       # optional
    "target_poi_id": int,
    "user_id": int,
    "sample_id": int,
}
```

## 12.2 Encoders

```python
h1 = history_encoder(history_inputs)
h2 = temporal_encoder(temporal_inputs)
h3 = spatial_encoder(spatial_inputs)
```

所有输出 shape 为：

```text
[B, hidden_dim]
```

## 12.3
## 12.3 Subset estimators

输入一个子集及对应表示：

``` python
logits_A = subset_estimators["Q1_Q3"](
    torch.cat([h1, h3], dim=-1)
)
probs_A = torch.softmax(logits_A, dim=-1)
```

每个 estimator 输出：

``` text
[B, num_pois]
```

## 12.4 Specific MI estimator

输入：

-   子集概率模型对 estimation samples 的输出；
-   真实 target POI labels；
-   训练集 POI prior；
-   数值稳定参数。

输出：

``` python
specific_mi[subset][poi_id] = float
sample_counts[poi_id] = int
```

## 12.5 SURD module

输入每个目标 POI 对所有变量子集的 specific information。

输出结构建议：

``` python
{
    poi_id: {
        "unique": {...},
        "redundant": {...},
        "synergistic": {...},
    }
}
```

注意：内部必须严格实现原始 SURD 算法。不要用临时的 MI 差值公式替代。

## 12.6 Weight builder

输入 POI-specific SURD 结果与样本数，输出：

``` python
{
    "unique_weights": {
        poi_id: {source_name: weight}
    },
    "synergy_weights": {
        poi_id: {subset_name: weight}
    },
}
```

同时输出：

-   原始 signed contributions；
-   归一化后权重；
-   是否触发低样本 fallback；
-   每个 POI 的样本数。

## 12.7 Auxiliary loss

输入：

-   batch labels；
-   单变量 head logits；
-   多变量 head logits；
-   当前 POI-specific weights。

输出：

``` python
loss_unique
loss_synergy
```

取权重时必须按 batch 中每个样本的真实 label 查询：

``` python
weight = weights[target_poi_id]
```

训练时可以这样做，因为训练标签已知；推理时不使用该操作。

------------------------------------------------------------------------

# 13. 训练伪代码

``` python
# 1. Build model
model = RecommendationModel(...)
subset_estimators = SubsetEstimators(...)

# 2. Warm-up recommendation model
for epoch in range(warmup_epochs):
    for batch in train_loader:
        outputs = model(batch)
        loss_rec = cross_entropy(outputs["rec_logits"], batch["target_poi_id"])

        optimizer_rec.zero_grad()
        loss_rec.backward()
        optimizer_rec.step()

# 3. Initialize MI estimators and estimate first SURD weights
fit_or_calibrate_subset_estimators(
    model=model,
    subset_estimators=subset_estimators,
    estimator_train_loader=estimator_train_loader,
)

specific_mi = estimate_specific_mi(
    subset_estimators=subset_estimators,
    estimation_loader=estimation_loader,
    poi_prior=train_poi_prior,
)

poi_surd = run_official_surd_decomposition(specific_mi)
weights = build_poi_specific_weights(
    poi_surd=poi_surd,
    poi_sample_counts=sample_counts,
)

# 4. Alternating training
for epoch in range(total_epochs):

    # Phase B: update recommendation encoders/head using fixed SURD weights
    for batch in train_loader:
        outputs = model(batch)

        loss_rec = recommendation_loss(
            outputs["rec_logits"],
            batch["target_poi_id"],
        )

        loss_unique = unique_auxiliary_loss(
            single_source_logits=outputs["single_source_logits"],
            target=batch["target_poi_id"],
            weights=weights["unique_weights"],
        )

        loss_synergy = synergy_auxiliary_loss(
            subset_logits=outputs["subset_logits"],
            target=batch["target_poi_id"],
            weights=weights["synergy_weights"],
        )

        loss = loss_rec + lambda_u * loss_unique + lambda_s * loss_synergy

        optimizer_rec.zero_grad()
        loss.backward()
        optimizer_rec.step()

    # Phase A: periodically refresh estimators and SURD
    if should_refresh_surd(epoch):
        fit_or_calibrate_subset_estimators(...)
        specific_mi = estimate_specific_mi(...)
        poi_surd = run_official_surd_decomposition(specific_mi)
        new_weights = build_poi_specific_weights(...)

        weights = ema_update(weights, new_weights, beta=ema_beta)

    validate(...)
    save_checkpoint(...)
```

伪代码中的 `run_official_surd_decomposition` 必须由真实 SURD
算法实现，不可用 placeholder 数学近似冒充最终结果。

------------------------------------------------------------------------

# 14. 必须实现的单元测试

## 14.1 Specific MI tests

-   对所有样本标签相同的 toy data，检查计算不会崩溃；
-   检查按 POI 分组的样本数正确；
-   检查 log-ratio 的分子确实取对应 POI 的预测概率；
-   检查概率为 0 时不会出现 `inf` / `NaN`；
-   检查先验使用 (N_y/N)，而不是 (1/K)。

## 14.2 SURD tests

-   使用原论文或其官方实现中的 toy examples 对齐结果；
-   检查分解项与总信息量之间的理论一致性；
-   检查变量重排时结果符合算法预期；
-   检查单变量、二变量和三变量情况；
-   检查负值/零值处理遵循原算法，不要在核心分解之前擅自截断。

## 14.3 Training tests

-   确认每个 subset head 只接收对应的信息源；
-   确认推荐损失可以正常反向传播；
-   确认 SURD weights 已 detach，不会建立错误计算图；
-   确认 batch 内不同 target POI 会查到不同权重；
-   确认推理路径不需要 target label；
-   确认 train/validation/test 时间边界没有泄漏。

------------------------------------------------------------------------

# 15. 实验设计

## 15.1 Baselines

至少比较：

1.  History-only next POI model；
2.  Multi-source recommendation model（无 SURD）；
3.  Multi-source + single-source auxiliary losses（不使用 SURD 权重）；
4.  Multi-source + global SURD weights；
5.  Multi-source + POI-specific SURD weights（完整模型）。

如果已有强 baseline（如 GETNext、STHGCN、Flashback
或其他数据集适用模型），按其官方协议实现或复现。不要为了比较而改变数据划分。

## 15.2 Ablation

建议：

-   w/o Specific MI：只用普通辅助任务；
-   w/o SURD：不进行 R/U/S 分解；
-   Global SURD vs POI-specific SURD；
-   w/o Unique loss；
-   w/o Synergy loss；
-   仅一阶子集 vs 加入二阶/高阶组合；
-   不同 SURD refresh interval；
-   不同 $\lambda_U,\lambda_S$；
-   不同低样本 POI 阈值和收缩强度。

## 15.3 指标

Next POI ranking：

-   Recall@K；
-   NDCG@K；
-   MRR@K（如适用）。

Long-tail：

-   按训练频率将 POI 分成 Head / Mid / Tail；
-   分别报告 Recall@K、NDCG@K；
-   可报告 macro-average POI performance；
-   报告不同频率 bucket 的覆盖率和样本数。

概率估计：

-   NLL；
-   Brier score；
-   ECE。

效率：

-   参数量；
-   训练时间；
-   MI/SURD 更新耗时；
-   推理 latency。

## 15.4 关键研究假设

建议围绕以下假设组织实验：

**H1：不同 POI 具有不同的 predictive information structure。**\
通过 POI-specific SURD 分布、频率分组和案例分析验证。

**H2：只使用 global information weights 会掩盖不同 POI 的信息需求。**\
比较 global SURD 和 POI-specific SURD。

**H3：POI-specific SURD-guided auxiliary supervision
能改善多源信息利用。**\
比较普通辅助监督与 POI-specific 加权辅助监督。

**H4：改善在 tail POI 上尤其明显。**\
按 POI 频率分组报告结果，并分析估计可靠性。

注意：这些是假设，不是预设结论。若实验不支持，应如实报告。

------------------------------------------------------------------------

# 16. 重要理论与实现注意事项

1.  **Specific MI 与 global MI 不同。** Specific MI 对某一个目标 POI
    状态取条件平均；global MI 是对所有目标状态按 (P(y)) 加权后的总体量。
2.  **SURD 也可以是 POI-specific。** 先逐个 POI 做 specific
    SURD，再加权聚合得到 global SURD。
3.  **SURD 不等于任意 MI 差分。** 必须严格实现原论文定义的分配规则。
4.  **观测预测信息不自动等于因果效应。** 本项目先研究 predictive
    information decomposition。
5.  **CE 不是 SURD 项。** SURD 权重调节辅助 CE，但辅助 CE 不等于纯
    unique 或 synergy 信息。
6.  **R 不意味着 embedding 应正交。** 不要将 embedding cosine similarity
    直接当成 redundancy。
7.  **训练标签与推理标签要区分。** 训练期可用真实 (y_n) 选择
    POI-specific 权重；推理期不能使用未知标签。
8.  **tail POI 的 specific MI 不稳定。**
    必须报告样本数、置信度/稳定性，并考虑 shrinkage。
9.  **概率校准是关键。** 不能仅凭分类准确率声称 MI 估计准确。
10. **避免数据泄漏。** 特别注意历史序列截断、时间划分、POI vocabulary
    和估计集的使用。
11. **先做可验证的小版本。** 先 3 个信息源、7 个子集、固定权重、标准
    CE；确认正确后再扩展。
12. **不要在第一版引入过多复杂机制。** 先验证 POI-specific SURD
    辅助监督是否有效，再考虑候选条件化推理、冗余正则或更复杂的动态
    gating。

------------------------------------------------------------------------

# 17. 推荐实现里程碑

## Milestone 1：数据与普通推荐模型

-   构造严格时间顺序的 next-POI 样本；
-   实现 3 个信息源；
-   实现 shared encoders + recommendation head；
-   完成训练和 Recall/NDCG/MRR 评估。

**验收标准**：普通多源推荐模型可稳定训练、复现实验结果。

## Milestone 2：Subset probability estimators

-   实现 7 个非空子集概率 head；
-   检查每个 head 的输入来源；
-   训练概率估计器；
-   实现概率校准评估。

**验收标准**：所有 head 输出合法概率，NLL/ECE 可计算。

## Milestone 3：Specific MI

-   按 POI 统计先验；
-   按 POI 和子集估计 specific MI；
-   检查数值稳定性；
-   生成 CSV 和可视化。

**验收标准**：toy example 的手算结果与代码一致。

## Milestone 4：Official SURD

-   根据原论文及其补充材料实现正式算法；
-   与论文 toy examples 或官方实现对齐；
-   对每个 POI 分别分解；
-   生成 global aggregation。

**验收标准**：分解满足原算法的恒等式/一致性检查。

## Milestone 5：SURD-guided training

-   构造 POI-specific U/S weights；
-   实现 weighted auxiliary losses；
-   实现 alternating update 和 EMA；
-   加入低样本 fallback。

**验收标准**：训练无 NaN，权重查询正确，推理不依赖标签。

## Milestone 6：完整实验

-   baseline；
-   global vs POI-specific；
-   ablation；
-   long-tail；
-   calibration；
-   efficiency；
-   case studies。

**验收标准**：可以清晰回答模型是否有效、为什么有效、额外成本是多少、tail
POI 是否受益。

------------------------------------------------------------------------

# 18. 给 AI Coding Agent 的执行顺序

请不要一次性生成所有代码。按照以下顺序逐步完成，每一步先解释设计，再修改代码，再运行最小测试：

1.  先检查现有项目结构、数据格式、数据集字段和已有 baseline。
2.  输出数据样本定义和时间划分方案，等待确认后实现
    preprocessing/dataset。
3.  实现普通多源 Next POI Recommendation，先不加入 SURD。
4.  实现 7 个 subset probability estimators，并验证输入隔离。
5.  实现 Specific MI 计算和单元测试。
6.  查阅并严格复现 SURD
    原论文的正式分解算法；不要自行猜公式。若原论文算法细节不足，应先定位补充材料/官方代码，并报告来源和差异。
7.  实现逐 POI SURD 与 global aggregation。
8.  实现权重平滑、低样本 fallback 和 weighted auxiliary losses。
9.  实现交替训练。
10. 实现评估、消融、long-tail 和效率统计。

每一步都需要提供：

-   修改/新增文件列表；
-   核心代码；
-   运行命令；
-   输入输出 shape；
-   最小测试结果；
-   当前已知限制。

如果发现现有数据集不包含某项特征，不要编造该特征；应说明缺失并调整对应信息源定义。不要未经说明改变任务目标、数据划分、Specific
MI 定义或 SURD 数学算法。
