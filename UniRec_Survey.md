# UniRec + TokenMixer 技术综述：从 Encode-Then-Interaction 到统一建模

> **目标读者**：熟悉 DIN、DCN 等"先编码序列、再做特征交叉"经典两段范式的工程师或研究者，希望了解 2024–2026 年"统一建模"浪潮的来龙去脉。

---

## 一、你熟悉的世界：Encode-Then-Interaction 范式

先从你已经知道的地方出发。

工业推荐系统的排序模型，长期以来遵循一套非常固定的流水线：

```
用户行为序列
  └─→ [序列编码器] (DIN / SIM / LONGER)
        └─→ 压缩为 1~K 个 Summary Token
              └─→ 拼接用户画像 / 物品属性 / 上下文
                    └─→ [特征交叉模块] (DCN / RankMixer / Wukong)
                          └─→ 预测点击概率
```

这个范式有一个非常朴素的直觉：**先把长序列"读懂"压缩，再让压缩结果和其他特征"互相影响"**。

DIN 是这个范式的代表作——用 Target-Attention 把用户历史行为压缩成一个与候选 Item 相关的向量，然后和画像特征拼接送入 MLP。DCNv2、Wukong、RankMixer 则是后来在"特征交叉"这一侧的持续进化。

这套方案有效，经过了多年生产验证。但随着模型越做越大、序列越来越长，一些根本性的裂缝开始显现。

---

## 二、问题在哪里：三条裂缝

### 裂缝 1：Q 太单薄，且 non-seq 从未 attend 过完整序列

用 Q-K-V 语言来描述这个问题最直接：

```
DIN 的 Target-Attention：
  Q   = target item embedding（只有候选物品信息）
  K/V = 行为序列

问题出在两处：
  1. Q 太单薄——用户画像、上下文、统计特征全部缺席，
     序列压缩的方向完全由物品决定，忽略了用户的当下状态。

  2. non-seq 特征永远见不到原始序列的 K/V——
     行为序列先被压缩成 1~K 个 Summary Token，
     non-seq 特征只能事后和这个"摘要"做晚期融合（Late Fusion），
     从未有机会直接去 attend 完整的行为历史。
```

结果：**non-seq 特征的 Q 永远 attend 的是序列的压缩摘要，而不是原始 K/V**。信息永远单向流动。

### 裂缝 2：过早聚合造成信息瓶颈

行为序列可能有数千条记录。为了让后续的特征交叉模块能处理，这些记录必须在序列编码器内被**强行压缩**成固定数量的 Token（比如 1 个 CLS Token，或者 K 个 Query Token）。

压缩不可避免地丢失细粒度的 Token 级信号。这些被丢掉的信号，恰恰是解锁 Scaling Law 所需要的——你不能指望一个已经丢失信息的模型，通过加大参数就变得更好。

> **Alibaba EST 论文（2026）的直接表述**：非统一或部分统一建模通过丢弃细粒度 Token 级信号，制造了阻止 Scaling 收益释放的信息瓶颈。

### 裂缝 3：模块分离阻碍整体 Scaling

两个独立模块（序列编码 + 特征交叉）意味着：
- 两套参数，两套优化目标，不能端到端地协同更新；
- 扩大某一侧的容量，另一侧无法受益；
- 无法使用 LLM 生态中成熟的工程优化（KV Cache、FlashAttention、混合精度）。

性能提升边际递减，计算和参数不能有效转化为收益。

---

## 三、两条并行的演进路线

2024–2026 年的这批论文，实际上沿着**两条相互独立但彼此呼应**的路线演进：

```
路线 A — "特征交叉 Scaling"（TokenMixer 系列）
  问题：如何让"特征交叉模块"本身能高效 Scaling？
  关键词：硬件感知、Token Mixing、MoE、Per-Token FFN、MFU
  代表：RankMixer → MTmixAtt → TokenMixer-Large → UniMixer
  定位：在 Encode-Then-Interaction 范式"交叉侧"做纵深

路线 B — "统一建模 UniRec"（序列+交叉统一）
  问题：如何彻底消除两段范式的信息瓶颈？
  关键词：统一 Tokenizer、双向交互、逐层 Cross-Attention
  代表：InterFormer → OneTrans → HyFormer → EST → TokenFormer
  定位：打破两段管道，重新定义推荐模型骨干
```

两条路线不是对立的：TokenMixer 系列解决的是"给定异构 Token，如何高效交叉"；UniRec 系列解决的是"序列和非序列 Token 根本就不应该分离"。**UniRec 中的 Token Mixing 设计，大量借鉴了 RankMixer 的思想**——OneTrans 的 Per-Token FFN、HyFormer 的 Query Boosting，都能看到 RankMixer 的影子。

---

## 四、路线 A：TokenMixer 系列（特征交叉侧的 Scaling 探索）

这条路线的核心问题是：**CPU 时代遗留的特征交叉算子（FM、DCN、DotProduct）在 GPU 上的 MFU 极低，既无法扩容又无法 Scaling，怎么破？**

---

### 4.1 RankMixer（ByteDance，2025.07）—— 硬件感知的 Token Mixing 开山之作

**arXiv**: [2507.15551](https://arxiv.org/abs/2507.15551)

**核心诊断：MFU 危机**

传统特征交叉模块的 MFU 仅有 **4.5%**（字节跳动实测）。原因是 FM 类算子、点积交叉等都是内存密集型操作，无法充分利用 GPU 的计算单元。结果是：算力买了，却用不上；加参数、加模块，延迟飙升但收益有限。

**三个核心设计**：

**① 用 Token Mixing 替代 Self-Attention**

把每个特征域视为一个 Token，用一个**多头 Token Mixing 算子**替代 O(T²) 的 Self-Attention。Token Mixing 的本质是：用一个固定（无参数）的混合矩阵在 Token 之间广播信息，计算量是 O(T) 而非 O(T²)。

```
标准 Self-Attention：
  Q = XW_Q, K = XW_K, V = XW_V
  Attn = softmax(QKᵀ/√d) · V    ← O(T²·d) 计算

RankMixer Token Mixing（无参数广播）：
  X_mixed = MixMatrix · X          ← O(T²) 但 T 很小（~16）
  然后通过 Per-Token FFN 做 domain-specific 变换
```

> **直觉**：T（特征 Token 数）通常只有 10~30，Token Mixing 矩阵的计算量微乎其微；真正的参数容量和计算量放在每个 Token 独立的 FFN 上。

**② Per-Token FFN（每个 Token 独立的 FFN）**

这是 RankMixer 最重要的贡献之一。不同特征域（用户画像、物品属性、上下文）的语义空间完全不同，用共享 FFN 会导致"异构空间主导问题"——某个主导特征域的梯度会压制其他域的学习。

解决方案：为每个 Token（每个特征域）分配**独立的 FFN 参数**，让每个域在自己的语义空间内自由变换，互不干扰。

```
共享 FFN（传统）：
  所有 Token 共用同一个 FFN(·)

Per-Token FFN（RankMixer）：
  Token_i → FFN_i(·)   每个 Token 有专属 FFN
  Token_j → FFN_j(·)
  ...
```

**③ Sparse MoE 扩展到 10 亿参数**

在 Per-Token FFN 的基础上，引入**稀疏 MoE（每个 Token 激活少量专家）**，使模型可以扩展到 1B+ 参数，同时保持推理延迟基本不变（稀疏激活 → 计算量不随总参数线性增长）。

**效果**：MFU 从 4.5% → **45%**（提升 10 倍）；参数规模扩大 100 倍，推理延迟不变；在线 A/B 测试用户活跃天数 +0.3%，总使用时长 +1.08%。

---

### 4.2 MTmixAtt（Meituan，2025.10）—— 多场景统一 + 自动 Tokenization

**arXiv**: [2510.15286](https://arxiv.org/abs/2510.15286)

**背景**：美团内部有多个推荐场景（首页、搜索、广告等），每个场景维护独立的特征工程和场景定制架构，维护成本极高，且难以迁移知识。

**两个核心模块**：

**① AutoToken（自动 Tokenization）**

传统做法：人工划定特征分组，比如"用户特征 Token"、"物品特征 Token"、"交叉特征 Token"，需要大量业务经验，且不同场景的分组方式不同。

MTmixAtt 的做法：用**可学习的聚类机制**，自动将异构特征聚合成语义连贯的 Token，无需人工干预。

```
传统（手工）：
  规则: [用户年龄, 城市, 性别] → UserToken
        [物品类目, 价格, 品牌] → ItemToken

AutoToken（自动）：
  学习: 哪些特征在语义上相关？
  输出: 自动聚类后的 Token 序列（跨域特征可能被聚到同一 Token）
```

**② MTmixAttBlock（三合一交互模块）**

在一个 Block 内整合三种机制：

| 组件 | 作用 | 参数类型 |
|------|------|---------|
| 可学习混合矩阵 | Token 间信息广播（替代固定的 MixMatrix） | **有参数**（vs RankMixer 无参数） |
| 共享 Dense Expert | 捕捉跨场景通用模式 | 所有场景共享 |
| 场景感知 Sparse Expert | 捕捉场景特定行为 | 每个场景独立激活 |

> **与 RankMixer 的关键区别**：RankMixer 的 Token Mixing 矩阵是无参数的（固定规则），MTmixAtt 引入了**可学习的混合矩阵**，让混合模式随数据优化。这一设计后来被 UniMixer 理论化。

**效果**：美团首页场景 Payment PV +3.62%，Actual Payment GTV +2.54%；扩展到 1B 参数后，性能单调提升。

---

### 4.3 TokenMixer-Large（ByteDance，2026.02）—— 解决深层 RankMixer 的工程瓶颈

**arXiv**: [2602.06563](https://arxiv.org/abs/2602.06563)

**背景**：RankMixer 在浅层配置下表现优秀，但当模型加深（堆叠更多 Block）时，出现三个工程问题：
1. **次优残差路径**：传统残差连接在深层网络中信息流不充分；
2. **梯度消失**：Block 数增加后，底层梯度传播困难；
3. **不完整的 MoE 稀疏化**：大规模专家训练时，部分专家利用率不均衡，无效参数占比高。

**三个关键修复**：

| 问题 | 解决方案 | 机制 |
|------|---------|------|
| 残差路径次优 | **Mixing-and-Reverting** 操作 | 在 Token Mixing 后增加一个"还原"步骤，保留原始特征的高频信息，防止过度平滑 |
| 梯度消失 | **跨层残差（Inter-layer Residuals）** | 每个 Block 的输出直接与非相邻层建立跳跃连接 |
| MoE 不均衡 | **辅助损失（Auxiliary Loss）** | 显式惩罚专家负载不均衡，确保每个专家得到充分训练 |

此外，引入 **Sparse Per-Token MoE**，在每个 Token 的独立 FFN 上再叠加稀疏专家，将模型扩展到 **70 亿和 150 亿参数**。

**效果**：字节跳动电商 GMV +2.98%，广告 ADSS +2.0%，直播打赏收入 +1.4%（均为线上 A/B 测试结果）。

---

### 4.4 UniMixer（快手，2026.04）—— 统一理论：三大架构殊途同归

**arXiv**: [2604.00590](https://arxiv.org/abs/2604.00590)

**核心贡献**：UniMixer 不是提出一个新的工程 trick，而是**从理论上证明推荐系统三大主流 Scaling 架构是等价的**，并据此推导出更优的统一设计。

**三大架构的统一视角**

推荐 Scaling 领域此前有三类主流方法，看起来差异很大：

| 类型 | 代表 | 基本思路 |
|------|------|---------|
| Attention-based | HiFormer, AutoInt | Self-Attention 学习 Token 间关系 |
| TokenMixer-based | RankMixer | 固定混合矩阵广播 + Per-Token FFN |
| FM-based | Wukong, DCN | 显式特征域两两/高阶交叉 |

UniMixer 的关键洞见：**RankMixer 的"固定混合矩阵"，其实是 Attention 的一个特殊情况（固定权重的 Attention）**，而 FM 的双线性交叉也可以被表示为特定参数化的 Token 混合。

> **数学直觉**：把 RankMixer 的规则混合矩阵替换成**可学习的参数化矩阵**，它就能退化成 Attention（当参数收敛到 softmax 权重时），也能退化成 FM（当参数学到双线性交叉模式时）。

**两个核心创新**：

**① Parameterized Token Mixing（参数化 Token 混合）**

```
RankMixer（固定规则）：
  X_out = W_fixed · X          W_fixed 不可学习

UniMixer（可学习）：
  X_out = W_learnable · X      W_learnable 随梯度更新
```

这一步解除了 RankMixer 中 `头数 == Token 数` 的硬约束（因为固定矩阵的维度是固定的），让混合模式可以被数据驱动地优化。

**② UniMixing-Lite（轻量版本）**

在保持理论统一性的同时，针对工业部署压缩参数量和计算量：通过低秩分解和稀疏激活，在性能基本不损失的前提下将计算开销降低 30-50%。

**效果**：Scaling 曲线优于 RankMixer、Wukong 等基线；同参数量下 AUC 更高，且 Scaling 斜率更大（参数利用效率更高）。

---

## 五、路线 B：UniRec 系列（序列与特征交叉的统一建模）

这批论文共同追求的目标是：

> **在单个骨干网络内，同时完成序列建模和特征交叉，让两类信息能够双向、实时、逐层地相互影响。**

核心变化：

```
旧范式（Encode-Then-Interaction）：
  序列 → [编码器] → Summary Token ──→ [交叉模块] → 预测
                                   ↑
                          画像/属性/上下文

新范式（UniRec）：
  序列 Token ──┐
               ├──→ [统一骨干 Block] × N ──→ 预测
  NS Token   ──┘   每一层内，两类 Token 双向交互
```

**用 Q-K-V 看三代演进，一张表说清楚：**

| 方法 | Q 是谁 | K/V 是谁 | 交互方向 | non-seq 能 attend 原始序列吗 |
|------|--------|---------|---------|:---------------------------:|
| **DIN** | target item | 行为序列 | 单向（item→seq） | ❌ non-seq 只和 Summary 融合 |
| **InterFormer** | seq summary ↔ non-seq summary | 对方 summary | 双向桥接 | ❌ 仍是 attend Summary |
| **OneTrans** | 所有 Token（Self-Attn） | 所有 Token | 双向 | ✅ 直接 attend 原始序列 |
| **HyFormer** | NS Token（专职 Q） | seq Token（专职 K/V） | 单向（NS→seq） | ✅ 直接 attend 原始序列 |
| **EST** | 所有 Token（LCA 剪枝） | 高质量 seq（CSA 筛选） | 双向但稀疏 | ✅ 无损输入，直接 attend |
| **TokenFormer** | 所有 Token（分层保护） | 所有 Token | 底层双向/上层收窄 | ✅ 但需要 BFTS 保护 |

> **核心进步可以用一句话概括**：UniRec 系列的本质，就是让 non-seq 特征的 Q，从"只能 attend 序列摘要"变为"能直接 attend 原始序列 K/V"。

---

### 5.1 InterFormer（Meta，2024.11）—— 第一步：双向桥接

**arXiv**: [2411.09852](https://arxiv.org/abs/2411.09852)

**侧重点**：发现并定义问题，提出双向交互的最小化改造。

**Q-K-V 视角**：
```
序列侧 Self-Attn：  Q = K = V = seq Token（序列内部自注意力）
非序列侧：          Q = K = V = non-seq Token（特征交叉内部）
桥接 Attention：    Q = non-seq summary，K/V = seq summary（双向互读摘要）

局限：non-seq 的 Q attend 的仍是 seq 的压缩摘要，不是原始序列。
```

InterFormer 没有直接把两个模块合并，而是在它们之间建了一座**双向的桥（Bridging Arch）**：

```
行为序列模块  ←──→  桥接结构  ←──→  非序列特征模块
  (保持完整)    双向信息选择    (保持完整)
```

两个核心设计：
1. **保留完整信息**：序列和非序列特征各自保持完整，不做早期压缩；
2. **交错式交互（Interleaving）**：两个模块的 Transformer Layer 交替堆叠，每隔几层通过桥接结构互换精炼摘要。

**评价**：这是范式转换的第一步，证明了双向交互有效。但两个模块仍然是分离的，架构复杂度偏高。

---

### 5.2 OneTrans（ByteDance + NTU，2025.10）—— 统一架构的第一个工业级方案

**arXiv**: [2510.26104](https://arxiv.org/abs/2510.26104) | **WWW 2026**

**侧重点**：彻底打通，用一个 Transformer 做所有事；工程落地。

**Q-K-V 视角**：
```
全局 Self-Attention：
  Q = K = V = [seq Token × N, NS Token × M]   所有 Token 混在一起

non-seq 的角色：既是 Q（主动读 seq），也是 K/V（被 seq 读）——双向
seq 的角色：既是 K/V（被 non-seq 读），也是 Q（读其他 seq 和 non-seq）——双向

突破：non-seq 的 Q 第一次能直接 attend 原始序列 K/V（非摘要）。
注意：不是专职的 Cross-Attention，双方角色平等，这点区别于 HyFormer。
```

OneTrans 做了三件关键的事：

**① 统一 Tokenizer**：序列特征和非序列特征全部转换成一条 Token Sequence，用 `[SEP]` 区分不同来源：

```
[SEP] 行为A 行为B ... 行为N [SEP] 行为C ... | 画像Token 物品Token 上下文Token
  └───── 序列 Token ──────────────────────┘  └───── 非序列(NS) Token ────┘
```

**② Mixed Parameterization（混合参数化）**：借鉴 RankMixer 的 Per-Token FFN 思想并推广到统一骨干：

| Token 类型 | 参数策略 | 原因 |
|-----------|---------|------|
| 序列 Token（行为历史） | **共享一套** QKV + FFN | 同质，数量大，共享参数高效 |
| 非序列 Token（画像/属性/上下文） | **每个独立一套** Token-Specific 参数 | 语义异质，需要保留各域独特表达 |

**③ Cross-Request KV Cache**：用户侧 Token 的 KV 在同一次请求的多个候选物品之间只计算一次，时间复杂度从 O(C) 降至 O(1)。

**线上效果**：GMV +5.68%（字节跳动，WWW 2026）。

---

### 5.3 HyFormer（ByteDance，2026.01）—— 让非序列特征主动"读"序列

**arXiv**: [2601.12681](https://arxiv.org/abs/2601.12681)

**侧重点**：重新思考 Query Token 的角色，让非序列特征从被动等待变为主动解码。

**Q-K-V 视角**：
```
Query Decoding 步（专职 Cross-Attention）：
  Q   = NS Token（非序列特征，num_queries 个，类似 DIN 的 target 但更丰富）
  K/V = seq Token（完整行为序列，不压缩）
  方向：单向（NS→seq），non-seq 主动读序列

Query Boosting 步（NS 内部 Token Mixing）：
  Q = K = V = NS Token（NS 之间做特征交叉）

与 DIN 的关系：DIN 是 HyFormer 的精神原型——
  DIN:      Q = 1个 target item，  K/V = seq
  HyFormer: Q = M个 NS Token（全部 non-seq 特征），K/V = seq
  进步：Q 从"只有物品"扩展到"所有非序列特征"，且 seq 不被压缩。
```

HyFormer 的核心洞见：**非序列特征应该成为解码行为序列的"查询"，而不是事后被动融合的对象。**

每个 HyFormer Block 循环执行两步：

```
第一步 — Query Decoding（用 NS Token 主动读序列）：
  非序列特征 → 扩展为 Global Tokens（这些 Token 的 Q 来自 NS 特征）
  Global Tokens 作为 Query，对行为序列的逐层 KV 做 Cross-Attention
  "让画像/上下文去主动理解行为历史"

第二步 — Query Boosting（Global Tokens 之间互相交叉）：
  Global Tokens 之间做高效 Token Mixing（借鉴 RankMixer）
  "不同维度的非序列特征再做一轮特征交叉"
```

**与 OneTrans 的区别**：

| | OneTrans | HyFormer |
|---|---|---|
| 交互模式 | 全局 Self-Attention（双向平等） | Cross-Attention（NS→seq）+ NS 内 Self-Attention |
| 设计哲学 | 工程简洁，统一到一个序列 | 角色分工，非序列是主动解码者 |
| Query Boosting | 无 | 有（借鉴 RankMixer Token Mixing） |

**线上效果**：已全量部署于字节跳动，服务数十亿用户。

---

### 5.4 EST（Alibaba Taobao，2026.02）—— 理论化 Scaling：无损输入 + 精准注意力

**arXiv**: [2602.10811](https://arxiv.org/abs/2602.10811)

**侧重点**：严格论证推荐系统与 LLM 的本质区别，提出保留幂律 Scaling 的最小修改。

**Q-K-V 视角**：
```
基础设置（无损输入）：
  Q = K = V = 全部原始 Token（seq + non-seq），无任何压缩

LCA 的作用（约束 Q-K 对）：
  剪掉低价值的 Q-K 对（如 seq Token 之间的自交互）
  只保留高价值的跨类型 Q-K 对（non-seq Q → seq K/V）
  → 相当于做了一个稀疏 Attention Mask

CSA 的作用（筛选 K/V 质量）：
  用内容相似度动态决定哪些 seq K/V 值得被 attend
  → 解决了"行为序列里有大量低质量噪声 K/V"的问题
```

EST 首先系统分析了"为什么推荐系统不能直接照搬 LLM 架构"：

| 属性 | LLM | CTR 预测 |
|------|-----|---------|
| 信息密度 | Token 间基本均匀 | 行为 Token 密度低，非行为 Token 密度高，严重不对称 |
| 模态先验 | 统一文本 | 行为信号 vs 内容信号有本质不同的结构先验 |

基于这个分析，EST 提出：**全部原始特征无损进入单一序列，但用两个专门设计的模块替代朴素的 Self-Attention**：

| 模块 | 作用 | 解决的问题 |
|------|------|----------|
| **LCA**（轻量跨特征注意力） | 剪除冗余自交互，聚焦跨特征高价值依赖 | 行为 Token 之间大量 Self-Attention 是低价值的 |
| **CSA**（内容稀疏注意力） | 用内容相似度动态筛选高信号行为 | 稠密注意力无法区分高/低质量行为，引入噪声 |

**最重要的结论**：去掉信息瓶颈之后，EST 验证了 CTR 预测模型的**稳定幂律 Scaling 关系**——AUC 随参数量对数线性增长，可预测、可规划。

**线上效果**：Taobao 广告 RPM +3.27%，CTR +1.22%。

---

### 5.5 TokenFormer（Tencent，2026.04）—— 发现统一建模的新隐患：SCP 崩溃

**arXiv**: [2604.13737](https://arxiv.org/abs/2604.13737)

**侧重点**：做了别人没做的反向实验，发现朴素统一建模的失效模式，并提出针对性修复。

**Q-K-V 视角**：
```
朴素统一建模的问题（SCP 的 Q-K-V 解释）：
  Q = K = V = [seq Token, non-seq Token]   直接混合
  non-seq Token 的维度分布与 seq 差异大
  → 混合后 Q·Kᵀ 的方差被拉低，softmax 趋于均匀
  → seq Token 的 V 输出退化为所有 V 的加权均值，不同维度趋同（SCP）

BFTS 的修复逻辑：
  底层：允许完全混合 Q/K/V，建立语义对齐
  上层：收窄 seq Token 的 K/V 感受野
        → 减少 non-seq 对 seq 的 Q·Kᵀ 干扰
        → 保护 seq Token 的 V 输出多样性
```

TokenFormer 发现了一个此前被忽视的现象：

> **Sequential Collapse Propagation（SCP，序列崩溃传播）**：  
> 当行为序列 Token 和维度不匹配的非序列字段直接做 Attention 交互时，序列特征的表示会发生**维度坍缩**——序列 Token 的不同维度趋于相同，失去区分能力，且这种坍缩沿网络深度向下传播，最终导致整个序列表达崩溃。

**两个修复方案**：

**① BFTS（Bottom-Full-Top-Sliding）注意力方案**

```
底层 Block（建立语义对齐）：全局 Full Attention
  → 所有 Token 充分交互，建立共享语义空间，对齐序列与非序列

上层 Block（保护序列表达）：收缩窗口滑动注意力
  → 逐渐缩小序列 Token 的感受野，减少非序列字段的维度干扰
```

**② NLIR（非线性交互表示）**

对 Hidden State 施加单侧非线性乘法变换，增强非线性特征交互，防止全局注意力下 Token 表示趋于线性均值、丧失判别性。

**评价**：这是这批论文中最具"防御性思维"的一篇——不只告诉你统一建模有多好，还告诉你做错了会怎么崩，以及如何避免。

---

## 六、两条路线的技术演进全景

### 6.1 路线 A：TokenMixer 系列演进

```
2025.07  RankMixer（ByteDance）
  ├─ 问题诊断：传统交叉算子 MFU 仅 4.5%，CPU 时代设计不适配 GPU
  ├─ 核心方案：Token Mixing（无参数广播）+ Per-Token FFN + Sparse MoE
  └─ 效果：MFU → 45%，参数 100x，延迟不变

2025.10  MTmixAtt（Meituan）
  ├─ 新问题：多场景需要统一，人工特征分组难以跨场维护
  ├─ 核心方案：AutoToken（自动 Tokenization）+ 可学习混合矩阵 + 场景 MoE
  └─ 关键进步：混合矩阵从"固定规则"变为"可学习参数"

2026.02  TokenMixer-Large（ByteDance）
  ├─ 新问题：深层网络的梯度消失、残差次优、MoE 不均衡
  ├─ 核心方案：Mixing-and-Reverting + 跨层残差 + 辅助损失
  └─ 效果：成功扩展到 7B / 15B 参数级别

2026.04  UniMixer（Kuaishou）
  ├─ 新贡献：从理论上统一三大 Scaling 架构（Attention / TokenMixer / FM）
  ├─ 核心方案：参数化 Token 混合矩阵（解除"头数==Token数"约束）
  └─ 效果：更高的 Scaling 效率，性能优于所有单一架构基线
```

### 6.2 路线 B：UniRec 系列演进

```
2024.11  InterFormer（Meta）
  ├─ 问题诊断：单向信息流 + 过早聚合是两段范式的根本痛点
  └─ 方案：双向桥接（仍是两个独立模块，最小化改造）

2025.10  OneTrans（ByteDance）
  ├─ 彻底统一：单一 Transformer 骨干 + 统一 Tokenizer
  ├─ 借鉴 RankMixer：序列 Token 共享参数，NS Token 各自独立（Per-Token）
  └─ 工程创新：KV Cache 使服务成本 O(C) → O(1)

2026.01  HyFormer（ByteDance）
  ├─ 深化统一：NS Token 主动充当 Query，Cross-Attention 解码序列
  ├─ 借鉴 RankMixer：Query Boosting 步使用 Token Mixing
  └─ 思想：非序列特征是"主动解码者"而非"被动融合对象"

2026.02  EST（Alibaba）
  ├─ 理论基础：识别 CTR vs LLM 的两大本质区别（信息密度不对称）
  ├─ 无损输入：全序列进入单一骨干，零信息丢失
  └─ LCA/CSA：精准处理不对称信息，验证幂律 Scaling

2026.04  TokenFormer（Tencent）
  ├─ 反向研究：发现 SCP 序列崩溃现象，统一建模有陷阱
  └─ 修复：BFTS 分层注意力 + NLIR 非线性变换
```

---

## 七、横向对比

### 7.1 TokenMixer 系列对比

| 维度 | RankMixer | MTmixAtt | TokenMixer-Large | UniMixer |
|------|:---------:|:--------:|:----------------:|:--------:|
| **混合矩阵** | 固定无参数 | **可学习** | 固定（继承 RankMixer） | **可学习**（统一理论） |
| **Token 分组** | 人工定义 | **AutoToken 自动聚类** | 人工定义 | 人工/自动 |
| **FFN 策略** | Per-Token 独立 FFN | 共享+场景稀疏 MoE | Sparse Per-Token MoE | Per-Token + Lite 压缩 |
| **参数规模** | ~1B | ~1B | **7B / 15B** | 不公开 |
| **核心贡献** | MFU 优化 + 奠基 | 多场景统一 | 深层稳定训练 | 理论统一三大架构 |
| **机构** | ByteDance | Meituan | ByteDance | Kuaishou |

### 7.2 UniRec 系列 Q-K-V 对比（核心）

| 方法 | non-seq 做 Q 吗 | Q 的内容 | K/V 是原始序列吗 | 交互方向 | 最大进步 |
|------|:--------------:|---------|:--------------:|---------|---------|
| **DIN** | ❌ | 只有 target item | ✅（但 non-seq 看不到） | 单向 | — 基线 — |
| **InterFormer** | ❌（仍用摘要） | non-seq summary | ❌ seq 摘要 | 双向桥接 | 双向，但 Q 仍 attend 摘要 |
| **OneTrans** | ✅（Self-Attn） | 所有 non-seq Token | ✅ | 双向平等 | non-seq Q 首次 attend 原始 seq |
| **HyFormer** | ✅（专职 Q） | 所有 non-seq Token | ✅ | 单向（NS→seq） | 专职 Cross-Attn，角色分工最清晰 |
| **EST** | ✅（LCA 剪枝） | 所有 non-seq Token | ✅（CSA 筛选高质量） | 双向稀疏 | Q-K 对和 K/V 质量同时优化 |
| **TokenFormer** | ✅（BFTS 保护） | 所有 non-seq Token | ✅（上层感受野收窄） | 底层双向/上层收窄 | 修复 SCP，保护 seq V 输出多样性 |

### 7.3 UniRec 系列综合对比

| 维度 | InterFormer | OneTrans | HyFormer | EST | TokenFormer |
|------|:-----------:|:--------:|:--------:|:---:|:-----------:|
| **统一化程度** | 部分（双向桥接） | 完全统一 | 完全统一 | 完全统一 | 完全统一 |
| **核心贡献类型** | 发现问题 | 工程落地 | 序列解码重设计 | 理论 Scaling | 失效模式分析 |
| **序列处理方式** | 保留完整，桥接 | 金字塔压缩 + Cache | Layer-wise KV 解码 | 全序列无损输入 | BFTS 分层注意力 |
| **non-seq 在 Attn 中的角色** | 被动融合（Q attend 摘要） | Q + K/V（Self-Attn） | 专职 Q（Cross-Attn） | Q（LCA 剪枝后） | Q（BFTS 保护后） |
| **最关注的问题** | 单向流 + 早聚合 | 工程可行性 | Q 角色设计 | Scaling Law | SCP 崩溃 |
| **机构** | Meta | ByteDance | ByteDance | Alibaba | Tencent |

### 7.4 两条路线的 Token 设计对比

| 方法 | Token 分组策略 | 序列 Token 参数 | 非序列 Token 参数 | 序列是否参与统一骨干 |
|------|--------------|:--------------:|:-----------------:|:------------------:|
| RankMixer | 人工分组 | — | Per-Token 独立 FFN | ❌ 仅特征交叉侧 |
| MTmixAtt | AutoToken 自动 | — | 共享+场景稀疏 | ❌ 仅特征交叉侧 |
| UniMixer | 人工/自动 | — | 可学习混合 | ❌ 仅特征交叉侧 |
| OneTrans | 统一 Tokenizer | 共享一套 | **每个独立一套** | ✅ |
| HyFormer | 统一骨干 | 被 Q 解码 | 充当 Q（Cross-Attn） | ✅ |
| EST | 全序列无损 | 稀疏注意力（CSA） | LCA 剪枝 | ✅ |

---

## 八、结论：这个方向告诉我们什么

### 8.1 两条路线殊途同归，但解决的问题不同

- **TokenMixer 系列**解决的是"给定异构 Token，如何高效、可 Scaling 地做特征交叉"——这是在两段范式框架内把"交叉侧"做到极致。
- **UniRec 系列**解决的是"两段范式本身的结构性缺陷"——打破隔离，让序列和特征交叉真正融合。

两条路线不是竞争关系：**UniRec 中的 Token Mixing 设计大量借鉴了 RankMixer**。HyFormer 的 Query Boosting、OneTrans 的 Per-Token FFN，都是 RankMixer 思想在统一骨干中的延续。

### 8.2 序列 Token 和非序列 Token 的本质不对等

这是贯穿所有论文的核心认知：

| | 序列 Token（行为历史） | 非序列 Token（画像/属性/上下文） |
|--|--|--|
| 数量 | 多（数十到数千） | 少（数个到数十） |
| 语义 | 同质（都是"行为事件"） | 异质（每个字段语义独立） |
| 信息密度 | 低 | 高 |
| 参数策略 | 适合**共享** | 适合**独立** |
| 理想角色 | 被解码的 K/V | 主动解码的 Q |

### 8.3 统一建模不是银弹，需要精心设计

你不能朴素地把所有 Token 扔进一个标准 Transformer——TokenFormer 的 SCP 实验告诉你这样做会崩。统一建模需要：
- 区分序列 Token 和非序列 Token 的参数策略（OneTrans）；
- 让非序列 Token 主动充当 Query（HyFormer）；
- 精心控制 K/V 的质量（EST 的 CSA）；
- 保护序列 Token 的表达不被 non-seq 污染（TokenFormer 的 BFTS）。

### 8.4 给实践者的建议

```
第一步（特征交叉侧优化，低风险）：TokenMixer 系列思路
  - 检查现有特征交叉模块的 MFU，评估是否存在 CPU 时代算子瓶颈
  - 引入 Per-Token FFN，让每个特征域有独立的变换空间
  - 用可学习混合矩阵（UniMixer 思路）替代固定规则交叉

第二步（低成本验证统一建模收益）：InterFormer 式双向桥接
  - 在现有序列编码器和特征交叉模块之间加桥接结构
  - 验证双向信息流的收益，评估全面改造的 ROI

第三步（架构统一）：OneTrans / HyFormer 式统一骨干
  - 统一 Tokenizer，让序列 Token 和非序列 Token 进入同一序列
  - 注意 KV Cache 等推理效率设计，确保延迟可控

第四步（稳定 Scaling）：EST 式无损输入 + 精准注意力
  - 验证信息瓶颈是否已消除，检查 Scaling 曲线是否符合幂律
  - 必要时引入 LCA/CSA 类稀疏注意力降低噪声

始终注意（TokenFormer 的警示）：
  - 监测序列 Token 表示的维度健康度（特征方差、有效秩）
  - 如有 SCP 症状，引入 BFTS 或等效的分层注意力保护措施
```

---

## 九、论文索引

### 路线 A：TokenMixer 系列

| 论文 | arXiv | 时间 | 机构 |
|------|-------|------|------|
| RankMixer | [2507.15551](https://arxiv.org/abs/2507.15551) | 2025.07 | ByteDance |
| MTmixAtt | [2510.15286](https://arxiv.org/abs/2510.15286) | 2025.10 | Meituan |
| TokenMixer-Large | [2602.06563](https://arxiv.org/abs/2602.06563) | 2026.02 | ByteDance |
| UniMixer | [2604.00590](https://arxiv.org/abs/2604.00590) | 2026.04 | Kuaishou |

### 路线 B：UniRec 系列

| 论文 | arXiv | 时间 | 机构 | 发表 |
|------|-------|------|------|------|
| InterFormer | [2411.09852](https://arxiv.org/abs/2411.09852) | 2024.11 | Meta | — |
| OneTrans | [2510.26104](https://arxiv.org/abs/2510.26104) | 2025.10 | ByteDance + NTU | WWW 2026 |
| HyFormer | [2601.12681](https://arxiv.org/abs/2601.12681) | 2026.01 | ByteDance | — |
| EST | [2602.10811](https://arxiv.org/abs/2602.10811) | 2026.02 | Alibaba | — |
| TokenFormer | [2604.13737](https://arxiv.org/abs/2604.13737) | 2026.04 | Tencent | — |

---

*相关背景论文（Encode-Then-Interaction 范式演进）：DIN (2018)、SIM (2020)、TWIN (2023)、LONGER (2025)、Wukong (2024)、DCNv2 (2021)*
