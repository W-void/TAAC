# TAAC 改动记录
---

## 2026-05-20

### r6：DIG-style 逐 fid 序列信息注入（`sid_mode='fid_order'`）

#### 背景与动机

参考 DIG（Discrimination Is Generation, arXiv 2605.14853）论文的核心设计：SID 通过残差量化构建层次化码本，每层 SID embedding 前缀累加（`e_sid^(1:l) = Σ eˡ[sˡ]`）可从粗到细逐步逼近完整 item 表示，作用类似于正则——浅层 block 只拿到粗粒度信息，被迫学习更泛化的表示。

TAAC 的序列每个 item 天然携带多个 sideinfo fid（schema.json 中每个序列域有 8–13 个 fid，如 item_id、shop_id、类目 id 等），这些 fid 在结构上与 DIG 的多层 SID token 对应。原实现将所有 fid concat 后一次性投影成一个 token，没有利用这种结构。

**改动目标**：将 DIG 的"逐层信息恢复"映射到 TAAC，让第 k 层 block 只看到前 `k_fids(k)` 个 fid 的前缀和，`k_fids` 随 block 深度线性增大（粗→细），同时保证下游 block 参数（Attention、FFN）完全共享，与 DIG 的 prefix-sum 逻辑对应。

#### 实现

**关键设计决策**：

- **不用 concat → 单 Linear**，改用每个 fid 各自一个独立 `Linear(emb_dim → D)`，输出 D 维后直接累加。这样无论当前 block 使用几个 fid，输出维度始终是 D，下游 block 参数完全共享，与 DIG prefix-sum 完全对应。
- 每层 block 前按线性调度计算 `k_fids`：

```
block 0:   k = ceil(S * 1 / N_blocks)
block 1:   k = ceil(S * 2 / N_blocks)
...
block N-1: k = S  (全量 fid)
```

- 时间特征（time_bucket / hour / weekday / diff）单独收集为 `time_delta`，加到每层的前缀和上后再过共享 `LayerNorm`。

**数据流**（`sid_mode='fid_order'`）：

```
_build_seq_tokens()
  ├─ _embed_seq_domain(..., fid_projs=...)
  │    └─ 每个 fid: emb(fid_i) → Linear_i(emb_dim→D) → gelu → fid_token_i
  │    └─ time_delta = time_bucket + hour + weekday + diff_*
  │    └─ 返回 (fid_token_list, time_delta)
  └─ full_token = LN(Σ fid_token_list + time_delta)   ← query_generator 用

_run_multi_seq_blocks()
  for blk_idx, block in blocks:
      k = ceil(S * (blk_idx+1) / N)
      seq_token = LN(Σ fid_token_list[:k] + time_delta)  ← 该层 block 用
      block(seq_token, ...)
```

#### 新增参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sid_mode` | `'none'` | `'none'` = 原有行为；`'fid_order'` = 逐 fid 前缀注入 |

- `sid_mode='none'`：行为与原始完全相同，旧 checkpoint 可直接加载。
- `sid_mode='fid_order'`：新增 `_seq_fid_projs`（每个域 S 个 `Linear(emb_dim→D)`）和共享 `LayerNorm`，替换原来的 `Linear(S*emb_dim→D) + LayerNorm`。

#### 改动文件

- `model.py`：
  - `PCVRHyFormer.__init__`：新增 `sid_mode` 参数；`_seq_fid_projs`（`nn.ModuleDict`）；`fid_order` 时 `_seq_proj` 改为共享 `LayerNorm`。
  - `_embed_seq_domain`：新增 `fid_projs` 参数，`fid_order` 路径返回 `(fid_token_list, time_delta)`。
  - `_build_seq_tokens`：`fid_order` 模式下额外返回 `fid_tokens_list` 和 `time_deltas_list`。
  - `_run_multi_seq_blocks`：新增 `fid_tokens_list` / `time_deltas_list` 参数；`fid_order` 模式下每层 block 前重建 seq token。
  - `forward` / `predict`：解包 `_build_seq_tokens` 返回值，透传给 `_run_multi_seq_blocks`。
- `train.py`：新增 `--sid_mode` CLI 参数（choices: `none` / `fid_order`，默认 `none`）；`model_args` 加 `sid_mode`。
- `infer.py`：`_FALLBACK_MODEL_CFG` 加 `'sid_mode': 'none'`。

---

## 2026-05-19

### 五项 UniRec 架构增强（r1–r5）

#### 总览

| ID | 改动 | 文件 | 关键字 |
|----|------|------|--------|
| r1 | Layer-wise Q 动态更新 | `model.py` | `DynamicQueryRefiner` |
| r2 | CSA 序列质量稀疏筛选 | `model.py` | `_csa_mask`, `csa_top_k` |
| r3 | SCP 防护（最后 Block） | `model.py` | `protect_seq` |
| r4 | Domain Embedding for Q | `model.py` | `domain_embs` (内含于 r1) |
| r5 | 全局 skip connection | `model.py` | `global_skip_proj` |

---

#### r1 + r4：Layer-wise Q 动态更新 + Domain Embedding

**动机**：原架构中初始 Q 由 `MultiSeqQueryGenerator` 生成后保持不变，随着 NS Token 在多层 Block 中迭代更新，Q 与 NS 之间的语义逐渐错位；不同序列域（click/cart/order）的 Q 使用相同的生成路径，缺乏域特化能力。

**实现**：新增 `DynamicQueryRefiner` 模块，在每个 `MultiSeqHyFormerBlock` 的 Query Decoding 步骤之前，用**当前已更新的 NS Token** 重新生成 Q：

```
ctx_i = LayerNorm(NS_mean + SeqMeanPool_i + DomainEmb_i)
Q_i   = reshape(Linear(ctx_i), [Nq, D])
```

- `DomainEmb_i`：每个序列域独立的可学习偏置向量（r4），使不同域的 Q 具有不同语义起点。
- 最后一个 Block（`protect_seq=True`）跳过 Q 刷新，防止 SCP（见 r3）。

**新增代码**：`DynamicQueryRefiner`（`model.py` 约 862 行）；`MultiSeqHyFormerBlock.__init__` 加 `self.dyn_q_refiner`；`forward` 在 Sequence Evolution 前调用。

---

#### r2：CSA 序列质量稀疏筛选

**动机**：Cross-Attention 中 Q 对所有序列 Token 均匀关注，但许多噪声行为（浏览时间极短、误触等）提供的信息量很低，增加计算量并可能引入噪声。

**实现**：在 Cross-Attention 前，用 Q mean 向量与 seq Token 做点积相似度打分，仅保留 Top-`csa_top_k` 个最相关 Token，其余位置加 padding mask：

```python
sim   = seq_tokens @ q_mean.T          # (B, L)
mask  = (sim < topk_threshold) | padding_mask
```

- `csa_top_k=0`（默认）：关闭，行为与原始相同。
- `csa_top_k>0`：启用稀疏筛选，典型值与 `--seq_top_k` 相同。
- `protect_seq` Block（最后一层）始终关闭 CSA，保证末层信息完整。

**新增参数**：`PCVRHyFormer(csa_top_k=...)` / `--csa_top_k`（`train.py`）。

---

#### r3：SCP 防护（最后一个 Block）

**动机**：Sequential Collapse Propagation（SCP）— 当 NS Token 与 Q 在末层共同参与 Seq Self-Attention 时，NS 的全局信号会覆盖序列局部模式，导致 seq Token 表示退化为 NS 的复制。

**实现**：末层 Block（`blk_idx == num_hyformer_blocks - 1`）设置 `protect_seq=True`：
- 跳过 `DynamicQueryRefiner`，保持 Q 不变；
- 跳过 CSA mask（full attention）；
- Seq Encoder 只做纯自注意力，不引入 NS 投影。

**新增参数**：`MultiSeqHyFormerBlock(protect_seq=True)`，由 `PCVRHyFormer.__init__` 自动为最后一个 Block 设置。

---

#### r5：全局 skip connection

**动机**：多层 Block 堆叠后，初始 NS Token 中的低频静态特征（用户长期偏好）信号容易被序列动态信息覆盖（梯度稀释）。全局残差提供直接梯度通路。

**实现**：在 `_run_multi_seq_blocks` 中，于 dropout 之前保存初始 NS Token 扁平向量，Block 栈执行完毕后加到最终输出：

```python
ns_init_flat  = ns_tokens.view(B, -1)          # (B, Nns*D)
...
output = output_proj(all_tokens) + global_skip_proj(ns_init_flat)
```

`global_skip_proj`：`Linear(Nns*D → D) + LayerNorm`，与 `output_proj` 维度相同。

---

#### 参数变更汇总

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `csa_top_k` | `0` | CSA 稀疏筛选保留 Token 数（0=关闭） |

其余改动均无需新参数（r1/r3/r4/r5 均自动生效）。

---

#### 改动文件

- `model.py`：`DynamicQueryRefiner` 新类；`MultiSeqHyFormerBlock` 加 `dyn_q_refiner` / `_csa_mask` / `protect_seq`；`PCVRHyFormer` 加 `csa_top_k` / `global_skip_proj` / `_run_multi_seq_blocks` 更新。
- `train.py`：新增 `--csa_top_k` CLI 参数；`model_args` 加 `csa_top_k`。
- `infer.py`：`_FALLBACK_MODEL_CFG` 加 `csa_top_k: 0`。

---

## 2026-05-18

### 新增时间特征（绝对时间 + 相对时间差）

**目标**：利用行为发生的时段规律（小时/星期）和序列与 target 的时间差（小时/天/周）提升模型表达力。

#### dataset.py

新增全局常量和辅助函数：

| 新增内容 | 说明 |
|---------|------|
| `NUM_HOUR_BUCKETS = 25` | 小时分桶数（0=padding, 1-24） |
| `NUM_WEEKDAY_BUCKETS = 8` | 星期分桶数（0=padding, 1-7，Mon=1） |
| `NUM_WEEK_OF_YEAR_BUCKETS = 54` | 全年第几周（0=padding, 1-53） |
| `HOUR_DIFF_BOUNDARIES` | 时间差-小时分桶边界（22个边界） |
| `DAY_DIFF_BOUNDARIES` | 时间差-天分桶边界（16个边界） |
| `WEEK_DIFF_BOUNDARIES` | 时间差-周分桶边界（9个边界） |
| `NUM_HOUR_DIFF_BUCKETS` | 小时差桶数 |
| `NUM_DAY_DIFF_BUCKETS` | 天差桶数 |
| `NUM_WEEK_DIFF_BUCKETS` | 周差桶数 |
| `_ts_to_hour(ts)` | Unix 时间戳 → 小时桶 id（1-24） |
| `_ts_to_weekday(ts)` | Unix 时间戳 → 星期桶 id（Mon=1..Sun=7） |
| `_ts_to_week_of_year(ts)` | Unix 时间戳 → 全年第几周（1-53） |

`_convert_batch` 新增输出字段：

| 字段 | shape | 说明 |
|------|-------|------|
| `req_hour` | (B,) | target 请求时间的小时 |
| `req_weekday` | (B,) | target 请求时间的星期几 |
| `req_week_of_year` | (B,) | target 请求时间的全年第几周 |
| `{domain}_seq_hour` | (B, L) | 序列历史行为的小时 |
| `{domain}_seq_weekday` | (B, L) | 序列历史行为的星期几 |
| `{domain}_diff_hours` | (B, L) | 序列行为与 target 的时间差（小时分桶） |
| `{domain}_diff_days` | (B, L) | 序列行为与 target 的时间差（天分桶） |
| `{domain}_diff_weeks` | (B, L) | 序列行为与 target 的时间差（周分桶） |

#### model.py

`ModelInput` 新增可选字段（默认 None，向后兼容）：

```python
req_hour, req_weekday, req_week_of_year   # target 时间
seq_hours, seq_weekdays                   # 序列绝对时间
seq_diff_hours, seq_diff_days, seq_diff_weeks  # 序列相对时间差
```

`PCVRHyFormer.__init__` 新增参数 `use_temporal_features=True`：

- **target 时间 NS token**：3 个独立 embedding（hour/weekday/week_of_year），每个 d_model 维，concat 后经 `Linear(3D→D) + LayerNorm` 投影成 **1 个额外 NS token**，信息不损失。
- **序列时间 embedding**（5 个 embedding table）：`seq_hour_emb`, `seq_weekday_emb`, `seq_diff_hour_emb`, `seq_diff_day_emb`, `seq_diff_week_emb`，均直接加到序列 token 上（与 `time_embedding` 相同的叠加方式）。

新增辅助方法 `_build_ns_tokens` 和 `_build_seq_tokens`，将 `forward` / `predict` 中重复的特征构建逻辑提取公用。

#### trainer.py

`_make_model_input` 从 `device_batch` 中读取所有新时间字段并传入 `ModelInput`，字段缺失时静默跳过（兼容旧数据）。

#### train.py

新增命令行参数：
- `--use_temporal_features`（默认 True）
- `--no_temporal_features`（关闭开关）

#### run.sh

由于新增 1 个 temporal NS token，`num_ns` 从 7 变为 8：
```
T = num_queries(2) × num_sequences(4) + num_ns(8) = 16
d_model(64) % T(16) = 0 ✓
```
`user_ns_tokens` 从 5 调整为 4 以保持 T=16。

#### 验证

用 `data/demo_1000.parquet` 冒烟测试通过：
- 所有新时间字段值域正常（无越界、无 NaN）
- `forward` 输出 shape `(32, 1)`，数值正常

---

## 2026-05-19

### UniRec 架构改进 + 泛化性增强

**目标**：
1. 对齐 OneTrans Mixed Parameterization 思想，让不同语义来源的 NS Token 有各自独立的参数；
2. 让经过多层迭代的 NS Token 也参与最终预测，补全信息遗漏；
3. 改善时间特征的泛化性，减少对训练集时间分布的过拟合。

#### dataset.py

| 改动 | 说明 |
|------|------|
| `NUM_WEEK_OF_YEAR_BUCKETS`: 54 → **5** | `week_of_year` 改为季度（Q1-Q4），粗粒度信号跨时间段更稳定 |
| `_ts_to_week_of_year()` | 返回值从 ISO 周（1-53）改为季度（1-4）；函数名保持不变，字段名无需改动 |

#### model.py

**方向一：Mixed Parameterization（NS Token 独立 FFN）**

- 新增 `user_ns_ffn`、`item_ns_ffn`、`user_dense_ffn`（可选）、`item_dense_ffn`（可选）、`temporal_ns_ffn` 五个独立 FFN（`LayerNorm + Linear + SiLU + Linear`，residual 连接）
- 在 `_build_ns_tokens` 中，每类 NS Token 经自己的独立 FFN 后再 concat，防止语义异质的 Token 共享早期参数（对应 OneTrans 的核心贡献）

**方向二：最终 NS Token 参与输出预测**

- `output_proj` 输入维度从 `Nq×S×D` 扩大为 `(Nq×S + Nns)×D`
- `_run_multi_seq_blocks` 输出时将最终的 `curr_ns` 与 Q Token 一起 concat 后投影
- 当前配置：`(2×4 + 8)×64 = 1024`，经 `Linear(1024→64) + LayerNorm` 输出 `D=64`

**泛化性：绝对时间特征 Dropout**

- 新增 `abs_time_dropout`（`p = min(dropout_rate × 5, 0.2)`，默认 `p=0.05`）
- **仅绝对时间特征**（`req_hour`、`req_weekday`、`req_week_of_year`、`seq_hour`、`seq_weekday`）在训练时加 Dropout
- **相对时间差特征**（`diff_hours`、`diff_days`、`diff_weeks`）不加额外 Dropout（跨时间段更稳定，无需正则）

#### infer.py

- `_FALLBACK_MODEL_CFG` 新增 `'use_temporal_features': True`，确保旧 checkpoint 推理时 fallback 与训练一致

#### 验证

冒烟测试通过（`data/demo_1000.parquet`，batch=32）：
- `forward` shape: `(32, 1)` ✓
- `predict` shape: `(32, 1)`, emb shape: `(32, 64)` ✓
- `output_proj.in_features = 1024`（= (2×4+8)×64）✓
- `num_ns = 8`（user_ns=4 + user_dense=1 + item_ns=2 + temporal=1）✓

---

### 泛化性补强（NS ID Dropout + Q-Gen LayerNorm + Dense Dropout）

**目标**：在上一批改动基础上，从三个正交角度进一步压缩训练-测试泛化缺口。

#### model.py

**1. NS 侧高基数 ID Dropout**（`GroupNSTokenizer` / `RankMixerNSTokenizer`）

- 两个 NS Tokenizer 新增参数 `id_threshold` 和 `id_dropout_rate`
- 对 `vocab > seq_id_threshold` 的 fid（User ID / Item ID 等高基数特征），embedding lookup 后在训练时施加 `Dropout(dropout_rate × 2)`
- 与 seq 侧的 `seq_id_emb_dropout` 机制完全对称，消除 NS 侧 ID 特征的正则盲区
- `PCVRHyFormer.__init__` 构造 Tokenizer 时自动传入 `id_threshold=seq_id_threshold`，无需新增外部参数

**2. Query Generator MeanPool 后加 LayerNorm**（`MultiSeqQueryGenerator`）

- 新增 `seq_pool_norms: ModuleList`（每个序列一个 `LayerNorm(d_model)`）
- MeanPool 后立即做 LayerNorm，防止短序列（有效长度 1-2）pooled vector 量级比长序列大幅偏低，避免 global_info concat 后梯度不均衡

**3. Dense 输入层 Dropout**（`PCVRHyFormer`）

- 新增 `dense_input_dropout: nn.Dropout(p=dropout_rate)`
- 在 `user_dense_proj` / `item_dense_proj` 投影前对原始 dense vector 施加 Dropout
- 防止模型记忆 dense 特征的绝对量级，提升对输入分布偏移的鲁棒性

#### 验证

冒烟测试通过（随机 batch=4，`rank_mixer_mode='simple'`）：
- `id_emb_dropout` 正确挂载，`_is_id[high_card_fid] = True` ✓
- `seq_pool_norms` 长度与序列数一致（=1）✓
- `dense_input_dropout` 正确挂载 ✓
- `forward` shape: `(4, 1)` train 和 eval 均正常 ✓
