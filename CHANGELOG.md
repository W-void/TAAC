# TAAC 改动记录
---

## 2026-05-20 (r7)

### r7：DIG 训练三项优化

#### 背景

r6 引入了 DIG 训练范式（多步 forward + reveal_ratio 从粗到细），但存在以下问题：

1. **Loss 权重平均不合理**：K 个 step 均等权重意味着粗粒度 step（低信息量）与细粒度 step（全特征）贡献相同，导致梯度偏向"用少量特征就能预测"的方向，浪费细粒度 step 的监督信号。
2. **div_loss 在粗粒度步噪声高**：reveal_ratio 较小时大量 fid embedding 被置零，Q token 主要来自零向量投影，pairwise cosine 偏高，div_loss 会向 query projection 注入虚假的"多样化"梯度。
3. **CSA 在粗粒度步不可靠**：reveal_ratio < 0.75 时 NS 特征稀疏，DynamicQueryRefiner 生成的 Q 质量差，用 Q·K^T top-k 稀疏化 attention mask 相当于随机剪枝，可能伤害 seq encoding。

#### 实现

**改动 A — trainer.py：DIG loss 线性递增加权**

```
weights[i] = (i+1) / (K*(K+1)/2)   # 线性递增，最后一步权重最大
loss = Σ weights[i] * step_loss[i]  # 已归一化，不再除以 K
```

例 K=4：weights = [0.1, 0.2, 0.3, 0.4]，全特征 step 权重是粗粒度 step 的 4 倍。

**改动 B — trainer.py：div_loss 只在最后一步累积**

```python
if step_idx == K - 1:
    div_loss_tensor = step_div_loss   # 只取 reveal_ratio=1.0 的 div_loss
```

**改动 C — model.py：`_run_multi_seq_blocks` + `MultiSeqHyFormerBlock.forward`**

- `_run_multi_seq_blocks` 新增 `reveal_ratio: float = 1.0` 参数，透传给每个 Block。
- `_run_multi_seq_blocks` 中 div_loss 计算门控：`reveal_ratio < 1.0` 时返回零张量。
- `MultiSeqHyFormerBlock.forward` 新增 `reveal_ratio` 参数，当 `reveal_ratio < 0.75` 时 CSA 退化为普通 padding mask（`csa_q_reliable = reveal_ratio >= 0.75`）。
- `PCVRHyFormer.forward` 将 `reveal_ratio` 传入 `_run_multi_seq_blocks`。

#### 影响

| 指标 | 预期方向 |
|------|---------|
| 细粒度 step 的梯度幅度 | ↑（相对权重提升） |
| div_loss 梯度噪声 | ↓（粗粒度步不再更新 query projection 的 orthogonal 方向） |
| 粗粒度步 CSA 引入的 attention mask 错误率 | ↓（直接禁用，退回全序列 attention） |

---

## 2026-05-20

### r6：DIG-style 全特征组逐步恢复（`sid_mode='fid_order'`）

#### 背景与动机

参考 DIG（Discrimination Is Generation, arXiv 2605.14853）论文的核心设计：SID 通过残差量化构建层次化码本，每层 SID embedding 前缀累加（`e_sid^(1:l) = Σ eˡ[sˡ]`）可从粗到细逐步逼近完整 item 表示，作用类似于正则——浅层 token 只拿到粗粒度信息，被迫学习更泛化的表示。

TAAC 的特征天然有粒度层次：
- **NS 侧**（user_int / item_int）：每个 fid 对应一个 embedding，vocab_size 越小越粗（shop 类目 < item_id）。
- **Seq 侧**：每个行为 item 携带 8–13 个 sideinfo fid（item_id、shop_id、类目 id 等），vocab_size 同样呈粗→细梯度。

**改动目标**：将 DIG 的"逐步信息恢复"训练范式映射到 TAAC 的全特征组，覆盖 `user_int`、`item_int`、`user_dense`、所有 `seq_domain`。训练时每个 step 执行 K 次 forward，每次揭示不同比例的特征子集（从粗到细），K 次 loss 平均后反传，迫使模型具备从不完整信息中恢复预测能力。

**关键区别（已纠正的误解）**：DIG 的"逐层加入"与模型的 block 数**无关**。正确范式是：
> 同一个 batch，运行 K 次 forward（每次 `reveal_ratio = k/K`），每次揭示不同粒度的特征子集，K 次 loss 都回传，模型学习"从粗到细"恢复预测能力。模型 block 参数完全共享（复用同一个前向路径）。

#### 实现

**特征揭示顺序（coarse → fine）**：

所有特征组按 `vocab_size` 升序排列，`vocab_size` 越小的特征越先被揭示（对应 DIG 中越粗粒度的 SID 前缀）：

```
user_int reveal order: sorted by vocab_size ascending
item_int reveal order: sorted by vocab_size ascending
seq_domain fid order:  sorted by vocab_size ascending (per domain)
user_dense / item_dense: 视为原子特征，reveal_ratio >= 0.5 时揭示
```

**训练数据流**（`sid_mode='fid_order'`）：

```
_train_step():
    for k in range(K):
        reveal_ratio = (k+1) / K          # 1/K, 2/K, ..., 1.0
        logits = model(batch, reveal_ratio)
        loss_sum += BCE(logits, label)
    loss = loss_sum / K
    loss.backward()

model.forward(inputs, reveal_ratio):
    ns_tokens = _build_ns_tokens(reveal_ratio)   # mask inactive fids → zero vec
    seq_tokens = _build_seq_tokens(reveal_ratio) # mask inactive fids → zero vec
    ... (same as none mode from here)

_build_ns_tokens(reveal_ratio):
    k_user = ceil(N_user * reveal_ratio)
    active_user = set(_user_fid_reveal_order[:k_user])
    user_fid_mask = [i in active_user for i in range(N_user)]
    # 传入 Tokenizer，被屏蔽的 fid 输出 zero float vector
    user_ns = user_ns_tokenizer(feats, fid_mask=user_fid_mask)

_embed_seq_domain(..., reveal_order, reveal_k):
    active_fids = set(reveal_order[:reveal_k])
    # fid not in active_fids → zero float vector (dtype=float)
```

**参数共享**：屏蔽操作发生在 embedding 层（zeroing），后续的 `concat → Linear → LN` 投影参数在所有揭示比例下完全共享，与 DIG 的 prefix-sum 语义对应。

#### 新增参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sid_mode` | `'none'` | `'none'` = 原有行为；`'fid_order'` = DIG-style 逐步特征恢复 |
| `dig_steps` | `4` | `fid_order` 模式下每个训练 step 的 forward 次数 K（K 越大正则越强但计算量越高） |

- `sid_mode='none'`：行为与原始完全相同，旧 checkpoint 可直接加载，无任何额外开销。
- `sid_mode='fid_order'`：无需新增 per-fid 参数，复用原有 `Linear(S×emb_dim → D) + LayerNorm`，训练时 reveal_ratio 控制哪些 fid embedding 为零向量。
- **推理时**：`predict()` 始终使用 `reveal_ratio=1.0`（等价于 `none` 模式），无额外推理开销。

#### 改动文件

- `model.py`：
  - `PCVRHyFormer.__init__`：新增 `sid_mode` / `dig_steps` 参数；`fid_order` 时计算并存储 `_user_fid_reveal_order`、`_item_fid_reveal_order`、`_seq_fid_reveal_order`（按 vocab_size 升序排列的 fid 索引列表）。
  - `GroupNSTokenizer.forward` / `RankMixerNSTokenizer.forward`：新增 `fid_mask: Optional[List[bool]]` 参数；被屏蔽 fid 输出 `new_zeros(..., dtype=torch.float)` 零向量（修复 Long/Float dtype 类型错误）。
  - `_embed_seq_domain`：新增 `reveal_order: Optional[List[int]]` / `reveal_k: int` 参数；根据 `reveal_k` 屏蔽序列特征（替换为 zero float vector）。
  - `_build_ns_tokens`：新增 `reveal_ratio: float = 1.0` 参数；`fid_order` 模式下计算 `user_fid_mask` / `item_fid_mask` 传给 Tokenizer；`user_dense` / `item_dense` 在 `reveal_ratio < 0.5` 时整体屏蔽。
  - `_build_seq_tokens`：新增 `reveal_ratio: float = 1.0` 参数；`fid_order` 模式下计算每个 domain 的 `reveal_k` 和 `reveal_order` 传给 `_embed_seq_domain`。
  - `forward`：新增 `reveal_ratio: float = 1.0` 参数，透传给 `_build_ns_tokens` 和 `_build_seq_tokens`。
  - `predict`：明确以 `reveal_ratio=1.0` 调用 `forward`，确保推理行为不变。
  - `_run_multi_seq_blocks`：**删除**旧的 `fid_tokens_list` / `time_deltas_list` / per-block 调度逻辑；接收已按 `reveal_ratio` 屏蔽好的 `seq_tokens_list`，无感知 DIG 细节。
- `train.py`：新增 `--sid_mode` / `--dig_steps` CLI 参数；`model_args` 加 `sid_mode` / `dig_steps`。
- `trainer.py`（`_train_step`）：新增 `_compute_loss` 辅助函数；`fid_order` 模式下循环 K 次 forward，平均 loss 后反传。
- `infer.py`：`_FALLBACK_MODEL_CFG` 加 `'dig_steps': 4`。
- `run.sh`：新增 `--dig_steps 4`；更新 `--sid_mode` 注释说明 DIG 训练范式。

#### 验证（smoke test `_smoke_test_r6.py`）

```
none mode forward: torch.Size([4, 1])                              ✓
fid_order mode: 0.25, 0.5, 0.75, 1.0 均输出 torch.Size([4, 1])   ✓
predict: logits=[4,1], emb=[4,32]                                  ✓
user_int reveal order: [0,1,2,3] (vocab 5<10<1000<50000)          ✓
seq reveal orders verified                                         ✓
DIG loss simulation (K=4): avg_loss=0.5816                        ✓
ALL TESTS PASSED
```

---

### r6 补充：时间特征也纳入 DIG 粗→细揭示

#### 背景

原始 r6 只对 sideinfo fid（user_int / item_int / seq_domain fid）做逐步揭示，时间特征始终以全量注入。但时间戳本身天然具备粒度层次（周 > 天 > 小时），应同样纳入 DIG 的粗→细流程。

#### 实现

**Seq 侧（`_embed_seq_domain`）**：

新增 `time_reveal_k` 参数，将 6 个时间特征按粒度排成固定顺序，通过 `time_reveal_k` 控制截止位置：

```
slot 0: time_bucket  (相对时间桶，最粗)
slot 1: diff_week    (相对时间差，周)
slot 2: diff_day     (相对时间差，天)
slot 3: weekday      (绝对星期，1-7)
slot 4: diff_hour    (相对时间差，小时)
slot 5: hour         (绝对小时，最细)
```

`_build_seq_tokens` 中根据 `reveal_ratio` 线性映射：
```python
time_reveal_k = max(1, ceil(6 * reveal_ratio))   # 1/K → 1 slot; K/K → 6 slots
```

**NS 侧（`_build_ns_tokens`）**：

`req_time` token 保持为 1 个 NS token（`num_ns` 不变，T 约束不受影响），但内部的 3 个分量按粒度逐步揭示：

```
reveal_ratio < 1/3  : 只用 week_of_year（最粗）；weekday & hour → zero vector
reveal_ratio < 2/3  : week_of_year + weekday；hour → zero vector
reveal_ratio >= 2/3 : 全量（week_of_year + weekday + hour）
```

**修复**：`zeros = inputs.user_int_feats.new_zeros(B, d_model, dtype=torch.float)`，避免 Long/Float dtype 错误。

#### 验证

```
ratio=0.167  time_slot=1/6  ns[week=✓ weekday=✗ hour=✗]  out=(4,1)   ✓
ratio=0.333  time_slot=2/6  ns[week=✓ weekday=✓ hour=✗]  out=(4,1)   ✓
ratio=0.500  time_slot=3/6  ns[week=✓ weekday=✓ hour=✗]  out=(4,1)   ✓
ratio=0.667  time_slot=4/6  ns[week=✓ weekday=✓ hour=✓]  out=(4,1)   ✓
ratio=0.833  time_slot=5/6  ns[week=✓ weekday=✓ hour=✓]  out=(4,1)   ✓
ratio=1.000  time_slot=6/6  ns[week=✓ weekday=✓ hour=✓]  out=(4,1)   ✓
none vs fid_order(ratio=1.0) max_diff = 0.00e+00                      ✓
ratio=1/6 vs ratio=1.0 max_diff = 0.1737 (> 0, 验证屏蔽有效)          ✓
ALL TIME-DIG TESTS PASSED
```

---

### r7：Query Collapse 防护（方案 A + B）

#### 背景与动机

当 `num_queries > 1` 时，`DynamicQueryRefiner` 原实现使用单个 `Linear(D → Nq*D) + reshape` 生成多个 Q token。由于所有 Q slot 共享同一投影参数，梯度路径对称，模型缺乏结构性激励将不同 Q 特化到不同语义方向，容易出现**坍塌**（各 Q 趋于相同方向）。

#### 实现

**方案 A — 独立投影头（结构性防坍塌）**

将 `q_projs` 从 1 个 `Linear(D → Nq*D)` 改为 `Nq` 个独立的 `Linear(D → D) + LayerNorm`：

```python
# 旧：共享投影
q_flat = self.q_projs[i](ctx)           # (B, Nq*D)
q_i = q_flat.view(B, Nq, D)

# 新：独立投影
qs = [proj(ctx) for proj in self.q_projs[i]]  # Nq × (B, D)
q_i = torch.stack(qs, dim=1)                  # (B, Nq, D)
```

每个 Q slot 有自己独立的参数和梯度路径，结构上保证不同 slot 可朝不同方向分化。参数量相同（`Nq × D×D` vs `D × Nq*D`）。

**方案 B — 多样性正则（辅助 Loss）**

新增 `_query_diversity_loss`，惩罚各 Q token 之间的**成对余弦相似度**：

```
L_div = mean_{域i} mean_{j≠k} |cosine_sim(Q_{i,j}, Q_{i,k})|
```

- `_run_multi_seq_blocks` 训练时计算并返回 `(output, div_loss)`
- `forward` 返回 `(logits, div_loss)`；`predict` 丢弃 div_loss（推理无影响）
- `loss_total = task_loss + query_div_weight × div_loss`

#### 新增参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--query_div_weight` | `0.01` | div_loss 权重；`0` = 关闭；仅 `num_queries > 1` 时有效 |

#### 打印输出

与主 loss 打印保持一致，三处同步：
- **tqdm 进度条**：`loss=0.62  div=0.31  ...`
- **TensorBoard**：新增 `Loss/train_div` 曲线
- **epoch logging**：`Average Div Loss: 0.3124 (weighted: 0.0031)`

#### 改动文件

- `model.py`：`DynamicQueryRefiner.q_projs` 改为独立投影头；新增 `_query_diversity_loss`；`_run_multi_seq_blocks` / `forward` 返回值扩展为 `(output/logits, div_loss)`；`predict` 解包并丢弃 div_loss。
- `train.py`：新增 `--query_div_weight` 参数；传给 trainer。
- `trainer.py`：`__init__` 接收 `query_div_weight`；`_train_step` 叠加 div_loss；三处打印同步。

---

### 训练 loss 分粒度打印

#### 背景

DIG 模式（`sid_mode='fid_order'`）每个 step 运行 K 次 forward，之前仅打印 K 次的均值，无法观察不同揭示比例下的学习情况。

#### 实现

`_train_step` 返回值从 `float` 扩展为 `Tuple[float, Optional[list], float]`：

```python
return avg_loss, dig_losses, div_loss_val
# dig_losses: List[float] 每个 reveal_ratio 的单独 loss；none 模式为 None
# div_loss_val: query diversity loss（未加权原始值）
```

三处打印同步（与原 loss 格式一致）：
- **tqdm 进度条**：`loss=0.62  div=0.31  r0.25=0.68  r0.50=0.61  r0.75=0.60  r1.00=0.59`
- **TensorBoard**：`Loss/train_dig_r0.25` 等独立曲线；`Loss/train_div`
- **epoch logging**：`DIG Average Loss: r0.25=0.68 | r0.50=0.61 | r0.75=0.60 | r1.00=0.59`

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
