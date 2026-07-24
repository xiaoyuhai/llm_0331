# Transformer 从入门到通关

## —— 图解 "Attention Is All You Need" 及其现代演进

> 本文档结合知识库 `NLP1.1.0.md`、`高频面试题` 及仓库 `01_Qwen3.py` 手写代码，从零构建对 Transformer 的完整理解。

---

# 第一篇 · 入门篇 — 为什么需要 Transformer？

## 1.1 前 Transformer 时代的困境

在 2017 年之前，NLP 领域的主流架构是 **RNN（循环神经网络）及其变体 LSTM / GRU**，通常以 **Seq2Seq（序列到序列）** 框架组织。

一个典型的机器翻译模型长这样：

```
输入: "我 爱 猫"  →  [编码器 RNN]  →  上下文向量  →  [解码器 RNN]  →  "I love cats"
```

这个架构有两个致命问题：

### 问题一：无法并行

RNN 的"循环"本质决定了它必须**逐个 token 顺序计算**：要算第 3 个位置的输出，必须先算完第 1、2 个位置。

```
t=1: h₁ = RNN(x₁, h₀)
t=2: h₂ = RNN(x₂, h₁)    ← 必须等 h₁ 算完
t=3: h₃ = RNN(x₃, h₂)    ← 必须等 h₂ 算完
```

这意味着 GPU 的并行能力完全被浪费了。训练一个长句子，计算时间随长度线性增长。

### 问题二：长距离依赖失效

假设句子是：

> "I grew up in France ... **（200个词后）** ... I speak fluent **French**"

RNN 需要把信息通过隐藏状态一步一步传递 200 步。信号在反复传递中会**指数级衰减**（梯度消失），导致模型"记不住"前面的 `France`，最终预测 `French` 时缺乏依据。

### 1.1.1 注意力机制（Attention）的初步引入

为了缓解问题二，研究者给 Seq2Seq + RNN 加上了 **注意力（Attention）机制**。其核心想法是：

> 解码器生成每个词时，不再只依赖一个固定的"上下文向量"，而是**动态地回头去看编码器的所有位置**，并加权提取最相关的信息。

```
解码器生成 "French" 时：
  → 计算当前状态与编码器每个位置的"相关性得分"
  → 对 "France" 位置给出最高分
  → 据此加权提取信息，辅助预测
```

但这仍然是**打在 RNN 身上的补丁**——RNN 的顺序计算瓶颈还在。

## 1.2 论文的核心理念

2017 年，Google 发表了论文 **《Attention Is All You Need》**，提出 **Transformer** 架构。

它做出了一个激进的决断：

> **把 RNN 整个扔掉，完全只用注意力机制来建模序列依赖。**

这个决定的逻辑来源于知识库 `NLP1.1.0.md` 中清晰的推演：

1. RNN 的作用是"建模序列中不同位置之间的依赖关系"
2. 注意力机制也能做这件事（每个位置可以关注其他任意位置）
3. 注意力机制还有额外好处：**可并行、长距离不衰减**
4. 结论：注意力机制可以完全替代 RNN

这就是标题 **"Attention Is All You Need"** 的真正含义——你不需要 RNN，注意力就够了。

---

# 第二篇 · 核心篇 — Transformer 逐块拆解

## 2.1 整体架构总览

Transformer 延续了 **Encoder-Decoder（编码器-解码器）** 结构：

```
输入序列 → [ 编码器堆叠 (N×) ] → 编码表示 → [ 解码器堆叠 (N×) ] → 输出序列
               ↓                                    ↑
        每个子层包含:                          每个子层包含:
        · Self-Attention                        · Masked Self-Attention
        · FFN                                   · Cross-Attention（编码器→解码器）
                                                · FFN
```

原始论文中 N=6。每个子层还包含 **残差连接** 和 **层归一化（LayerNorm）**。

## 2.2 核心组件一：自注意力机制（Self-Attention）

### 2.2.1 直观理解

想象你在读一句话：

> "**那只动物**没有过马路，因为**它**太累了。"

当你读到"它"时，你需要知道"它"指代什么。自注意力就是让模型在处理"它"这个 token 时，**去看一眼输入序列中所有其他位置**，然后用加权的方式把相关信息融合进来。

### 2.2.2 数学上的三要素：Query、Key、Value

每个 token 会生成三个向量：

| 角色 | 类比 | 作用 |
|------|------|------|
| **Query (Q)** | 你在图书馆前台问的问题 | "我要找关于动物的书" |
| **Key (K)** | 每本书的标签 | "动物学"、"历史"、"编程"... |
| **Value (V)** | 书的内容 | 匹配成功后提取的信息 |

计算过程分三步：

**Step 1：Q 与所有 K 做点积 → 相关性得分**

```text
score(i,j) = Qᵢ · Kⱼ
```

**Step 2：Softmax 归一化 → 注意力权重**

```text
attention_weights = softmax(score / √dₖ)
```

> 除以 √dₖ 是为了防止点积值随维度增长过大，导致 softmax 梯度消失。这就是论文中 **scaled dot-product attention** 的"scaled"的含义。

**Step 3：加权求和 V → 输出**

```text
outputᵢ = Σⱼ weight(i,j) · Vⱼ
```

**矩阵形式（核心公式）：**

```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V
```

### 2.2.3 代码视角：从公式到实现

在你的 `01_Qwen3.py` 中，RoPE 旋转之前的 Q、K 生成逻辑就是这一步：

```python
# 伪代码：投影生成 Q、K、V
self.q_proj = nn.Linear(dim, num_heads * head_dim)
self.k_proj = nn.Linear(dim, num_kv_heads * head_dim)  # GQA：KV 头比 Q 头少
self.v_proj = nn.Linear(dim, num_kv_heads * head_dim)

# 每个 token 的 Q, K, V
Q = self.q_proj(x)  # (batch, seq_len, num_heads * head_dim)
K = self.k_proj(x)  # (batch, seq_len, num_kv_heads * head_dim)
V = self.v_proj(x)
```

注意力得分计算：

```python
# score = Q @ Kᵀ / √d_k
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
# mask 处理（因果 mask / padding mask）
scores = scores.masked_fill(mask == 0, float('-inf'))
# softmax 归一化
attn_weights = F.softmax(scores, dim=-1)
# 加权求和
output = torch.matmul(attn_weights, V)
```

## 2.3 核心组件二：多头注意力（Multi-Head Attention）

### 为什么需要多头？

自然语言的语义关系是**多层**的。同一句话：

> "那只动物没有过马路，因为它太累了。"

- **头1** 可能关注代指关系："它" → "那只动物"
- **头2** 可能关注因果关系："因为" → "所以没过马路"
- **头3** 可能关注短语结构："过马路"

一组 Q/K/V 投影只能捕捉一种关系模式。**多头注意力**让模型用**多组独立的 Q/K/V 投影**并行捕捉不同类型的语义关系。

### 计算过程

```
输入 X
  ├─→ Q₁, K₁, V₁ ─→ Head₁ 输出  ─┐
  ├─→ Q₂, K₂, V₂ ─→ Head₂ 输出  ─┤
  ├─→ ...                         ┤
  └─→ Qₕ, Kₕ, Vₕ ─→ Headₕ 输出  ─┘
                                     ↓
                            拼接所有头 → 线性投影 W_O → 最终输出
```

### 代码视角：`01_Qwen3.py` 中的多头

```python
# 拆分成多个头
def reshape_for_attention(x, num_heads):
    batch, seq_len, _ = x.shape
    x = x.view(batch, seq_len, num_heads, -1)
    return x.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)

# 对每个头独立算注意力，然后合并
Q = reshape_for_attention(self.q_proj(x), num_heads)     # (B, H, S, D)
K = reshape_for_attention(self.k_proj(x), num_kv_heads)  # (B, KV_H, S, D)
V = reshape_for_attention(self.v_proj(x), num_kv_heads)

# GQA: KV 头 repeat_interleave 到与 Q 头数一致
K = K.repeat_interleave(num_heads // num_kv_heads, dim=1)
V = V.repeat_interleave(num_heads // num_kv_heads, dim=1)

# 每个头独立算注意力
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
# ... mask, softmax, 加权求和 ...

# 合并所有头
output = output.transpose(1, 2).reshape(batch, seq_len, -1)
# 最终投影
output = self.o_proj(output)
```

## 2.4 核心组件三：位置编码（Positional Encoding）

### 为什么要位置编码？

自注意力是**置换不变（permutation invariant）** 的——如果把输入 token 的顺序打乱，注意力输出完全不变。因为 Q @ Kᵀ 只依赖向量内容，不依赖位置。

但在语言里，"我打你"和"你打我"完全不同。所以必须给模型注入位置信息。

### 原始 Transformer：正弦/余弦位置编码

论文使用固定公式：

```text
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

- `pos`：位置索引（0, 1, 2, ...）
- `i`：维度索引（0, 1, 2, ..., d_model/2）
- 不同频率的正弦波让每个位置有唯一的"编码指纹"
- 模型可以**外推**到训练时未见过的长度

### 现代演进：RoPE（Rotary Position Embedding）

你的 `01_Qwen3.py` 中使用的是 **RoPE（旋转位置编码）**，这是当前 LLM 的主流选择（Qwen、LLaMA、Gemini 等）。

RoPE 的核心思想是：**对 Q 和 K 向量的特定维度对进行旋转变换，旋转角度与位置相关。**

```
对第 m 个位置的 token，其 Q/K 向量的 (dim 2i, dim 2i+1) 对旋转 m·θᵢ 弧度
其中 θᵢ = base^(-2i/d)
```

RoPE 的优雅之处在于：**旋转后的 Q·K 点积自动包含了相对位置信息**——两个 token 相距越远，旋转角度差越大，点积衰减越自然。

代码形态（`01_Qwen3.py`）：

```python
# 预计算 cos/sin
cos = torch.cos(pos * freq).to(dtype)
sin = torch.sin(pos * freq).to(dtype)

# rotate_half：配对 (d/2, d/2) 而非相邻
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

# 应用旋转
q_embed = (q * cos) + (rotate_half(q) * sin)
k_embed = (k * cos) + (rotate_half(k) * sin)
```

> **💡 与原始 Transformer 的区别**：原始 PE 是加在输入上，RoPE 是旋转 Q/K 本身，这使得相对位置信息直接参与注意力计算。

## 2.5 核心组件四：前馈神经网络（FFN）

每个注意力层后面接一个 FFN，对每个 token **独立**做非线性变换。

### 原始 FFN

```text
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

一个升维 + ReLU + 降维：`d_model → d_ff → d_model`，其中 `d_ff = 4 * d_model`。

### 现代演进：SwiGLU

你的 `01_Qwen3.py` 中使用的是 **SwiGLU**（LLaMA / Qwen 系列标配）：

```python
def swiglu(gate, x):
    return F.silu(gate) * x

# FFN 有两个投影：
gate = self.gate_proj(x)   # d_model → d_ff
up   = self.up_proj(x)     # d_model → d_ff
down = self.down_proj(F.silu(gate) * up)  # d_ff → d_model
```

SwiGLU = Swish（SiLU）门控 + 线性单元，比 ReLU 有更好的梯度流，通常能提升 1-2 个百分点的下游任务效果。

## 2.6 核心组件五：残差连接与层归一化

### 残差连接（Residual Connection）

每个子层的**输入**直接与**输出**相加：

```text
output = x + SubLayer(x)
```

确保梯度有一条"高速公路"直接回传到浅层，让数十层甚至上百层的 Transformer 也能稳定训练。

### 层归一化（LayerNorm）

对每个 token 的**所有特征维度**做标准化：

```text
LayerNorm(x) = γ · (x - μ) / σ + β
```

使每个 token 的表示具有稳定的均值和方差，避免深层网络中的分布漂移。

### 两种排列方式

| 方式 | 公式 | 代表模型 |
|------|------|----------|
| **Post-Norm**（原始论文） | `LayerNorm(x + SubLayer(x))` | 原始 Transformer |
| **Pre-Norm**（现代标准） | `x + SubLayer(LayerNorm(x))` | GPT、LLaMA、**Qwen3** |

你的 `01_Qwen3.py` 使用 Pre-Norm：

```python
def forward(self, x, mask, cos, sin):
    # Pre-Norm
    normed = self.input_layernorm(x)
    attn_out = self.self_attn(normed, mask, cos, sin)
    x = x + attn_out       # 残差

    normed = self.post_attention_layernorm(x)
    ffn_out = self.mlp(normed)
    x = x + ffn_out        # 残差
    return x
```

Pre-Norm 更稳定，允许更大学习率而不会梯度爆炸。

## 2.7 编码器 vs 解码器的区别

```
┌──────────────────────────────────────────────────────────────┐
│                        编码器 (Encoder)                       │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐ │
│  │ Self-Attention│ →  │    FFN      │ →  │  ... (N层)   │ │
│  │  (双向，无mask)│     │              │     │              │ │
│  └──────────────┘     └──────────────┘     └──────────────┘ │
└──────────────────────────────────────────────────────────────┘
                             ↓ 输出编码表示

┌──────────────────────────────────────────────────────────────┐
│                        解码器 (Decoder)                       │
│  ┌────────────────┐   ┌──────────────────┐   ┌────────────┐ │
│  │ Masked Self-Attn│ → │ Cross-Attention  │ → │    FFN     │ │
│  │  (因果，只看过去)│   │  (看编码器输出)   │   │            │ │
│  └────────────────┘   └──────────────────┘   └────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

关键区别：

1. **编码器**：Self-Attention 是**双向的**，每个 token 能看到前后的所有 token（BERT 风格）
2. **解码器**：Self-Attention 是**因果（causal）的**，每个 token 只能看到自己和之前的 token（GPT 风格），通过 **causal mask** 实现
3. **解码器多一个 Cross-Attention**：Q 来自解码器，K、V 来自编码器输出——让解码器在生成时"回头看"输入

### 因果掩码（Causal Mask）

```text
原始注意力矩阵 (4×4)：    加了因果掩码后：
  [1, 1, 1, 1]            [1, 0, 0, 0]
  [1, 1, 1, 1]     →      [1, 1, 0, 0]
  [1, 1, 1, 1]            [1, 1, 1, 0]
  [1, 1, 1, 1]            [1, 1, 1, 1]
```

右上角被 mask 为 `-inf`，经过 softmax 后变为 0——当前位置就"看不见"未来 token 了。

---

# 第三篇 · 进阶篇 — 为什么 Transformer 这么强？

## 3.1 并行计算：训练速度的革命

这是 Transformer 相对于 RNN 最大的优势。

| 架构 | 计算方式 | 时间复杂度 | 能否并行 |
|------|---------|-----------|---------|
| RNN | 逐个 token 顺序 | O(n) 步，每步 O(d²) | ❌ |
| Transformer | 所有 token 一起 | O(1) 步，每步 O(n²·d) | ✅ |

Transformer 的一次前向传播中，所有 token 的 Q、K、V 是一次性算出来的，注意力矩阵也是一次性算出来的。这让 GPU 的矩阵运算能力得到充分利用。

> **代价**：注意力复杂度是 O(n²)（n 是序列长度）。这就是为什么长文本（如 100k tokens）需要特殊优化（稀疏注意力、Flash Attention 等）。

## 3.2 长距离依赖：任意 token 直接连接

在 RNN 中，位置 i 和位置 j 的信息传递需要经过 `|i-j|` 步。而在 Transformer 中，**任意两个 token 之间只隔了一层注意力**，距离为 1。

```
RNN:    x₁ → x₂ → x₃ → ... → x₁₀₀  (100步衰减)
Trans:  x₁ ←——直接注意力——→ x₁₀₀      (1步直达)
```

这解决了长文本中"记不住前面内容"的根本问题。

## 3.3 可扩展性：Transformer 的三驾马车

知识库 `高频面试题` 指出：**"大模型是指参数规模巨大（十亿到千亿级）的深度神经网络。"**

Transformer 的核心优势在于它**能像乐高一样堆叠**：

1. **堆叠层数**：从 6 层到 70+ 层（如 PaLM），每多一层增加模型容量
2. **扩大维度**：从 512 到 12288（GPT-3），增加表示能力
3. **增加数据**：注意力机制在更多数据上持续受益（不像小模型会饱和）

这三点共同支撑了"scaling law"——随着模型、数据、计算量的增长，性能持续提升。

---

# 第四篇 · 精通篇 — 从 Transformer 到现代 LLM

## 4.1 架构路线的三大分支

```
原始 Transformer (Encoder-Decoder)
    ├── Encoder-Only → BERT、RoBERTa (理解型任务)
    │
    ├── Encoder-Decoder → T5、BART (条件生成)
    │
    └── Decoder-Only → GPT系列、LLaMA、Qwen (自回归生成) ← 🌟 当今主流
```

当前的大语言模型几乎全部是 **Decoder-Only** 架构，原因在知识库中有总结：

> "GPT 使用 Transformer 架构，具备全局自注意力机制，能够有效建模长距离依赖信息。同时，Transformer 的并行计算特性使得模型能够高效处理长文本序列。"

Decoder-Only 还能更高效地支持 **in-context learning（上下文学习）**和 **chain-of-thought（思维链）**——这些能力在 Encoder 结构中难以自然涌现。

## 4.2 注意力的工程进化

### MHA → MQA → GQA

这是现代 LLM 的注意力机制演进路线，也是你 `01_Qwen3.py` 实现的重点。

| 方案 | 头数 | 特点 | 代表模型 |
|------|------|------|---------|
| **MHA**（原始） | Q=H, K=H, V=H | 每头独立 | 原始 Transformer |
| **MQA** | Q=H, K=1, V=1 | 推理快，但质量稍降 | PaLM |
| **GQA**（折中方案） | Q=H, K=G, V=G | 兼顾效率和质量 | LLaMA 2/3、**Qwen3** |

你代码中的实现：

```python
# Q 和 KV 头数不同
self.num_heads = 16        # H
self.num_kv_heads = 8      # G（GQA 的 G）

# KV 通过 repeat_interleave 扩展到和 Q 一样多头
# K: (B, 8, S, D) → (B, 16, S, D)
# V: (B, 8, S, D) → (B, 16, S, D)
```

GQA 的优势：**训练质量接近 MHA，推理速度接近 MQA**。

### KV Cache：推理时的加速利器

在自回归生成时，每步只生成 **一个 token**，但注意力需要看**之前所有的 token**。

```
Step 1: [token_1]                       → 算 Q₁K₁V₁
Step 2: [token_1, token_2]              → 重新算 Q₁K₁V₁ + 算 Q₂K₂V₂ ❌ 浪费
Step 3: [token_1, token_2, token_3]     → 重复计算更浪费 ❌
```

KV Cache 的核心思想：**把之前所有 step 的 K、V 缓存起来，每步只算当前 token 的 Q、K、V，然后把 K、V 追加到缓存中。**

```
Step 1: 算 K₁V₁ → 缓存 [K₁V₁]          → 注意力用 cache
Step 2: 算 K₂V₂ → 缓存 [K₁V₁, K₂V₂]    → 注意力用 cache
Step 3: 算 K₃V₃ → 缓存 [K₁V₁, K₂V₂, K₃V₃]
```

你代码中的实现：

```python
# prefill：一次性处理所有输入 token
if self.kv_cache is not None:
    k = torch.cat([self.kv_cache[layer_idx][0], k], dim=2)
    v = torch.cat([self.kv_cache[layer_idx][1], v], dim=2)
self.kv_cache[layer_idx] = (k, v)

# decode：只需关注最后一个位置
q = q[:, :, -1:, :]  # 只取当前步的 Q
# K、V 已经是完整的历史
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
```

## 4.3 位置编码的现代演进

```
原始 Transformer:  Sinusoidal PE（固定的 sin/cos）
    ↓
BERT、GPT:          Learnable PE（可学习的嵌入）
    ↓
LLaMA、Qwen:        RoPE（旋转位置编码） ← 🏆 当前主流
    ↓
ALiBi、XL:          偏置式位置编码（长文本友好）
```

RoPE 之所以成为主流，因为它：
1. 自然地编码了**相对位置**（比绝对位置更合理）
2. 可以**外推**到训练时没见过的长度
3. 与线性注意力兼容

## 4.4 规范化层的演进

```
原始 Transformer:  Post-Norm LayerNorm
    ↓
GPT-2/3:            Pre-Norm LayerNorm ← 🏆 当前标准
    ↓
LLaMA/Qwen:         Pre-Norm RMSNorm   ← 🏆 更高效
```

**RMSNorm** 是 LayerNorm 的简化版，去掉了均值归零的步骤，只保留方差缩放：

```text
RMSNorm(x) = γ · x / RMS(x)
RMS(x) = √(mean(x²) + ε)
```

你的代码中的实现：

```python
class RMSNorm:
    def forward(self, x):
        # 计算 RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # 缩放
        return x / rms * self.weight
```

RMSNorm 比 LayerNorm 快 10-15%，且效果相当。

## 4.5 超越 Transformer 的尝试

知识库提到：

> "为克服 Transformer 处理长文本的计算瓶颈，出现更高效的架构（如 **Mamba**），追求线性复杂度和更优的推理性能。"

| 架构 | 复杂度 | 核心思想 | 代表 |
|------|--------|---------|------|
| Transformer | O(n²) | 全局注意力 | GPT、LLaMA |
| **Linear Attention** | O(n) | 核函数近似注意力 | Performer |
| **Mamba (SSM)** | O(n) | 状态空间模型 | Mamba、Jamba |
| **Hybrid** | 混合 | 注意力 + SSM | Jamba、Qwen3-MoE 部分变体 |

目前来看，Transformer 仍然是绝对主流，但长文本场景正在推动新架构的发展。

---

# 第五篇 · 实战篇 — 和你的 01_Qwen3.py 对答案

## 5.1 代码架构与论文的对应

| 论文组件 | `01_Qwen3.py` 中的实现 | 说明 |
|----------|----------------------|------|
| Scaled Dot-Product Attention | `scaled_dot_product_attention()` | 核心公式 `softmax(QKᵀ/√d)V` |
| Multi-Head + GQA | `Qwen3Attention` | 16 Q-heads, 8 KV-heads |
| RoPE | `precompute_freqs_cis()` + `apply_rotary_emb()` | 半维配对旋转 |
| FFN + SwiGLU | `Qwen3MLP` | `silu(gate) * up` |
| Pre-Norm RMSNorm | `Qwen3RMSNorm` | 去掉均值计算 |
| Residual Connection | `x = x + sublayer(x)` | 每个子层后的加法 |
| KV Cache | `Qwen3Model.kv_cache` | dict 缓存 K/V |
| Decoder Layer | `Qwen3DecoderLayer` | Attention + FFN + Pre-Norm |
| Final Norm + LM Head | `Qwen3Model` 末尾 | RMSNorm + Linear → logits |

## 5.2 从论文到代码的思维地图

当你在 `01_Qwen3.py` 中看到一段代码时，可以这样追溯它的论文来源：

```
看到 scaled_dot_product_attention() → 论文第 3.2.1 节 (Scaled Dot-Product Attention)
看到 Q/K 拆分多头 → 论文第 3.2.2 节 (Multi-Head Attention)
看到 RoPE 旋转 → 论文《RoFormer》(2021) + 现代实践
看到 SwiGLU → 论文《GLU Variants Improve Transformer》(2020)
看到 Pre-Norm / RMSNorm → GPT-3 + LLaMA 实践
看到 GQA → 论文《GQA: Training Generalized Multi-Query Transformer》(2023)
看到 KV Cache → 推理加速的通用工程实践
```

---

# 附录 · 论文精读路线图

如果你想真正读透《Attention Is All You Need》，按这个顺序来：

1. **先读图**：论文的 Figure 1（整体架构）和 Figure 2（缩放点积注意力 + 多头）
2. **再读 Section 3**（模型结构）：3.1 → 3.2 → 3.3 → 3.4 → 3.5
3. **读 Section 4**（为什么用自注意力）：Table 1 对比了计算复杂度
4. **读 Section 5**（训练）：了解 warmup 学习率策略
6. **回头看 Section 2**（背景）：了解 Transformer 出现前的工作
7. **最后读实验 Section 5.4**：看它在翻译任务上的表现

> 📍 论文路径：`https://arxiv.org/pdf/1706.03762`

---

# 参考文献

- 《Attention Is All You Need》Vaswani et al., 2017
- 尚硅谷大模型课程笔记：`NLP1.1.0.md` — Transformer 模型章节
- 尚硅谷大模型高频面试题：`高频面试题-V2.1.9.md` — 大模型原理章节
- `01_Qwen3.py` — Qwen3 手写实现
