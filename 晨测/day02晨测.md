## 1、GPT系列的发展所推动的大语言模型训练的三阶段，分别是什么，作用是什么？

```
1. 预训练（Pre-training）
   - 做什么：海量无标注文本上做自监督（如 next-token prediction）
   - 作用：学语言规律、世界知识、基础推理，得到通用基座模型（Base）

2. 有监督微调 / 指令微调（SFT）
   - 做什么：用「指令–回答」高质量标注数据微调
   - 作用：把基座从“会补全”变成“会按指令对话/完成任务”（Chat / Instruct）

3. 对齐（Alignment，经典是 RLHF；也可用 DPO 等）
   - 做什么：用人类偏好数据，通过奖励模型 + RL（或直接偏好优化）继续优化
   - 作用：让输出更符合人类偏好：更有用、更安全、更少胡编

一句话：预训练学能力 → SFT 学会听话 → 对齐学得更靠谱。
```

## 2、LLM的基础架构包含哪些层，分别的作用是什么

```
以 Decoder-only Transformer（GPT / Qwen）为例：

1. Tokenizer（模型外但必需）
   - 文本 → token ids

2. Token Embedding
   - id → hidden 向量
   - 形状：[B, T] → [B, T, hidden_size]

3. 位置编码（RoPE / 绝对 PE 等）
   - 注入位置/相对位置信息（RoPE 通常作用在 Q/K 上）

4. Transformer Block × N（主干，重复堆叠）
   每层通常包含：
   (1) Self-Attention（MHA / GQA / MQA）
       - 建模 token 间依赖，聚合上下文
   (2) FFN / MLP（常为 SwiGLU）
       - 每个位置做非线性变换，加工/存储特征
   (3) Norm（LayerNorm / RMSNorm）+ 残差
       - 稳定训练；残差缓解梯度消失、便于堆深
   - 现代 LLM 多为 Pre-Norm：
     x = x + Attn(Norm(x))
     x = x + FFN(Norm(x))

5. Final Norm
   - 全网最后再归一化一次，稳定送入输出头

6. LM Head（输出头）
   - Linear(hidden → vocab)，得到 logits
   - 常与 Embedding 权重共享（tie_word_embeddings）

7. 采样（生成策略，架构外）
   - greedy / temperature / top-k / top-p 等选出下一个 token，循环直到 EOS

一句话：Embedding 进 → 多层「Attention 读上下文 + FFN 想内容」→ LM Head 说出下一个词。
```

## 3、正余弦位置编码是怎么样的，它的缺陷是什么？旋转位置编码和正余弦位置编码相比区别是什么

```
【正余弦位置编码 Sinusoidal PE】
- 来源：原版 Transformer
- 做法：给每个绝对位置 pos、每个维度 i 一组固定 sin/cos，直接加到 token embedding 上
  PE(pos, 2i)   = sin(pos / 10000^(2i/d))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
  x_input = token_emb + PE(pos)
- 低频刻画粗位置，高频刻画细位置；理论上相对位置可通过线性变换关联

【缺陷】
1. 长文本外推弱：训练长度外的位置分布模型没见过，推理易掉点
2. 绝对位置为主，相对距离不直观，模型要自己学“相距 k”
3. 位置与语义直接相加混在一起，经多层后位置信息可能变糊
4. 对长度变化不够稳健

【RoPE 旋转位置编码】
- 不加在 embedding 上，而对 Attention 的 Q、K 做旋转变换
- 把 head 向量两两一组（或前后半段配对）按位置旋转
- 关键性质：Q 在 m、K 在 n 的内积只依赖相对位置 m-n
  ⟨R_m q, R_n k⟩ = ⟨q, R_{n-m} k⟩
- 一般不旋转 V；decode 时用 offset 取对应 cos/sin

【对比】
| 维度     | 正余弦 PE              | RoPE                      |
| 注入位置 | 加在 embedding         | 乘在 Q/K（旋转）          |
| 位置类型 | 绝对为主               | 相对位置更自然            |
| 长文外推 | 较弱                   | 通常更好（可配 NTK/YaRN） |
| 现代 LLM | 早期常见               | LLaMA / Qwen 等主流       |

一句话：sin/cos 是“位置向量加进去”；RoPE 是“把位置写进 QK 内积的相对几何里”。
```

## 4、什么是KVCache，是用在训练阶段还是推理阶段？使用它的目的是什么？为什么只缓存K,V，不需要缓存Q

```
【是什么】
推理时把每一层 Attention 已算过的 Key、Value 存起来；
下一步生成只算新 token 的 K/V，与历史 cache 拼接后复用。

【训练还是推理】
- 主要用于推理（自回归生成）
- 训练一般整段并行前向，通常不需要逐步 KV Cache
- Prefill：整段 prompt 算完写入 cache
- Decode：每次输入 [B,1]，读 cache 并追加

【目的】
不用 cache：每步都要把完整序列重跑，历史 K/V 反复重算，极慢
用 cache：每个 token 的 K/V 只算一次，大幅降延迟、提吞吐

【为什么只缓存 K、V，不缓存 Q】
Attention: softmax(Q K^T / √d) V
- 当前步只需要“当前 token 的 Q”去查全部历史
- 历史 K：当“索引/标签”，后面还要被匹配 → 要存
- 历史 V：当“内容”，匹配上要取出来 → 要存
- 历史 Q：只在当时那一步用于产生当时输出，之后不再参与计算 → 不必存
- 再存 Q 还会多占显存，无收益

类比：K=书架标签，V=书内容，Q=这次来问的问题（问完即弃）。
```

## 5、讲一下什么是GQA，为什么现代大语言模型选择GQA，而不是传统MHA

```
【三种结构】（以 Q 头数为例）
1. MHA：Q 头 = K 头 = V 头，每个 Q 头独立一套 K/V（最强、最贵）
2. MQA：Q 多头，K/V 全局只有 1 组，所有 Q 共享（最省，可能掉点）
3. GQA：Q 多头，K/V 少头；Q 分组，每组共享一套 K/V（折中）

例（Qwen3-0.6B）：Q=16，KV=8，每 2 个 Q 头共享 1 组 K/V
实现：K/V 算出后 repeat_interleave 对齐到 Q 头数，再做标准注意力

【现代 LLM 为何选 GQA 而非 MHA】
主因在推理 + KV Cache，不在训练本身：
1. Cache 显存 ∝ layers × seq × kv_heads × head_dim
   - GQA 把 kv_heads 从 Hq 降到 Hq/G，cache 约降为 1/G
   - 长上下文时 MHA 的 cache 经常比权重大
2. decode 常是内存带宽瓶颈，K/V 更少 → 更快
3. 质量上 GQA ≈ MHA，明显好于过狠的 MQA
→ GQA = 速度/显存 与 效果 的甜点；LLaMA/Qwen/Mistral 等普遍采用

关系：MHA --减 KV 头--> GQA --减到 1--> MQA
```

## 6、讲一下什么是GLU，其作用是什么?它与普通的前馈神经网络相比增加了什么结构？

```
【是什么】
GLU（Gated Linear Unit，门控线性单元）：
一路做门控，一路做内容，逐元素相乘后再投影。

原始形式：
  GLU(x) = σ(W_g x) ⊙ (W_u x)

现代 LLM 常用 SwiGLU（门控激活为 SiLU/Swish）：
  h = SiLU(W_gate x) ⊙ (W_up x)
  out = W_down h

【作用】
1. 按输入动态开关特征通道（门≈0 压掉，门≈1 放行）
2. 比普通 ReLU/GELU 两层 FFN 表达力更强
3. 实证上质量更好，LLaMA/Qwen 等标配

【相对普通 FFN 多了什么】
普通 FFN：
  x → Linear↑ → Activation → Linear↓
  （通常 2 个矩阵）

GLU/SwiGLU FFN：
  x → gate_proj → SiLU ─┐
  x → up_proj ──────────┼─ ⊙ → down_proj
  （通常 3 个矩阵：gate / up / down）

核心增量：
1. 第二条上行分支（gate）
2. 逐元素门控相乘 ⊙
（为控制总参数，intermediate 常略缩小，如 8/3 d，而不是简单 4d）
```

## 7、讲一下什么是MoE，其优势是什么

```
【是什么】
MoE（Mixture of Experts，混合专家）：
- 不是只有一套 FFN，而是多个 Expert（专家 FFN）
- Router（门控）为每个 token 打分，选 Top-K 个专家（稀疏激活）
- 只计算被选中的专家，再按权重合并输出
- 通常替换 Transformer 里的 FFN 位置

流程：
  x → Router → Top-K 专家 → 加权求和 → 输出
  （其余专家本步不计算）

【优势】
1. 总参数可以很大，但单次只激活一小部分 → 算力一定时容量更大
2. 相对同等质量的超大稠密模型，推理/训练 FLOPs 更可控
3. 专家可分工（代码/数学/语言等），路由自动分流
4. 扩展性好：加专家≈加容量
5. 工业界冲大模型能力时性价比高（Mixtral、DeepSeek-MoE 等）

注意：激活参数省 ≠ 显存一定省（总参数往往仍要加载）；
还有负载均衡、通信、实现复杂等代价。

和 GLU/GQA 不冲突：专家内部可以是 SwiGLU，Attention 仍可用 GQA。
```

## 8、讲一下RMSNorm是什么，它与LayerNorm的核心区别是什么？

```
【RMSNorm 是什么】
Root Mean Square Layer Normalization：
用均方根做特征缩放的归一化，稳定深层训练。

  RMS(x) = sqrt(mean(x^2) + eps)
  RMSNorm(x) = γ ⊙ (x / RMS(x))
（通常只有可学习 scale γ，没有 bias β）

【与 LayerNorm 的核心区别】
LayerNorm：
  先减均值 μ，再除标准差 σ，再 γ ⊙ · + β
  → 中心化 + 缩放

RMSNorm：
  不减均值，只除 RMS，再乘 γ
  → 只缩放，不中心化

| 对比       | LayerNorm        | RMSNorm              |
| 减均值     | 要               | 不要（核心区别）     |
| 缩放统计量 | 标准差 σ         | 均方根 RMS           |
| 参数       | 常有 γ 和 β      | 通常只有 γ           |
| 计算量     | 更多             | 更少、更快           |
| 现代 LLM   | 早期常用         | LLaMA/Qwen 等主流    |

直觉：对 Transformer，控制尺度往往比强制均值归零更关键；
RMSNorm 更省且效果足够好，故成标配。
```

## 9、讲一下Pre-Norm和Post-Norm的区别是什么

```
【定义】
Post-Norm（原版 Transformer）：
  x = Norm( x + Attention(x) )
  x = Norm( x + FFN(x) )
  → 先子层与残差，再 Norm

Pre-Norm（现代 LLM 主流）：
  x = x + Attention( Norm(x) )
  x = x + FFN( Norm(x) )
  → 先 Norm，再子层，再残差
  （整网出口常再加 Final Norm）

【核心区别】
1. Norm 放在残差前还是后
2. Pre-Norm 有更干净的恒等通路：原始 x 可一路往下加
3. Post-Norm 主路径被每层 Norm 缩放，深层梯度更容易不稳

| 对比     | Post-Norm              | Pre-Norm                    |
| 稳定性   | 深了易不稳，要小心调参 | 更稳、更好训                |
| 深度扩展 | 较难                   | 容易堆很多层                |
| 使用现状 | 早期 BERT/原版         | GPT-2 后、LLaMA/Qwen 主流   |
| 出口     | 层内已 Norm            | 常需额外 final_norm         |

一句话：Pre-Norm 残差通路更直，深层更好练，所以现在基本都用 Pre-Norm。
```
