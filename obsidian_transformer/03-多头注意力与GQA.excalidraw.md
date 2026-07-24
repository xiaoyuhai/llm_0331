---

excalidraw-plugin: parsed
tags: [excalidraw]

---
==⚠  Switch to EXCALIDRAW VIEW in the MORE OPTIONS menu of this document. ⚠== You can decompress Drawing data with the command palette: 'Decompress current Excalidraw file'. For more info check in plugin settings under 'Saving'


# Excalidraw Data

## Text Elements
多头注意力 (Multi-Head Attention) 与 GQA
输入: X (B, S, D)
拆分为 H 个头
Head 1
(Q₁, K₁, V₁)
Head 2
(Q₂, K₂, V₂)
Head 3
(Q₃, K₃, V₃)
Head 4
(Q₄, K₄, V₄)
Head 1 输出
Head 2 输出
Head 3 输出
Head 4 输出
拼接所有 Head (Concat)
线性投影 W_O → 最终输出
GQA vs MHA vs MQA
MHA:  Q=H, K=H, V=H
每头独立，质量最高，推理最慢
MQA:  Q=H, K=1, V=1
推理极快，质量略降
GQA:  Q=H, K=G, V=G
兼顾质量与效率
你代码中: H=16, G=8 ⭐
GQA KV 扩展:

K: (B, 8, S, D) → (B, 16, S, D)
V: (B, 8, S, D) → (B, 16, S, D)
repeat_interleave(dim=1)
%%
## Drawing
```compressed-json
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
```
%%
