---

excalidraw-plugin: parsed
tags: [excalidraw]

---
==⚠  Switch to EXCALIDRAW VIEW in the MORE OPTIONS menu of this document. ⚠== You can decompress Drawing data with the command palette: 'Decompress current Excalidraw file'. For more info check in plugin settings under 'Saving'


# Excalidraw Data

## Text Elements
旋转位置编码 (RoPE: Rotary Position Embedding)
预计算 cos/sin 频率矩阵
θᵢ = base^(-2i/d)
cos, sin = f(pos, θᵢ)  # 每个位置每维一对
Q / K 向量
对 Q/K 应用旋转变换 (rotate_half)
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

q_embed = q*cos + rotate_half(q)*sin
k_embed = k*cos + rotate_half(k)*sin
RoPE 配对方式 (半维配对):
dim 0
dim 2
配对旋转
dim 1
dim 3
配对旋转
💡 关键性质:
• RoPE 使 Q·K 点积自动包含相对位置信息
• 两 token 距离越远，旋转角度差越大，点积衰减
• 支持长度外推 (extrapolation)
• 与原始 Transformer sin/cos PE 不同：RoPE 旋转 Q/K 本身，而非加在输入上
%%
## Drawing
```compressed-json
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
```
%%
