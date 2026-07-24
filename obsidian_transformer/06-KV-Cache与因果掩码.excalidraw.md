---

excalidraw-plugin: parsed
tags: [excalidraw]

---
==⚠  Switch to EXCALIDRAW VIEW in the MORE OPTIONS menu of this document. ⚠== You can decompress Drawing data with the command palette: 'Decompress current Excalidraw file'. For more info check in plugin settings under 'Saving'


# Excalidraw Data

## Text Elements
KV Cache 与 因果掩码 (Causal Mask)
因果掩码 (Causal Mask)
无 mask (双向注意力)
Token1  Token2  Token3  Token4
Token1   ✓      ✓      ✓      ✓
Token2   ✓      ✓      ✓      ✓
Token3   ✓      ✓      ✓      ✓
Token4   ✓      ✓      ✓      ✓
因果 mask (自回归)
Token1  Token2  Token3  Token4
Token1   ✓      ❌     ❌     ❌
Token2   ✓      ✓      ❌     ❌
Token3   ✓      ✓      ✓      ❌
Token4   ✓      ✓      ✓      ✓
每个 token 只能看到自己和之前的 token
右上三角 mask 为 -inf → softmax 后 = 0
KV Cache: 推理加速的核心优化
❌ 无 KV Cache (重复计算)
每步重新计算所有历史 K, V
✅ 有 KV Cache (缓存复用)
每步只算当前 token 的 K, V
Decode 步骤流程:
输入: [token_1]
算 K₁, V₁ → 缓存
Q₁ 与 cache 做注意力
输入: [token_2]
算 K₂, V₂ → 追加缓存
Q₂ 与 cache 做注意力
输入: [token_3]
算 K₃, V₃ → 追加缓存
Q₃ 与 cache 做注意力
你代码中的 KV Cache:

# prefill: 一次性处理全部 prompt token
k = torch.cat([cache_k, k_new], dim=2)
v = torch.cat([cache_v, v_new], dim=2)
cache[layer_idx] = (k, v)

# decode: 只取最后一位的 Q
q = q[:, :, -1:, :]
scores = Q @ Kᵀ / √d  # K 包含全部缓存
%%
## Drawing
```compressed-json
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
tBlNEu3xsy0+IqHRcx6L1tYPNTHxP9plHO4=
```
%%
