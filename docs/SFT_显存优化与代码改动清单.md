# `02_sft_demo.py` 训练显存/性能优化文档

## 0. 问题现象与诊断

报错关键信息：
```
CUDA out of memory. Tried to allocate 726.00 MiB.
GPU 0 has a total capacity of 8.00 GiB of which 0 bytes is free.
20.98 GiB is allocated by PyTorch
```

- 物理显存 8GB,PyTorch 累计申请 20.98GB(靠反复释放/重分配硬撑到第 22 步)。
- 训练显存 = **权重 W + 梯度 ∇W + 优化器状态 + 激活值 + logits/loss 中间量**。

---

## 1. 当前显存占用估算(fp32,seq=2500,batch=1)

| 项目 | 公式 | 大小 |
|---|---|---|
| 模型权重 (W) | 0.6B × 4B (fp32) | 2.4 GB |
| 梯度 (∇W) | 同 W | 2.4 GB |
| AdamW 状态 (m, v) | 2 × W | 4.8 GB |
| **静态合计** | | **9.6 GB** ← 已超 8GB |
| logits 张量 | B×seq×vocab×4 = 1×2500×151936×4 | 1.5 GB |
| `log_softmax(logits)` | 同上 | 1.5 GB |
| attention / MLP 激活 | 随 seq² 增长 | 数 GB |
| **峰值总量** | | **≈ 20+ GB** |

结论:**静态部分就已经超过 8GB**,所以无论怎样调 seq 都救不回全 fp32 + 全参数 + AdamW,必须从静态部分入手。

---

## 2. 优化方案分两类

- **A. 参数层**(不改代码结构,只改 `SFTConfig` 或函数默认值)
- **B. 代码层**(需要改 `02_sft_demo.py` 实现)

---

## 3. 参数层优化(改 `SFTConfig`,行 9-21)

### 3.1 `max_length: 2500 → 512`(`get_train_data`,第 33 行)

- **为什么**:激活随 `seq²` 增长,logits 按 `seq×vocab` 线性增长。2500→512 可让激活减少约 24×,logits 减少 4.9×。
- **代价**:长样本被截断。
- **⚠️ 关键陷阱(你遇到的梯度报错就在这)**:`create_answer_mask`(行 56-81)依赖 `<|im_end|>` 的 eos 位置来切分 user/assistant 轮次。**截断到 512 可能正好把某一轮的 `<|im_end|>` 砍掉**,导致 `_parse_conversation_turns` 拿到的 eos 数量是奇数,或 user/assistant 轮数不匹配,`_set_answer_masks` 里 `zip(user_ends, assistant_ends)` 出现长度不一致 → mask 设置错位 → 梯度异常。
  - **应对**:
    1. 截断前先检查样本的 eos 数量是否为偶数,不是就跳过或保留更长 seq。
    2. 或者把 `max_length` 设到 768/1024 留出余量。
    3. 或在 `create_answer_mask` 里加保护:eos 数 < 2 或为奇数时整条置 0,跳过该样本。
- **建议值**:`max_length = 512`(教学场景)或 `1024`(覆盖更多样本且降低截断风险)。

### 3.2 `batch_size: 1`

- 已是最小,无需再降。

### 3.3 `lr: 3e-5`

- 全参数 SFT 合理,无需改。若改用 LoRA 应调到 `1e-4 ~ 2e-4`。

### 3.4 `warmup_ratio: 0.1`

- 合理,保持。

### 3.5 `train_data_size / eval_data_size`

- 与显存无直接关系,但 `get_train_data` 一次性 tokenize 全部数据塞进 Python list(行 30-34),CPU 内存随 `train_data_size` 线性增长。

---

## 4. 代码层优化点(按收益排序)

### 4.1 模型用 bf16 加载(最高优先级) — 第 265 行

```python
model = AutoModelForCausalLM.from_pretrained(
    ModelConfig.REMOTE_MODEL_NAME_BASE,
    torch_dtype=torch.bfloat16,
)
```

- **为什么**:W / ∇W / 激活 / logits 全部减半。静态 9.6GB → **4.8GB**,瞬间低于 8GB 上限。
- **代价**:精度略低,对 0.6B SFT 够用;与 AGENTS.md "默认 bfloat16" 一致。
- 注意:`compute_loss` 里 `log_softmax` 仍建议 fp32 以稳定数值,可加 `logits.float()`。

### 4.2 开启梯度检查点 — 第 269 行附近

```python
model.gradient_checkpointing_enable()
model.enable_input_require_grads()   # 配合冻结 embedding 时用
```

- **为什么**:前向只存关键点输入,反向重算中间激活,省约 50% 激活显存。
- **代价**:训练变慢约 30%。8GB 卡上几乎必选。

### 4.3 缩短 `max_length` — 第 33 行

```python
result = tokenizer.apply_chat_template(
    message_list, tokenize=True, truncation=True, max_length=512
)["input_ids"]
```

- 必改项。**配合 §3.1 的截断陷阱一起处理**。

### 4.4 `compute_loss` 的 logits 内存爆炸 — 第 154-186 行

当前两次 materialize 大张量(logits 1.5GB + log_probs 1.5GB 同时存在):

```python
log_probs = torch.log_softmax(logits, dim=-1)
```

优化方式二选一:

- **方式 A(最小改动)**:
  ```python
  logits = logits.float()
  log_probs = torch.log_softmax(logits, dim=-1)
  del logits
  ```
  峰值减半。
- **方式 B(推荐)**:chunked cross-entropy,分块算 log_softmax 再 gather,峰值从 `[B, seq, vocab]` 降到 `[B, chunk, vocab]`。

### 4.5 换 8-bit 优化器 — 第 264, 270 行

```python
from bitsandbytes.optim import AdamW8bit
optimizer = AdamW8bit(model.parameters(), lr=sft_config.lr)
```

- **为什么**:AdamW 的 (m, v) 从 fp32 压成 int8,4.8GB → **1.2GB**。
- **依赖**:`uv pip install bitsandbytes`,仅 CUDA。

### 4.6 LoRA / QLoRA(8GB 卡最理想) — 第 265 行之后

```python
from peft import LoraConfig, get_peft_model
lora_cfg = LoraConfig(r=8, lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora_cfg)
```

- **为什么**:只训 ~1% 参数,梯度与优化器状态从全模型降到几十 MB;可再 `load_in_4bit=True`(QLoRA),权重 2.4GB → 0.6GB。
- **代价**:教学上"全参数微调"演示性减弱,但 8GB 卡实际跑全参数不现实。

### 4.7 数据加载流程重构 — 第 24-36, 276-292 行

当前问题:
1. `get_train_data` 一次性 tokenize 全部 10000 条保存在内存。
2. 训练循环里**原地 `sample.extend([...])` padding**(行 290, 230),会污染数据且每 epoch 越_pad 越长。
3. 没有 `attention_mask`,padding 的 token 仍参与前向,浪费显存且影响 loss。

建议用 `Dataset + DataLoader` 配合 tokenizer 的 `padding=True`。不直接解决 CUDA OOM,但解决数据污染 + CPU 内存隐患。

### 4.8 加 `attention_mask` 并屏蔽 pad — 第 292 行

```python
attention_mask = (data_tensor != tokenizer.pad_token_id).long()
logits = model(input_ids, attention_mask=attention_mask).logits
```

### 4.9 评估时显存释放 — 第 218, 326 行

eval 之后加 `torch.cuda.empty_cache()` 缓解碎片。

### 4.10 环境变量 — 启动前设置

```powershell
$env:PYTORCH_ALLOC_CONF="expandable_segments:True"
```

- **为什么**:让分配器跨段使用碎片显存。
- 注意:只解决碎片,**无法解决总量超限**,必须配合 §4.1/4.3 才有效。

---

## 5. 推荐组合方案

| 方案 | 改动 | 8GB 能否跑通 | 训练速度 |
|---|---|---|---|
| **最小集** | §4.1 (bf16) + §4.3 (seq=512) | ✅ 大概率可跑 | 快 |
| **稳妥集** | 最小集 + §4.2 (grad ckpt) + §4.10 | ✅ 稳定可跑 | 中等 |
| **极限集** | 稳妥集 + §4.5 (8bit AdamW) | ✅ 余量大 | 中等 |
| **生产集** | §4.6 (LoRA) + bf16 + seq=1024 | ✅ 余量巨大 | 快 |

建议按 **最小集 → 稳妥集 → 极限集** 顺序逐项加,每加一项重新跑观察显存与 loss。

---

## 6. 不推荐的做法

- `torch.cuda.empty_cache()` 每步清:治标不治本,反而增加调度开销。
- 叠加 `autocast`:HF `torch_dtype=bf16` 已等价,重复。
- 改用 fp16 训练:范围窄,易 NaN,bf16 更稳。
- `load_in_8bit=True` 做全参数微调:8bit 基座配全参数训不稳定,应配 LoRA。

---

## 7. 改动清单速查(按文件位置)

| 位置 | 行号 | 建议改动 | 优先级 |
|---|---|---|---|
| `get_train_data` | 33 | `max_length=512` + 截断保护 | ★★★ |
| `get_eval_data` | 47 | 同上 | ★★★ |
| `train` 模型加载 | 265 | 加 `torch_dtype=torch.bfloat16` | ★★★ |
| `train` 梯度检查点 | 269 后 | `model.gradient_checkpointing_enable()` | ★★ |
| `train` 优化器 | 270 | 换 `AdamW8bit`(可选) | ★ |
| `train` 前向调用 | 308 | 传 `attention_mask` | ★★(正确性) |
| `compute_loss` | 162 | `logits.float()` + `del logits` | ★ |
| `eval_model` padding | 230 | 改为构造副本,不污染原数据 | ★(正确性) |
| `create_answer_mask` | 56-81 | 加 eos 数量奇偶/轮次保护 | ★★★(配合截断) |
| `main` | 348 | `SFTConfig(train_data_size=200, max_length=512)` | ★★★ |
| 运行环境 | — | `PYTORCH_ALLOC_CONF=expandable_segments:True` | ★ |

---

## 8. 截断导致梯度报错的排查指引(你刚遇到的问题)

**根因**:`max_length=512` 把某条样本截断到 `<|im_end|>` 中间,导致 `create_answer_mask` 里:

1. `eos_position` 数量不对(奇数或为 0)。
2. `_parse_conversation_turns` 返回的 `user_ends` / `assistant_ends` 长度不匹配。
3. `_set_answer_masks` 的 `for user_end, assistant_end in zip(...)` 静默丢掉了多余的那一端。
4. 最终 `final_mask` 的覆盖范围错位,把不属于 assistant 的位置标成 1,梯度方向错误 → loss 异常或 NaN。

**验证方法**:
```python
# 在 train 循环里加一行打印
print("eos count:", final_mask.sum().item(), "seq len:", input_ids.shape[1])
```
若 `final_mask.sum() == 0` 或异常大,就是截断把 eos 搞丢了。

**最小修复**:把 `max_length` 临时设回 1024 或 2048(配合 §4.1 bf16 已经能省很多显存),再观察 loss 是否正常。等后面加了 bf16/grad ckpt,显存够了再逐步降 seq。