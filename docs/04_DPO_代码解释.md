# `04_dpo_demo.py` 逐行代码解释文档

> 本文配合 `04_dpo_demo.py` 阅读,目的是让你理清 DPO(直接偏好优化)这条算法线的脉络:
> **它到底在学什么 + 每段代码做什么 + 每个参数含义**。

---

## 一、先理脉络:DPO 在学什么?为什么需要它?

### 1.1 三阶段训练流水线回顾

```
01_Qwen3.py   从零实现 Qwen3 架构(只懂结构,不会对话)
      ↓
02_sft_demo.py  SFT 监督微调(学会"怎么生成回复",模仿标准答案)
      ↓
03_model_generate.py  推理验证(看看微调后模型说话通不通)
      ↓
04_dpo_demo.py  DPO 直接偏好优化(学会"哪个回复更好",对齐人类偏好)  ← 你现在学到这
      ↓
(后续 RLHF 体系: PPO / GRPO ...)
```

### 1.2 SFT 的局限 → 为什么需要 DPO

SFT 教模型"复读标准答案",但:
- 一个问题可以有多种好答案,SFT 学不到"哪种风格更受人喜欢"。
- 模型可能输出语法正确但价值观/实用性差的答案。

DPO 的核心思想:
> 给模型同一个 prompt 的两个回答——人类偏好的(chosen)和人类不喜欢的(rejected),
> 让模型"偏好 chosen 的程度"相对"偏好 rejected 的程度"变得比参考模型更大。

### 1.3 DPO 公式(记住这一条,代码就围绕它转)

$$
\mathcal{L}_{DPO} = -\log\sigma\Big(\beta \cdot \big[\underbrace{(\log\pi_\theta(y_w|x) - \log\pi_\theta(y_l|x))}_{\text{训练模型偏好差}} - \underbrace{(\log\pi_{ref}(y_w|x) - \log\pi_{ref}(y_l|x))}_{\text{参考模型偏好差}}\big]\Big)
$$

符号含义:
- $\pi_\theta$:正在训练的模型(参数逐步更新)
- $\pi_{ref}$:冻结不训练的参考模型(通常是 SFT 后的模型快照),作用是"防止训练模型偏离太远"
- $y_w$:chosen(winning)偏好回答
- $y_l$:rejected(losing)被拒回答
- $\beta$:控制偏好强度,越小越保守,越大越激进
- $\sigma$:sigmoid 函数;$-\log\sigma(\cdot)$ 即 binary cross-entropy,把"想让训练模型偏好差 > 参考模型偏好差"变成最大似然

---

## 二、文件结构概览

| 行号 | 段落 | 作用 |
|---|---|---|
| 1-4 | 导入 + 加载 tokenizer | Qwen3-0.6B-Base 的 tokenizer |
| 9-23 | `DPOConfig` 数据类 | 所有超参数 |
| 26-71 | `get_train_data / get_eval_data` | 取 chosen / rejected 偏好对 |
| 77-170 | `create_answer_mask` + 子函数 | 标记 assistant 回答位置(同 SFT) |
| 173-187 | `compute_loss` | **DPO 损失函数核心** |
| 189-214 | `compute_log_probs` | 由 logits 算回答的对数概率 |
| 220-237 | `cosine_decay` | 学习率余弦衰减(同 SFT) |
| 241-350 | `eval_model` | 评估函数 |
| 357-528 | `train` | 训练主循环 |
| 531-537 | `main` | 入口 |

---

## 三、逐段代码解释

### 3.1 导入与 tokenizer(行 1-4)

```python
from device_utils import get_device
from transformers import AutoTokenizer
from llm_config import ModelConfig
tokenizer = AutoTokenizer.from_pretrained(ModelConfig.REMOTE_MODEL_NAME_BASE)
```
- `ModelConfig.REMOTE_MODEL_NAME_BASE` = `"Qwen/Qwen3-0.6B-Base"`(见 `llm_config.py:7`)。
- 只加载 **Base 版**(非 Instruct 版)的 tokenizer,因为 SFT 后才有对话能力。

---

### 3.2 `DPOConfig`(行 9-23)

```python
@dataclass
class DPOConfig:
    train_data_size: int  = 10000
    eval_data_size: int = 500
    lr: float = 3e-6     # ⚠ DPO 学习率比 SFT 小一个数量级(SFT 是 3e-5)
    batch_size: int = 4
    warmup_ratio: float = 0.1
    save_dir: str = "./finetuned/03_dpo_demo"
    log_dir :str = "./logs/03_dpo_demo"
    eval_iter: int = 100
    log_iter: int = 100
    beta: float = 0.1    # ⭐ DPO 独有
```

参数逐条含义:

| 参数 | 含义 | 备注 |
|---|---|---|
| `train_data_size` | 训练样本数,默认 10000,`main()` 里被覆盖成 20000 | DPO 用偏好对,每条要 chosen + rejected 两次前向,batch_size 也大,显存压力大 |
| `eval_data_size` | 评估样本数,500 | |
| `lr` | 学习率,`3e-6` | **DPO 必须远小于 SFT**(SFT 是 3e-5)。因为 DPO 是在已学会的模型上做精修,步子大了会破坏 SFT 学到的能力 |
| `batch_size` | 批大小,4 | DPO 一条样本要做 4 次前向(训练模型 chosen/rejected + 参考模型 chosen/rejected),有效显存 ≈ SFT 的 4 倍 |
| `warmup_ratio` | 预热比例,0.1 | 前 10% 的 batch 走线性 warmup,之后余弦衰减 |
| `save_dir` | 模型保存目录 | `./finetuned/03_dpo_demo` |
| `log_dir` | TensorBoard 日志目录 | |
| `eval_iter` | 每 100 batch 评估一次 | |
| `log_iter` | 每 100 batch 记一次训练 loss | |
| `beta` | **DPO 专属 KL 控制系数**,默认 0.1 | ⭐ 见下 |

#### `beta` 详细解释

回顾 DPO 公式:
$$\mathcal{L} = -\log\sigma\big(\beta \times \text{margin}\big)$$

- `margin = (训练模型偏好差) - (参考模型偏好差)`
- `beta` 是 margin 前的系数,**控制偏好学习的"激进程度"**:
  - `beta` 小(如 0.01):grad 对 margin 不敏感,训练慢但稳,不容易过拟合/破坏 SFT
  - `beta` 大(如 1.0):放大差距,学得快,但容易把模型训崩
- 论文推荐 0.1~0.5,这里取 0.1 是保守值。
- 物理意义:`beta` 越大,等价于在 KL 约束里给"偏离参考模型"的惩罚越轻。

---

### 3.3 `get_train_data / get_eval_data`(行 26-71)

```python
from datasets import load_dataset
train_data = load_dataset("./data/ultrafeedback_binarized")["train_prefs"]
```

- 数据集是 **UltraFeedback 的"二值化"版本**(已在本地 `./data/ultrafeedback_binarized`)。
- 每条样本有两个字段:
  - `chosen`:人类偏好的回答(结构是对话 list)
  - `rejected`:人类不喜欢的回答(结构是对话 list)

```python
message_list = train_data[i]["chosen"]
result = tokenizer.apply_chat_template(message_list, tokenize=True)["input_ids"]
chosen_result.append(result)
```

- `apply_chat_template`:把 `[{"role":"user",...},{"role":"assistant",...}]` 转 token_ids。
- ⚠ 注意:**这里没有 `truncation=True` / `max_length=...`**(与 02_sft_demo.py 不同)。
  - 长样本会原样保留全长,后面 padding 会让单批最大长度很大 → 显存压力更大。
  - 想省显存需自己加 `truncation=True, max_length=512`。

---

### 3.4 `create_answer_mask`(行 77-170) — 与 SFT 完全相同

复用 SFT 的逻辑,详细见 02 的文档。简要:
- 全 0 初始化 `answer_mask = torch.zeros_like(labels)`
- 找 `<|im_end|>` 的位置 → 奇偶分组区分 user / assistant 轮次边界
- 把 assistant 轮次覆盖的 token 标 1

> DPO 里同样关键:loss 只该算在"assistant 实际回答的 token"上,不能把 prompt / user 部分也算进去。

---

### 3.5 `compute_loss`(行 173-187) — ⭐ DPO 核心

```python
def compute_loss(chosen_log_probs, rejected_log_probs,
                  ref_chosen_log_probs, ref_rejected_log_probs, beta):
    margin = chosen_log_probs - rejected_log_probs \
           - (ref_chosen_log_probs - ref_rejected_log_probs)
    loss = - torch.nn.functional.logsigmoid(beta * margin)
    return loss.mean()
```

逐行对照 DPO 公式:

| 代码 | 对应公式 |
|---|---|
| `chosen_log_probs - rejected_log_probs` | $\log\pi_\theta(y_w\|x) - \log\pi_\theta(y_l\|x)$,训练模型的偏好差 |
| `ref_chosen_log_probs - ref_rejected_log_probs` | $\log\pi_{ref}(y_w\|x) - \log\pi_{ref}(y_l\|x)$,参考模型的偏好差 |
| `margin = ... - ...` | 两个偏好差之差 |
| `beta * margin` | 公式中的 $\beta \cdot [\cdot]$ |
| `-logsigmoid(...)` | $-\log\sigma(\cdot)$,即 BCE |
| `.mean()` | 全 batch 平均 |

输入 4 个参数都是 `[batch_size, ]` 向量(每个样本一个标量 log prob)。

**为什么这样设计?**
- 想让训练模型"比参考模型更偏好 chosen"。
- 当 `margin > 0`(训练模型偏好差 > 参考模型偏好差),$` \sigma(\beta\cdot\text{margin}) \to 1`$,loss → 0(已学好)。
- 当 `margin < 0`(反过来偏好 rejected),loss 变大,梯度推回正确方向。
- `ref_*` 项相当于 **KL 正则**:防止训练模型为了放大偏好差而瞎改输出分布。

---

### 3.6 `compute_log_probs`(行 189-214) — 从 logits 算 log_prob

```python
log_probs = torch.log_softmax(logits, dim=-1)
label_token_log_prob = torch.gather(
    input=log_probs,
    dim=-1,
    index=labels.unsqueeze(-1)
).squeeze(-1)
masked_label_token_log_prob = assistant_mask * label_token_log_prob
log_probs = masked_label_token_log_prob.sum(dim=-1)
return log_probs
```

| 步骤 | 含义 |
|---|---|
| `log_softmax(logits, dim=-1)` | 把 logits 归一化成对数概率,形状 `[B, seq, vocab]` |
| `gather` | 按 `labels` 指定的 token_id 收集对应位置的对数概率,得到 `[B, seq]` |
| `* assistant_mask` | 屏蔽非回答部分,只保留 assistant 回答 token 的 log_prob |
| `.sum(dim=-1)` | 把整条回答的所有 token 的 log_prob 相加 = 整条回答的对数似然 $\log p(y\|x)$ |

⚠ **重要区别(对比 SFT)**:
- SFT 的 `compute_loss` 是**逐 token 算 NLL 再平均**,输出一个 token 级 loss。
- DPO 的 `compute_log_probs` 是**把整条回答的所有 token log_prob 加起来**,输出 `整条回答的 log 似然`,形状从 `[B, seq]` 压成 `[B,]`。
- 因为 DPO 比的是"整条回答的好坏",不是单个 token。

---

### 3.7 `cosine_decay`(行 220-237) — 与 SFT 完全相同

学习率调度:
- 前 `warmup_ratio` 比例:线性升到 `lr`
- 之后:余弦曲线衰减到 0

```python
decay_level = (np.cos(np.pi * progress) + 1) * 0.5  # 1 → 0
return lr * decay_level
```

---

### 3.8 `eval_model`(行 241-350)

逻辑和 `train` 主体几乎一样,区别:
- `model.eval()` + `torch.no_grad()`,不计算梯度
- 不 backward / 不 step
- 返回平均 loss

padding / mask 构造流程与训练相同(同样有 `sample.extend` 原地污染问题)。

---

### 3.9 `train` 主循环(行 357-528) — 重点段

#### 3.9.1 初始化(行 368-380)

```python
model = AutoModelForCausalLM.from_pretrained("finetuned/02_sft_demo_backup")
ref_model = AutoModelForCausalLM.from_pretrained("finetuned/02_sft_demo_backup")
model.to(device); ref_model.to(device)
model.train(); ref_model.eval()
optimizer = AdamW(model.parameters(), lr=dpo_config.lr)
```

关键点:
- **两个模型从同一个 SFT 后的快照加载**:这是 DPO 的标准做法,参考模型 = SFT 模型快照。
- 注意路径是 `02_sft_demo_backup` 不是 `02_sft_demo`,你需要先有这个目录(可能是 SFT 训练完另存一份)。
- `ref_model` 永远 `.eval()` 且不接收梯度,只做前向提供参照。
- ⚠ **两个模型同时在显存里**:8GB 卡极易爆,这是 DPO 比 SFT 显存翻倍的主因。

#### 3.9.2 取一个 batch(行 391-414)

对 chosen 和 rejected 分别:
1. 按 batch 取切片
2. 找当前 batch 最大长度,max_length
3. 用 `pad_token_id` padding 到等长(⚠ 原地 `extend` 会污染数据)
4. 转 tensor → 拆 `input_ids = data_tensor[:,:-1]` 和 `labels = data_tensor[:,1:]`(next-token 偏移)
5. **构造 `padding_mask` 和 `assistant_mask`,交集 = `final_mask`**

> ⚠ **没有 `truncation` + 没有 `attention_mask` 传给 model()**:与 02 同样的潜在问题。- 建议加 `attention_mask` 让模型正确处理 pad。

#### 3.9.3 四次前向传播(行 445-450) — DPO 的关键开销

```python
# 训练模型,2 次前向(要算梯度)
chosen_output_logits = model(chosen_input_ids).logits
rejected_output_logits = model(rejected_input_ids).logits

# 参考模型,2 次前向(无梯度,只读)
with torch.no_grad():
    ref_chosen_output_logits = ref_model(chosen_input_ids).logits
    ref_rejected_output_logits = ref_model(rejected_input_ids).logits
```

DPO 显存为什么是 SFT 的 4 倍:
- 同一批数据要算 **chosen** 和 **rejected** 两条序列(2×)
- 每条序列都要走 **训练模型** 和 **参考模型** 两次模型(2×)
- 训练模型两次前向的激活值要保存为了反向

#### 3.9.4 算 4 个 log_prob + loss(行 453-487)

```python
chosen_log_prob       = compute_log_probs(chosen_output_logits,      chosen_labels,  final_chosen_mask)
rejected_log_prob     = compute_log_probs(rejected_output_logits,    rejected_labels,final_rejected_mask)
ref_chosen_log_prob   = compute_log_probs(ref_chosen_output_logits,  chosen_labels,  final_chosen_mask)
ref_rejected_log_prob = compute_log_probs(ref_rejected_output_logits,rejected_labels,final_rejected_mask)

loss = compute_loss(
    chosen_log_prob, rejected_log_prob,
    ref_chosen_log_prob, ref_rejected_log_prob,
    dpo_config.beta
)
```

注意:**参考模型的 log_prob 直接用它的 logits 算**,不需要 `ref_loss.backward()`——它不更新参数。

#### 3.9.5 反向 + 更新(行 492-503)

```python
loss.backward()
current_lr = cosine_decay(current_batch, total_batch, dpo_config.warmup_ratio, dpo_config.lr)
optimizer.param_groups[0]["lr"] = current_lr
optimizer.step()
optimizer.zero_grad()
```

- 只对 `model` 的参数算梯度(因为 `ref_model` 在 `no_grad` 下做的)。- LR 调度写在 `step` 之前其实是设下次用的新 lr。
- `zero_grad` 清掉本轮梯度。

#### 3.9.6 日志 + 保存(行 506-528)

- 每 100 batch 评估一次 → TensorBoard 写 `eval/loss`
- 每 100 batch 写 `train/loss` 和 `train/current_lr`
- 训练完保存到 `dpo_config.save_dir`

---

### 3.10 `main`(行 531-537)

```python
def main():
    dpo_config = DPOConfig(train_data_size=20000, batch_size=4)
    train(dpo_config=dpo_config)
```

- `train_data_size=20000`(覆盖默认 10000)
- `batch_size=4` → 实际显存压力是 SFT 的 8~16 倍,8GB 几乎不可能跑通,见第六章避坑。

---

## 四、DPO vs SFT 关键差异对照表

| 维度 | SFT (02) | DPO (04) |
|---|---|---|
| 学习目标 | 复读标准答案 | 让 chosen 的概率比 rejected 上升 |
| 数据集 | ultrachat 单条对话 | ultrafeedback 偏好对 (chosen+rejected) |
| 模型数量 | 1 个 | 2 个 (训练 + 参考) |
| 前向次数/batch | 1 | 4 |
| Loss 公式 | `F.cross_entropy` 类 | `-logsigmoid(beta * margin)` |
| log_prob 单位 | 单 token | 整条回答(求和) |
| `lr` 量级 | `3e-5` | `3e-6`(小一个量级) |
| 参考模型约束 | 无 | 有(KL 正则) |
| `beta` 超参 | 无 | 有(0.1) |

---

## 五、代码里的关键参数清单(直接对应你能调的旋钮)

| 参数 / 变量 | 出处 | 作用 | 调节建议 |
|---|---|---|---|
| `lr` | `DPOConfig.lr` | 学习率 | DPO 经验值 1e-6~5e-6,过大易崩 |
| `beta` | `DPOConfig.beta` | DPO 偏好强度 | 0.1~0.5,小保守大激进 |
| `batch_size` | `DPOConfig.batch_size` | 批大小 | 8GB 卡建议改 1 |
| `train_data_size` | `DPOConfig` | 训练样本数 | 显存不够可减到 1000 |
| `warmup_ratio` | `DPOConfig` | 预热比例 | 0.1 合理 |
| `max_length` | (代码没设置!) | 序列上限 | **建议补上 `truncation=True, max_length=512`** |
| `02_sft_demo_backup` | 行 373 | ref 模型来源 | 必须是你 SFT 之后另存一份的目录 |

---

## 六、8GB 显存避坑清单(预测你会遇到的 OOM)

DPO 的显存需求大约是 SFT 的 **4 倍**,因为你一个 batch 跑 4 次模型前向 + 训练模型要存激活做反向。

按以下顺序逐项加,每加一项重跑观察:

| 优先级 | 改动 | 位置 | 为什么 |
|---|---|---|---|
| ★★★ | `batch_size: 4 → 1` | `main():532` | 直接把显存 / 4 |
| ★★★ | 加 `torch_dtype=torch.bfloat16` | 行 373-374 | 权重/激活减半 |
| ★★★ | 加 `truncation=True, max_length=512` | `get_train_data` 和 `get_eval_data` 的 `apply_chat_template` | 激活随 seq² 增长 |
| ★★ | 训练模型开 `gradient_checkpointing_enable()` | 行 375 之后 | 省激活 ~50% |
| ★★ | `ref_model` 可在 CPU 上跑(只读不训) | 行 374-376 | 省一份权重的显存 |
| ★ | 用 LoRA 包住训练模型(参考模型不动) | 行 375 后 | 只训 1% 参数 |
| ★ | `PYTORCH_ALLOC_CONF=expandable_segments:True` | 启动前环境变量 | 缓解碎片 |

> 最经济的"让代码能跑通"的最小集:batch_size=1 + bf16 + max_length=512,大概率能在 8GB 上跑起来。

---

## 七、跑之前的前置检查清单

跑 `04_dpo_demo.py` 前确认:

1. `./data/ultrafeedback_binarized` 数据集存在(代码用的本地路径,不是 HF 仓库名):
   ```
   D:\workspace\py_ws\llm_0331\data\ultrafeedback_binarized
   ```
   如果不存在,要么改成 `load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned")` 之类 HF 数据集,要么先下载。

2. SFT 备份目录存在:
   ```
   D:\workspace\py_ws\llm_0331\finetuned\02_sft_demo_backup
   ```
   代码从这加载 **训练模型** 和 **参考模型**(注意不是 `02_sft_demo`,是 `_backup`)。  
   如果只有 `02_sft_demo`,要么改第 373-374 行的路径,要么把 SFT 输出复制一份重命名。

3. tokenizer / chat_template 与 SFT 时一致。

---

## 八、一句话脉络总结

> **DPO 的本质 = 给同一个 prompt 的"好答案"和"坏答案",让训练模型对好答案的概率相对坏答案的概率,比参考模型做得更明显;通过 `-logσ(beta × margin)` 这个 BCE-style loss 来优化。**
>
> 代码骨架就是:
> 1. 加载 SFT 后的模型两份(一份训练,一份参考冻结)
> 2. 取一批 chosen + rejected 数据,4 次前向算 4 个 log_prob
> 3. `margin = 训练偏好差 - 参考偏好差`,`loss = -logsigmoid(beta * margin)`
> 4. backward 只更新训练模型
> 5. 循环到收敛,保存

理解这一段,你就掌握了 DPO 的全貌。