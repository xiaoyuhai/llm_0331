# macOS 上 Python 访问 HuggingFace 失败：SSL 证书问题

> 现象：浏览器和终端 `curl` 都能打开 HF / 镜像，但 notebook / `transformers` / `huggingface_hub` 报连不上。  
> 结论：多半不是网络挂了，而是 **python.org 安装的 Python 没装根证书**。

---

## 1. 报错长什么样

典型栈（notebook 里常见）：

```text
FileMetadataError: Distant resource does not seem to be on huggingface.co.
...
OSError: We couldn't connect to 'https://hf-mirror.com' to load the files,
and couldn't find them in the cached files.
```

可能同时看到 PyCharm 警告：

```text
您已将 JVM 属性 https.proxyHost 设置为 127.0.0.1 ...
```

以及代码：

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("model/Qwen3-0.6B-Base")
```

---

## 2. 先分清：其实有两层问题

| 层级 | 问题 | 表现 |
|------|------|------|
| A. 本地路径 | `model/Qwen3-0.6B-Base` 不存在或不是完整 HF 目录 | `from_pretrained` 会把字符串当 Hub repo，去网上拉 |
| B. Python SSL | python.org 的 Python 默认 CA 为空 | HTTPS 校验失败，被包装成“连不上 Hub” |

本仓库当时的真实情况：

```text
model/
└── Qwen3-0.6B/
    └── tokenizer.json          # 只有 tokenizer，给 01_Qwen3.py 用

# notebook 期望的是另一套完整 HF 布局：
model/Qwen3-0.6B-Base/          # 当时不存在
```

所以：**本地没有 Base 目录 → 必走网络；网络请求又因证书失败。**

和手写模型路径不要混：

| 路径 | 用途 |
|------|------|
| `model/Qwen3-0.6B/` | `01_Qwen3.py` + `tokenizers.Tokenizer` |
| `model/Qwen3-0.6B-Base/` | `02_sft_demo.ipynb` 的 `AutoTokenizer` / `AutoModelForCausalLM` |

---

## 3. 为什么“浏览器能上，代码不能上”

| 客户端 | 用的信任链 | 结果 |
|--------|------------|------|
| 浏览器 / `curl` / 系统 `openssl` | macOS 系统证书 | 正常 |
| 项目 `.venv` 里的 Python | 基座 Python 自带的 OpenSSL CA 路径 | 可能是空的 → 失败 |

关键自检：

```bash
# 1) 系统侧：通常 OK
curl -I https://hf-mirror.com
curl -I https://huggingface.co

# 2) Python 默认 SSL：出问题时长这样
.venv/bin/python - <<'PY'
import ssl, os, urllib.request
print(ssl.get_default_verify_paths())
print(ssl.create_default_context().cert_store_stats())
# 坏的时候常见：
#   openssl_cafile=.../etc/openssl/cert.pem 但文件不存在
#   stats → {'x509': 0, 'x509_ca': 0}
try:
    urllib.request.urlopen("https://hf-mirror.com", timeout=15)
    print("HTTPS OK")
except Exception as e:
    print("HTTPS FAIL:", e)
PY
```

失败时的真实异常往往是：

```text
SSLCertVerificationError: certificate verify failed:
unable to get local issuer certificate
```

`huggingface_hub` 会把它包装成更笼统的 “couldn't connect / Distant resource...”，所以第一眼像网络问题。

补充：若开了 Clash 等代理，DNS 可能落到 `198.18.x.x`（fake-ip）。  
**能 ping/curl 通只说明代理链路在，不代表 Python 证书库是好的。**

---

## 4. 根因：python.org 版 Python 的老坑

macOS 上若用官网安装包：

```text
/Library/Frameworks/Python.framework/Versions/3.12/
```

它期望的 CA 文件在：

```text
.../Versions/3.12/etc/openssl/cert.pem
```

但安装后这个文件经常 **不存在**。  
官网附带了补救脚本：

```text
/Applications/Python 3.12/Install Certificates.command
```

脚本逻辑本质是：

1. `pip install -U certifi`
2. 把 `.../etc/openssl/cert.pem` **符号链接**到 `certifi` 的 `cacert.pem`

鸡生蛋问题：

- 没证书 → `pip install certifi` 自己也 SSL 失败  
- 所以双击脚本也可能报：`Install Certificates failed`

这不是“每个项目配置错了”，而是 **基座解释器缺 CA**。  
从它建出来的所有 venv（包括本仓库 `.venv`）都会继承这个问题。

---

## 5. 怎么修（从治本到兜底）

### 5.1 治本：给每个 python.org 版本补 CA（推荐）

若 `Install Certificates.command` 能跑通，优先用它。  
跑不通时，直接把系统 CA 拷进 Python 的 OpenSSL 目录（admin 组可写时无需折腾 sudo 密码）：

```bash
# 以 3.12 为例；3.13/3.14 同理
PY_VER=3.12
OPENSSL_DIR="/Library/Frameworks/Python.framework/Versions/${PY_VER}/etc/openssl"
mkdir -p "$OPENSSL_DIR"
rm -f "$OPENSSL_DIR/cert.pem"

# 若该版本 base 里已有 certifi，可 ln -s 到 certifi 的 cacert.pem
# 否则直接用系统 bundle：
cp /etc/ssl/cert.pem "$OPENSSL_DIR/cert.pem"
chmod 644 "$OPENSSL_DIR/cert.pem"
```

修好后应类似：

```text
stats → {'x509': 128, 'crl': 0, 'x509_ca': 128}
HTTPS OK
```

本机已处理版本：

- `3.12` / `3.13`：写入系统 CA 到 `cert.pem`
- `3.14`：链接到自带 `certifi/cacert.pem`

**现有 venv 一般不用重建**，会复用基座的 `openssl_cafile` 路径。

### 5.2 以后少踩坑：换更省心的基座 Python

| 来源 | 证书体验 |
|------|----------|
| python.org Framework | 常要手动补证书 |
| Homebrew：`/opt/homebrew/bin/python3` | 通常更省事 |
| `uv python` 管理的解释器 | 通常更省事 |

新项目示例：

```bash
uv venv --python /opt/homebrew/bin/python3
# 或
uv venv --python 3.12   # 让 uv 自己拉/管理 Python
```

### 5.3 全局兜底（shell 层保险）

已写入 `~/.zshrc` 的思路：

```bash
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/cert.pem}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$SSL_CERT_FILE}"
export CURL_CA_BUNDLE="${CURL_CA_BUNDLE:-$SSL_CERT_FILE}"
```

注意：

- 新开终端才会读到 zshrc  
- **PyCharm / Jupyter kernel 不一定继承 zshrc**，改完后应 **重启 IDE 或重载 kernel**  
- 治本仍是 5.1；环境变量只是双保险

### 5.4 不要用的“假修法”

- 业务代码里 `ssl._create_unverified_context()` / 关校验  
- 每个项目手写一遍临时 `os.environ["SSL_CERT_FILE"]=...` 当长期方案  
- 把责任全推给 HF 镜像或盲目关代理（先证明是不是证书）

---

## 6. 证书修好后，模型怎么落地

notebook 需要完整 HF 布局，不只是一个 `tokenizer.json`。

```bash
cd /Users/yuhaifeng/PycharmProjects/0331_llm

# 下载到 notebook 期望目录
uv run huggingface-cli download Qwen/Qwen3-0.6B-Base \
  --local-dir model/Qwen3-0.6B-Base
```

或先走 Hub id（依赖网络）：

```python
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
```

本地目录就绪后再用：

```python
tokenizer = AutoTokenizer.from_pretrained("model/Qwen3-0.6B-Base")
```

---

## 7. 和镜像 / 代理的关系（次要，但会叠加）

常见环境：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

说明：

1. **主因是证书时**：修 CA 后 `urllib`/`httpx` 访问镜像应恢复  
2. **镜像行为**：`hf-mirror.com` 有时对资源返回 `308` 跳回 `huggingface.co`  
   - 裸 `httpx` 跟着跳也许还能下  
   - `huggingface_hub` 的元数据/HEAD 逻辑可能更挑，偶发仍报 `LocalEntryNotFoundError`  
3. 不稳时可临时：

```bash
unset HF_ENDPOINT
# 或
export HF_ENDPOINT=https://huggingface.co
```

4. PyCharm 的 `https.proxyHost=127.0.0.1` 是 **JVM 旧代理属性** 提示；  
   它主要影响 IDE 自身，不是这次 Python SSL 空 CA 的根因。  
   若本机没有代理听在该端口，应在  
   `Settings → Appearance & Behavior → System Settings → HTTP Proxy`  
   改成不代理/正确代理，避免叠 buff。

---

## 8. 排障清单（以后按这个过）

```text
1. 本地路径在不在？是不是完整 HF 目录？
   ls model/Qwen3-0.6B-Base

2. Python CA 是不是空的？
   .venv/bin/python -c "import ssl; print(ssl.create_default_context().cert_store_stats())"
   # x509_ca 应为 > 0

3. 纯 HTTPS 通不通？
   .venv/bin/python -c "import urllib.request; urllib.request.urlopen('https://hf-mirror.com', timeout=15); print('ok')"

4. Hub 下载通不通？
   uv run python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('Qwen/Qwen3-0.6B-Base','config.json'))"

5. 仍失败再看：HF_ENDPOINT、系统代理/Clash、HF_TOKEN、IDE 是否没重启
```

---

## 9. 一句话总结

- **不是 mac 上不了 HuggingFace，是 python.org 装的 Python 常常没根证书。**  
- 浏览器走系统信任链；裸 Python 走自己的 `cert.pem`，文件缺失就全军覆没。  
- **修基座解释器（或换 Homebrew/`uv` Python）一次，而不是每个项目修一次。**  
- 本仓库 notebook 还要额外保证：`model/Qwen3-0.6B-Base` 是完整模型目录，和 `01_Qwen3.py` 用的 `model/Qwen3-0.6B` 不是同一套。

## 10. 本次已落地的修复（更新）

> 记录时间：2026-07-21。下面这些已经在本机做过，不只是“建议”。

### 10.1 基座 Python 证书（治本）

已给 python.org Framework 写入/链接 CA：

| 版本 | 路径 | 处理 |
|------|------|------|
| 3.12 | `/Library/Frameworks/Python.framework/Versions/3.12/etc/openssl/cert.pem` | 复制系统 `/etc/ssl/cert.pem` |
| 3.13 | `.../3.13/etc/openssl/cert.pem` | 同上 |
| 3.14 | `.../3.14/etc/openssl/cert.pem` | 符号链接到该版本自带 `certifi/cacert.pem` |

验证（应 `x509_ca > 0`，且 HTTPS 成功）：

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 - <<'PY'
import ssl, urllib.request
print(ssl.create_default_context().cert_store_stats())
urllib.request.urlopen("https://hf-mirror.com", timeout=15)
print("ok")
PY
```

说明：

- 本仓库 `.venv` 基于 3.12，**不用重建**也会跟着好。
- Jupyter / PyCharm **即使不继承 zsh 环境变量**，只要用的是这套基座 Python，默认 SSL 也应正常。
- 官方 `Install Certificates.command` 当时因没证书装不了 `certifi` 而失败，所以改用了直接写入 `cert.pem` 的方式。

### 10.2 shell 兜底环境变量

已写入 `~/.zshrc`：

```bash
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/cert.pem}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$SSL_CERT_FILE}"
export CURL_CA_BUNDLE="${CURL_CA_BUNDLE:-$SSL_CERT_FILE}"
```

这只是双保险。notebook 不读 zshrc 时，仍主要靠 10.1。

### 10.3 本仓库模型目录

notebook 需要的完整 HF 布局已就绪：

```text
model/Qwen3-0.6B-Base/
  config.json
  generation_config.json
  tokenizer.json
  tokenizer_config.json
  vocab.json
  merges.txt
  model.safetensors
  ...
```

可用本地-only 方式验证：

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("model/Qwen3-0.6B-Base", local_files_only=True)
```

与手写脚本目录区分：

```text
model/Qwen3-0.6B/         # 01_Qwen3.py
model/Qwen3-0.6B-Base/    # 02_sft_demo.ipynb
```

### 10.4 调试残留已清理

已删除下载/调试过程中的临时文件，避免污染仓库：

- `model/Qwen3-0.6B-Base/.cache`
- `model/Qwen3-0.6B-Base/._____temp`
- `model/Qwen3-0.6B-Base/.msc` / `.mv` / `.gitattributes`
- 项目根目录 `__pycache__/`
- PyCharm 调试残留目录 `261_*_pycharm-support-libs/`
- 用户目录临时 CA 副本 `~/.local/share/python-certs/`

### 10.5 以后新项目还会不会因为证书报错？

**就“python.org 3.12/3.13/3.14 缺 CA”这一类问题：本机已经修过，正常不会再犯。**

更细一点：

| 场景 | 还会不会因证书挂 |
|------|------------------|
| 继续用现有 3.12/3.13/3.14 Framework 建 venv | 一般 **不会** |
| PyCharm / Jupyter 不带 zsh 环境变量 | 一般 **不会**（已修基座 `cert.pem`） |
| 新装一个 python.org 大版本（如 3.15）且忘了补证书 | **会**，要再补一次 |
| 重装/覆盖安装同一版本 Python，把 `etc/openssl/cert.pem` 清掉 | **会**，要再补一次 |
| 公司 MITM 代理、自定义根证书、错误系统代理 | 仍可能，但是另一类问题 |
| 本地模型路径不存在、镜像 308、没网 | 仍可能报 Hub 错误，但不是 CA 空 |

新项目建议习惯：

```bash
# 优先用 Homebrew 或 uv 管理的 Python，少碰 python.org 安装包坑
uv venv --python /opt/homebrew/bin/python3
# 或
uv venv --python 3.12
```

若以后又怀疑证书，先跑：

```bash
.venv/bin/python -c "import ssl; print(ssl.create_default_context().cert_store_stats())"
# 期望 x509_ca > 0
```

### 10.6 一句话

本机已把 python.org 3.12/3.13/3.14 的根证书补上，并给 shell 加了兜底；本仓库 `Qwen3-0.6B-Base` 也已本地化。  
**之后新项目默认不应再踩“浏览器能开 HF、Python 却 SSL 失败”这个坑**；只有重装 Python、新大版本、或代理/证书环境大变时才需要再检查。
