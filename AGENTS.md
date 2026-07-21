# AGENTS.md

Teaching repo: hand-written Qwen3-0.6B + SFT notebook. No package layout, tests, lint, or CI.

## Layout

| Path | Role |
|------|------|
| `01_Qwen3.py` | From-scratch Qwen3 (GQA, RoPE, SwiGLU, RMSNorm, KV cache, weight map) |
| `02_sft_demo.ipynb` | Chat template, answer mask, SFT loss, train loop sketch |
| `model/` | Local model assets (often incomplete) |
| `晨测/` | Quiz notes only |
| `requirements.txt` | Pinned deps; torch wheels are **CUDA win/linux only** |

## Run

```bash
# use project venv
.venv/bin/python 01_Qwen3.py

# weight load + short greedy generate (needs safetensors on disk)
.venv/bin/python -c "import importlib.util as u; s=u.spec_from_file_location('q','01_Qwen3.py'); m=u.module_from_spec(s); s.loader.exec_module(m); m.test_load_qwen3_safetensors()"
```

- Prefer `uv pip install …` into `.venv`.
- Device helper: `cuda` → `mps` → `cpu` (`get_device()` in `01_Qwen3.py`).
- `main()` does **not** load weights → garbage text is expected.
- Default dtype is `bfloat16`.

## Model assets (easy to miss)

- `01_Qwen3.py` expects:
  - `model/Qwen3-0.6B/tokenizer.json`
  - `model/Qwen3-0.6B/model.safetensors` (for real generation / `test_load_qwen3_safetensors`)
- Repo may only ship the tokenizer. Missing weights → `FileNotFoundError` on load.
- `02_sft_demo.ipynb` uses **`model/Qwen3-0.6B-Base`** (HF layout via `AutoTokenizer` / `AutoModelForCausalLM`), not the hand-written path above. That directory is separate and may also be absent.

Weight loader maps HF names → teaching names with `strict=True` (`load_qwen3_safetensors_weights`). Do not invent alternate key maps without checking that function.

## macOS gotchas

- `requirements.txt` does not provide a macOS torch wheel; install a Mac-capable torch separately if recreating the venv.
- Building `xformers` from source fails with Apple Clang (`-fopenmp`). Use Homebrew GCC, e.g.:
  ```bash
  brew install libomp ninja gcc
  CC=/opt/homebrew/bin/gcc-16 CXX=/opt/homebrew/bin/g++-16 \
    uv pip install xformers==0.0.35 --no-build-isolation
  ```
- MPS is fine for this small model; do not hardcode `cuda`.

## Code conventions in `01_Qwen3.py`

- **Pre-Norm** blocks; final `RMSNorm` before LM head.
- **GQA**: 16 query heads, 8 KV heads; K/V `repeat_interleave` to match Q.
- **RoPE**: half-dimension pairing `(0,d/2), (1,d/2+1), …` via `rotate_half`, not adjacent pairs; decode uses `offset=start_pos` / `current_pos`.
- **KV cache**: dict `layer_idx → (K, V)`; inference only. Prefill full prompt once, decode feeds shape `[B, 1]`.
- **Tokenizer**: `tokenizers.Tokenizer` + optional chat wrap (`<|im_start|>` …). Base vs chat EOS depends on tokenizer filename (`base` → `<|endoftext|>`, else `<|im_end|>`).
- Keep Chinese teaching comments unless asked to strip them.

## SFT notebook notes

- Loss should be masked to assistant answer tokens only (`create_answer_mask` / `assistant_answer_mask`).
- Labels are typically next-token shifted (`input_ids[:, 1:]`).
- Notebook train path is Transformers + AdamW sketch, not the hand-written `Qwen3Model`.

## Out of scope unless asked

- No commit/push conventions in-repo.
- No formal test suite; verification = run script / notebook cells.
- Do not add production packaging unless requested.
