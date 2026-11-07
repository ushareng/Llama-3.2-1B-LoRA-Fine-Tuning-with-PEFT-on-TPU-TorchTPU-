# Llama 3.2 1B LoRA Fine-Tuning with PEFT on TPU (TorchTPU)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ushareng/Llama-3.2-1B-LoRA-Fine-Tuning-with-PEFT-on-TPU-TorchTPU-/blob/main/llama32_1b_lora_torchtpu_%282%29.ipynb)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Fine-tune **Llama 3.2 1B Instruct** with **LoRA** (Hugging Face [`peft`](https://github.com/huggingface/peft)) on a Cloud TPU using **[TorchTPU](https://github.com/google-pytorch/torch_tpu)**, the native PyTorch backend for TPUs. It's standard `transformers` + `peft` code. The TPU-specific changes are `device="tpu"`, `torch.compile`, and a few TPU-friendly habits (fixed shapes, few host syncs).

**Task: text-to-SQL.** Given a table schema and a question, the model should answer with only the SQL query. On a single **TPU v5e** chip, 400 LoRA steps (**53 seconds** of training) take exact-match accuracy from **12.5% to 79.0%**.

---

## Results (TPU v5e-1, Colab)

| | |
|---|---|
| Model | `unsloth/Llama-3.2-1B-Instruct` (ungated copy of `meta-llama/Llama-3.2-1B-Instruct`), bf16 |
| LoRA | r=16, α=32, dropout 0.05 on `q,k,v,o,gate,up,down`. **11.3M trainable params (0.75%)** |
| Data | [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context), 5,182 train examples, 200 held-out |
| Training | 400 steps × batch 8 × 256 tokens, AdamW, lr 2e-4, warmup + cosine |
| Software | TorchTPU on `torch 2.14.1` |

### Accuracy (exact match, 200 held-out questions)

| | exact match |
|---|---:|
| Base Llama 3.2 1B Instruct | 12.5% |
| **+ LoRA (400 steps)** | **79.0%** |

```
Q:    What is the lowest unlock order for the athens circuit?
REF:  SELECT MIN(unlock_order) FROM table_24463470_1 WHERE circuit = "Athens"
BASE: SELECT MIN unlock_order FROM table_24463470_1 WHERE circuit = 'athens'
LoRA: SELECT MIN(unlock_order) FROM table_24463470_1 WHERE circuit = "Athens"

Q:    What is the highest pick number of a defensive back with less than 136 overall points?
REF:  SELECT MAX(pick__number) FROM table_name_99 WHERE overall < 136 AND position = "defensive back"
BASE: SELECT MAX pick__number FROM table_name_99 WHERE overall < '136' AND position = 'defensive back'
LoRA: SELECT MAX(pick__number) FROM table_name_99 WHERE overall < 136 AND position = "defensive back"
```

### Speed and memory

| metric | value |
|---|---:|
| First step (incl. `torch.compile` + XLA compilation) | 36.8 s |
| 399 training steps | **52.9 s** |
| Throughput (padded / real tokens) | 15,460 / 6,830 tokens/s |
| Model FLOPs (4·N per token estimate) | ~93 TFLOP/s (≈47% of v5e's 197 bf16 TFLOP/s) |
| Peak HBM | 15.74 GiB of 16 GiB |
| Saved adapter | 59.5 MB |

Training loss (answer tokens only) falls from 0.17 to 0.055.

## How it works

| Step | What the notebook does | Why it matters on TPU |
|---|---|---|
| Install | Installs `torch_tpu` only if missing (Colab TPU runtimes include it) | |
| Execution mode | `torch.compile(dynamic=False)` for forward + loss, with the backward compiled through AOTAutograd. The optimizer runs in **defer-and-fuse** eager mode (`TORCH_TPU_DEFER_AND_FUSE=1`) | One XLA program per step instead of thousands of small ops |
| Model | Llama 3.2 1B in bf16 + PEFT LoRA adapters in fp32 | |
| Data | Chat-template formatting, labels only on the SQL answer, right-padded to a fixed `[8, 256]` batch with `drop_last` | **Static shapes, so the step compiles once.** Causal attention means end padding needs no attention mask |
| Loss | Gathers hidden states at the ≤64 answer positions and applies the LM head **only there** | Full logits would be `[8, 256, 128256]` (~2 GB in fp32). This is 4× smaller and avoids HBM OOM |
| Logging | Losses are copied to the host only every 25 steps | Avoids stalling the TPU with `.item()` every step |
| Evaluation | Fixed-shape greedy generator (`[B, 256+96]` buffer, one compiled forward reused for every token) | HF `generate()` changes tensor shapes every token and would recompile each time |
| Base vs LoRA | `with model.disable_adapter():` | Compare without loading a second model |
| Save | Copy only LoRA tensors to the CPU, then `save_pretrained`. Optional `merge_and_unload()` / `push_to_hub` | The base weights never leave the TPU |

## Quick start

1. Click **Open in Colab** above.
2. **Runtime → Change runtime type → TPU** (v5e-1 or v6e-1).
3. *(Optional)* To use the official gated `meta-llama/Llama-3.2-1B-Instruct`, accept the license on Hugging Face and add your token as a Colab secret named **`HF_TOKEN`**. Without a token, the notebook uses the ungated copy `unsloth/Llama-3.2-1B-Instruct`, which has the same weights.
4. **Runtime → Run all**.

Without a TPU, the notebook runs a short CPU smoke test with `HuggingFaceTB/SmolLM2-135M-Instruct`.

### Main settings

```python
MAX_LEN, BATCH_SIZE, MAX_STEPS = 256, 8, 400
ANSWER_LEN = 64                   # max supervised tokens per example (~99% of the dataset)
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LR, WARMUP_STEPS = 2e-4, 20
GRADIENT_CHECKPOINTING = False    # True if you raise BATCH_SIZE / MAX_LEN and hit HBM OOM
USE_TORCH_COMPILE = True          # False -> eager (defer-and-fuse)
```

Peak memory at these settings is 15.74 of 16 GiB on a v5e. To use a larger batch or longer sequences, turn on `GRADIENT_CHECKPOINTING`.

### Use the adapter

```python
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", dtype=torch.bfloat16).to("tpu")
model = PeftModel.from_pretrained(base, "llama32-1b-sql-lora").to("tpu")
merged = model.merge_and_unload()   # optional: plain transformers model, no PEFT needed
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `AttributeError: '_TpuDeviceProperties' object has no attribute ...` | `torch.tpu.get_device_properties()` is experimental and its fields differ between TorchTPU builds. The notebook reads them defensively |
| `CompileTimeHbmOom` / `RuntimeBufferAllocationFailure` | Lower `BATCH_SIZE` or `MAX_LEN`, or set `GRADIENT_CHECKPOINTING = True` |
| `torch.compile failed, falling back to eager` | The notebook frees memory and continues in eager mode. Set `TORCH_TPU_LAUNCH_BLOCKING=1` (before `import torch`) to debug op by op |
| `HF_TOKEN` secret timed out | Secrets can only be read when running from the Colab UI. Without a token, the notebook uses the ungated model copy |
| Slow first step | That's XLA compilation. Set `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT` to a directory to cache compiled programs across runs |

## Next steps

- **Sequence packing:** about 55% of tokens are padding (real 6.8k vs padded 15.5k tokens/s). Packing several examples per row would roughly double useful throughput.
- **Bounded dynamism** (`torch._dynamo.mark_dynamic(x, 1, min=..., max=...)`) for variable-length data without a recompile per shape.
- **Multi-chip:** TorchTPU supports `torch.distributed` (`backend="tpu_dist"`), DDP, FSDP2 / SimpleFSDP and DTensor.
- **Better SQL evaluation:** execution accuracy (running the queries) instead of exact string match.

## References

- [TorchTPU](https://github.com/google-pytorch/torch_tpu) and its [documentation](https://torchtpu.readthedocs.io/)
- Hu et al., [*LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685), 2021
- [Hugging Face PEFT](https://github.com/huggingface/peft)
- [Llama 3.2 model card](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context)

## License

Code: [Apache 2.0](LICENSE). Llama 3.2 weights are under the [Llama 3.2 Community License](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/blob/main/LICENSE.txt). The dataset has its own license terms.
