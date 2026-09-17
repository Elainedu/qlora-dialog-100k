# qlora-dialog-100k

**QLoRA (4-bit NF4 + LoRA) fine-tuning of BLOOM** on the BELLE 100K
Chinese dialogue corpus, with two Gradio front-ends for the resulting
adapter and merged model.

---

## Table of Contents

- [What This Does](#what-this-does)
- [Model & Dataset](#model--dataset)
- [Training Setup](#training-setup)
- [System Architecture](#system-architecture)
- [Repository Layout](#repository-layout)
- [Key Files](#key-files)
- [Requirements](#requirements)
- [Configuration](#configuration)
- [Reproducing Training](#reproducing-training)
- [Running Inference](#running-inference)
- [Notes](#notes)
- [Files Not in Repo](#files-not-in-repo)
- [References](#references)
- [License](#license)

---

## What This Does

This project applies **QLoRA** (Dettmers et al. 2023) to fine-tune a
Chinese-optimised BLOOM model (`YeungNLP/bloom-1b1-zh`) on ~100 000
Human/Assistant conversation pairs.

QLoRA quantises the frozen base to 4-bit **NF4** via `bitsandbytes`, then
inserts **LoRA** low-rank adapters (r = 8) into the attention projection
(`query_key_value`) — so only ~6–7 % of the model's parameters are
trainable and the whole training loop fits into ~10 GB of VRAM.

Two Gradio front-ends are provided:

- **v1 (adapter mode)** — loads the 4-bit base + LoRA adapter on top,
  needs `bitsandbytes`.
- **v2 (merged mode)** — loads a pre-merged full-precision checkpoint,
  no `bitsandbytes` required (Windows-friendly).

---

## Model & Dataset

**Base model**

| Field    | Value                    |
| -------- | ------------------------ |
| Model id | `YeungNLP/bloom-1b1-zh`  |
| Family   | BLOOM, Chinese-optimised |
| Params   | ~1.1 B                   |

**Dataset**

| Field       | Value                                                    |
| ----------- | -------------------------------------------------------- |
| Source      | BELLE Dialog 100K (YeungNLP redistribution)              |
| Train dir   | `train_dataset/data_train/` (~351 MB Arrow)              |
| Val dir     | `train_dataset/data_val/`   (~7 MB Arrow)                |
| Sample count| ~100 000 dialogue pairs                                  |
| Format      | Human/Assistant multi-turn, tokenised by BloomTokenizerFast |

**Prompt template used in the demos:**

```
<Human>:<user turn 1>

<Assistant>:<bot turn 1>

<Human>:<user turn 2>

<Assistant>:
```

---

## Training Setup

Method: **QLoRA** (4-bit NF4 base + trainable LoRA adapter).

### QLoRA configuration

```python
lora_config = {
    "r": 8,
    "lora_alpha": 32,
    "target_modules": ["query_key_value"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

bnb_config = {
    "load_in_4bit":            True,
    "bnb_4bit_quant_type":     "nf4",
    "bnb_4bit_compute_dtype":  "float16",
    "bnb_4bit_use_double_quant": True,
}
```

### Training arguments

| Setting              | Value                                    |
| -------------------- | ---------------------------------------- |
| Epochs               | 1                                        |
| Best-step checkpoint | step 640                                 |
| Trainable params     | ~25 M (~6.7 % of 396 M-scale model)      |
| Compute dtype        | fp16                                     |
| GPU tested           | NVIDIA T4 (~10 GB), also DGX-1           |
| Wall time            | ~2–3 h on a single T4                    |
| Final loss           | ~0.77                                    |

VRAM comparison:

| Method              | VRAM required | Training speed |
| ------------------- | ------------- | -------------- |
| Full fine-tuning    | ~40 GB        | Fast           |
| **QLoRA (this)**    | ~10 GB        | Medium         |
| LoRA (8-bit)        | ~15 GB        | Faster         |

---

## System Architecture

```
                ┌────────────────────────────────┐
                │ BELLE 100K dialogue corpus     │
                │ (YeungNLP redistribution)      │
                └───────────────┬────────────────┘
                                │  BloomTokenizerFast
                                ▼
   ┌─────────────────────────────────────────────────────┐
   │  train_dataset/                                     │
   │    ├─ data_train/  (~351 MB Arrow)                  │
   │    └─ data_val/    (~7 MB Arrow)                    │
   └───────────────┬─────────────────────────────────────┘
                   │
                   ▼
   ┌─────────────────────────────────────────────────────┐
   │  Notebook 01: QLoRA fine-tune                       │
   │  - YeungNLP/bloom-1b1-zh loaded 4-bit (NF4)         │
   │  - LoRA on query_key_value (r=8, alpha=32)          │
   │  - 1 epoch, fp16 compute, ~10 GB VRAM               │
   └───────────────┬─────────────────────────────────────┘
                   │  best step 640
                   ▼
   ┌─────────────────────────────────────────────────────┐
   │  my-lora-model-1epoch-YeungNLP/                     │
   │    adapter_model.bin  +  adapter_config.json        │
   └───────┬─────────────────────────────────────────────┘
           │
           │  Notebook 02: merge_and_unload()
           ▼
   ┌─────────────────────────────────────────────────────┐
   │  my-lora-merged-model-steps640-YeungNLP/            │
   │    pytorch_model.bin  +  config / tokenizer         │
   └───────┬─────────────────────────────────────────────┘
           │
           ▼
   ┌─────────────────────────────────────────────────────┐
   │  Serve (Gradio)                                     │
   │   v1  app_qlora_chat.py   4-bit base + adapter      │
   │   v2  app_merged_chat.py  merged model, no bnb      │
   │   CLI demo.py             one-shot smoke test       │
   └─────────────────────────────────────────────────────┘
```

---

## Repository Layout

```
qlora-dialog-100k/
├── app_qlora_chat.py                       # v1: 4-bit base + LoRA (needs bitsandbytes)
├── app_merged_chat.py                      # v2: merged model, streaming
├── demo.py                                 # CLI smoke test (also documents QLoRA design)
├── requirements.txt
├── README.md
├── notebooks/
│   ├── 01-qlora-finetune-bloomz.ipynb      # QLoRA training
│   └── 02-merge-lora-weights.ipynb         # LoRA -> base merge
├── train_dataset/
│   ├── data_train/                         # Arrow train split (~351 MB, gitignored)
│   └── data_val/                           # Arrow val split (~7 MB, gitignored)
├── my-lora-model-1epoch-YeungNLP/          # LoRA adapter (~97 MB, gitignored)
└── my-lora-merged-model-steps640-YeungNLP/ # Merged full model (~667 MB, gitignored)
```

---

## Key Files

| File | Purpose |
| ---- | ------- |
| `app_qlora_chat.py` | **v1 Gradio UI.** Loads `YeungNLP/bloom-1b1-zh` in 4-bit via `BitsAndBytesConfig(nf4, double-quant, fp16 compute)`, attaches the LoRA adapter under `my-lora-model-1epoch-YeungNLP/` with `PeftModel.from_pretrained`, then `merge_and_unload()`. Custom queue-based streamer, `StoppingCriteria` on EOS, OpenCC s2t/t2s conversion. |
| `app_merged_chat.py` | **v2 Gradio UI.** Loads the merged checkpoint under `my-lora-merged-model-steps640-YeungNLP/` via plain `AutoModelForCausalLM` — no `bitsandbytes`. Uses `TextIteratorStreamer` + background `Thread` for token streaming. |
| `demo.py` | Documentation-heavy CLI script explaining QLoRA/LoRA design; runs a minimal generation loop for smoke-testing. |
| `notebooks/01-qlora-finetune-bloomz.ipynb` | QLoRA training notebook — dataset loading, tokenisation, `BitsAndBytesConfig`, `LoraConfig`, `Trainer` loop; writes the adapter to `my-lora-model-1epoch-YeungNLP/`. |
| `notebooks/02-merge-lora-weights.ipynb` | Loads the base + adapter, calls `merge_and_unload()`, writes the merged checkpoint to `my-lora-merged-model-steps640-YeungNLP/` so v2 can be served without `bitsandbytes`. |
| `train_dataset/` | Arrow-format train/val splits of BELLE 100K, produced by the tokenisation cells of notebook 01. |
| `requirements.txt` | `torch`, `transformers`, `datasets`, `peft`, `accelerate`, `bitsandbytes`, `gradio`, `pandas`, `numpy`, `tqdm`, `pyyaml`. |

---

## Requirements

From `requirements.txt`:

```
torch>=2.0.0
transformers>=4.32.0
datasets>=2.14.0
peft>=0.5.0
accelerate>=0.23.0
bitsandbytes>=0.41.1
gradio>=4.0.0
pandas>=2.0.0
numpy>=1.24.0
tqdm>=4.65.0
pyyaml>=6.0
```

Hardware / OS:

- Python 3.10+, CUDA 11.7+
- **At least 10 GB VRAM** for QLoRA training/inference (T4 / V100 / A100).
- `bitsandbytes` (4-bit) is only reliably supported on **Linux**. Windows
  users should use the **v2 merged** demo instead.
- OpenCC (`opencc-python-reimplemented`) for Simplified/Traditional
  conversion in both UIs.

Install:

```bash
pip install -r requirements.txt
```

---

## Configuration

There are no environment variables. Model / adapter paths are set inline
in each launcher (edit these if you moved the folders):

```python
# app_qlora_chat.py
base_model    = "YeungNLP/bloom-1b1-zh"
adapter_path  = "my-lora-model-1epoch-YeungNLP"

# app_merged_chat.py
model_name_or_path = "my-lora-merged-model-steps640-YeungNLP"
```

Runtime sliders in the Gradio UI expose `temperature`, `top_p`, `top_k`,
`repetition_penalty`, and `max_new_tokens`.

---

## Reproducing Training

```bash
# 1. Clone
git clone https://github.com/Elainedu/qlora-dialog-100k.git
cd qlora-dialog-100k

# 2. Install deps (Linux recommended for bitsandbytes)
pip install -r requirements.txt

# 3. Prepare the dataset — download BELLE 100K, then run the tokenisation
#    cells of notebook 01 to write train_dataset/data_train + data_val.

# 4. Fine-tune with QLoRA
jupyter notebook notebooks/01-qlora-finetune-bloomz.ipynb
#    -> writes  my-lora-model-1epoch-YeungNLP/  (adapter only, ~97 MB)

# 5. Merge the LoRA adapter into the base model
jupyter notebook notebooks/02-merge-lora-weights.ipynb
#    -> writes  my-lora-merged-model-steps640-YeungNLP/  (~667 MB)
```

After step 4 you can already run **v1**; after step 5 you can run **v2**
without `bitsandbytes`.

---

## Running Inference

### v1 — QLoRA (adapter + 4-bit base)

```bash
python app_qlora_chat.py
```

Loads `YeungNLP/bloom-1b1-zh` in 4-bit and attaches the LoRA adapter.
Needs `bitsandbytes` + CUDA.

### v2 — Merged model

```bash
python app_merged_chat.py
```

Loads the merged checkpoint via plain `AutoModelForCausalLM`. Uses
`TextIteratorStreamer` for streaming output. Works on Windows /
CPU-only.

### CLI smoke test

```bash
python demo.py
```

Both Gradio apps default to `http://localhost:7860` and expose:

- Multi-turn conversation history
- Sliders for `temperature`, `top_p`, `top_k`, `repetition_penalty`,
  `max_new_tokens`
- Preset example prompts
- Streaming output (queue-based streamer in v1, `TextIteratorStreamer`
  in v2)
- OpenCC-based Simplified ↔ Traditional conversion

---

## Notes

- **`bitsandbytes` on Windows is fragile.** If v1 fails to import
  `bitsandbytes`, run v2 (merged) instead.
- **CPU inference** is possible for the merged model but very slow — a
  CUDA-capable GPU is strongly recommended.
- **Model paths** in the Gradio scripts (`model_name_or_path`,
  `adapter_path`) are relative to the repo root; adjust them if the
  checkpoints live elsewhere.
- **BLOOM padding side.** BLOOM tokenizers use left-side padding for
  causal-LM training — do not flip it to right.
- **Adapter vs merged trade-off.** Adapter mode (v1) is smaller and lets
  you hot-swap adapters; merged mode (v2) is a single self-contained
  checkpoint that ships without `bitsandbytes`.

---

## Files Not in Repo

Weights and dataset shards are excluded from git due to size. If any of
these folders are missing after cloning, obtain them as follows:

| Excluded                                     | Size    | How to obtain / regenerate |
| -------------------------------------------- | ------- | -------------------------- |
| `my-lora-model-1epoch-YeungNLP/`             | ~97 MB  | Run `notebooks/01-qlora-finetune-bloomz.ipynb`. |
| `my-lora-merged-model-steps640-YeungNLP/`    | ~667 MB | Run `notebooks/02-merge-lora-weights.ipynb` (after 01). |
| `train_dataset/data_train/` + `data_val/`    | ~358 MB | Download BELLE 100K from Hugging Face and tokenise via notebook 01. See <https://huggingface.co/datasets/BelleGroup/train_1M_CN> and <https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M>. |
| `YeungNLP/bloom-1b1-zh` base weights         | ~2.2 GB | Pulled automatically by `transformers` / `peft` on first run. |

---

## References

- Dettmers et al. **QLoRA: Efficient Finetuning of Quantized LLMs.**
  arXiv:2305.14314. <https://arxiv.org/abs/2305.14314>
- Hu et al. **LoRA: Low-Rank Adaptation of Large Language Models.**
  arXiv:2106.09685. <https://arxiv.org/abs/2106.09685>
- Le Scao et al. **BLOOM: A 176B-Parameter Open-Access Multilingual
  Language Model.** arXiv:2211.05100. <https://arxiv.org/abs/2211.05100>
- Hugging Face **PEFT**: <https://huggingface.co/docs/peft>
- **bitsandbytes**: <https://github.com/TimDettmers/bitsandbytes>
- Base model: <https://huggingface.co/YeungNLP/bloom-1b1-zh>
- BELLE dataset: <https://huggingface.co/BelleGroup>

---

## License

Educational use only. Base model weights and dataset follow their upstream
licenses (BLOOM RAIL License, BELLE dataset terms). Demo code and
notebooks in this repository may be freely used, modified, and
redistributed for teaching and research.
