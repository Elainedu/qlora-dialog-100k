# qlora-dialog-100k

QLoRA fine-tuning of BLOOM on a 100K Chinese dialogue dataset, with two Gradio
front-ends for the resulting model.

## Overview

This project applies QLoRA (4-bit quantisation + LoRA) to fine-tune a
BLOOM-based Chinese language model on ~100,000 Human/Assistant conversation
pairs. Two Jupyter notebooks cover training and LoRA-to-base merging, and two
Gradio applications are provided: one that loads the LoRA adapter on top of
the quantised base, and one that serves the fully merged checkpoint.

## Model / Approach

- Base model: `YeungNLP/bloom-1b1-zh` (Chinese-optimised BLOOM)
- Fine-tuning method: QLoRA
  - 4-bit NF4 quantisation via `bitsandbytes`
  - LoRA adapters via `peft`
  - Trainable parameter share: ~6-7% of the full model
- Dataset: BELLE Dialog 100K (YeungNLP redistribution)
  - Train split: `train_dataset_belle_100k_YeungNLP/data_train/` (~351 MB Arrow)
  - Val split:   `train_dataset_belle_100k_YeungNLP/data_val/` (~7 MB Arrow)
- Training: 1 epoch, best checkpoint at step 640
- Hardware tested: NVIDIA T4 (~10 GB VRAM), DGX-1
- Training time: ~2-3 hours on a single T4

### QLoRA configuration used

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
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_use_double_quant": True,
}
```

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

Additional notes:

- Python 3.10+
- CUDA 11.7+
- A CUDA GPU with **at least 10 GB VRAM** for QLoRA inference/training
  (T4 / V100 / A100)
- `bitsandbytes` is required for 4-bit loading and is only reliably supported
  on Linux (Windows users can still run the *merged* v2 demo without
  `bitsandbytes`)

Install:

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Elainedu/qlora-dialog-100k.git
cd qlora-dialog-100k

# 2. Install dependencies
pip install -r requirements.txt

# 3. Choose one of the demos below
```

### Run the QLoRA (adapter + 4-bit base) demo - v1

```bash
python gradio-app-v1-qlora-chat-custom-streaming.py
```

Loads `YeungNLP/bloom-1b1-zh` in 4-bit and attaches the LoRA adapter in
`my-lora-model-1epoch-YeungNLP/`.

### Run the merged-model demo - v2

```bash
python gradio-app-v2-merged-model-iterator-streaming.py
```

Loads the fully merged checkpoint in `my-lora-merged-model-steps640-YeungNLP/`
via `AutoModelForCausalLM` (uses `TextIteratorStreamer` for streaming output).
No `bitsandbytes` required.

### Simple non-Gradio smoke test

```bash
python demo.py
```

## Training / Merging Workflow

```bash
# 1. Fine-tune with QLoRA
jupyter notebook 10-10-qlora-finetune-bloomz-YeungNLP.ipynb

# 2. Merge the LoRA weights back into the base model
jupyter notebook 10-15-merge-lora-weights-to-base-model.ipynb
```

The merge step writes the standalone checkpoint that
`gradio-app-v2-merged-model-iterator-streaming.py` consumes.

## Project Structure

```
qlora-dialog-100k/
├── 10-10-qlora-finetune-bloomz-YeungNLP.ipynb          # QLoRA training notebook
├── 10-15-merge-lora-weights-to-base-model.ipynb        # LoRA + base merge notebook
├── demo.py                                             # CLI smoke test
├── gradio-app-v1-qlora-chat-custom-streaming.py        # v1: 4-bit base + LoRA adapter
├── gradio-app-v2-merged-model-iterator-streaming.py    # v2: merged model, no bitsandbytes
├── my-lora-model-1epoch-YeungNLP/                      # LoRA adapter (~97 MB) *
│   ├── adapter_model.bin
│   └── adapter_config.json
├── my-lora-merged-model-steps640-YeungNLP/             # Merged full model (~667 MB) *
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── generation_config.json
│   └── tokenizer.json
├── train_dataset_belle_100k_YeungNLP/                  # Arrow-format dataset *
│   ├── data_train/
│   └── data_val/
├── configs/ data/ models/ notebooks/ src/
├── requirements.txt
└── README.md
```

`*` = large artefact, not necessarily present in a fresh clone; see Notes.

## Gradio UI

Both apps share the same feature set:

- Streaming output (custom queue-based streamer in v1,
  `TextIteratorStreamer` in v2)
- Multi-turn conversation history
- Sliders for `temperature`, `top_p`, `top_k`, `repetition_penalty`
- Preset example prompts
- OpenCC-based Simplified <-> Traditional conversion

## Memory Footprint

| Method                    | VRAM required | Training speed |
| ------------------------- | ------------- | -------------- |
| Full fine-tuning          | ~40 GB        | Fast           |
| **QLoRA (this repo)**     | ~10 GB        | Medium         |
| LoRA (8-bit)              | ~15 GB        | Faster         |

## Notes

- **Weights and datasets are typically not stored in git** due to size.
  If any of the following folders are missing after cloning, obtain them as
  described:

  - `my-lora-model-1epoch-YeungNLP/` (~97 MB) - reproduce by running
    `10-10-qlora-finetune-bloomz-YeungNLP.ipynb`.
  - `my-lora-merged-model-steps640-YeungNLP/` (~667 MB) - reproduce by
    running the merge notebook `10-15-merge-lora-weights-to-base-model.ipynb`
    after training.
  - `train_dataset_belle_100k_YeungNLP/` - download the BELLE dialog dataset
    from HuggingFace (e.g. [`YeungNLP/firefly-train-1.1M`](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)
    or the original [`BELLE 100K`](https://huggingface.co/datasets/BelleGroup/train_1M_CN))
    and tokenise it with the training notebook.
  - `YeungNLP/bloom-1b1-zh` base model is pulled on demand by
    `transformers`/`peft`; no manual download needed if you have internet.

- **Model paths** in the two Gradio scripts (`model_name_or_path`) are
  relative to the repo root. Adjust them if you move the checkpoints.
- **`bitsandbytes` on Windows** is fragile. If the v1 (QLoRA) demo fails to
  import `bitsandbytes`, run the v2 (merged) demo instead - it does not
  require 4-bit loading.
- **CPU inference** is possible for the merged model but very slow; a
  CUDA-capable GPU is strongly recommended.

## References

- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [PEFT documentation](https://huggingface.co/docs/peft)
- [YeungNLP model collection](https://huggingface.co/YeungNLP)

## License

Educational use only. Base model weights and dataset follow their upstream
licenses (BLOOM RAIL License, BELLE dataset terms). Demo code and notebooks
in this repository may be freely used, modified, and redistributed for
teaching and research.
