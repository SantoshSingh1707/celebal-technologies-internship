# Mini GPT-2 Style Language Model from Scratch (TensorFlow)

A decoder-only transformer (GPT-2 style) — tokenizer, model, training loop, and
sampling — implemented entirely from scratch in TensorFlow. It trains on
public-domain corpora (English tiny shakespeare + a Hindi discourse corpus) and
generates coherent text entirely on a laptop CPU with ~1M parameters.

Inspired by Andrej Karpathy's educational material on building GPT from first
principles.

## Features

- **BPE tokenizer from scratch** — byte-level Byte-Pair Encoding trained on the
  corpus itself, so the same tokenizer works for English and Hindi (and any
  other language) without changes.
- **Decoder-only transformer** — token + learned position embeddings, N
  transformer blocks of causal multi-head self-attention and a GELU MLP with
  pre-norm LayerNorm and residual connections, and a weight-tied output head.
- **Training loop** — AdamW with weight decay, linear warmup + cosine LR decay,
  gradient clipping, periodic train/val evaluation, and checkpointing.
- **Autoregressive generation** — temperature + top-k sampling for text
  generation, plus an interactive chat mode.
- **Bonus work** — non-English (Hindi) corpus, a Grouped-Query Attention (GQA)
  flag, and a config scaffold for scaling up to the ~124M GPT-2 scale.

## Repository structure

```
Final_Project/
├── main.py                 # CLI: prepare / train / generate / chat
├── gpt2/
│   ├── config.py           # ModelConfig, TrainConfig, DataConfig dataclasses
│   ├── tokenizer.py        # BPE tokenizer from scratch (numpy-accelerated)
│   ├── data.py             # corpus loading + (x, y) batching
│   ├── model.py            # GPT: embeddings, blocks, causal attention, MLP
│   ├── train.py            # AdamW, LR schedule, eval, checkpointing
│   ├── generate.py         # temperature + top-k sampling
│   └── __init__.py         # package entry points
├── data/
│   ├── tinyshakespeare.txt # English corpus (public domain)
│   └── hindi_discourse.txt # Hindi discourse corpus (bonus)
├── tests/                  # test directory (add your own here)
├── gpt_dev.ipynb           # concept walk-through notebook (self-attention etc.)
├── FinalProject_Spec.txt   # project specification / write-up
└── requirements.txt
```

## Requirements

Python 3.10+.

```bash
pip install -r requirements.txt
```

Key dependencies: `tensorflow>=2.15`, `numpy>=1.24`, `matplotlib>=3.7`,
`tqdm>=4.60`, `regex>=2023.6`.

## Usage

Run all commands from the `Final_Project/` directory.

### 1. Prepare — train the BPE tokenizer and inspect the corpus

```bash
python main.py prepare --corpus tinyshakespeare
```

### 2. Train the model

```bash
python main.py train --max-iters 5000
```

Train on the Hindi corpus (bonus):

```bash
python main.py train --corpus hindi_discourse --max-iters 3000
```

Enable grouped-query attention (bonus) or scale up:

```bash
python main.py train --gqa-num-kv-heads 1
```

### 3. Generate text from a trained checkpoint

```bash
python main.py generate \
  --checkpoint models/final.weights.h5 \
  --prompt "To be, or not to be"
```

Sampling is controlled with `--temperature` (default 0.8) and `--top-k`
(default 200).

### 4. Interactive continuation

```bash
python main.py chat --checkpoint models/final.weights.h5
```

## Model configuration

The default "mini" config (~1M parameters) is a good CPU-training size:

| Parameter       | Value  | Default flag        |
|-----------------|--------|---------------------|
| vocab_size      | 768    | `--vocab-size`      |
| block_size      | 256    | `--block-size`      |
| n_layer         | 4      | `--n-layer`         |
| n_head          | 4      | `--n-head`          |
| n_embd          | 128    | `--n-embd`          |
| dropout         | 0.1    | `--dropout`         |
| gqa_num_kv_heads| 0      | `--gqa-num-kv-heads`|

`ModelConfig` (in `gpt2/config.py`) also carries a `bias` switch and the
structure needed to describe a full ~124M GPT-2 ("small") configuration for the
scaling-up bonus.

## How it works

1. **Tokenization** — the corpus is split into UTF-8 bytes; Byte-Pair Encoding
   merges the most frequent adjacent byte pairs until the target vocabulary
   size is reached. `encode()` / `decode()` round-trip text and token ids.
2. **Encoded dataset** — the corpus becomes a long integer tensor, split into
   train/validation (default 90/10), and sampled into batches of `(x, y)` pairs
   where `y` is `x` shifted one token right.
3. **Model** — token and position embeddings feed N pre-norm transformer blocks
   (causal multi-head attention → GELU MLP), then a final LayerNorm and a
   weight-tied projection to vocabulary logits.
4. **Training** — cross-entropy loss over all positions, AdamW with decoupled
   weight decay, cosine LR schedule with linear warmup, gradient clipping, and
   snapshot checkpoints whenever validation loss improves.
5. **Generation** — for each step the model predicts the next-token
   distribution; temperature and top-k refine it before sampling, and the new
   token is appended to the context.

## Training logs and checkpoints

Every `--eval-interval` steps (default 250) the loop prints:

```
step  250/5000 | loss 5.3214 | val 5.2031 | lr 8.75e-05 | 23s
```

Checkpoints are saved under the `--checkpoint-dir` directory (default `models/`):

- `best.weights.h5` + `best.weights.json` — lowest validation loss
- `final.weights.h5` + `final.weights.json` — end of training

The `.json` files store the model config, training hyperparameters, and
parameter count for reproducibility.

## Reproducibility

Seeding is controlled with `--seed` (default 42) on every subcommand; the
tokenizer training, dataset split, and training loop all follow it, so runs with
identical arguments reproduce the same checkpoints.

## Bonus work highlights

- **Non-English data** — train or generate on the Hindi corpus with
  `--corpus hindi_discourse`; the byte-level BPE handles it with no code changes.
- **Custom attention** — enable Grouped-Query Attention with
  `--gqa-num-kv-heads > 0` (shares K/V heads across query-head groups, as in
  Llama-2 / Mistral).
- **Scaling up** — `ModelConfig` can describe the full ~124M GPT-2
  configuration if you want to push past the mini default.