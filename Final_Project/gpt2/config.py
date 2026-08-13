"""Configuration dataclasses for the mini GPT-2 style model.

ModelConfig mirrors nanoGPT's Config and lets us define everything from a
~1M parameter "mini" model up to the full 127M GPT-2 configuration.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int = 768          # BPE vocabulary size (trained on corpus)
    block_size: int = 256          # max context length (sequence window)
    n_layer: int = 4               # number of transformer blocks
    n_head: int = 4                # number of attention heads
    n_embd: int = 128              # embedding / hidden dimension
    dropout: float = 0.1           # dropout probability (0 = off)
    bias: bool = False             # layer norms / projections use bias? (GPT-2 uses bias)
    gqa_num_kv_heads: int = 0      # >0 enables Grouped-Query Attention (bonus)

    def __post_init__(self) -> None:
        if self.gqa_num_kv_heads > 0 and self.n_head % self.gqa_num_kv_heads != 0:
            raise ValueError("n_head must be divisible by gqa_num_kv_heads")

    @property
    def n_kv_heads(self) -> int:
        """Key/value heads: one per head (MHA) or grouped (GQA)."""
        return self.gqa_num_kv_heads or self.n_head

    def param_count(self) -> int:
        """Rough parameter estimate for the decoder-only stack."""
        n = self.vocab_size * self.n_embd                    # token embedding
        n += self.block_size * self.n_embd                   # position embedding
        for _ in range(self.n_layer):
            n += 3 * self.n_embd * self.n_embd               # q, k, v projections
            n += self.n_embd * self.n_embd                   # attention out projection
            n += 4 * self.n_embd * self.n_embd               # MLP in
            n += 4 * self.n_embd * self.n_embd               # MLP out
        return n

    def describe(self) -> str:
        return (
            f"GPT-mini  params~{self.param_count()/1e6:.2f}M  "
            f"layers={self.n_layer} heads={self.n_head} "
            f"embd={self.n_embd} block={self.block_size} vocab={self.vocab_size}"
        )


@dataclass
class TrainConfig:
    max_iters: int = 5000         # total training steps
    batch_size: int = 8           # sequences per batch
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_iters: int = 200       # linear warmup steps
    lr_decay_iters: int = 5000    # cosine decay over this many steps (0 = constant)
    eval_interval: int = 250      # steps between validation
    eval_iters: int = 50          # batches averaged per evaluation
    seed: int = 1337


@dataclass
class DataConfig:
    corpus: str = "tinyshakespeare"      # selects data/<name>.txt
    split: float = 0.9                   # train/validation split ratio
    data_dir: str = "data"
    seed: int = 42
