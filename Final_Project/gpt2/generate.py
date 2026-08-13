"""Autoregressive text generation: temperature + top-k sampling.
"""
from __future__ import annotations

import tensorflow as tf

from .config import ModelConfig
from .model import GPT
from .tokenizer import BPE


def generate(
    model: GPT,
    tokenizer: BPE,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 200,
    seed: int = 42,
) -> str:
    """Generate ``max_new_tokens`` tokens continuing from ``prompt``."""
    idx = tf.constant([tokenizer.encode(prompt)], dtype=tf.int32)
    tf.random.set_seed(seed)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size:]          # crop context
        logits = model.forward(idx_cond)                      # (1, T, vocab)
        logits = logits[:, -1, :] / temperature               # last token
        if top_k is not None:
            k = min(top_k, logits.shape[-1])
            top = tf.math.top_k(logits, k)
            logits = tf.where(
                logits < tf.reduce_min(top.values), -float("inf"), logits
            )
        probs = tf.nn.softmax(logits, axis=-1)
        next_id = tf.cast(tf.random.categorical(probs, num_samples=1), tf.int32)  # (1, 1)
        idx = tf.concat([idx, next_id], axis=1)

    return tokenizer.decode(idx.numpy().tolist()[0])
