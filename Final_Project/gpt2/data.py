"""Dataset loading and batching for the mini GPT-2.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf

from .config import DataConfig
from .tokenizer import BPE


class Corpus:
    """Holds the raw text, the trained tokenizer, and the encoded arrays."""

    def __init__(self, text: str, tokenizer: BPE):
        self.text = text
        self.tokenizer = tokenizer
        self.ids: List[int] = tokenizer.encode(text)

    @property
    def n_tokens(self) -> int:
        return len(self.ids)


def load_text(data_dir: str, corpus_name: str) -> str:
    path = Path(data_dir) / f"{corpus_name}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Corpus not found: {path}. Put a text file there or prepare it first."
        )
    return path.read_text(encoding="utf-8")


def build_corpus(cfg: DataConfig, vocab_size: int, special_tokens: tuple = ()) -> Corpus:
    text = load_text(cfg.data_dir, cfg.corpus)
    tokenizer = BPE(vocab_size=vocab_size, special_tokens=special_tokens)
    tokenizer.train(text, verbose=False)
    return Corpus(text, tokenizer)


def encode_dataset(
    ids: List[int], split: float, block_size: int, seed: int = 42
) -> tuple[tf.Tensor, tf.Tensor]:
    """Split the token stream into train/val numpy arrays, then tf Tensors."""
    data = np.array(ids, dtype=np.int32)
    n = len(data)
    n_train = int(split * n)
    train_data = data[:n_train]
    val_data = data[n_train:]
    return tf.constant(train_data), tf.constant(val_data)


def get_batch(
    split_data: tf.Tensor, batch_size: int, block_size: int, seed: int = 0
) -> tuple[tf.Tensor, tf.Tensor]:
    """Random (x, y) batch. x is a (B, T) context, y is x shifted by one token."""
    rng = tf.random.Generator.from_seed(seed)
    n = tf.size(split_data).numpy()
    max_start = n - block_size - 1
    idxs = rng.uniform([batch_size], minval=0, maxval=max_start, dtype=tf.int32)
    x = tf.stack([split_data[i : i + block_size] for i in idxs.numpy()], axis=0)
    y = tf.stack([split_data[i + 1 : i + block_size + 1] for i in idxs.numpy()], axis=0)
    return x, y
