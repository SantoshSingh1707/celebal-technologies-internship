"""Byte-Pair Encoding tokenizer from scratch (works on any language)."""
from __future__ import annotations

import json
import regex as re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def get_stats(ids: List[int], counts: Dict[Tuple[int, int], int] | None = None
              ) -> Dict[Tuple[int, int], int]:
    """Count consecutive pairs: [1,2,3,1,2] -> {(1,2):2, (2,3):1, (3,1):1}."""
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
    """Replace all consecutive occurrences of ``pair`` with token ``idx``."""
    new_ids: List[int] = []
    i = 0
    while i < len(ids):
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


def _numpy_pairs(stats: Dict[Tuple[int, int], int]) -> Tuple[int, int]:
    """Most frequent pair from a numpy-backed stats dict."""
    return max(stats, key=stats.get)


class BPE:
    """Byte-Pair Encoding tokenizer, trained on a raw text corpus."""

    GPT2_SPLIT_PATTERN = (
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

    def __init__(self, vocab_size: int = 768,
                 special_tokens: Tuple[str, ...] = ("<|endoftext|>",)):
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256")
        self.vocab_size = vocab_size
        self.special_tokens = list(special_tokens)
        self.pattern = self.GPT2_SPLIT_PATTERN
        self.compiled = re.compile(self.pattern)

        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # ------------------------------------------------------------------ #
    # training (numpy-accelerated, same BPE algorithm)
    # ------------------------------------------------------------------ #
    def train(self, text: str, verbose: bool = True) -> "BPE":
        """Learn merges on ``text`` until vocab reaches self.vocab_size."""
        num_merges = self.vocab_size - 256 - len(self.special_tokens)
        if num_merges < 0:
            raise ValueError("vocab_size must be >= 256 + number of special tokens")

        chunks = re.findall(self.compiled, text)
        ids, boundaries = self._flatten(chunks)

        merges: Dict[Tuple[int, int], int] = {}
        vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}

        for i in range(num_merges):
            stats = self._count_pairs(ids, boundaries)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids, boundaries = self._merge_pair(ids, pair, idx, boundaries)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose and (i % 500 == 0 or i == num_merges - 1):
                print(f"merge {i + 1:5d}/{num_merges}: {pair!r} -> {idx} "
                      f"({vocab[idx]!r}) x{stats[pair]}")

        self.merges = merges
        self.vocab = vocab
        self.special_ids = {
            sp: self.vocab_size - len(self.special_tokens) + j
            for j, sp in enumerate(self.special_tokens)
        }
        return self

    # ------------------------------------------------------------------ #
    # numpy helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _flatten(chunks: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Concatenate chunks into one int array + mask of chunk boundaries."""
        parts = [np.frombuffer(ch.encode("utf-8"), dtype=np.uint8).astype(np.int32)
                 for ch in chunks]
        if not parts:
            return np.array([], dtype=np.int32), np.array([], dtype=bool)
        ids = np.concatenate(parts)
        ends = np.cumsum([len(p) for p in parts]) - 1
        boundaries = np.zeros(len(ids), dtype=bool)
        boundaries[ends[:-1]] = True   # position is the last byte of a chunk
        return ids, boundaries

    @staticmethod
    def _count_pairs(ids: np.ndarray, boundaries: np.ndarray) -> Dict[Tuple[int, int], int]:
        """Count adjacent pairs, ignoring pairs that span chunk boundaries."""
        if len(ids) < 2:
            return {}
        a, b = ids[:-1], ids[1:]
        mask = ~boundaries[:-1]
        if not mask.any():
            return {}
        keys = a[mask].astype(np.int64) * (1 << 20) + b[mask]
        unique, counts = np.unique(keys, return_counts=True)
        return {(int(k >> 20), int(k & ((1 << 20) - 1))): int(c)
                for k, c in zip(unique, counts)}

    @staticmethod
    def _merge_pair(ids: np.ndarray, pair: Tuple[int, int], new_id: int,
                    boundaries: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Replace all occurrences of ``pair`` with ``new_id`` in one vectorized pass."""
        if len(ids) < 2:
            return ids, boundaries
        a, b = ids[:-1], ids[1:]
        match = (a == pair[0]) & (b == pair[1]) & (~boundaries[:-1])
        if not match.any():
            return ids, boundaries

        out = ids.copy()
        # new chunk ids: 0 = no change, 1 = start of a merged pair, 2 = consumed
        step = np.zeros(len(ids), dtype=np.int32)
        step[:-1][match] = 1
        step[1:][match] = 2

        new_bdry = boundaries.copy()
        new_bdry[:-1][match] = new_bdry[1:][match]  # chunk end follows the pair

        keep = np.zeros(len(ids), dtype=bool)
        keep[:-1][match] = True                    # keep pair-start positions
        keep |= (step != 2)                        # plus all non-consumed positions

        new_ids = np.zeros(int(keep.sum()), dtype=np.int32)
        new_ids[step[keep] == 0] = out[keep][step[keep] == 0]
        new_ids[step[keep] == 1] = new_id
        new_bdry = new_bdry[keep]
        return new_ids, new_bdry

    # ------------------------------------------------------------------ #
    # encode / decode
    # ------------------------------------------------------------------ #
    def _encode_chunk(self, text_bytes: bytes) -> List[int]:
        """Greedily apply merges (lowest index first) to a byte chunk."""
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text: str) -> List[int]:
        """Text -> list of token ids (special tokens preserved verbatim)."""
        if self.special_tokens:
            pattern = "(" + "|".join(re.escape(s) for s in self.special_tokens) + ")"
            parts = [p for p in re.split(pattern, text) if p]
        else:
            parts = [text]

        tokens: List[int] = []
        for part in parts:
            if part in self.special_ids:
                tokens.append(self.special_ids[part])
                continue
            chunks = re.findall(self.compiled, part)
            for chunk in chunks:
                tokens.extend(self._encode_chunk(chunk.encode("utf-8")))
        return tokens

    def decode(self, ids: List[int]) -> str:
        """List of token ids -> text."""
        parts = []
        for idx in ids:
            if idx in self.special_ids.values():
                parts.append([s for s, i in self.special_ids.items() if i == idx][0])
            else:
                parts.append(self.vocab[idx].decode("utf-8", errors="replace"))
        return "".join(parts)

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def save(self, file_path: str | Path) -> None:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "pattern": self.pattern,
            "merges": [[a, b] for (a, b) in self.merges],
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, file_path: str | Path) -> "BPE":
        with open(file_path, encoding="utf-8") as f:
            payload = json.load(f)
        tok = cls(vocab_size=payload["vocab_size"],
                  special_tokens=tuple(payload["special_tokens"]))
        tok.pattern = payload["pattern"]
        tok.compiled = re.compile(tok.pattern)
        merges = {}
        vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for a, b in payload["merges"]:
            idx = len(vocab)
            merges[(a, b)] = idx
            vocab[idx] = vocab[a] + vocab[b]
        tok.merges = merges
        tok.vocab = vocab
        tok.special_ids = {
            sp: tok.vocab_size - len(tok.special_tokens) + j
            for j, sp in enumerate(tok.special_tokens)
        }
        return tok
