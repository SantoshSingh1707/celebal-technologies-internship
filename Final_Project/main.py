"""CLI for the mini GPT-2 from scratch project.

Commands:
  prepare   download / show corpus stats and train the BPE tokenizer
  train     train the mini model on a corpus (english or hindi)
  generate  sample text from a trained checkpoint
  chat      interactive continuation of a prompt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from gpt2.config import DataConfig, ModelConfig, TrainConfig
from gpt2.data import build_corpus, encode_dataset
from gpt2.generate import generate
from gpt2.model import GPT
from gpt2.train import load_checkpoint, train

if hasattr(sys.stdout, "reconfigure"):   # Windows cp1252 console can't print UTF-8
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SPECIAL_TOKENS = ("<|endoftext|>",)
BASE_DIR = Path(__file__).resolve().parent


def _model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--corpus", default="tinyshakespeare",
                   choices=["tinyshakespeare", "hindi_discourse"],
                   help="corpus name in data/ (default: %(default)s)")
    p.add_argument("--vocab-size", type=int, default=768)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--gqa-num-kv-heads", type=int, default=0,
                   help=">0 enables grouped-query attention (bonus)")
    p.add_argument("--seed", type=int, default=42)


def _train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--max-iters", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-iters", type=int, default=200)
    p.add_argument("--lr-decay-iters", type=int, default=5000)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-iters", type=int, default=50)
    p.add_argument("--checkpoint-dir", default="models")


def _args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Train a mini GPT-2 style language model from scratch in TensorFlow.",
    )
    sub = p.add_subparsers(dest="command")

    prep = sub.add_parser("prepare", help="train BPE tokenizer + show corpus stats")
    _model_args(prep)

    tr = sub.add_parser("train", help="train the model on a corpus")
    _model_args(tr)
    _train_args(tr)

    gen = sub.add_parser("generate", help="sample text from a checkpoint")
    _model_args(gen)
    gen.add_argument("--checkpoint", default="models/final.weights.h5")
    gen.add_argument("--prompt", default="")
    gen.add_argument("--max-new-tokens", type=int, default=200)
    gen.add_argument("--temperature", type=float, default=0.8)
    gen.add_argument("--top-k", type=int, default=200)

    chat = sub.add_parser("chat", help="interactive continuation session")
    _model_args(chat)
    chat.add_argument("--checkpoint", default="models/final.weights.h5")
    chat.add_argument("--temperature", type=float, default=0.8)
    chat.add_argument("--top-k", type=int, default=200)
    return p


def _build_and_prepare(args) -> tuple[ModelConfig, TrainConfig, DataConfig]:
    mcfg = ModelConfig(
        vocab_size=args.vocab_size, block_size=args.block_size,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        dropout=args.dropout, gqa_num_kv_heads=args.gqa_num_kv_heads,
    )
    dcfg = DataConfig(corpus=args.corpus, data_dir=str(BASE_DIR / "data"))
    tcfg = TrainConfig(seed=args.seed)
    return mcfg, tcfg, dcfg


def cmd_prepare(args) -> None:
    mcfg, tcfg, dcfg = _build_and_prepare(args)
    corpus = build_corpus(dcfg, mcfg.vocab_size, SPECIAL_TOKENS)
    print(f"[prepare] corpus  : {dcfg.corpus}.txt  ({corpus.n_tokens:,} tokens)")
    print(f"[prepare] vocab   : {mcfg.vocab_size} BPE tokens trained from scratch")
    print(f"[prepare] config  : {mcfg.describe()}")
    samples = [1000, 3000, 5000]
    for s in samples:
        enc = corpus.ids[s : s + 40]
        print(f"[prepare] sample @{s}: {corpus.tokenizer.decode(enc)!r}")


def cmd_train(args) -> None:
    mcfg, tcfg, dcfg = _build_and_prepare(args)
    for k in ("max_iters", "batch_size", "learning_rate", "weight_decay",
              "warmup_iters", "lr_decay_iters", "eval_interval", "eval_iters"):
        setattr(tcfg, k, getattr(args, k))

    print(f"[train] {mcfg.describe()}")
    corpus = build_corpus(dcfg, mcfg.vocab_size, SPECIAL_TOKENS)
    train_data, val_data = encode_dataset(corpus.ids, dcfg.split,
                                          mcfg.block_size, seed=dcfg.seed)
    print(f"[train] dataset  : {corpus.n_tokens:,} tokens "
          f"| train {train_data.shape} | val {val_data.shape}")

    model = GPT(mcfg)
    print(f"[train] parameters: {model.num_params():,}")
    train(model, train_data, val_data, tcfg, tokenizer=corpus.tokenizer,
          checkpoint_dir=args.checkpoint_dir)


def cmd_generate(args) -> None:
    mcfg, tcfg, dcfg = _build_and_prepare(args)
    corpus = build_corpus(dcfg, mcfg.vocab_size, SPECIAL_TOKENS)
    model = GPT(mcfg)
    load_checkpoint(model, args.checkpoint)
    if not args.prompt:
        args.prompt = "To be, or not to be" if args.corpus == "tinyshakespeare" \
            else "जीवन "
    out = generate(model, corpus.tokenizer, args.prompt,
                   max_new_tokens=args.max_new_tokens,
                   temperature=args.temperature, top_k=args.top_k, seed=args.seed)
    print("\n" + "=" * 70)
    print(out)
    print("=" * 70)


def cmd_chat(args) -> None:
    mcfg, tcfg, dcfg = _build_and_prepare(args)
    corpus = build_corpus(dcfg, mcfg.vocab_size, SPECIAL_TOKENS)
    model = GPT(mcfg)
    load_checkpoint(model, args.checkpoint)
    print("Interactive continuation (Ctrl+C or 'quit' to exit)\n")
    while True:
        try:
            prompt = input("\nPrompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not prompt:
            continue
        if prompt.lower() in {"quit", "exit", "q"}:
            break
        text = generate(model, corpus.tokenizer, prompt, max_new_tokens=150,
                        temperature=args.temperature, top_k=args.top_k, seed=args.seed)
        print(f"\nGPT-mini: {text}")


def main() -> None:
    args = _args().parse_args()
    if args.command is None:
        _args().print_help()
        print("\nQuick start:\n"
              "  python main.py prepare               # train BPE tokenizer + stats\n"
              "  python main.py train                 # train the model\n"
              "  python main.py generate              # sample text from a checkpoint\n"
              "  python main.py chat                  # interactive continuation")
        return
    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "chat":
        cmd_chat(args)


if __name__ == "__main__":
    main()
