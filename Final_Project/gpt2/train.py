"""Training loop: AdamW, LR schedule, validation, checkpointing."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import tensorflow as tf

from .config import ModelConfig, TrainConfig
from .model import GPT
from .tokenizer import BPE


def lr_schedule(step: int, cfg: TrainConfig) -> float:
    """Linear warmup then cosine decay to near-zero (nanoGPT-style)."""
    if cfg.lr_decay_iters <= 0:
        return cfg.learning_rate
    if step < cfg.warmup_iters:
        return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_iters)
    progress = (step - cfg.warmup_iters) / max(1, cfg.lr_decay_iters - cfg.warmup_iters)
    progress = min(1.0, max(0.0, progress))
    coeff = 0.5 * (1 + np.cos(np.pi * progress))
    return cfg.learning_rate * coeff


def build_optimizer(cfg: TrainConfig):
    """AdamW with decoupled weight decay."""
    opt = tf.keras.optimizers.AdamW(
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        beta_1=cfg.beta1,
        beta_2=cfg.beta2,
    )
    return opt


def train(
    model: GPT,
    train_data: tf.Tensor,
    val_data: tf.Tensor,
    cfg: TrainConfig,
    tokenizer: Optional[BPE] = None,
    checkpoint_dir: str | Path = "models",
    on_step: Optional[Callable[[int, dict], None]] = None,
    use_tf_function: bool = True,
) -> dict:
    """Run the full training loop. Returns the logged metrics."""
    opt = build_optimizer(cfg)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log = {"iter": [], "loss": [], "val_loss": [], "lr": [], "time_s": []}
    tf.random.set_seed(cfg.seed)
    best_val = float("inf")
    start = time.time()

    def step_fn(x, y, step):
        with tf.GradientTape() as tape:
            loss = model.loss(x, y)
        grads = tape.gradient(loss, model.variables())
        grads, _ = tf.clip_by_global_norm(grads, cfg.grad_clip)
        opt.lr.assign(lr_schedule(int(step), cfg))
        opt.apply_gradients(zip(grads, model.variables()))
        return loss

    if use_tf_function:
        step_fn = tf.function(step_fn)

    from .data import get_batch

    for step in range(cfg.max_iters):
        x, y = get_batch(train_data, cfg.batch_size, model.config.block_size,
                         seed=cfg.seed + step)
        loss = step_fn(x, y, step)

        if step % cfg.eval_interval == 0 or step == cfg.max_iters - 1:
            est = model.estimate_loss(train_data, val_data, cfg.batch_size,
                                      cfg.eval_iters, seed=cfg.seed + step)
            elapsed = time.time() - start
            log["iter"].append(step)
            log["loss"].append(float(loss.numpy()))
            log["val_loss"].append(est["val"])
            log["lr"].append(float(opt.lr.numpy()))
            log["time_s"].append(elapsed)
            print(f"step {step:5d}/{cfg.max_iters} | loss {log['loss'][-1]:.4f} | "
                  f"val {est['val']:.4f} | lr {log['lr'][-1]:.2e} | {elapsed:.0f}s")
            if on_step is not None:
                on_step(step, {"loss": log["loss"][-1], "val_loss": est["val"],
                               "lr": log["lr"][-1]})

            if est["val"] < best_val:
                best_val = est["val"]
                _save_checkpoint(model, tokenizer, cfg, checkpoint_dir / "best.weights.h5",
                                 "best")

    _save_checkpoint(model, tokenizer, cfg, checkpoint_dir / "final.weights.h5", "final")
    _save_metrics(log, checkpoint_dir / "metrics.json")
    return log


def _save_checkpoint(model: GPT, tokenizer: Optional[BPE], cfg: TrainConfig,
                     path: Path, tag: str) -> None:
    weights = {v.name: v.numpy() for v in model.variables()}
    meta = {
        "tag": tag,
        "config": {
            "vocab_size": model.config.vocab_size,
            "block_size": model.config.block_size,
            "n_layer": model.config.n_layer,
            "n_head": model.config.n_head,
            "n_embd": model.config.n_embd,
            "dropout": model.config.dropout,
            "bias": model.config.bias,
            "gqa_num_kv_heads": model.config.gqa_num_kv_heads,
        },
        "train": {"max_iters": cfg.max_iters, "learning_rate": cfg.learning_rate,
                  "batch_size": cfg.batch_size},
        "num_params": model.num_params(),
    }
    np.savez_compressed(path, **weights)
    meta_path = path.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[save] {tag} checkpoint -> {path}")


def _save_metrics(log: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)


def load_checkpoint(model: GPT, path: str | Path) -> dict:
    """Restore weights saved by _save_checkpoint. Returns meta dict."""
    path = Path(path)
    with open(path.with_suffix(".json"), encoding="utf-8") as f:
        meta = json.load(f)
    npz_path = Path(str(path) + ".npz")
    data = np.load(npz_path if npz_path.exists() else path, allow_pickle=False)
    for v in model.variables():
        if v.name in data.files:
            v.assign(data[v.name])
    print(f"[load] restored weights from {path} ({meta['num_params']} params)")
    return meta
