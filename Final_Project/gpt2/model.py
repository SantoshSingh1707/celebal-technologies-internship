"""Mini GPT-2 decoder-only transformer in plain TensorFlow ops."""
from __future__ import annotations

import tensorflow as tf

from .config import ModelConfig


def gelu(x: tf.Tensor) -> tf.Tensor:
    """GPT-2 uses GELU activation on the MLP."""
    coeff = tf.constant(0.7978845608, dtype=tf.float32)  # sqrt(2/pi)
    return 0.5 * x * (1.0 + tf.tanh(coeff * (x + 0.044715 * x**3)))


class LayerNorm:
    """Pre-norm layer normalisation (element-wise scale + shift)."""

    def __init__(self, config: ModelConfig, name: str = "ln"):
        self.name = name
        self.gamma = tf.Variable(tf.ones([config.n_embd]), trainable=True, name=f"{name}/gamma")
        self.beta = tf.Variable(tf.zeros([config.n_embd]), trainable=True, name=f"{name}/beta")
        self.eps = 1e-5

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        var = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
        x = (x - mean) / tf.sqrt(var + self.eps)
        return x * self.gamma + self.beta

    def variables(self):
        return [self.gamma, self.beta]


class CausalSelfAttention:
    """Causal multi-head attention; optional GQA shares K/V heads."""

    def __init__(self, config: ModelConfig, name: str = "attn"):
        self.config = config
        self.name = name
        d = config.n_embd
        n_kv = config.n_kv_heads
        head_dim = d // config.n_head

        # causal mask: lower-triangular ones
        mask = tf.linalg.band_part(tf.ones((config.block_size, config.block_size)), -1, 0)
        self.mask = tf.reshape(mask, (1, 1, config.block_size, config.block_size))

        self.c_attn = tf.Variable(
            tf.random.normal([d, 3 * d], stddev=0.02), trainable=True, name=f"{name}/c_attn"
        )
        self.c_attn_kv = None
        if config.gqa_num_kv_heads > 0:  # GQA path
            kv_dim = n_kv * head_dim
            self.c_attn_kv = tf.Variable(
                tf.random.normal([d, 2 * kv_dim], stddev=0.02), trainable=True,
                name=f"{name}/c_attn_kv",
            )
        self.c_proj = tf.Variable(
            tf.random.normal([d, d], stddev=0.02), trainable=True, name=f"{name}/c_proj"
        )
        self.dropout = tf.keras.layers.Dropout(config.dropout) if config.dropout > 0 else None

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        B, T, C = tf.shape(x)[0], tf.shape(x)[1], self.config.n_embd
        n_head, head_dim = self.config.n_head, C // self.config.n_head

        if self.config.n_kv_heads == self.config.n_head:  # full MHA
            qkv = tf.matmul(x, self.c_attn)
            q, k, v = tf.split(qkv, 3, axis=-1)
        else:  # GQA: broadcast shared K/V heads
            qkv = tf.matmul(x, self.c_attn)
            q, _, _ = tf.split(qkv, 3, axis=-1)
            kv = tf.matmul(x, self.c_attn_kv)
            k, v = tf.split(kv, 2, axis=-1)

        def reshape_heads(t, heads):
            return tf.reshape(t, (B, T, heads, head_dim))

        q = reshape_heads(q, n_head)         # (B, T, n_head, head_dim)
        k = reshape_heads(k, self.config.n_kv_heads)
        v = reshape_heads(v, self.config.n_kv_heads)

        q = tf.transpose(q, perm=[0, 2, 1, 3])   # (B, nh, T, hd)
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        v = tf.transpose(v, perm=[0, 2, 1, 3])

        att = tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(head_dim, tf.float32))
        mask = tf.slice(self.mask, [0, 0, 0, 0], [1, 1, T, T])
        att = tf.where(mask == 1, att, -1e9 * tf.ones_like(att))
        att = tf.nn.softmax(att, axis=-1)
        if self.dropout is not None:
            att = self.dropout(att, training=True)

        y = tf.matmul(att, v)                    # (B, nh, T, hd)
        y = tf.transpose(y, perm=[0, 2, 1, 3])   # (B, T, nh, hd)
        y = tf.reshape(y, (B, T, C))

        y = tf.matmul(y, self.c_proj)
        if self.dropout is not None:
            y = self.dropout(y, training=True)
        return y

    def variables(self):
        vs = [self.c_attn, self.c_proj]
        if self.c_attn_kv is not None:
            vs.append(self.c_attn_kv)
        return vs


class MLP:
    """Position-wise feed-forward network: Linear -> GELU -> Linear."""

    def __init__(self, config: ModelConfig, name: str = "mlp"):
        self.name = name
        d = config.n_embd
        self.c_fc = tf.Variable(
            tf.random.normal([d, 4 * d], stddev=0.02), trainable=True, name=f"{name}/c_fc"
        )
        self.c_proj = tf.Variable(
            tf.random.normal([4 * d, d], stddev=0.02), trainable=True, name=f"{name}/c_proj"
        )
        self.dropout = tf.keras.layers.Dropout(config.dropout) if config.dropout > 0 else None

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        x = tf.matmul(x, self.c_fc)
        x = gelu(x)
        x = tf.matmul(x, self.c_proj)
        if self.dropout is not None:
            x = self.dropout(x, training=True)
        return x

    def variables(self):
        return [self.c_fc, self.c_proj]


class Block:
    """One transformer block: causal attention + MLP with pre-norm residuals."""

    def __init__(self, config: ModelConfig, name: str = "block"):
        self.name = name
        self.ln_1 = LayerNorm(config, name=f"{name}/ln1")
        self.attn = CausalSelfAttention(config, name=f"{name}/attn")
        self.ln_2 = LayerNorm(config, name=f"{name}/ln2")
        self.mlp = MLP(config, name=f"{name}/mlp")

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def variables(self):
        return self.ln_1.variables() + self.attn.variables() \
            + self.ln_2.variables() + self.mlp.variables()


class GPT:
    """Decoder-only transformer: embeddings + blocks + head."""

    def __init__(self, config: ModelConfig, name: str = "gpt"):
        self.config = config
        self.name = name
        self.wte = tf.Variable(
            tf.random.normal([config.vocab_size, config.n_embd], stddev=0.02),
            trainable=True, name="wte",
        )
        self.wpe = tf.Variable(
            tf.random.normal([config.block_size, config.n_embd], stddev=0.02),
            trainable=True, name="wpe",
        )
        self.blocks = [Block(config, name=f"block{i}") for i in range(config.n_layer)]
        self.ln_f = LayerNorm(config, name="ln_f")
        self.dropout = tf.keras.layers.Dropout(config.dropout) if config.dropout > 0 else None

    def forward(self, idx: tf.Tensor) -> tf.Tensor:
        """Forward pass. idx: (B, T). Returns logits (B, T, vocab)."""
        B, T = idx.shape
        assert T <= self.config.block_size, "sequence longer than block_size"

        tok_emb = tf.gather(self.wte, idx)                       # (B, T, C)
        pos = tf.range(T, dtype=tf.int32)
        pos_emb = tf.gather(self.wpe, pos)                       # (T, C)
        x = tok_emb + pos_emb
        if self.dropout is not None:
            x = self.dropout(x, training=True)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = tf.matmul(x, self.wte, transpose_b=True)        # weight tying
        return logits

    def __call__(self, idx: tf.Tensor) -> tf.Tensor:
        return self.forward(idx)

    def loss(self, idx: tf.Tensor, targets: tf.Tensor) -> tf.Tensor:
        """Cross-entropy over the last timestep for a (B, T) batch."""
        logits = self.forward(idx)
        logits = tf.reshape(logits, [-1, self.config.vocab_size])
        targets = tf.reshape(targets, [-1])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=targets, logits=logits
        )
        return tf.reduce_mean(loss)

    def variables(self):
        vs = [self.wte, self.wpe, self.ln_f.gamma, self.ln_f.beta]
        for b in self.blocks:
            vs += b.variables()
        return vs

    def num_params(self) -> int:
        return sum(int(tf.size(v).numpy()) for v in self.variables())

    def estimate_loss(self, train_data: tf.Tensor, val_data: tf.Tensor,
                      batch_size: int, eval_iters: int, seed: int) -> dict[str, float]:
        """Average loss over ``eval_iters`` batches of train and val."""
        out: dict[str, float] = {}
        for split_name, split_data in [("train", train_data), ("val", val_data)]:
            losses = []
            for i in range(eval_iters):
                x, y = get_batch_local(split_data, batch_size, self.config.block_size, seed + i)
                losses.append(self.loss(x, y))
            out[split_name] = float(tf.reduce_mean(losses).numpy())
        return out


def get_batch_local(split_data, batch_size, block_size, seed):
    """Module-local import to avoid a circular import from gpt2.data."""
    from .data import get_batch
    return get_batch(split_data, batch_size, block_size, seed)
