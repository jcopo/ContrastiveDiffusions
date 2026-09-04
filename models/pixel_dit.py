"""PixelDiT: pixel-space conditional velocity transformer.

Self-contained (jax, flax, einops only) so the design loop runs without the
private training stack. Verified bit-identical to the training implementation.

    from huggingface_hub import hf_hub_download
    import importlib.util
    spec = importlib.util.spec_from_file_location("cfg", hf_hub_download("jcopo/mnist", "config.py"))
    cfg = importlib.util.module_from_spec(spec); spec.loader.exec_module(cfg)
    net = cfg.load_model()
    v = net(x, t=t, cond=c).output   # x: (B,28,28,1) in [-1,1], c: (B,28,28,2)
    v = net(x, t=t).output          # cond omitted = unconditional

Predicts the flow-matching velocity v(z_t, t | c) for MNIST, conditioned on
c = [masked image, mask]. Trained for CoDiff (arXiv:2410.11826).

Architecture after github.com/Lyy-iiis/pMF, single-head. The blocks below are
transcribed from the training code so this file stands alone; parameter names
and numerics match the published checkpoint exactly.
"""

import math

import jax
import jax.numpy as jnp
from einops import rearrange
from flax import nnx, struct
from jax import Array
from jax.typing import ArrayLike, DTypeLike


def _linear(in_features, out_features, *, bias, init_constant, param_dtype, dtype, rngs, zero_init=False):
    """normal(std = c / sqrt(in)) kernel, zero bias."""
    kernel_init = nnx.initializers.zeros if zero_init else nnx.initializers.normal(stddev=init_constant / math.sqrt(in_features))
    return nnx.Linear(
        in_features,
        out_features,
        use_bias=bias,
        kernel_init=kernel_init,
        bias_init=nnx.initializers.zeros,
        param_dtype=param_dtype,
        dtype=dtype,
        rngs=rngs,
    )


def precompute_rope_freqs_2d(dim: int, seq_len: int, theta: float = 10000.0) -> Array:
    """2D rotary frequencies, complex64 of shape (seq_len, dim // 2)."""
    dim = dim // 2  # for 2d rotary embeddings
    T = int(seq_len**0.5)
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    positions = jnp.arange(T, dtype=jnp.float32)
    freqs_h = jnp.einsum("i,j->ij", positions, freqs)
    freqs_w = jnp.einsum("i,j->ij", positions, freqs)
    freqs = jnp.concatenate([jnp.tile(freqs_h[:, None, :], (1, T, 1)), jnp.tile(freqs_w[None, :, :], (T, 1, 1))], axis=-1)
    real = jnp.cos(freqs).reshape(seq_len, dim)
    imag = jnp.sin(freqs).reshape(seq_len, dim)
    return jax.lax.complex(real, imag)


def apply_rotary_pos_emb(x: Array, freqs_cis: Array) -> Array:
    """Rotate the LAST T tokens of x (prefix tokens stay unrotated)."""
    b, n, h, d = x.shape
    pairs = x.astype(jnp.float32).reshape(b, n, h, d // 2, 2)
    x_complex = jax.lax.complex(pairs[..., 0], pairs[..., 1])
    T = freqs_cis.shape[0]
    rotated = x_complex.at[:, -T:, :, :].multiply(freqs_cis[None, :, None, :])
    out = jnp.stack([jnp.real(rotated), jnp.imag(rotated)], axis=-1).reshape(b, n, h, d)
    return out.astype(x.dtype)


class BottleneckPatchEmbedder(nnx.Module):
    """patch conv -> pca_channels, then 1x1 conv -> hidden."""

    def __init__(self, input_size, patch_size, pca_channels, in_channels, hidden_size, *, param_dtype, dtype, rngs: nnx.Rngs):
        self.num_patches = (input_size // patch_size) ** 2
        conv_kw = {
            "use_bias": True,
            "kernel_init": nnx.initializers.xavier_uniform(in_axis=(0, 1, 2), out_axis=-1),
            "param_dtype": param_dtype,
            "dtype": dtype,
            "rngs": rngs,
        }
        self.proj1 = nnx.Conv(in_channels, pca_channels, kernel_size=(patch_size, patch_size), strides=(patch_size, patch_size), **conv_kw)
        self.proj2 = nnx.Conv(pca_channels, hidden_size, kernel_size=(1, 1), **conv_kw)

    def __call__(self, x: Array) -> Array:
        x = self.proj2(self.proj1(x))
        return x.reshape(x.shape[0], -1, x.shape[-1])


class PMFTimestepEmbedder(nnx.Module):
    """Sinusoidal embedding (freq 256) + 2-layer silu MLP."""

    def __init__(self, hidden_size, *, init_constant=1.0, frequency_embedding_size=256, param_dtype=jnp.float32, dtype=jnp.float32, rngs: nnx.Rngs):
        self.frequency_embedding_size = frequency_embedding_size
        half = frequency_embedding_size // 2
        self.freqs = jnp.exp(-math.log(10000.0) * jnp.arange(half, dtype=jnp.float32) / half)
        kw = {"bias": True, "init_constant": init_constant, "param_dtype": param_dtype, "dtype": dtype, "rngs": rngs}
        self.fc1 = _linear(frequency_embedding_size, hidden_size, **kw)
        self.fc2 = _linear(hidden_size, hidden_size, **kw)

    def __call__(self, t: Array) -> Array:
        # sinusoid in f32 for phase precision; fc1 downcasts to compute dtype
        args = t[:, None].astype(jnp.float32) * self.freqs[None]
        emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        return self.fc2(nnx.silu(self.fc1(emb)))


class RoPEAttention(nnx.Module):
    """Multi-head self-attention with 2D RoPE and QK RMSNorm."""

    def __init__(self, hidden_size, num_heads, *, init_constant, param_dtype, dtype, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        kw = {"bias": False, "init_constant": init_constant, "param_dtype": param_dtype, "dtype": dtype, "rngs": rngs}
        self.q_proj = _linear(hidden_size, hidden_size, **kw)
        self.k_proj = _linear(hidden_size, hidden_size, **kw)
        self.v_proj = _linear(hidden_size, hidden_size, **kw)
        self.out_proj = _linear(hidden_size, hidden_size, **kw)
        self.q_norm = nnx.RMSNorm(self.head_dim, epsilon=1e-6, param_dtype=param_dtype, dtype=dtype, rngs=rngs)
        self.k_norm = nnx.RMSNorm(self.head_dim, epsilon=1e-6, param_dtype=param_dtype, dtype=dtype, rngs=rngs)

    def __call__(self, x: Array, rope_freqs: Array) -> Array:
        b, n, hidden = x.shape
        q = self.q_proj(x).reshape(b, n, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(b, n, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(b, n, self.num_heads, self.head_dim)
        q = apply_rotary_pos_emb(self.q_norm(q), rope_freqs)
        k = apply_rotary_pos_emb(self.k_norm(k), rope_freqs)
        attn = jax.nn.dot_product_attention(q, k, v, implementation="xla").reshape(b, n, hidden)
        return self.out_proj(attn)


class SwiGLUMlp(nnx.Module):
    def __init__(self, in_features, hidden_features, *, init_constant, param_dtype, dtype, rngs: nnx.Rngs):
        kw = {"bias": False, "init_constant": init_constant, "param_dtype": param_dtype, "dtype": dtype, "rngs": rngs}
        self.w1 = _linear(in_features, hidden_features, **kw)
        self.w3 = _linear(in_features, hidden_features, **kw)
        self.w2 = _linear(hidden_features, in_features, **kw)

    def __call__(self, x: Array) -> Array:
        return self.w2(nnx.silu(self.w1(x)) * self.w3(x))


class PMFBlock(nnx.Module):
    """Transformer block with zero-initialized per-channel gates on residuals."""

    def __init__(self, hidden_size, num_heads, *, mlp_ratio, init_constant, param_dtype, dtype, rngs: nnx.Rngs):
        kw = {"init_constant": init_constant, "param_dtype": param_dtype, "dtype": dtype, "rngs": rngs}
        self.norm1 = nnx.RMSNorm(hidden_size, epsilon=1e-6, param_dtype=param_dtype, dtype=dtype, rngs=rngs)
        self.attn = RoPEAttention(hidden_size, num_heads, **kw)
        self.norm2 = nnx.RMSNorm(hidden_size, epsilon=1e-6, param_dtype=param_dtype, dtype=dtype, rngs=rngs)
        self.mlp = SwiGLUMlp(hidden_size, int(hidden_size * mlp_ratio), **kw)
        self.attn_scale = nnx.Param(jnp.zeros((hidden_size,), param_dtype))
        self.mlp_scale = nnx.Param(jnp.zeros((hidden_size,), param_dtype))

    def __call__(self, x: Array, rope_freqs: Array) -> Array:
        x = x + self.attn(self.norm1(x), rope_freqs) * self.attn_scale.value.astype(x.dtype)
        x = x + self.mlp(self.norm2(x)) * self.mlp_scale.value.astype(x.dtype)
        return x


class PMFFinalLayer(nnx.Module):
    """RMSNorm + zero-init linear projection."""

    def __init__(self, hidden_size, patch_size, out_channels, *, param_dtype, dtype, rngs: nnx.Rngs):
        self.norm = nnx.RMSNorm(hidden_size, epsilon=1e-6, param_dtype=param_dtype, dtype=dtype, rngs=rngs)
        self.linear = _linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True, init_constant=1.0, zero_init=True,
            param_dtype=param_dtype, dtype=dtype, rngs=rngs,
        )

    def __call__(self, x: Array) -> Array:
        return self.linear(self.norm(x))


@struct.dataclass


class PixelDiTOutput:
    """``output`` is the velocity v(z_t, t | cond)."""

    output: Array


class PixelDiT(nnx.Module):
    """Shared trunk of (depth - head_depth) blocks, then a velocity head."""

    def __init__(
        self,
        *,
        input_size: int = 28,
        patch_size: int = 4,
        in_channels: int = 1,
        cond_channels: int = 2,
        hidden_size: int = 256,
        depth: int = 8,
        num_heads: int = 4,
        mlp_ratio: float = 8 / 3,
        head_depth: int = 3,
        num_time_tokens: int = 4,
        pca_channels: int | None = None,
        token_init_constant: float = 1.0,
        embedding_init_constant: float = 1.0,
        weight_init_constant: float = 0.32,
        t_floor: float = 0.05,
        param_dtype: DTypeLike = jnp.float32,
        dtype: DTypeLike = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.input_size = input_size
        self.patch_size = patch_size
        self.out_channels = in_channels
        self.cond_channels = cond_channels
        self.num_time_tokens = num_time_tokens
        self.t_floor = t_floor
        if pca_channels is None:
            pca_channels = 128 if hidden_size <= 1024 else 256

        self.x_embedder = BottleneckPatchEmbedder(
            input_size,
            patch_size,
            pca_channels,
            in_channels + cond_channels,
            hidden_size,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.h_embedder = PMFTimestepEmbedder(
            hidden_size, init_constant=embedding_init_constant, param_dtype=param_dtype, dtype=dtype, rngs=rngs
        )
        token_init = nnx.initializers.normal(stddev=token_init_constant / math.sqrt(hidden_size))
        self.time_tokens = nnx.Param(token_init(rngs.params(), (1, num_time_tokens, hidden_size), param_dtype))

        self.prefix_tokens = num_time_tokens
        self.spatial_tokens = self.x_embedder.num_patches
        total_tokens = self.spatial_tokens + self.prefix_tokens
        self.pos_embed = nnx.Param(nnx.initializers.normal(stddev=0.02)(rngs.params(), (1, total_tokens, hidden_size), param_dtype))

        # complex64 constant, stored rather than recomputed per forward
        self._rope_freqs = precompute_rope_freqs_2d(hidden_size // num_heads, self.spatial_tokens)

        block_kw = {
            "mlp_ratio": mlp_ratio,
            "init_constant": weight_init_constant,
            "param_dtype": param_dtype,
            "dtype": dtype,
            "rngs": rngs,
        }
        self.shared_blocks = nnx.List([PMFBlock(hidden_size, num_heads, **block_kw) for _ in range(depth - head_depth)])
        self.heads = nnx.List([PMFBlock(hidden_size, num_heads, **block_kw) for _ in range(head_depth)])
        self.final_layer = PMFFinalLayer(hidden_size, patch_size, self.out_channels, param_dtype=param_dtype, dtype=dtype, rngs=rngs)

    def unpatchify(self, x: Array) -> Array:
        h = w = int(x.shape[1] ** 0.5)
        return rearrange(x, "b (h w) (p1 p2 c) -> b (h p1) (w p2) c", h=h, w=w, p1=self.patch_size, p2=self.patch_size, c=self.out_channels)

    def __call__(self, x: ArrayLike, t: ArrayLike, cond: ArrayLike | None = None, s: ArrayLike | None = None) -> PixelDiTOutput:
        """Velocity v(x_t, t | cond), as ``.output``.

        ``cond`` defaults to zeros, the null conditioning seen during training's
        conditioning dropout — i.e. the unconditional model. ``s`` sets the
        conditioning offset h = t - s and defaults to ``t`` (h = 0), the only
        regime this model is used in.
        """
        squeeze = x.ndim < 4
        if squeeze:
            x = jnp.expand_dims(x, 0)
        cond = jnp.zeros((*x.shape[:-1], self.cond_channels), x.dtype) if cond is None else jnp.asarray(cond)
        if cond.ndim < 4:
            cond = jnp.expand_dims(cond, 0)
        if cond.shape[:-1] != x.shape[:-1] or cond.shape[-1] != self.cond_channels:
            raise ValueError(f"cond must have shape {(*x.shape[:-1], self.cond_channels)}, got {cond.shape}")

        batch = x.shape[0]
        t = jnp.asarray(t).reshape(-1).astype(jnp.float32)
        s = t if s is None else jnp.asarray(s).reshape(-1).astype(jnp.float32)

        x_embed = self.x_embedder(jnp.concatenate([x, cond.astype(x.dtype)], axis=-1))
        h_embed = self.h_embedder(t - s)
        time_tokens = self.time_tokens.value.astype(x_embed.dtype) + h_embed[:, None, :].astype(x_embed.dtype)
        time_tokens = jnp.broadcast_to(time_tokens, (batch, self.num_time_tokens, x_embed.shape[-1]))

        seq = jnp.concatenate([time_tokens, x_embed], axis=1) + self.pos_embed.value.astype(x_embed.dtype)
        for block in self.shared_blocks:
            seq = block(seq, self._rope_freqs)
        for block in self.heads:
            seq = block(seq, self._rope_freqs)

        image_slice = slice(self.prefix_tokens, self.prefix_tokens + self.spatial_tokens)
        x_pred = self.unpatchify(self.final_layer(seq[:, image_slice]))
        # boundary condition: v = (z - x_pred) / t, floored so t -> 0 stays finite
        v = (x - x_pred) / jnp.clip(t, self.t_floor, 1.0).reshape(-1, 1, 1, 1).astype(x.dtype)
        return PixelDiTOutput(output=v.squeeze(axis=0) if squeeze else v)


MODEL_CONFIG = dict(
    input_size=28,
    patch_size=4,
    in_channels=1,
    cond_channels=2,
    hidden_size=256,
    depth=8,
    num_heads=4,
    head_depth=3,
    t_floor=0.05,
)


def get_model(rngs=None):
    """Un-initialized PixelDiT with the architecture of the published weights."""
    return PixelDiT(**MODEL_CONFIG, rngs=rngs or nnx.Rngs(0))


def load_model(weights_path=None, revision=None):
    """PixelDiT with the published weights loaded; downloads them if not given."""
    from flax import serialization

    if weights_path is None:
        from huggingface_hub import hf_hub_download

        weights_path = hf_hub_download("jcopo/mnist", "model.msgpack", revision=revision)
    net = get_model()
    _, state = nnx.split(net)
    with open(weights_path, "rb") as f:
        state.replace_by_pure_dict(serialization.from_bytes(state.to_pure_dict(), f.read()))
    nnx.update(net, state)
    net.eval()
    return net
