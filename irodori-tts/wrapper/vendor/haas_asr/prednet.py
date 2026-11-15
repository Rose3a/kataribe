"""Kimi Delta Attention prediction network for NeMo's transducer (RNN-T / TDT) decoder.

``KDATransducerDecoder`` replaces the LSTM stack inside NeMo's
``RNNTDecoder`` with pre-norm residual blocks of Kimi Delta Attention (KDA)
from flash-linear-attention:

    x = x + dropout(kda(norm(x)))

followed by a final LayerNorm. The label embedding, the joint
network and every NeMo decoding algorithm are unchanged. ("RNNT" in NeMo's
names is the transducer framework; nothing in this model is an LSTM.)

This module's import path, its class names and its ``kda_*`` config keys are
written into every exported ``.nemo``, so none of them may ever change.

Decoding state layout
---------------------
Only matters while decoding: the KDA math is unchanged, and the layout is not
saved in checkpoints. Training carries no state, so none of this runs then.

Why: in batched decoding, samples that emit blank must keep their old state
while the others advance. NeMo selects and copies states with helpers on
``RNNTDecoder`` that expect an LSTM's ``(h, c)``: two tensors of the same shape
``[num_layers, batch, width]``. KDA's state per layer is four differently shaped
tensors instead, so it is flattened into that shape. The helpers only slice the
batch dimension, so they then work unchanged. For the released config
(2 layers, W = 20,480), the last dimension of each component holds:

    component 0 [2, B, W]:  | recurrent [5, 64, 64] 20,480 |
    component 1 [2, B, W]:  | conv q [320, 4] 1,280 | conv k 1,280 | conv v 1,280 | zeros 16,640 |

Before each step the state is unpacked into fla's shapes, and packed again after
it. The recurrent memory gets a component of its own so that unpacking it is a
free ``.view()``. Dropping the zeros would split the memory across both
components and cost a concatenation every step.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from nemo.collections.asr.modules.rnnt import RNNTDecoder
from torch import nn

from haas_asr._fla import load_kda_class

# Prefix of this module's keys in ``model.decoder.prednet``.
CONFIG_PREFIX = "kda_"


@dataclass(frozen=True)
class KDAConfig:
    """The ``kda_*`` keys of ``model.decoder.prednet``, prefix removed. All required."""

    num_heads: int
    head_dim: int
    expand_v: float
    conv_size: int
    final_norm: bool

    @classmethod
    def from_prednet(cls, prednet) -> KDAConfig:
        values = {
            key.removeprefix(CONFIG_PREFIX): value
            for key, value in prednet.items()
            if key.startswith(CONFIG_PREFIX)
        }
        expected = {f.name for f in fields(cls)}
        if unknown := sorted(values.keys() - expected):
            raise ValueError(f"unknown prednet keys: {[CONFIG_PREFIX + k for k in unknown]}")
        if missing := sorted(expected - values.keys()):
            raise ValueError(f"missing prednet keys: {[CONFIG_PREFIX + k for k in missing]}")
        return cls(**values)


class _LayerCache:
    """The subset of ``fla.models.utils.Cache`` that fla layers use.

    Layers read their state with ``len(cache)`` and ``cache[layer_idx]`` and
    write it with ``cache.update(layer_idx=..., **state)``.
    """

    def __init__(self, states: list[dict] | None = None):
        self.states = states if states is not None else []

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, layer_idx: int) -> dict:
        return self.states[layer_idx]

    def update(self, layer_idx: int = 0, offset=None, **state) -> dict:
        # `offset` is fla's variable-length bookkeeping, unused here.
        while len(self.states) <= layer_idx:
            self.states.append({})
        self.states[layer_idx].update(state)
        return self.states[layer_idx]


@dataclass(frozen=True)
class _StateLayout:
    """Packs fla's per-layer cache into NeMo's two-tensor state and back (see module docstring)."""

    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    key_dim: int
    value_dim: int
    conv_size: int

    @classmethod
    def from_layer(cls, layer: nn.Module) -> _StateLayout:
        """Read the dimensions off a constructed layer, so fla's own math is used."""
        return cls(**{f.name: int(getattr(layer, f.name)) for f in fields(cls)})

    @property
    def recurrent_size(self) -> int:
        return self.num_v_heads * self.head_k_dim * self.head_v_dim

    @property
    def conv_sizes(self) -> tuple[int, int, int]:
        """Flat sizes of the q, k and v short-convolution caches."""
        key, value = self.key_dim * self.conv_size, self.value_dim * self.conv_size
        return key, key, value

    @property
    def conv_total(self) -> int:
        return sum(self.conv_sizes)

    @property
    def width(self) -> int:
        return max(self.recurrent_size, self.conv_total)

    def to_cache(self, state: tuple[torch.Tensor, torch.Tensor]) -> _LayerCache:
        recurrent, conv = state
        batch_size = recurrent.shape[1]
        conv_dims = (self.key_dim, self.key_dim, self.value_dim)
        layers = []
        for recurrent_i, conv_i in zip(recurrent, conv, strict=True):
            caches = torch.split(conv_i[:, : self.conv_total], self.conv_sizes, dim=-1)
            layers.append(
                {
                    # A view whenever the component is exactly recurrent_size wide.
                    "recurrent_state": recurrent_i[:, : self.recurrent_size].reshape(
                        batch_size, self.num_v_heads, self.head_k_dim, self.head_v_dim
                    ),
                    # Strided slices, so reshape copies; fla updates these in place.
                    "conv_state": tuple(
                        cache.reshape(batch_size, dim, self.conv_size)
                        for cache, dim in zip(caches, conv_dims, strict=True)
                    ),
                }
            )
        return _LayerCache(layers)

    def from_cache(
        self, cache: _LayerCache, num_layers: int, batch_size: int, device, dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # empty, not zeros: only the padding and any missing entries need clearing.
        shape = (num_layers, batch_size, self.width)
        recurrent = torch.empty(shape, device=device, dtype=dtype)
        conv = torch.empty(shape, device=device, dtype=dtype)
        recurrent[..., self.recurrent_size :].zero_()
        conv[..., self.conv_total :].zero_()
        for i in range(num_layers):
            layer_state = cache[i] if i < len(cache) else {}
            recurrent_dst = recurrent[i, :, : self.recurrent_size]
            if (src := layer_state.get("recurrent_state")) is None:
                recurrent_dst.zero_()
            else:
                recurrent_dst.copy_(src.reshape(batch_size, -1))
            conv_dst = conv[i, :, : self.conv_total]
            if (srcs := layer_state.get("conv_state")) is None:
                conv_dst.zero_()
            else:
                for dst, src in zip(
                    torch.split(conv_dst, self.conv_sizes, dim=-1), srcs, strict=True
                ):
                    dst.copy_(src.reshape(batch_size, -1))
        return recurrent, conv


class KDAPredictionNetwork(nn.Module):
    """Stands in for the ``nn.LSTM`` that NeMo's prednet calls ``dec_rnn``.

    Same call contract as NeMo uses it: ``forward(y, state) -> (g, new_state)``
    with time-first ``(U, B, H)`` input and output.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        config: KDAConfig,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        kda_cls = load_kda_class()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # The state layout needs the short convolution. "chunk" is the training
        # kernel; fla switches to its recurrent kernel for one-label decoding steps.
        self.layers = nn.ModuleList(
            kda_cls(
                hidden_size=hidden_size,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                expand_v=config.expand_v,
                conv_size=config.conv_size,
                mode="chunk",
                use_short_conv=True,
                layer_idx=i,
            )
            for i in range(num_layers)
        )
        self.norms = nn.ModuleList(
            nn.LayerNorm(hidden_size, eps=norm_eps) for _ in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(hidden_size, eps=norm_eps) if config.final_norm else None
        self.layout = _StateLayout.from_layer(self.layers[0])

    def empty_state(self, batch_size: int, *, device=None, dtype=None):
        shape = (self.num_layers, batch_size, self.layout.width)
        return (
            torch.zeros(shape, device=device, dtype=dtype),
            torch.zeros(shape, device=device, dtype=dtype),
        )

    def forward(self, y: torch.Tensor, state=None):
        _, batch_size, hidden_size = y.shape
        if hidden_size != self.hidden_size:
            raise ValueError(f"input width {hidden_size} != hidden_size {self.hidden_size}")
        x = y.transpose(0, 1)  # fla is batch-first

        # Training without an incoming state skips the cache, which keeps fla's
        # fused q/k/v convolution available. Every other path needs the state.
        use_cache = state is not None or not self.training
        if state is not None:
            cache = self.layout.to_cache(state)
        else:
            cache = _LayerCache() if use_cache else None

        for layer, norm in zip(self.layers, self.norms, strict=True):
            out, _, _ = layer(hidden_states=norm(x), past_key_values=cache, use_cache=use_cache)
            x = x + self.dropout(out)
        if self.final_norm is not None:
            x = self.final_norm(x)

        if use_cache:
            new_state = self.layout.from_cache(
                cache, self.num_layers, batch_size, x.device, x.dtype
            )
        else:
            # NeMo discards the state in training; zeros keep the shape contract.
            new_state = self.empty_state(batch_size, device=x.device, dtype=x.dtype)
        return x.transpose(0, 1), new_state


class KDATransducerDecoder(RNNTDecoder):
    """``RNNTDecoder`` whose prediction network is a KDA stack.

    Configured under ``model.decoder.prednet`` next to NeMo's usual keys::

        prednet:
          pred_hidden: 640
          pred_rnn_layers: 2
          dropout: 0.2
          kda_num_heads: 5
          kda_head_dim: 64
          kda_expand_v: 1.0
          kda_conv_size: 4
          kda_final_norm: true
    """

    def __init__(
        self,
        prednet,
        vocab_size,
        normalization_mode=None,
        random_state_sampling=False,
        blank_as_pad=True,
    ):
        if normalization_mode:
            raise ValueError("normalization_mode must be null: the blocks carry their own norms")
        if random_state_sampling:
            raise ValueError("random_state_sampling is not supported by this prednet")
        # Set before super().__init__(), which calls _predict_modules().
        self._kda_config = KDAConfig.from_prednet(prednet)
        super().__init__(
            prednet=prednet,
            vocab_size=vocab_size,
            normalization_mode=None,
            random_state_sampling=False,
            blank_as_pad=blank_as_pad,
        )

    def _predict_modules(
        self,
        vocab_size,
        pred_n_hidden,
        pred_rnn_layers,
        dropout,
        rnn_hidden_size=-1,
        **_lstm_kwargs,  # forget_gate_bias, t_max, ...: no counterpart here
    ):
        if rnn_hidden_size > 0 and rnn_hidden_size != pred_n_hidden:
            raise ValueError("rnn_hidden_size must equal pred_hidden: there is no projection")
        if self.blank_as_pad:
            embed = nn.Embedding(vocab_size + 1, pred_n_hidden, padding_idx=self.blank_idx)
        else:
            embed = nn.Embedding(vocab_size, pred_n_hidden)
        dec_rnn = KDAPredictionNetwork(
            hidden_size=pred_n_hidden,
            num_layers=pred_rnn_layers,
            config=self._kda_config,
            dropout=dropout,
        )
        return nn.ModuleDict({"embed": embed, "dec_rnn": dec_rnn})

    def initialize_state(self, y: torch.Tensor):
        return self.prediction["dec_rnn"].empty_state(y.size(0), device=y.device, dtype=y.dtype)
