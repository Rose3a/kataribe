"""Haas-ja の書き起こし用: Triton を使わない Kimi Delta Attention（fla 0.5.2 の KimiDeltaAttention と同じ計算・同じ重み名）。

Haas-ja-v1.0 の prednet（2層・5ヘッド×64）は1トークンずつしか呼ばれないので、
素の PyTorch の逐次計算で十分速い。fla の Triton カーネルの式をそのまま写している:
- 畳み込み: 深さ方向の因果 conv1d + SiLU。cache は直近 W 個の入力 [B, D, W]
- ゲート: g = -exp(A_log) * softplus(f_proj(x) + dt_bias)、beta = sigmoid(b_proj(x))
- q, k は L2 正規化（eps 1e-6）、q に K^-0.5
- 状態は state_v_first（[B, HV, V, K]）。Haas は状態を自分で詰め直すだけなので並びは何でもよいが、fla に合わせる
- 出力: RMSNorm(o) * weight * sigmoid(g_proj(x)) → o_proj

`install()` で haas_asr._fla.load_kda_class をこのクラスに差し替える。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ShortConvolution(nn.Conv1d):
    """fla.modules.ShortConvolution と同じ重み（[D, 1, W]）の因果 conv1d。"""

    def __init__(self, hidden_size: int, kernel_size: int, bias: bool = False, activation: str | None = "silu"):
        super().__init__(hidden_size, hidden_size, kernel_size, groups=hidden_size, bias=bias, padding=0)
        self.hidden_size = hidden_size
        self.activation = activation

    def forward(self, x, cache=None, output_final_state=False, **kwargs):
        # x: [B, T, D] → [B, D, T]
        b, t, d = x.shape
        w = self.kernel_size[0]
        xt = x.transpose(1, 2)
        past = cache if cache is not None else xt.new_zeros(b, d, w)
        full = torch.cat([past.to(xt.dtype), xt], dim=-1)  # [B, D, W + T]
        y = F.conv1d(full[..., 1:], self.weight, self.bias, groups=d)  # 窓 = 直前 W-1 個 + 今の入力
        if self.activation is not None:
            y = F.silu(y)
        new_cache = None
        if output_final_state or cache is not None:
            new_cache = full[..., -w:]
            if cache is not None:
                cache.copy_(new_cache)  # fla は cache をその場で更新する
                new_cache = cache
        return y.transpose(1, 2), new_cache


class RMSNormGated(nn.Module):
    """fla.modules.FusedRMSNormGated（activation=sigmoid）と同じ式。"""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x, g):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()
        return (y * torch.sigmoid(g.float())).to(x.dtype)


def kda_recurrent(q, k, v, g, beta, state):
    """q, k: [B, T, H, K]、v: [B, T, HV, V]、g: [B, T, HV, K]（ゲート済み対数減衰）、beta: [B, T, HV]、
    state: [B, HV, V, K] か None。戻り値 o [B, T, HV, V]、新しい state（float32）。"""
    b, t, h, kd = q.shape
    hv, vd = v.shape[2], v.shape[3]
    rep = hv // h
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * kd**-0.5
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    if rep > 1:
        q, k = q.repeat_interleave(rep, 2), k.repeat_interleave(rep, 2)
    s = state.float() if state is not None else q.new_zeros(b, hv, vd, kd)
    out = []
    for i in range(t):
        s = s * g[:, i].exp()[:, :, None, :]
        ki = k[:, i]
        vi = v[:, i] - (s * ki[:, :, None, :]).sum(-1)
        s = s + (beta[:, i, :, None] * vi)[..., None] * ki[:, :, None, :]
        out.append((s * q[:, i][:, :, None, :]).sum(-1))
    return torch.stack(out, 1), s


class KimiDeltaAttention(nn.Module):
    """fla.layers.kda.KimiDeltaAttention の推論用の代わり（Haas が使う引数だけ）。"""

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 1,
        head_dim: int = 128,
        num_heads: int = 16,
        num_v_heads: int | None = None,
        mode: str = "chunk",
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int | None = None,
        norm_eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__()
        if not use_short_conv or allow_neg_eigval or safe_gate or lower_bound is not None:
            raise NotImplementedError("Haas-ja-v1.0 の設定（short conv あり・既定のゲート）だけ対応")
        self.mode = mode
        self.hidden_size = hidden_size
        self.conv_size = conv_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads
        self.head_k_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.key_dim = num_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.q_conv1d = ShortConvolution(self.key_dim, conv_size, bias=conv_bias)
        self.k_conv1d = ShortConvolution(self.key_dim, conv_size, bias=conv_bias)
        self.v_conv1d = ShortConvolution(self.value_dim, conv_size, bias=conv_bias)
        gate_dim = self.num_v_heads * self.head_k_dim
        self.f_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, gate_dim, bias=False),
        )
        self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.zeros(gate_dim, dtype=torch.float32))
        self.g_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True),
        )
        self.o_norm = RMSNormGated(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        if attention_mask is not None:
            raise NotImplementedError("attention_mask は未対応（Haas は渡さない）")
        x = hidden_states
        b, t, _ = x.shape
        last = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last = past_key_values[self.layer_idx]
        cq = ck = cv = None
        if last is not None and last.get("conv_state") is not None:
            cq, ck, cv = last["conv_state"]
        q, cq = self.q_conv1d(self.q_proj(x), cache=cq, output_final_state=use_cache)
        k, ck = self.k_conv1d(self.k_proj(x), cache=ck, output_final_state=use_cache)
        v, cv = self.v_conv1d(self.v_proj(x), cache=cv, output_final_state=use_cache)

        g = self.f_proj(x).float().view(b, t, self.num_v_heads, self.head_k_dim)
        g = -self.A_log.float().exp()[:, None] * F.softplus(g + self.dt_bias.float().view(self.num_v_heads, -1))
        beta = torch.sigmoid(self.b_proj(x).float())
        q = q.view(b, t, self.num_heads, self.head_k_dim)
        k = k.view(b, t, self.num_heads, self.head_k_dim)
        v = v.view(b, t, self.num_v_heads, self.head_v_dim)

        state = last.get("recurrent_state") if last is not None else None
        o, new_state = kda_recurrent(q, k, v, g, beta, state)
        o = o.to(x.dtype)

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=new_state.to(x.dtype) if use_cache else None,
                conv_state=(cq, ck, cv) if use_cache else None,
                layer_idx=self.layer_idx,
                offset=t,
            )

        gate = self.g_proj(x).view(b, t, self.num_v_heads, self.head_v_dim)
        o = self.o_norm(o, gate).reshape(b, t, self.value_dim)
        return self.o_proj(o), None, past_key_values


def install() -> None:
    """haas_asr が fla の代わりにこのクラスを使うようにする（haas_asr を読み込む前でも後でもよい）。"""
    import haas_asr._fla as fla_mod
    import haas_asr.prednet as prednet_mod

    def load_kda_class():
        return KimiDeltaAttention

    fla_mod.load_kda_class = load_kda_class
    prednet_mod.load_kda_class = load_kda_class
