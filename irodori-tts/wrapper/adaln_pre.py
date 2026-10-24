"""adaLN の事前計算（TensorRT の plan に t の代わりに mod を渡す）。

DiT の各層の adaLN（attention_adaln / mlp_adaln）は、時刻 t から作る cond_embed だけで決まる
shift / scale / gate を使う。t は合成の最初に全ステップ分が決まるので、全ステップ分を einsum で一括に
計算し、plan には t の代わりに mod: (B, 層, 2, 3, D) を渡す。plan の中では adaLN の行列積をしない
（Blockwise Small・W8A8 で DiT 1回 8.7→7.3ms、sample_rf −6〜14%、plan −85MB）。

- 2 は attention_adaln / mlp_adaln、3 は shift / scale / gate（gate は tanh をかける前）。
- plan 側の変換は bf16-fallback/adaln_builder.py、plan の作成とキャッシュは trt_adaln.py。
- 実行時: AdalnPrecompute が重みを（plan に捨てられる前に）重ねて持ち、sample_euler_rf_cfg の前に表を作り、
  AdalnAdapter が plan を呼ぶ直前に kw['t'] を mod に差し替える。
- 省VRAM（IRODORI_LOW_VRAM=1）では重み（約94MB）を CPU に置き、表を作るあいだだけ GPU に載せる。
"""
import copy
import math
import os
import sys
import types

import torch
import torch.nn.functional as F

CHECK_ENV = "IRODORI_ADALN_CHECK"  # 1 のとき、呼び出しごとに t が表の t と合っているか確かめる（同期が入る）
PARTS = ("shift", "scale", "gate")


def plan_takes_mod(engine) -> bool:
    """plan の t が mod（5 次元）か。"""
    try:
        return len(engine.engine.get_tensor_shape("t")) == 5
    except Exception:  # noqa: BLE001 - t が無い・読めない plan は従来どおり
        return False


def t_schedule(num_steps, mode, sway_coeff, device, init_scale=0.999):
    """rf.sample_euler_rf_cfg と同じ式（t_schedule の全 num_steps + 1 個）。"""
    u = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    if str(mode).strip().lower() == "sway":
        u = u + float(sway_coeff) * (torch.cos(0.5 * math.pi * u) + u - 1.0)
        u = u.clamp(0.0, 1.0)
    return (1.0 - u) * init_scale


def patch_blocks(blocks):
    """adaLN の forward を、cond_embed = (B, 2, 3, D) の事前計算値を受け取る版に差し替える（変換用）。"""

    def make(idx):
        def forward(self, x, cond_embed):
            shift = cond_embed[:, idx, 0][:, None, :]
            scale = cond_embed[:, idx, 1][:, None, :]
            gate = cond_embed[:, idx, 2][:, None, :]
            x_dtype = x.dtype
            x = x.float()
            x = x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + self.eps)
            x = x * (1.0 + scale) + shift
            gate = torch.tanh(gate)
            return x.to(x_dtype), gate
        return forward

    for block in blocks:
        block.attention_adaln.forward = types.MethodType(make(0), block.attention_adaln)
        block.mlp_adaln.forward = types.MethodType(make(1), block.mlp_adaln)


def supports(model) -> str | None:
    """事前計算できないモデルならその理由、できれば None。"""
    if getattr(model, "delta_cond_module", None) is not None:
        return "MeanFlow（delta_t あり）のモデルは未対応です"
    blocks = getattr(model, "blocks", None)
    if not blocks or not all(hasattr(b, "attention_adaln") and hasattr(b, "mlp_adaln") for b in blocks):
        return "adaLN を持たないモデルです"
    return None


def stack_weights(model):
    """全層の adaLN の重みを (層, 2, 3, ...) に重ねる。返り値: wd (L,2,3,R,D), wu (L,2,3,D,R), bu (L,2,3,D)。"""
    def per(attr, getter):
        return torch.stack([torch.stack([torch.stack([getter(getattr(ad, f"{n}_{attr}")) for n in PARTS])
                                         for ad in (b.attention_adaln, b.mlp_adaln)])
                            for b in model.blocks])

    with torch.no_grad():
        wd = per("down", lambda m: m.weight.detach())
        wu = per("up", lambda m: m.weight.detach())
        bu = per("up", lambda m: m.bias.detach())
    return wd.contiguous(), wu.contiguous(), bu.contiguous()


def compute_table(cond_module, wd, wu, bu, time_dim, t_values):
    """t_values: (S,) → (S, 層, 2, 3, D)。数回の einsum で、層ごとの計算と同じ値を出す。"""
    from irodori_tts import model as model_module
    with torch.no_grad():
        emb = model_module.get_timestep_embedding(t_values, time_dim).to(wd.dtype)
        c = torch.stack(cond_module(emb).chunk(3, dim=-1), dim=1)       # (S, 3, D)
        h = torch.einsum("spd,lapkd->slapk", F.silu(c), wd)               # (S, L, 2, 3, R)
        return (torch.einsum("slapk,lapdk->slapd", h, wu) + bu + c[:, None, None]).contiguous()


class AdalnPrecompute:
    """adaLN の重みを持ち、合成ごとに全ステップ分の表を作る。plan に重みを捨てられる前に作ること。"""

    def __init__(self, model, low_vram=False):
        reason = supports(model)
        if reason:
            raise RuntimeError(reason)
        self.device = next(model.parameters()).device
        self.dtype = next(model.cond_module.parameters()).dtype
        self.time_dim = model.cfg.timestep_embed_dim
        self.low_vram = bool(low_vram)
        with torch.inference_mode(False):
            cond = copy.deepcopy(model.cond_module).eval()
            weights = stack_weights(model)
            for p in cond.parameters():
                p.requires_grad_(False)
            if self.low_vram:
                cond = cond.to("cpu")
                weights = tuple(w.to("cpu").pin_memory() if self.device.type == "cuda" else w.to("cpu")
                                for w in weights)
            else:
                cond = cond.to(self.device)
        self._cond, self._weights = cond, weights
        self.bytes = sum(w.numel() * w.element_size() for w in weights)
        self.state = None
        self.check = os.environ.get(CHECK_ENV) == "1"

    # -- 表の計算
    def table(self, t_values):
        """t_values: (S,) の tensor（モデルの dtype）→ (S, 層, 2, 3, D)（GPU）。"""
        if not self.low_vram:
            return compute_table(self._cond, *self._weights, self.time_dim, t_values)
        with torch.inference_mode(False):
            cond = copy.deepcopy(self._cond).to(self.device)
            weights = [w.to(self.device, non_blocking=True) for w in self._weights]
        try:
            return compute_table(cond, *weights, self.time_dim, t_values)
        finally:
            del cond, weights

    # -- 1回の合成
    def begin(self, num_steps, mode="linear", sway_coeff=-1.0, independent=True):
        steps = t_schedule(int(num_steps), mode, sway_coeff, self.device)[:-1].to(self.dtype)
        table = self.table(steps)
        self.state = {"table": table, "steps": steps if self.check else None, "by_batch": {}, "n": 0,
                      "independent": bool(independent), "count": table.shape[0]}

    def end(self):
        self.state = None

    def mod_for(self, t):
        """このステップの mod: (B, 層, 2, 3, D)。t: (B,)。"""
        state, batch = self.state, t.shape[0]
        if state is not None and state["independent"] and state["n"] < state["count"]:
            # independent の CFG は 1 ステップ = 1 回の呼び出し。t を GPU から読み戻す同期を避けて数える。
            idx = state["n"]
            state["n"] += 1
            if self.check and not bool((t.to(state["steps"].dtype) == state["steps"][idx]).all()):
                raise RuntimeError(f"adaLN の表と t が合いません（step {idx}: t={t.tolist()}）")
            wide = state["by_batch"].get(batch)
            if wide is None:
                table = state["table"]
                wide = state["by_batch"][batch] = table[:, None].expand(
                    table.shape[0], batch, *table.shape[1:]).contiguous()
            return wide[idx]
        # 表に無い呼び出し（joint / alternating の CFG、合成の外からの呼び出し）は、その t で直接計算する。
        return self.table(t.to(self.dtype))


class AdalnAdapter:
    """plan を呼ぶ直前に kw['t'] を mod に差し替える。run_engine.Adapter の前に挟む。"""

    def __init__(self, adapter, precompute):
        self.adapter = adapter
        self.precompute = precompute

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    def __call__(self, **kw):
        kw["t"] = self.precompute.mod_for(kw["t"])
        return self.adapter(**kw)


def install_sample_hook(precompute, module=None):
    """sample_euler_rf_cfg の前に表を作る。module は irodori_tts.inference_runtime（テストでは差し替える）。"""
    if module is None:
        import irodori_tts.inference_runtime as module
    original = getattr(module.sample_euler_rf_cfg, "_adaln_original", module.sample_euler_rf_cfg)

    def sample(*args, **kwargs):
        if "num_steps" not in kwargs:
            return original(*args, **kwargs)
        precompute.begin(kwargs["num_steps"], kwargs.get("t_schedule_mode", "linear"),
                         kwargs.get("sway_coeff", -1.0),
                         str(kwargs.get("cfg_guidance_mode", "independent")).strip().lower() == "independent")
        try:
            return original(*args, **kwargs)
        finally:
            precompute.end()

    sample._adaln_original = original
    module.sample_euler_rf_cfg = sample
    return original
