"""Resident DirectML diffusion + ONNX codec worker for AMD Radeon."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys
import time
import traceback
import hashlib

from radeon_bridge import send, receive, pack, unpack

parser = argparse.ArgumentParser()
parser.add_argument("--project", required=True)
parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
parser.add_argument("--checkpoint", required=True, type=Path)
args = parser.parse_args()
wire_out = sys.stdout.buffer
sys.stdout = sys.stderr
root = Path(args.project)
checkpoint = args.checkpoint.resolve()
if root not in checkpoint.parents or not checkpoint.is_file():
    raise SystemExit("checkpoint must be a file inside project root")
checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
sys.path.insert(0, str(root / "runtime" / "trt-lab" / "repo"))

try:
    import torch
    import torch_directml
    from safetensors import safe_open
    from irodori_tts import model as im
    from irodori_tts.config import ModelConfig

    torch.set_num_threads(2)

    def real_rope(x, freqs):
        pair = x.float().reshape(*x.shape[:3], -1, 2)
        c = freqs[None, :, None, :, 0]
        s = freqs[None, :, None, :, 1]
        a, b = pair[..., 0], pair[..., 1]
        return torch.stack((a * c - b * s, a * s + b * c), dim=-1).reshape_as(x).type_as(x)

    im.apply_rotary_emb = real_rope

    if args.precision == "fp16":
        # The model source normally promotes these numerically sensitive helpers
        # to FP32.  The Radeon opt-in path deliberately keeps their activations
        # in FP16; masks remain bool and are never cast to FP16.
        def fp16_timestep(timestep, dim):
            # 角度は 1000*t まで大きくなり、FP16（1000 付近の刻みは 0.5）では cos/sin が意味を失う。
            # [batch, dim] の小さな計算なので FP32 で求めてから FP16 にする。
            half = dim // 2
            freqs = 1000.0 * torch.exp(
                -torch.log(torch.tensor(10000.0, device=timestep.device, dtype=torch.float32))
                * torch.arange(half, device=timestep.device, dtype=torch.float32) / half)
            args_ = timestep[:, None].float() * freqs[None, :]
            return torch.cat([torch.cos(args_), torch.sin(args_)], dim=-1).to(timestep.dtype)

        def fp16_rope(x, freqs):
            pair = x.reshape(*x.shape[:3], -1, 2)
            c = freqs[None, :, None, :, 0]
            s = freqs[None, :, None, :, 1]
            a, b = pair[..., 0], pair[..., 1]
            return torch.stack((a * c - b * s, a * s + b * c), dim=-1).reshape_as(x)

        # 残差の値は 256 を超えることがあり、FP16 の x*x はあふれて ∞ になる（正規化後が 0 になり、
        # 出力が out_proj のバイアスだけになって音声にならない）。二乗平均だけは FP32 で求める。
        def fp16_inv_rms(x, eps):
            return torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps).to(x.dtype)

        def fp16_rms(self, x):
            return (x * fp16_inv_rms(x, self.eps) * self.weight).to(dtype=x.dtype)

        def fp16_adaln(self, x, cond_embed):
            # model.LowRankAdaLN と同じく SiLU は down の前（逆にすると別の関数になり、音声にならない）。
            shift, scale, gate = cond_embed.chunk(3, dim=-1)
            shift = self.shift_up(self.shift_down(torch.nn.functional.silu(shift))) + shift
            scale = self.scale_up(self.scale_down(torch.nn.functional.silu(scale))) + scale
            gate = self.gate_up(self.gate_down(torch.nn.functional.silu(gate))) + gate
            x = x * fp16_inv_rms(x, self.eps)
            return x * (1.0 + scale) + shift, torch.tanh(gate)

        im.get_timestep_embedding = fp16_timestep
        im.apply_rotary_emb = fp16_rope
        im.RMSNorm.forward = fp16_rms
        im.LowRankAdaLN.forward = fp16_adaln

    class Core(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            self.cond_module = torch.nn.Sequential(
                torch.nn.Linear(cfg.timestep_embed_dim, cfg.model_dim, bias=False), torch.nn.SiLU(),
                torch.nn.Linear(cfg.model_dim, cfg.model_dim, bias=False), torch.nn.SiLU(),
                torch.nn.Linear(cfg.model_dim, cfg.model_dim * 3, bias=False))
            # MeanFlow 蒸留モデルだけが持つ delta_t 条件付け（RF モデルでは None のまま）。
            self.delta_cond_module: torch.nn.Module | None = None
            self.in_proj = torch.nn.Linear(cfg.patched_latent_dim, cfg.model_dim)
            self.blocks = torch.nn.ModuleList(im.DiffusionBlock(cfg) for _ in range(cfg.num_layers))
            self.out_norm = im.RMSNorm(cfg.model_dim, eps=cfg.norm_eps)
            self.out_proj = torch.nn.Linear(cfg.model_dim, cfg.patched_latent_dim)

        def forward(self, x, t, context, freqs, latent_mask, delta=None):
            cond = self.cond_module(im.get_timestep_embedding(t, self.cfg.timestep_embed_dim))[:, None, :]
            if self.delta_cond_module is not None and delta is not None:
                delta_embed = im.get_timestep_embedding(
                    delta, self.cfg.timestep_embed_dim).to(dtype=cond.dtype)
                cond = cond + self.delta_cond_module(delta_embed)[:, None, :]
            h = self.in_proj(x)
            for i, block in enumerate(self.blocks):
                # context_kv があれば条件の state は使われない。K/V の数はモデルの条件で変わる
                # （500M v3 は caption が無く4つ）ので、位置で取り出さない。
                h = block(x=h, cond_embed=cond, text_state=None, text_mask=context["text_mask"],
                          speaker_state=None, speaker_mask=context["speaker_mask"],
                          caption_state=None, caption_mask=context["caption_mask"],
                          freqs_cis=freqs, self_mask=latent_mask, context_kv=context["kv"][i])
            return self.out_proj(self.out_norm(h))

    names = [torch_directml.device_name(i).rstrip("\0") for i in range(torch_directml.device_count())]
    idx = next(i for i, name in enumerate(names) if "AMD" in name.upper())
    device = torch_directml.device(idx)
    with safe_open(str(checkpoint), framework="pt") as f:
        config = json.loads(f.metadata()["config_json"])
        cfg = ModelConfig(**{k: v for k, v in config.items()
                             if k in {field.name for field in fields(ModelConfig)}})
        prefixes = ("cond_module.", "delta_cond_module.", "in_proj.", "blocks.", "out_norm.", "out_proj.")
        # DirectML は BFloat16 を扱えず、プロセスごと落ちる（Blockwise の bf16 版など）。FP32 で読む。
        state = {k: (t.float() if t.dtype == torch.bfloat16 else t)
                 for k in f.keys() if k.startswith(prefixes) for t in (f.get_tensor(k),)}
    with torch.device("meta"):
        core = Core(cfg)
    if any(key.startswith("delta_cond_module.") for key in state):
        core.delta_cond_module = torch.nn.Sequential(
            torch.nn.Linear(cfg.timestep_embed_dim, cfg.model_dim, bias=False), torch.nn.SiLU(),
            torch.nn.Linear(cfg.model_dim, cfg.model_dim, bias=False), torch.nn.SiLU(),
            torch.nn.Linear(cfg.model_dim, cfg.model_dim * 3, bias=False))
    core.load_state_dict(state, strict=True, assign=True)
    core = core.eval().to(device)
    if args.precision == "fp16":
        core = core.half()
    del state
    contexts = {}
    freq_cache = {}
    codec = None
    send(wire_out, {"torch": torch.__version__, "adapter": names[idx], "device": str(device),
                    "parameters": sum(p.numel() for p in core.parameters()),
                    "layers": len(core.blocks), "precision": args.precision,
                    "checkpoint_sha256": checkpoint_sha256})

    with torch.no_grad():
        while True:
            message = receive(sys.stdin.buffer)
            op = message["op"]
            if op == "quit":
                break
            if op == "reset":
                contexts.clear()
                send(wire_out, {"ok": True})
            elif op == "load_codec_onnx":
                from radeon_codec import DirectMLCodec, ort
                codec = DirectMLCodec(message["path"], idx, precision=args.precision)
                send(wire_out, {"ort": ort.__version__, "providers": codec.session.get_providers(),
                                "sample_rate": codec.sample_rate, "hop_length": codec.hop_length,
                                "precision": codec.precision})
            elif op == "decode":
                if codec is None:
                    raise RuntimeError("Radeon ONNX codec is not loaded")
                # The exported decoder accepts [batch, frames, 32] and
                # performs the DACVAE layout transpose inside Decoder.forward.
                latent = unpack(message["latent"]).to(dtype=torch.float16 if args.precision == "fp16" else torch.float32).contiguous().numpy()
                begin = time.perf_counter()
                audio = codec.decode(latent)
                send(wire_out, {"audio": pack(torch.from_numpy(audio)),
                                "compute_s": time.perf_counter() - begin})
            elif op == "context":
                c = message["context"]
                context = {k: unpack(c[k]).to(device=device, dtype=torch.bool) if c[k] is not None else None
                           for k in ("text_mask", "speaker_mask", "caption_mask")}
                context["kv"] = [tuple(unpack(t).to(device=device, dtype=torch.float16 if args.precision == "fp16" else torch.float32) for t in layer) for layer in c["kv"]]
                contexts[message["key"]] = context
                send(wire_out, {"ok": True})
            elif op == "forward":
                work_dtype = torch.float16 if args.precision == "fp16" else torch.float32
                x = unpack(message["x"]).to(device=device, dtype=work_dtype)
                t = unpack(message["t"]).to(device=device, dtype=work_dtype)
                delta = message.get("delta")
                if delta is not None:
                    delta = unpack(delta).to(device=device, dtype=work_dtype)
                mask = unpack(message["latent_mask"])
                if mask is not None:
                    mask = mask.to(device=device, dtype=torch.bool)
                length = x.shape[1]
                if length not in freq_cache:
                    freq_cache[length] = torch.view_as_real(
                        im.precompute_freqs_cis(cfg.model_dim // cfg.num_heads, length)).to(device=device, dtype=work_dtype)
                begin = time.perf_counter()
                output = core(x, t, contexts[message["key"]], freq_cache[length], mask, delta).float().cpu()
                send(wire_out, {"output": pack(output), "compute_s": time.perf_counter() - begin})
            else:
                raise ValueError(op)
except Exception:
    error = traceback.format_exc()
    print(error, file=sys.stderr, flush=True)
    send(wire_out, {"error": error})
