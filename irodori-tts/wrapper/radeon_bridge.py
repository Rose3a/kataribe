"""Parent side of the resident DirectML worker used by Radeon Engine."""
from __future__ import annotations

import pickle
import struct
import subprocess
import time
import hashlib
from pathlib import Path


def send(stream, obj):
    payload = pickle.dumps(obj, protocol=4)
    stream.write(struct.pack("<Q", len(payload)))
    stream.write(payload)
    stream.flush()


def _read_exact(stream, count):
    chunks = []
    while count:
        part = stream.read(count)
        if not part:
            raise EOFError("Radeon worker pipe closed")
        chunks.append(part)
        count -= len(part)
    return b"".join(chunks)


def receive(stream):
    count = struct.unpack("<Q", _read_exact(stream, 8))[0]
    return pickle.loads(_read_exact(stream, count))


def pack(tensor):
    if tensor is None:
        return None
    arr = tensor.detach().cpu().contiguous().numpy()
    return (str(arr.dtype), arr.shape, arr.tobytes())


def unpack(obj):
    import numpy as np
    import torch
    if obj is None:
        return None
    dtype, shape, data = obj
    return torch.from_numpy(np.frombuffer(data, dtype=dtype).reshape(shape).copy())


class Bridge:
    def __init__(self, python: Path, root: Path, runtime, compact: bool = True,
                 worker_script: Path | None = None, precision: str = "fp32",
                 checkpoint: Path | None = None):
        self.compact = compact
        self.runtime = runtime
        self.precision = precision
        self.checkpoint = Path(checkpoint).resolve() if checkpoint else None
        if self.checkpoint is None or not self.checkpoint.is_file():
            raise RuntimeError("Radeon checkpoint identity is not proven")
        self.checkpoint_sha256 = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        worker_script = worker_script or Path(__file__).with_name("radeon_worker.py")
        log_path = Path(__file__).with_name("radeon_worker.log")
        self.log = log_path.open("ab")
        self.proc = subprocess.Popen(
            [str(python), str(worker_script), "--project", str(root),
             "--precision", precision, "--checkpoint", str(self.checkpoint)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
        )
        try:
            self.info = receive(self.proc.stdout)
        except Exception:
            self.close()
            raise RuntimeError("Radeon worker failed during startup")
        if "error" in self.info:
            self.close()
            raise RuntimeError(self.info["error"])
        if self.info.get("precision") != precision:
            self.close()
            raise RuntimeError(
                f"Radeon worker precision mismatch: requested {precision}, "
                f"worker reported {self.info.get('precision')!r}")
        if self.info.get("checkpoint_sha256") != self.checkpoint_sha256:
            self.close()
            raise RuntimeError("Radeon worker checkpoint identity mismatch")
        self.contexts = {}
        self.stats = {}

    def command(self, obj):
        try:
            send(self.proc.stdin, obj)
            result = receive(self.proc.stdout)
        except (EOFError, OSError, struct.error, ValueError) as exc:
            self.close()
            raise RuntimeError("Radeon worker pipe failed") from exc
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def reset_stats(self):
        self.contexts = {}
        self.stats = {"calls": 0, "roundtrip_s": 0.0,
                      "worker_compute_s": 0.0, "codec_s": 0.0}
        self.command({"op": "reset"})

    def forward(self, **kw):
        start = time.perf_counter()
        cache = kw["context_kv_cache"]
        if cache is None:
            raise ValueError("Radeon bridge requires context_kv_cache=True")
        key = id(cache)
        if key not in self.contexts:
            self.contexts[key] = cache
            masks = [kw.get(k) for k in ("text_mask", "speaker_mask", "caption_mask")]
            lengths = []
            for mask in masks:
                if mask is None:
                    lengths.append(None)
                elif self.compact:
                    positions = mask.any(dim=0).nonzero()
                    lengths.append(int(positions[-1].item()) + 1 if positions.numel() else 1)
                else:
                    lengths.append(mask.shape[1])
            context = {k: pack(mask[:, :n] if mask is not None else None)
                       for k, mask, n in zip(("text_mask", "speaker_mask", "caption_mask"), masks, lengths)}
            # K/V は text、[speaker]、[caption] の順に K と V が並ぶ。無い条件は詰められているので、
            # モデルの条件の有無から各 K/V に対応するマスクの長さを決める。
            attention = self.runtime.model.blocks[0].attention
            kv_lengths = [lengths[0]]
            if attention.has_speaker_condition:
                kv_lengths.append(lengths[1])
            if attention.has_caption_condition:
                kv_lengths.append(lengths[2])
            context["kv"] = [[pack(t[:, :kv_lengths[i // 2]]) for i, t in enumerate(layer)]
                              for layer in cache]
            self.command({"op": "context", "key": key, "context": context})
        result = self.command({"op": "forward", "key": key, "x": pack(kw["x_t"]),
                               "t": pack(kw["t"]), "latent_mask": pack(kw.get("latent_mask")),
                               # MeanFlow サンプラーだけが delta_t を渡す（RF は None）。
                               "delta": pack(kw["delta_t"]) if kw.get("delta_t") is not None else None})
        output = unpack(result["output"])
        self.stats["calls"] += 1
        self.stats["roundtrip_s"] += time.perf_counter() - start
        self.stats["worker_compute_s"] += result["compute_s"]
        if not output.isfinite().all():
            raise RuntimeError("Non-finite Radeon diffusion output")
        return output.float()

    def load_codec(self, path: Path):
        return self.command({"op": "load_codec_onnx", "path": str(Path(path).resolve())})

    def decode(self, latent):
        # DACVAECodec uses [B, frames, D], while some runtime paths expose
        # the native codec layout [B, D, frames]. Normalize before ONNX.
        if latent.ndim != 3:
            raise ValueError(f"Radeon codec latent must be rank-3, got {tuple(latent.shape)}")
        if latent.shape[-1] != 32 and latent.shape[1] == 32:
            latent = latent.transpose(1, 2).contiguous()
        wire_latent = latent if self.precision == "fp16" else latent.float()
        result = self.command({"op": "decode", "latent": pack(wire_latent)})
        audio = unpack(result["audio"])
        self.stats["codec_s"] = self.stats.get("codec_s", 0.0) + result["compute_s"]
        if not audio.isfinite().all():
            raise RuntimeError("Non-finite Radeon codec output")
        return audio

    def close(self):
        if getattr(self, "proc", None) is not None and self.proc.poll() is None:
            try:
                send(self.proc.stdin, {"op": "quit"})
                self.proc.wait(timeout=15)
            except Exception:
                self.proc.kill()
                self.proc.wait(timeout=5)
        self.proc = None
        if getattr(self, "log", None) is not None:
            self.log.close()
