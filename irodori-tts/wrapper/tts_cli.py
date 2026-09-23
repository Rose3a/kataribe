"""BF16 TensorRT Irodori-TTS wrapper with speaker cassette and HTTP server.

Single-process resident engine; switching speakers reuses the GPU plan and
reloads only the .speaker.safetensors embedding. Falls back to PyTorch when
CUDA is unavailable, the fallback plan cannot be loaded, or the user opts
in via IRODORI_BACKEND=torch.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import threading
from pathlib import Path
from typing import Iterable, Optional

import torch

# ---------------------------------------------------------------- defaults
# 既定値はこのリポジトリからの相対で決める。開発機の絶対パスは置かない。
# 別の場所に置いた資産を使うときは IRODORI_* 環境変数で明示する。
BOX_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMBED_DIRS: tuple[Path, ...] = (
    BOX_ROOT / "speakers",
    Path(__file__).resolve().parents[1] / "embeddings",
)
DEFAULT_PLAN_PATH = BOX_ROOT / "irodori-tts" / "bf16-fallback" / "fallback_bf16.plan"
DEFAULT_CHECKPOINT = BOX_ROOT / "models" / "model.safetensors"
DEFAULT_HF_HOME = BOX_ROOT / ".cache" / "huggingface"
DEFAULT_RUNTIME_DIR = BOX_ROOT / "runtime"
DEFAULT_PYTHON = BOX_ROOT / ".local" / "venv" / "Scripts" / "python.exe"
DEFAULT_CACHE_DIR = BOX_ROOT / ".cache" / "irodori-tts-cache"
MAX_HTTP_BODY = 16 * 1024 * 1024


def parse_radeon_precision(value: str | None = None) -> str:
    """Return the opt-in Radeon precision, defaulting safely to FP32."""
    value = (value if value is not None else os.environ.get(
        "IRODORI_RADEON_PRECISION", "fp32")).strip().lower()
    if value not in {"fp32", "fp16"}:
        raise ValueError(
            "Invalid IRODORI_RADEON_PRECISION={!r}; expected fp32 or fp16. "
            "Unset it to use the default FP32 Radeon path.".format(value))
    return value


# ---------------------------------------------------------------- helpers
def env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


def env_path_list(name: str) -> list[Path]:
    """Parse a path list env var. Accepts ';' or ',' as separator."""
    raw = os.environ.get(name)
    if not raw:
        return []
    sep = ";" if ";" in raw else ("," if "," in raw else None)
    if sep:
        return [Path(p).expanduser() for p in raw.split(sep) if p.strip()]
    return [Path(raw).expanduser()]


def _import_runtime():
    """Wire sys.path so both the irodori_tts package and the lab helpers
    are importable. Both live under IRODORI_RUNTIME_DIR (default:
    <repo>/runtime): the package in Irodori-TTS-Aratako/irodori_tts/,
    the lab helpers in trt-lab-20260905/. Both directories are added
    explicitly.
    """
    runtime_dir = env_path("IRODORI_RUNTIME_DIR", DEFAULT_RUNTIME_DIR)
    candidates = (
        runtime_dir / "Irodori-TTS-Aratako",
        runtime_dir / "Irodori-TTS-Aratako" / "source",
        runtime_dir / "trt-lab-20260905",
        runtime_dir / "trt-lab" / "repo",
        runtime_dir / "trt-lab",
    )
    # A launcher from an older installation may leave IRODORI_RUNTIME_DIR set
    # to a path that no longer exists. Always keep this checkout self-contained.
    local_runtime = Path(__file__).resolve().parents[2] / "runtime"
    candidates += (
        local_runtime / "trt-lab" / "repo",
        local_runtime / "trt-lab",
    )
    for candidate in candidates:
        if (candidate / "irodori_tts").is_dir():
            sys.path.insert(0, str(candidate))
    from irodori_tts.inference_runtime import (
        InferenceRuntime, RuntimeKey, SamplingRequest, save_wav,
    )
    return runtime_dir, InferenceRuntime, RuntimeKey, SamplingRequest, save_wav


def _import_engine():
    runtime_dir = env_path("IRODORI_RUNTIME_DIR", DEFAULT_RUNTIME_DIR)
    for candidate in (runtime_dir / "trt-lab-20260905", runtime_dir / "trt-lab"):
        if candidate.exists():
            sys.path.insert(0, str(candidate))
    from run_engine import Engine, Adapter
    return Engine, Adapter


def resolve_embed_dirs(extra: Iterable[Path] = ()) -> list[Path]:
    """Search order for *.speaker.safetensors:

    1. Any paths passed via the CLI (relative paths resolve against cwd).
    2. IRODORI_EMBED_DIRS / IRODORI_EMBED_DIR environment variables.
    3. The built-in defaults, relative to this repository:
       <repo>/speakers and <repo>/irodori-tts/embeddings.
    """
    seen: list[Path] = []
    seen_resolved: set[str] = set()
    candidates: list[Path] = []
    candidates.extend(Path(p).expanduser() for p in extra)
    candidates.extend(env_path_list("IRODORI_EMBED_DIRS"))
    candidates.extend(env_path_list("IRODORI_EMBED_DIR"))
    for raw in candidates:
        p = Path(raw)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        else:
            p = p.resolve()
        key = str(p).lower()
        if key in seen_resolved:
            continue
        seen_resolved.add(key)
        seen.append(p)
    for d in DEFAULT_EMBED_DIRS:
        p = d.resolve()
        key = str(p).lower()
        if key not in seen_resolved:
            seen_resolved.add(key)
            seen.append(p)
    return seen


# ---------------------------------------------------------------- cassette
class SpeakerCassette:
    """Lazy GPU cache of *.speaker.safetensors. Validates shape and dtype."""

    def __init__(self, embed_dirs: Iterable[Path]):
        self.embed_dirs = [Path(d).expanduser().resolve() for d in embed_dirs]
        self._cache: dict[str, torch.Tensor] = {}
        self._name_to_path: dict[str, Path] = {}
        self._name_to_root: dict[str, Path] = {}
        self._scan()

    def _scan(self) -> None:
        for d in self.embed_dirs:
            if not d.exists():
                continue
            for p in sorted(d.rglob("*.safetensors")):
                if p.stem not in self._name_to_path:
                    name = p.name[:-len(".speaker.safetensors")] if p.name.endswith(".speaker.safetensors") else p.stem
                    self._name_to_path[name] = p
                    self._name_to_root[name] = d
        # Convenience aliases: also expose each embedding under its
        # bare "fairy" name so callers can pass either "fairy" or
        # "fairy.speaker". The first one wins.
        aliases = {}
        for name in self._name_to_path:
            if name.endswith(".speaker"):
                alias = name[: -len(".speaker")]
                aliases[alias] = (self._name_to_path[name], self._name_to_root[name])
        self._name_to_path.update({name: path for name, (path, _root) in aliases.items()})
        self._name_to_root.update({name: root for name, (_path, root) in aliases.items()})

    @property
    def speakers(self) -> list[str]:
        return list(self._name_to_path.keys())

    @property
    def dirs(self) -> list[Path]:
        return list(self.embed_dirs)

    def path_for(self, name: str) -> Path:
        if name not in self._name_to_path:
            raise FileNotFoundError(
                f"speaker embedding not found: {name}\n"
                f"searched in: {[str(d) for d in self.embed_dirs]}\n"
                f"available: {self.speakers}"
            )
        return self._name_to_path[name]

    def folder_for(self, name: str) -> str | None:
        """Return the speaker's path below its configured root, if any.

        This is presentation metadata only: a file directly below ``speakers``
        has no folder, while every nested file below ``speakers/idolmaster`` is
        reported as ``idolmaster``.  The embedding lookup name remains flat so
        existing projects and API clients stay compatible.
        """
        path = self.path_for(name)
        root = self._name_to_root[name]
        try:
            parent = path.relative_to(root).parent
        except ValueError:
            return None
        # Speaker assets commonly live one level deeper, for example
        # ``speakers/idolmaster/haruka/haruka.speaker.safetensors``.  The
        # first path component is the user-facing series/group tab; the
        # per-speaker asset directory must not split that series into tabs.
        return parent.parts[0] if parent.parts else None

    def get(self, name: str) -> torch.Tensor:
        cached = self._cache.get(name)
        if cached is not None:
            return cached
        from safetensors import safe_open
        path = self.path_for(name)
        with safe_open(str(path), framework="pt", device="cpu") as f:
            shape = f.get_slice("speaker_embedding").get_shape()
            if len(shape) not in (2, 3) or shape[-1] != 768 or not 1 <= shape[-2] <= 64:
                raise ValueError(f"unsupported speaker shape {shape} from {path}")
            tensor = f.get_tensor("speaker_embedding").to(torch.bfloat16).contiguous()
        self._cache[name] = tensor
        return tensor

    def has(self, name: str) -> bool:
        return name in self._name_to_path

    def refresh(self) -> None:
        """Re-scan the search paths and drop the GPU cache."""
        self._cache.clear()
        self._name_to_path.clear()
        self._name_to_root.clear()
        self._scan()


# ---------------------------------------------------------------- engines
class TorchBackend:
    """CPU/GPU PyTorch reference backend. Always available."""

    name = "torch"

    def __init__(self, device: Optional[str] = None):
        os.environ.setdefault("HF_HOME", str(env_path("IRODORI_HF_HOME", DEFAULT_HF_HOME)))
        _, InferenceRuntime, RuntimeKey, _, _ = _import_runtime()
        checkpoint = env_path("IRODORI_CHECKPOINT", DEFAULT_CHECKPOINT)
        if device not in (None, "cpu", "cuda"):
            raise ValueError(f"unsupported torch device: {device}")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA backend requested, but torch.cuda.is_available() is False")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.runtime = InferenceRuntime.from_key(RuntimeKey(
            checkpoint=str(checkpoint),
            model_device=self.device,
            model_precision="bf16" if self.device == "cuda" else "fp32",
            codec_device=self.device,
            codec_precision="bf16" if self.device == "cuda" else "fp32",
            codec_deterministic_encode=True,
            codec_deterministic_decode=True,
            compile_model=False,
            compile_dynamic=False,
        ))

    def synthesize(self, request, speaker_tensor: torch.Tensor, out_wav: Path, log_fn=None) -> dict:
        if speaker_tensor is not None:
            request.ref_embed = self._speaker_path(speaker_tensor)
        result = self.runtime.synthesize(request, log_fn=log_fn)
        save_wav = _import_runtime()[4]
        save_wav(str(out_wav), result.audio, result.sample_rate)
        return {"backend": self.name, "wall_s": result.total_to_decode,
                "audio_s": result.audio.shape[-1] / result.sample_rate}

    @staticmethod
    def _speaker_path(tensor: torch.Tensor) -> str:
        from safetensors.torch import save_file
        cache_dir = env_path("IRODORI_CACHE_DIR", DEFAULT_CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / "active_speaker.speaker.safetensors"
        cpu = tensor.detach().to("cpu").contiguous()
        save_file({"speaker_embedding": cpu}, str(target))
        return str(target)


class TrtBackend:
    """Resident bf16 TensorRT engine, single plan, batch1 only."""

    name = "trt"

    def __init__(self, plan_path: Path):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "TrtBackend requires CUDA, but torch.cuda.is_available() is False "
                f"(torch={torch.__version__}, torch CUDA={torch.version.cuda!r}). "
                "Run setup_venv.bat again on the RTX machine, or install a "
                "CUDA-enabled PyTorch build and update the NVIDIA driver."
            )
        if not plan_path.exists():
            raise FileNotFoundError(plan_path)
        from tensorrt import __version__ as trt_version
        os.environ.setdefault("HF_HOME", str(env_path("IRODORI_HF_HOME", DEFAULT_HF_HOME)))
        _, InferenceRuntime, RuntimeKey, _, _ = _import_runtime()
        Engine, Adapter = _import_engine()

        self.plan_path = plan_path
        self.engine = Engine(plan_path)
        self.runtime = InferenceRuntime.from_key(RuntimeKey(
            checkpoint=str(env_path("IRODORI_CHECKPOINT", DEFAULT_CHECKPOINT)),
            model_device="cuda", model_precision="bf16",
            codec_device="cuda", codec_precision="bf16",
            codec_deterministic_encode=True,
            codec_deterministic_decode=True,
            compile_model=False,
            compile_dynamic=False,
        ))
        self.original_forward = self.runtime.model.forward_with_encoded_conditions
        self.adapter = Adapter(self.runtime.model, self.engine, compact=True)
        self.runtime.model.forward_with_encoded_conditions = self.adapter
        self.stream = torch.cuda.Stream()
        self.stream.wait_stream(torch.cuda.current_stream())
        self.trt_version = trt_version

    def synthesize(self, request, speaker_tensor: torch.Tensor, out_wav: Path, log_fn=None) -> dict:
        if speaker_tensor is not None:
            from safetensors.torch import save_file
            cache_dir = env_path("IRODORI_CACHE_DIR", DEFAULT_CACHE_DIR)
            cache_dir.mkdir(parents=True, exist_ok=True)
            target = cache_dir / "active_speaker.speaker.safetensors"
            save_file({"speaker_embedding": speaker_tensor.detach().to("cpu").contiguous()},
                      str(target))
            request.ref_embed = str(target)
        with torch.cuda.stream(self.stream):
            result = self.runtime.synthesize(request, log_fn=log_fn)
            torch.cuda.synchronize()
        save_wav = _import_runtime()[4]
        save_wav(str(out_wav), result.audio, result.sample_rate)
        return {"backend": self.name, "wall_s": result.total_to_decode,
                "audio_s": result.audio.shape[-1] / result.sample_rate}


class RadeonBackend:
    """Resident DirectML diffusion + ONNX codec backend for AMD Radeon."""

    name = "radeon"

    def __init__(self, project_root: Optional[Path] = None,
                 worker_python: Optional[Path] = None,
                 codec_path: Optional[Path] = None):
        # The wrapper lives in <box>\irodori-tts\wrapper; the worker needs the
        # box root so it can see runtime/, models/, work/ and speakers/.
        root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
        self.precision = parse_radeon_precision()
        worker_python = worker_python or Path(os.environ.get(
            "IRODORI_RADEON_PYTHON",
            str(root / "work" / "amd-dml-venv" / "Scripts" / "python.exe")))
        if codec_path is None:
            codec_name = ("IRODORI_RADEON_CODEC_FP16" if self.precision == "fp16"
                          else "IRODORI_RADEON_CODEC")
            codec_default = (root / "work" / "codec_decoder_fp16.onnx"
                             if self.precision == "fp16"
                             else root / "work" / "codec_decoder.onnx")
            codec_path = Path(os.environ.get(codec_name, str(codec_default)))
        if not worker_python.exists():
            raise FileNotFoundError(
                f"Radeon DirectML Python not found: {worker_python}. "
                "Set IRODORI_RADEON_PYTHON to the amd-dml-venv interpreter.")
        if not codec_path.exists():
            hint = ("Set IRODORI_RADEON_CODEC_FP16 to a validated FP16 ONNX codec, "
                    "or export one with tools/export_radeon_codec.py --precision fp16."
                    if self.precision == "fp16" else
                    "Set IRODORI_RADEON_CODEC to codec_decoder.onnx.")
            raise FileNotFoundError(
                f"Radeon ONNX codec not found: {codec_path}. "
                + hint)
        os.environ.setdefault("HF_HOME", str(env_path("IRODORI_HF_HOME", DEFAULT_HF_HOME)))
        _, InferenceRuntime, RuntimeKey, _, _ = _import_runtime()
        checkpoint = env_path("IRODORI_CHECKPOINT", DEFAULT_CHECKPOINT)
        self.runtime = InferenceRuntime.from_key(RuntimeKey(
            checkpoint=str(checkpoint), model_device="cpu", model_precision="fp32",
            codec_device="cpu", codec_precision="fp32",
            codec_deterministic_encode=True, codec_deterministic_decode=True,
            compile_model=False, compile_dynamic=False,
        ))
        from radeon_bridge import Bridge
        self.bridge = Bridge(worker_python, root, self.runtime, compact=True,
                             precision=self.precision, checkpoint=checkpoint)
        try:
            self.codec_info = self.bridge.load_codec(codec_path)
            if self.codec_info.get("precision") != self.precision:
                raise ValueError(
                    f"Radeon {self.precision} requires a matching ONNX codec; "
                    f"got {self.codec_info.get('precision')!r} from {codec_path}")
            if (self.codec_info["sample_rate"] != self.runtime.codec.sample_rate
                    or self.codec_info["hop_length"] != int(self.runtime.codec.model.hop_length)):
                raise ValueError("Radeon ONNX codec sample rate/hop length does not match the runtime")
        except Exception:
            self.bridge.close()
            raise
        self.runtime.model.forward_with_encoded_conditions = self.bridge.forward
        # Reference audio encoding remains on CPU; waveform reconstruction
        # uses the resident variable-length ONNX decoder on the Radeon GPU.
        self.runtime.codec.decode_latent = self.bridge.decode

        # Reduce padding transferred over the CPU/DirectML pipe.
        seen = set()
        for tokenizer in (self.runtime.tokenizer, self.runtime.caption_tokenizer):
            if tokenizer is None or id(tokenizer) in seen:
                continue
            seen.add(id(tokenizer))
            original_encode = tokenizer.batch_encode
            def compact_encode(*pos, _original=original_encode, **kw):
                ids, mask = _original(*pos, **kw)
                positions = mask.any(dim=0).nonzero()
                length = int(positions[-1].item()) + 1 if positions.numel() else 1
                return ids[:, :length].contiguous(), mask[:, :length].contiguous()
            tokenizer.batch_encode = compact_encode

    def synthesize(self, request, speaker_tensor: torch.Tensor, out_wav: Path, log_fn=None) -> dict:
        if speaker_tensor is not None:
            request.ref_embed = TorchBackend._speaker_path(speaker_tensor)
            request.no_ref = False
        self.bridge.reset_stats()
        result = self.runtime.synthesize(request, log_fn=log_fn)
        save_wav = _import_runtime()[4]
        save_wav(str(out_wav), result.audio, result.sample_rate)
        return {"backend": self.name, "wall_s": result.total_to_decode,
                "audio_s": result.audio.shape[-1] / result.sample_rate,
                "bridge": self.bridge.stats}

    def close(self):
        self.bridge.close()


# ---------------------------------------------------------------- wrapper
class IrodoriTTS:
    """Top-level wrapper. Resolves backend at construction time."""

    def __init__(self,
                 backend: str = "auto",
                 embed_dirs: Optional[Iterable[Path]] = None,
                 plan_path: Optional[Path] = None,
                 allow_no_speaker: bool = False):
        self.embed_dirs = list(embed_dirs) if embed_dirs is not None else resolve_embed_dirs()
        self.cassette = SpeakerCassette(self.embed_dirs)
        if not self.cassette.speakers and not allow_no_speaker:
            raise FileNotFoundError(
                f"no *.speaker.safetensors in {self.embed_dirs}"
            )
        if backend == "auto":
            backend = self._select_backend(plan_path)
        if backend == "trt":
            self.backend: object = TrtBackend(plan_path or DEFAULT_PLAN_PATH)
        elif backend == "cpu":
            self.backend = TorchBackend(device="cpu")
        elif backend == "cuda":
            self.backend = TorchBackend(device="cuda")
        elif backend == "radeon":
            self.backend = RadeonBackend(project_root=Path(__file__).resolve().parents[2])
        else:
            self.backend = TorchBackend()
        self.default_speaker = self._pick_default_speaker()

    @staticmethod
    def _select_backend(plan_path: Optional[Path]) -> str:
        if os.environ.get("IRODORI_BACKEND") == "torch":
            return "torch"
        if os.environ.get("IRODORI_BACKEND") == "trt":
            return "trt"
        if not torch.cuda.is_available():
            return "torch"
        if plan_path is None:
            plan_path = DEFAULT_PLAN_PATH
        if not plan_path.exists():
            return "torch"
        return "trt"

    def _pick_default_speaker(self) -> Optional[str]:
        for preferred in (
            "tsukuyomi",
            "ureshun",
            "unleashguang",
            "fairy",
            "kugimiya",
        ):
            if self.cassette.has(preferred):
                return preferred
        return self.cassette.speakers[0] if self.cassette.speakers else None

    def speakers(self) -> list[str]:
        return self.cassette.speakers

    def search_dirs(self) -> list[str]:
        return [str(d) for d in self.cassette.dirs]

    def refresh_cassette(self) -> None:
        self.cassette.refresh()

    def synthesize(self,
                   text: str,
                   speaker: Optional[str] = None,
                   out_wav: Optional[Path] = None,
                   seed: Optional[int] = 1001,
                   num_steps: int = 8,
                   seconds: Optional[float] = None,
                   duration_scale: float = 1.0,
                   cfg_scale_text: float = 3.0,
                   cfg_scale_speaker: float = 5.0,
                   cfg_scale_caption: float = 3.0,
                   caption: Optional[str] = None,
                   caption_strength: float = 1.0,
                   reference_strength: float = 1.0,
                   speaker_strength: float = 1.0,
                   secondary_speaker: Optional[str] = None,
                   secondary_speaker_strength: float = 0.5,
                   additional_speakers: Optional[list[tuple[str, float]]] = None,
                   ref_wav: Optional[str] = None,
                   t_schedule_mode: str = "sway",
                   sway_coeff: float = -1.0,
                   log_fn=None,
                   speaker_tensor_override: Optional[torch.Tensor] = None) -> dict:
        if not text.strip():
            raise ValueError("text must not be empty")
        speaker_name = self.default_speaker if speaker is None else speaker
        if ref_wav is not None and speaker_tensor_override is not None:
            raise ValueError("reference audio and speaker mix cannot be combined")
        if additional_speakers is None:
            additional_speakers = ([] if secondary_speaker is None else
                                   [(secondary_speaker, secondary_speaker_strength)])
        if len(additional_speakers) > 3:
            raise ValueError("追加話者は最大3人までです")
        if ref_wav is not None and additional_speakers:
            raise ValueError("音声リファレンスと追加話者は同時に使えません")
        speaker_strength = float(speaker_strength)
        if not math.isfinite(speaker_strength) or not 0 <= speaker_strength <= 1:
            raise ValueError("speaker strength must be between 0 and 1")
        speaker_tensor = (speaker_tensor_override if speaker_tensor_override is not None
                          else None if ref_wav or speaker_strength == 0
                          else (self.cassette.get(speaker_name) if speaker_name else None))
        if speaker_tensor is not None:
            speaker_tensor = None if speaker_strength == 0 else speaker_tensor * speaker_strength
        def token_rows(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.ndim == 3 and tensor.shape[0] == 1:
                tensor = tensor[0]
            if tensor.ndim != 2 or tensor.shape[1] != 768:
                raise ValueError("speaker cassette has an unsupported shape")
            return tensor

        for additional_name, raw_strength in additional_speakers:
            strength = float(raw_strength)
            if not math.isfinite(strength) or not 0 <= strength <= 1:
                raise ValueError("additional speaker strength must be between 0 and 1")
            if strength == 0:
                continue
            second = token_rows(self.cassette.get(additional_name))
            if speaker_tensor is None:
                speaker_tensor = second * strength
                continue
            primary = token_rows(speaker_tensor)
            rows = max(primary.shape[0], second.shape[0])
            combined = torch.zeros((rows, 768), dtype=torch.float32,
                                   device=primary.device)
            combined[:primary.shape[0]] += primary.float()
            combined[:second.shape[0]] += second.to(
                device=primary.device, dtype=torch.float32) * strength
            speaker_tensor = combined.to(primary.dtype)
        out_wav = Path(out_wav) if out_wav else Path(f"outputs/{speaker_name}_{seed}.wav")
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        _, _, _, SamplingRequest, _ = _import_runtime()
        request = SamplingRequest(
            text=text, caption=caption,
            caption_strength=float(caption_strength),
            reference_strength=float(reference_strength),
            # Empty string is not the same as an omitted embedding to the
            # runtime; use None until a cassette is actually staged below.
            ref_wav=ref_wav, ref_embed=None, no_ref=False,
            num_candidates=1, decode_mode="batch",
            seconds=seconds, duration_scale=float(duration_scale),
            num_steps=num_steps, t_schedule_mode=t_schedule_mode, sway_coeff=float(sway_coeff),
            cfg_scale_text=float(cfg_scale_text),
            cfg_scale_speaker=float(cfg_scale_speaker), cfg_scale_caption=float(cfg_scale_caption),
            seed=seed,
        )
        if ref_wav is not None:
            pass
        elif speaker_tensor is not None:
            from safetensors.torch import save_file
            cache_dir = env_path("IRODORI_CACHE_DIR", DEFAULT_CACHE_DIR)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / "active_speaker.speaker.safetensors"
            save_file({"speaker_embedding": speaker_tensor.detach().to("cpu").contiguous()},
                      str(cache_file))
            request.ref_embed = str(cache_file)
        else:
            request.no_ref = True
        start = time.perf_counter()
        result = self.backend.synthesize(request, speaker_tensor, out_wav, log_fn=log_fn)
        wall = time.perf_counter() - start
        return {
            "backend": result["backend"],
            "wall_s": wall,
            "engine_wall_s": result["wall_s"],
            "audio_s": result["audio_s"],
            "speaker": speaker_name,
            "seed": seed,
            "out_wav": str(out_wav),
        }

    def close(self) -> None:
        close = getattr(self.backend, "close", None)
        if close is not None:
            close()


# ---------------------------------------------------------------- CLI / serve
def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Irodori bf16 trt + speaker cassette")
    ap.add_argument("--mode", choices=("cli", "serve"), default="cli")
    ap.add_argument("--speaker", help="speaker name; required in cli mode")
    ap.add_argument("--text", help="text to synthesise; required in cli mode")
    ap.add_argument("--text-file", type=Path,
                    help="UTF-8 file with one utterance per non-empty line")
    ap.add_argument("--out", type=Path, help="output wav path; default outputs/<name>_<seed>.wav")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs"))
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--num-steps", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--backend", choices=("auto", "trt", "torch", "cpu", "cuda", "radeon"), default="auto")
    ap.add_argument("--embed-dir", action="append", type=Path, default=[],
                    help="additional search root for *.speaker.safetensors "
                         "(relative paths resolve against cwd); repeat to add more")
    ap.add_argument("--list-speakers", action="store_true")
    ap.add_argument("--list-search-dirs", action="store_true",
                    help="print resolved search roots and exit")
    ap.add_argument("--refresh-cassette", action="store_true",
                    help="re-scan the search paths and drop the GPU cache before listing")
    ap.add_argument("--port", type=int, default=8765, help="port for serve mode")
    return ap


def _build_tts(args: argparse.Namespace) -> IrodoriTTS:
    embed_dirs = resolve_embed_dirs(args.embed_dir) if args.embed_dir else None
    tts = IrodoriTTS(backend=args.backend, embed_dirs=embed_dirs)
    if args.refresh_cassette:
        tts.refresh_cassette()
    return tts


def _print_speaker_index(tts: IrodoriTTS) -> None:
    print("search roots:")
    for d in tts.search_dirs():
        print(f"  - {d}")
    print("available speakers:")
    for name in tts.speakers():
        mark = " (default)" if name == tts.default_speaker else ""
        print(f"  - {name}{mark}")


def _run_cli(args: argparse.Namespace) -> int:
    tts = _build_tts(args)
    if args.list_search_dirs:
        print("search roots:")
        for d in tts.search_dirs():
            print(f"  - {d}")
        return 0
    if args.list_speakers or (not args.speaker and not args.text and not args.text_file):
        _print_speaker_index(tts)
        return 0
    if not args.speaker:
        print("error: --speaker is required (use --list-speakers to see options)",
              file=sys.stderr)
        return 2
    if not args.text and not args.text_file:
        print("error: --text or --text-file is required", file=sys.stderr)
        return 2
    if args.text_file:
        texts = [t for t in args.text_file.read_text(encoding="utf-8-sig").splitlines()
                 if t.strip()]
    else:
        texts = [args.text]
    for index, text in enumerate(texts):
        if args.out is not None:
            out_wav = args.out
        elif len(texts) > 1:
            out_wav = args.out_dir / f"{args.speaker}_{args.seed}_{index:03}.wav"
        else:
            out_wav = args.out_dir / f"{args.speaker}_{args.seed}.wav"
        result = tts.synthesize(
            text=text, speaker=args.speaker, out_wav=out_wav,
            seed=args.seed, num_steps=args.num_steps, seconds=args.seconds,
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


def _run_serve(args: argparse.Namespace, tts: IrodoriTTS) -> int:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        slots = threading.BoundedSemaphore(2)
        def log_message(self, fmt, *a):
            sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % a))

        def do_GET(self):
            if self.path == "/speakers":
                payload = json.dumps({
                    "speakers": tts.speakers(),
                    "default": tts.default_speaker,
                    "backend": tts.backend.name,
                    "search_dirs": tts.search_dirs(),
                }, ensure_ascii=False).encode("utf-8")
                self._send(200, payload)
                return
            if self.path == "/refresh":
                tts.refresh_cassette()
                payload = json.dumps({"speakers": tts.speakers()},
                                     ensure_ascii=False).encode("utf-8")
                self._send(200, payload)
                return
            self._send(404, json.dumps({"error": "not found"}, ensure_ascii=False).encode("utf-8"))

        def do_POST(self):
            length_header = self.headers.get("Content-Length")
            if not length_header:
                self._send(411, json.dumps({"error": "Content-Length required"},
                                           ensure_ascii=False).encode("utf-8"))
                return
            try:
                length = int(length_header)
            except ValueError:
                self._send(400, json.dumps({"error": "invalid Content-Length"},
                                           ensure_ascii=False).encode("utf-8"))
                return
            if length < 0 or length > MAX_HTTP_BODY:
                self._send(413, json.dumps({"error": "request body is too large"},
                                           ensure_ascii=False).encode("utf-8"))
                return
            try:
                body = self.rfile.read(length).decode("utf-8")
            except UnicodeDecodeError as e:
                self._send(400, json.dumps({"error": f"body is not valid utf-8: {e}"},
                                           ensure_ascii=False).encode("utf-8"))
                return
            try:
                req = json.loads(body)
            except json.JSONDecodeError as e:
                self._send(400, json.dumps({"error": f"invalid json: {e}"},
                                           ensure_ascii=False).encode("utf-8"))
                return
            try:
                if not self.slots.acquire(timeout=0.1):
                    self._send(429, json.dumps({"error": "synthesis queue is full"},
                                               ensure_ascii=False).encode("utf-8"))
                    return
                try:
                    result = tts.synthesize(
                    text=req["text"],
                    speaker=req.get("speaker"),
                    out_wav=Path(req["out_wav"]) if req.get("out_wav") else None,
                    seed=req.get("seed", 1001),
                    num_steps=req.get("num_steps", 8),
                    seconds=req.get("seconds"),
                    )
                finally:
                    self.slots.release()
            except (KeyError, ValueError, FileNotFoundError) as e:
                self._send(400, json.dumps({"error": str(e)},
                                           ensure_ascii=False).encode("utf-8"))
                return
            payload = json.dumps(result, ensure_ascii=False).encode("utf-8")
            self._send(200, payload)

        def _send(self, code: int, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"serving on http://127.0.0.1:{args.port}  backend={tts.backend.name}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    tts = _build_tts(args)
    try:
        if args.mode == "serve":
            return _run_serve(args, tts)
        return _run_cli(args)
    finally:
        tts.close()


if __name__ == "__main__":
    sys.exit(main())
