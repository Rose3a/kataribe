"""TensorRT FP16 build of the DACVAE decoder (latent -> waveform).

The PyTorch codec runs in BF16 and is the largest single stage of a TRT
synthesis.  This module builds a variable-length FP16 TensorRT plan of the same
decoder in an isolated child process, verifies it against an FP32 PyTorch
reference, and swaps it into ``runtime.codec.decode_latent``.  A plan is only
published when it is at least as close to FP32 as the BF16 PyTorch path it
replaces, so the switch never lowers quality.  Anything unusual at run time
(shape outside the profile, non-finite output) falls back to PyTorch.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from trt_cache import cache_key, digest

BOX = Path(__file__).resolve().parents[2]
PLAN_NAME = 'codec_fp16.plan'
# 30 s at 25 latent frames/s, the runtime's max_seconds.
MAX_FRAMES = 750
OPT_FRAMES = 200
# Verification lengths; the 750-frame case only checks shape/finiteness to
# keep the FP32 reference within small-GPU memory.
VERIFY_FRAMES = (1, 25, 200, 400)
# Allowed SNR shortfall against the BF16 PyTorch codec on the same input.
SNR_MARGIN_DB = 1.0


def _runtime_paths():
    repo = BOX / 'runtime' / 'trt-lab' / 'repo'
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def codec_weights(repo_id=None):
    """Local path of the codec weights the runtime loads (same HF cache)."""
    os.environ.setdefault('HF_HOME', str(BOX / '.cache' / 'huggingface'))
    _runtime_paths()
    from huggingface_hub import hf_hub_download
    from irodori_tts.inference_runtime import RuntimeKey
    repo_id = repo_id or RuntimeKey.__dataclass_fields__['codec_repo'].default
    if Path(repo_id).exists():
        return Path(repo_id)
    try:
        # The runtime has already fetched it; do not ask the Hub on every start.
        return Path(hf_hub_download(repo_id=repo_id, filename='weights.pth',
                                    local_files_only=True))
    except Exception:  # noqa: BLE001 - not cached yet
        return Path(hf_hub_download(repo_id=repo_id, filename='weights.pth'))


def _weights_digest(path):
    """sha256 of the 430 MB weights, remembered by size/mtime between starts."""
    memo = BOX / '.cache' / 'trt-codec' / 'weights-digest.json'
    stat = path.stat()
    stamp = dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    try:
        record = json.loads(memo.read_text(encoding='utf-8'))
        if {k: record.get(k) for k in stamp} == stamp:
            return record['sha256']
    except (OSError, ValueError, KeyError):
        pass
    value = digest(path)
    memo.parent.mkdir(parents=True, exist_ok=True)
    memo.write_text(json.dumps(dict(stamp, sha256=value)), encoding='utf-8')
    return value


def cache_identity(weights):
    import torch
    import tensorrt as trt
    from importlib.metadata import version
    if not torch.cuda.is_available():
        raise RuntimeError('TensorRT codec requires CUDA')
    gpu = torch.cuda.get_device_properties(torch.cuda.current_device())
    sources = [Path(__file__), BOX / 'runtime/trt-lab/repo/irodori_tts/codec.py']
    return dict(codec=_weights_digest(weights), gpu=gpu.name,
                capability=[gpu.major, gpu.minor], device=str(getattr(gpu, 'uuid', '')),
                trt=trt.__version__, torch=torch.__version__, cuda=torch.version.cuda,
                dacvae=version('dacvae'), precision='fp16', max_frames=MAX_FRAMES,
                sources={str(p.relative_to(BOX)): digest(p) for p in sources})


def valid_cache(folder, identity):
    try:
        record = json.loads((folder / 'ready.json').read_text(encoding='utf-8'))
        plan = folder / PLAN_NAME
        return (record['identity'] == identity and plan.stat().st_size > 0
                and record['plan_sha256'] == digest(plan))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def ensure_codec_plan(progress, log):
    """Return a verified codec plan, building it once per GPU/codec/toolchain."""
    progress('TensorRT: codec を確認中', 76)
    weights = codec_weights()
    identity = cache_identity(weights)
    cache = BOX / '.cache' / 'trt-codec' / cache_key(identity)
    if valid_cache(cache, identity):
        log('TensorRT: 対応する codec plan を再利用')
        return cache / PLAN_NAME
    cache.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='build-', dir=cache.parent))
    logfile = work / 'build.log'
    log(f'TensorRT codec build log: {logfile}')
    progress('TensorRT: codec plan 構築（初回のみ）', 78)
    env = os.environ.copy()
    env.update(PYTHONIOENCODING='utf-8', PYTHONUTF8='1', PYTHONUNBUFFERED='1')
    with logfile.open('w', encoding='utf-8') as output:
        result = subprocess.run([sys.executable, str(Path(__file__)), '--build', str(work),
                                 str(weights)], env=env, cwd=BOX, stdout=output,
                                stderr=subprocess.STDOUT,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result.returncode:
        tail = logfile.read_text(encoding='utf-8', errors='replace')[-6000:]
        log('TensorRT codec child process failed:\n' + tail)
        lines = [line.strip() for line in tail.splitlines() if line.strip()]
        raise RuntimeError(f'TensorRT codec の構築に失敗しました: '
                           f'{lines[-1][:500] if lines else result.returncode}\nログ: {logfile}')
    cache.mkdir(exist_ok=True)
    plan = work / PLAN_NAME
    record = dict(identity=identity, plan_sha256=digest(plan),
                  verification=json.loads((work / 'verify.json').read_text(encoding='utf-8')))
    plan.replace(cache / PLAN_NAME)
    marker = work / 'ready.json'
    marker.write_text(json.dumps(record, ensure_ascii=False), encoding='utf-8')
    marker.replace(cache / 'ready.json')
    return cache / PLAN_NAME


# ------------------------------------------------------------------ runtime
class CodecEngine:
    """FP16 decoder plan; scratch memory is borrowed from PyTorch per call.

    A STATIC context would pin ~1 GB for the 750-frame worst case.  With
    USER_MANAGED memory the scratch comes from the caching allocator on the
    current stream, like the PyTorch codec's own temporaries.
    """

    def __init__(self, plan):
        import tensorrt as trt
        import torch
        self.torch = torch
        # TensorRT keeps one process-wide logger; share the DiT engine's when
        # it is loaded.  (Importing run_engine here would turn TF32 on.)
        logger = getattr(sys.modules.get('run_engine'), 'LOGGER', None)
        self.runtime = trt.Runtime(logger or trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(Path(plan).read_bytes())
        if self.engine is None:
            raise RuntimeError(f'codec plan deserialization failed: {plan}')
        self.context = self.engine.create_execution_context(
            trt.ExecutionContextAllocationStrategy.USER_MANAGED)
        low, _, high = self.engine.get_tensor_profile_shape('latent', 0)
        self.min_shape, self.max_shape = tuple(low), tuple(high)

    def accepts(self, latent):
        return (latent.ndim == 3
                and all(lo <= d <= hi for d, lo, hi in zip(latent.shape, self.min_shape, self.max_shape)))

    def __call__(self, latent):
        torch = self.torch
        latent = latent.to(device='cuda', dtype=torch.float16).contiguous()
        context = self.context
        if not context.set_input_shape('latent', tuple(latent.shape)):
            raise ValueError(f'latent shape {tuple(latent.shape)} outside codec profile')
        audio = torch.empty(tuple(context.get_tensor_shape('audio')), device='cuda',
                            dtype=torch.float16)
        scratch = torch.empty(max(1, context.update_device_memory_size_for_shapes()),
                              device='cuda', dtype=torch.uint8)
        context.set_device_memory(scratch.data_ptr(), scratch.numel())
        context.set_tensor_address('latent', latent.data_ptr())
        context.set_tensor_address('audio', audio.data_ptr())
        if not context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError('TensorRT codec execution failed')
        # Freed blocks are only reused by later work on this same stream.
        return audio


class TrtCodecDecoder:
    """Drop-in for ``DACVAECodec.decode_latent`` with a PyTorch fallback."""

    def __init__(self, engine, fallback, log=None):
        self.engine, self.fallback = engine, fallback
        self.log = log or (lambda message: print(message, file=sys.stderr, flush=True))
        self.calls = self.fallbacks = 0

    def __call__(self, latent):
        if not self.engine.accepts(latent):
            self.fallbacks += 1
            return self.fallback(latent)
        audio = self.engine(latent).float()
        # The caller copies the waveform to the CPU next, so this sync is free.
        if not bool(self.engine.torch.isfinite(audio).all()):
            self.fallbacks += 1
            self.log('[trt-codec] non-finite FP16 output; decoding this line with PyTorch')
            return self.fallback(latent)
        self.calls += 1
        return audio


def install(runtime, plan, log=None):
    """Route ``runtime.codec.decode_latent`` through the TensorRT plan."""
    engine = CodecEngine(plan)
    hop = int(runtime.codec.model.hop_length)
    probe = engine(engine.torch.zeros((1, 2, runtime.codec.latent_dim), device='cuda'))
    if tuple(probe.shape) != (1, 1, 2 * hop):
        raise RuntimeError(f'codec plan output {tuple(probe.shape)} does not match hop {hop}')
    decoder = TrtCodecDecoder(engine, runtime.codec.decode_latent, log)
    runtime.codec.decode_latent = decoder
    return decoder


# ------------------------------------------------------------------ build (child process)
def _load_codec(weights, dtype):
    import torch
    _runtime_paths()
    from irodori_tts.codec import DACVAECodec
    # Same loader and patches (watermark passthrough, fixed message) as the runtime.
    return DACVAECodec.load(repo_id=str(weights), device='cuda', dtype=dtype,
                            deterministic_encode=True, deterministic_decode=True)


def _decoder_module(model):
    """(B, T, D) latent -> (B, 1, samples), matching DACVAECodec.decode_latent."""
    import torch

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, latent):
            return self.model.decode(latent.transpose(1, 2).contiguous())

    return Decoder().eval()


def _snr(reference, actual):
    import torch
    reference, actual = reference.float().flatten(), actual.float().flatten()
    error = (reference - actual).square().sum().clamp_min(1e-30)
    return float(10 * torch.log10(reference.square().sum().clamp_min(1e-30) / error))


def build(work, weights):
    import torch
    import tensorrt as trt
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    onnx_path = work / 'codec_fp16.onnx'
    codec = _load_codec(weights, None)
    for module in codec.model.modules():
        if hasattr(module, 'weight_g'):
            torch.nn.utils.remove_weight_norm(module)
    model = codec.model.half()
    with torch.inference_mode():
        torch.onnx.export(_decoder_module(model), torch.zeros(1, 64, codec.latent_dim, device='cuda',
                                                       dtype=torch.float16),
                          str(onnx_path), dynamo=False, input_names=['latent'],
                          output_names=['audio'], opset_version=18, do_constant_folding=True,
                          dynamic_axes={'latent': {1: 'frames'}, 'audio': {2: 'samples'}})
    del codec, model
    torch.cuda.empty_cache()
    print('ONNX_EXPORTED', flush=True)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    # Level 5 was measured ~5% faster at 3x the build time; not worth it here.
    config.builder_optimization_level = 3
    profile = builder.create_optimization_profile()
    profile.set_shape('latent', (1, 1, 32), (1, OPT_FRAMES, 32), (1, MAX_FRAMES, 32))
    config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError('TensorRT returned no codec engine')
    (work / PLAN_NAME).write_bytes(bytes(plan))
    onnx_path.unlink()
    print('PLAN_BUILT', flush=True)
    verify(work, weights)


def verify(work, weights):
    """Publishable only if at least as close to FP32 as the BF16 PyTorch codec."""
    import torch
    reference = _load_codec(weights, None)
    current = _load_codec(weights, torch.bfloat16)
    engine = CodecEngine(work / PLAN_NAME)
    hop = int(reference.model.hop_length)
    generator = torch.Generator(device='cuda').manual_seed(1234)
    rows = []
    with torch.inference_mode():
        for frames in VERIFY_FRAMES:
            latent = torch.randn((1, frames, reference.latent_dim), device='cuda',
                                 generator=generator)
            expected = reference.decode_latent(latent)
            actual = engine(latent)
            baseline = current.decode_latent(latent)
            if tuple(actual.shape) != (1, 1, frames * hop):
                raise RuntimeError(f'codec shape {tuple(actual.shape)} at {frames} frames')
            if not torch.isfinite(actual).all():
                raise RuntimeError(f'non-finite codec output at {frames} frames')
            row = dict(frames=frames, snr_trt_db=_snr(expected, actual),
                       snr_bf16_db=_snr(expected, baseline))
            rows.append(row)
            print('VERIFY', json.dumps(row), flush=True)
            if frames > 1 and row['snr_trt_db'] < row['snr_bf16_db'] - SNR_MARGIN_DB:
                raise RuntimeError(f'TensorRT codec is less accurate than BF16 PyTorch: {row}')
            if frames > 1 and not math.isfinite(row['snr_trt_db']):
                raise RuntimeError(f'invalid SNR: {row}')
        for scale in (1.0, 3.0):
            latent = scale * torch.randn((1, MAX_FRAMES, reference.latent_dim), device='cuda',
                                         generator=generator)
            actual = engine(latent)
            if tuple(actual.shape) != (1, 1, MAX_FRAMES * hop) or not torch.isfinite(actual).all():
                raise RuntimeError(f'codec failed at {MAX_FRAMES} frames, scale {scale}')
    (work / 'verify.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print('VERIFIED', flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] != '--build':
        raise SystemExit('usage: trt_codec.py --build WORK_DIR WEIGHTS')
    build(Path(sys.argv[2]), Path(sys.argv[3]))
