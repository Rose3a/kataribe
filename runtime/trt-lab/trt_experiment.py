"""Isolated Irodori v4.1 BF16 cached-DiT ONNX/TensorRT experiment."""
from pathlib import Path
import argparse
import json
import os
import sys
import time

SOURCE_DIR = Path(__file__).resolve().parent
LAB = Path(os.environ.get('IRODORI_TRT_WORK_DIR', SOURCE_DIR))
os.environ.setdefault('HF_HOME', str(LAB.parent.parent / '.cache' / 'huggingface'))
sys.path.insert(0, str(SOURCE_DIR / 'repo'))
import torch
from numerical_checks import validate_export
from irodori_tts import model as model_module
from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest, save_wav

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
TEXT = 'こんにちは、私はAIです。これは音声合成のテストです。ばらお、聞こえる？'
NAMES = ['x', 't', 'text_mask', 'speaker_mask', 'caption_mask', 'rope_cos', 'rope_sin'] + [f'kv_{i}' for i in range(6)]


def runtime():
    return InferenceRuntime.from_key(RuntimeKey(
        checkpoint=os.environ['IRODORI_CHECKPOINT'],
        model_device='cuda', model_precision='bf16', codec_device='cuda',
        codec_precision='bf16', codec_deterministic_encode=True,
        codec_deterministic_decode=True, compile_model=False, compile_dynamic=False))


def request(text=TEXT, seed=1001, seconds=None):
    # SamplingRequest distinguishes "not supplied" (None) from an empty
    # string.  Passing '' is still treated as ref_embed being present and
    # conflicts with no_ref=True, which made a clean export fail before the
    # ONNX/TRT work started.
    ref_embed = os.environ.get('IRODORI_EXPORT_REF_EMBED') or None
    return SamplingRequest(text=text, caption=None,
        ref_embed=ref_embed, no_ref=not bool(ref_embed), num_candidates=1, decode_mode='batch', seconds=seconds,
        duration_scale=1.0, num_steps=8, t_schedule_mode='sway', sway_coeff=-1.0,
        cfg_scale_speaker=5.0, cfg_scale_caption=3.0, seed=seed)


def real_rope(x, freqs):
    # Match the original float32 complex multiply, then round back to BF16.
    cos, sin = freqs[..., 0][None, :, None, :], freqs[..., 1][None, :, None, :]
    pairs = x.float().reshape(*x.shape[:3], -1, 2)
    a, b = pairs[..., 0], pairs[..., 1]
    return torch.stack((a*cos-b*sin, a*sin+b*cos), dim=-1).reshape_as(x).to(x.dtype)


class CachedDiT(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.in_proj = model.in_proj
        self.cond_module = model.cond_module
        self.blocks = model.blocks
        self.out_norm = model.out_norm
        self.out_proj = model.out_proj
        self.time_dim = model.cfg.timestep_embed_dim
        self.delta_cond_module = model.delta_cond_module

    def forward(self, x, t, text_mask, speaker_mask, caption_mask, rope_cos, rope_sin, *kv):
        cond = self.cond_module(model_module.get_timestep_embedding(t, self.time_dim).to(x.dtype))[:, None, :]
        if self.delta_cond_module is not None:
            cond = cond + self.delta_cond_module(model_module.get_timestep_embedding(kv[6], self.time_dim).to(x.dtype))[:, None, :]
            kv = kv[:6]
        freqs = torch.stack((rope_cos, rope_sin), dim=-1)
        h = self.in_proj(x)
        for i, block in enumerate(self.blocks):
            cache = tuple(k[i] for k in kv)
            h = block(h, cond, cache[0], text_mask, cache[2], speaker_mask,
                      cache[4], caption_mask, freqs, context_kv=cache)
        return self.out_proj(self.out_norm(h)).to(x.dtype)


def inputs_from(model, kw, compact=False):
    freq = model._rope_freqs(kw['x_t'].shape[1], kw['x_t'].device)
    cache = kw['context_kv_cache']
    masks = [kw[k] for k in ('text_mask', 'speaker_mask', 'caption_mask')]
    if masks[2] is None:
        # caption を持たないモデル（v3 など）: 全部無効な長さ1の caption を補い、
        # plan の入力（caption_mask / kv_4, kv_5）は v4 系と同じ形のままにする。
        masks[2] = torch.zeros((kw['x_t'].shape[0], 1), dtype=torch.bool, device=kw['x_t'].device)
    # 各層の kv は (text k, v, speaker k, v[, caption k, v])。足りない caption 分はゼロで埋める。
    cache = [tuple(layer) + tuple(torch.zeros_like(layer[0][:, :1]) for _ in range(6 - len(layer)))
             for layer in cache]
    lengths = [mask.shape[1] for mask in masks]
    if compact:
        lengths = []
        for mask in masks:
            positions = mask.any(dim=0).nonzero()
            lengths.append(int(positions[-1].item()) + 1 if positions.numel() else 1)
    return (kw['x_t'], kw['t'], *(mask[:, :length].contiguous() for mask, length in zip(masks, lengths)),
            freq.real.contiguous(), freq.imag.contiguous(),
            *(torch.stack([layer[i][:, :lengths[i//2]] for layer in cache]).contiguous() for i in range(6)),
            *((kw['delta_t'].contiguous(),) if model.delta_cond_module is not None else ()))


@torch.inference_mode()
def export(capture_only=False):
    rt = runtime()
    names = NAMES + (['delta_t'] if rt.model.delta_cond_module is not None else [])
    captured = {}
    reference = {}
    original = rt.model.forward_with_encoded_conditions
    def tap(**kw):
        b = kw['x_t'].shape[0]
        if b not in captured:
            captured[b] = tuple(x.detach().clone() for x in inputs_from(rt.model, kw))
            reference[b] = original(**kw).detach().clone()
            return reference[b]
        return original(**kw)
    rt.model.forward_with_encoded_conditions = tap
    req = request()
    if rt.model.delta_cond_module is not None:
        req.num_steps = 4
    result = rt.synthesize(req, log_fn=None)
    save_wav(str(LAB / 'export_reference.wav'), result.audio, result.sample_rate)
    rt.model.forward_with_encoded_conditions = original
    print('captured', {b: [list(x.shape) for x in xs] for b, xs in captured.items()}, flush=True)
    wrapper = CachedDiT(rt.model).eval()
    old_rope = model_module.apply_rotary_emb
    model_module.apply_rotary_emb = real_rope
    try:
        for b, xs in captured.items():
            torch.save(xs, LAB / f'inputs_b{b}.pt')
        xs = captured[max(captured)]
        out = wrapper(*xs)
        comparison = validate_export(out, reference[max(captured)])
        print('EXPORT_COMPARISON', json.dumps(comparison), flush=True)
        (LAB / 'export_comparison.json').write_text(json.dumps(comparison, indent=2), encoding='utf-8')
        torch.save(reference[max(captured)], LAB / 'export_expected.pt')
        shapes = {n: list(x.shape) for n, x in zip(names, xs)}
        (LAB / 'shapes.json').write_text(json.dumps(shapes, indent=2))
        if capture_only:
            print('INPUTS_CAPTURED', flush=True)
            return
        dynamic = {'x': {0: 'batch', 1: 'latent'}, 't': {0: 'batch'},
                   'text_mask': {0: 'batch', 1: 'text'}, 'speaker_mask': {0: 'batch', 1: 'speaker'},
                   'caption_mask': {0: 'batch', 1: 'caption'},
                   'rope_cos': {0: 'latent'}, 'rope_sin': {0: 'latent'},
                   'v': {0: 'batch', 1: 'latent'}}
        for i in range(6):
            dynamic[f'kv_{i}'] = {1: 'batch', 2: ['text', 'speaker', 'caption'][i//2]}
        if 'delta_t' in names:
            dynamic['delta_t'] = {0: 'batch'}
        torch.onnx.export(wrapper, xs, str(LAB / 'dit_bf16.onnx'),
                          input_names=names, output_names=['v'], opset_version=18,
                          dynamo=False, dynamic_axes=dynamic, do_constant_folding=True)
        print('ONNX_EXPORTED', flush=True)
    finally:
        model_module.apply_rotary_emb = old_rope


def build(compact=False):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(LAB / 'dit_bf16.onnx')):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    config.builder_optimization_level = 3
    profile = builder.create_optimization_profile()
    shapes = json.loads((LAB / 'shapes.json').read_text())
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        opt = shapes[inp.name]
        if compact:
            if inp.name == 'text_mask':
                opt[1] = 48
            elif inp.name == 'caption_mask':
                opt[1] = 1
            elif inp.name in ('kv_0', 'kv_1'):
                opt[2] = 48
            elif inp.name in ('kv_4', 'kv_5'):
                opt[2] = 1
        low, high = opt.copy(), opt.copy()
        # First experiment: fixed text/speaker lengths, variable batch and audio length.
        if inp.name in ('x', 't', 'text_mask', 'speaker_mask', 'caption_mask'):
            low[0], high[0] = 1, 3
        if inp.name.startswith('kv_'):
            low[1], high[1] = 1, 3
        if inp.name == 'x':
            low[1], high[1] = 1, 750
        if inp.name.startswith('rope_'):
            low[0], high[0] = 1, 750
        if compact:
            for name, axis, maximum in [('text_mask', 1, 256), ('speaker_mask', 1, 64), ('caption_mask', 1, 512),
                                         ('kv_0', 2, 256), ('kv_1', 2, 256), ('kv_2', 2, 64), ('kv_3', 2, 64),
                                         ('kv_4', 2, 512), ('kv_5', 2, 512)]:
                if inp.name == name:
                    low[axis], high[axis] = 1, maximum
        profile.set_shape(inp.name, low, opt, high)
        print('profile', inp.name, low, opt, high, flush=True)
    config.add_optimization_profile(profile)
    t0 = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError('TensorRT returned no engine')
    (LAB / ('dit_compact.plan' if compact else 'dit_bf16.plan')).write_bytes(bytes(plan))
    print('ENGINE_BUILT', plan.nbytes, 'seconds', time.perf_counter()-t0, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['capture', 'export', 'build', 'build_compact'])
    args = ap.parse_args()
    if args.action == 'capture':
        export(capture_only=True)
    elif args.action == 'build_compact':
        build(compact=True)
    else:
        globals()[args.action]()
