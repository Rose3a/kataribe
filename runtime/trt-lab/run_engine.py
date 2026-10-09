"""Run a serialized cached-DiT engine through PyTorch CUDA pointers."""
from collections import OrderedDict
import argparse
import hashlib
import json
import statistics
import time

from trt_experiment import LAB, NAMES, inputs_from, request, runtime, torch, save_wav
import tensorrt as trt

LOGGER = trt.Logger(trt.Logger.WARNING)

class Engine:
    def __init__(self, path, graph=False, input_names=NAMES, output_name='v'):
        self.logger = LOGGER
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError('Engine deserialization failed')
        self.context = self.engine.create_execution_context()
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.outputs = {}
        self.graph_enabled = graph
        self.graphs = OrderedDict()
        self.input_names = list(input_names)
        if 'delta_t' in self.names and 'delta_t' not in self.input_names:
            self.input_names.append('delta_t')
        self.output_name = output_name
        self.max_batch = self.engine.get_tensor_profile_shape(self.input_names[0], 0)[2][0]
        self.dtype_map = {trt.float32: torch.float32, trt.float16: torch.float16,
                          trt.bfloat16: torch.bfloat16, trt.bool: torch.bool,
                          trt.int32: torch.int32, trt.int64: torch.int64}

    def __call__(self, xs):
        if len(xs) != len(self.input_names):
            raise ValueError('Wrong number of engine inputs')
        for name, tensor in zip(self.input_names, xs):
            # caption を持たないモデル（v3 など）では caption_mask / kv_4, kv_5 が未使用で
            # ONNX 変換時に削られる。エンジンに無い入力は検証も束縛もしない。
            if name not in self.names:
                continue
            expected = self.dtype_map[self.engine.get_tensor_dtype(name)]
            if not tensor.is_cuda or not tensor.is_contiguous() or tensor.dtype != expected:
                raise ValueError(f'{name} must be contiguous CUDA {expected}')
            low, _, high = self.engine.get_tensor_profile_shape(name, 0)
            if len(tensor.shape) != len(low) or any(d < lo or d > hi for d, lo, hi in zip(tensor.shape, low, high)):
                raise ValueError(f'{name} shape {tuple(tensor.shape)} outside profile {low}..{high}')
        if self.graph_enabled:
            key = tuple(tuple(x.shape) for x in xs)
            if key not in self.graphs:
                if len(self.graphs) >= 4:
                    torch.cuda.synchronize()
                    self.graphs.popitem(last=False)
                # Each captured shape owns a context and its scratch memory.
                direct_context, direct_outputs = self.context, self.outputs
                try:
                    self.context = self.engine.create_execution_context()
                    self.outputs = {}
                    static = tuple(x.clone() for x in xs)
                    for _ in range(2):
                        self.execute(static)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
                        output = self.execute(static)
                    self.graphs[key] = (static, graph, output, self.context)
                finally:
                    # Eager shape changes must never mutate captured contexts.
                    self.context, self.outputs = direct_context, direct_outputs
            static, graph, output, context = self.graphs[key]
            for dst, src in zip(static, xs):
                dst.copy_(src)
            graph.replay()
            return output
        return self.execute(xs)

    def execute(self, xs):
        tensors = dict(zip(self.input_names, xs))
        for name in self.names:
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                x = tensors[name]
                if not x.is_cuda or not x.is_contiguous():
                    raise ValueError(f'{name} must be contiguous CUDA tensor')
                if x.dtype != self.dtype_map[self.engine.get_tensor_dtype(name)]:
                    raise ValueError(f'dtype mismatch for {name}: {x.dtype}')
                if not self.context.set_input_shape(name, tuple(x.shape)):
                    raise ValueError(f'Shape outside profile: {name} {x.shape}')
                if not self.context.set_tensor_address(name, x.data_ptr()):
                    raise RuntimeError(f'Binding failed for {name}')
        shape = tuple(self.context.get_tensor_shape(self.output_name))
        if min(shape) < 0:
            raise ValueError(f'Unresolved output shape: {shape}')
        if shape not in self.outputs:
            self.outputs[shape] = torch.empty(shape, device='cuda', dtype=self.dtype_map[self.engine.get_tensor_dtype(self.output_name)])
        output = self.outputs[shape]
        self.context.set_tensor_address(self.output_name, output.data_ptr())
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError('TensorRT execution failed')
        return output


class Adapter:
    def __init__(self, model, engine, compact=False):
        self.model, self.engine = model, engine
        self.compact = compact
        self.prepared = OrderedDict()
        self.calls = 0

    def __call__(self, **kw):
        if kw.get('latent_mask') is not None or kw.get('context_kv_cache') is None:
            raise ValueError('Experimental engine requires cached KV and no latent mask')
        cache = kw['context_kv_cache']
        key = (id(cache), kw['x_t'].shape[1])
        if key not in self.prepared:
            xs = inputs_from(self.model, kw, compact=self.compact)
            self.prepared[key] = (cache, xs[2:13])
            if len(self.prepared) > 2:
                self.prepared.popitem(last=False)
        self.calls += 1
        delta = (kw['delta_t'].contiguous(),) if 'delta_t' in self.engine.input_names else ()
        if self.model.delta_cond_module is not None and not delta:
            raise ValueError('MeanFlow requires a plan with delta_t input')
        return self.run((kw['x_t'].contiguous(), kw['t'].contiguous(), *self.prepared[key][1], *delta))

    def run(self, xs):
        # Independent CFG with text, speaker and caption needs batch 4, one
        # more than the plan profile.  Rows never interact, so run the batch
        # in profile-sized slices.  kv_* stack layers first (batch on axis 1)
        # and the RoPE tables carry no batch axis.
        limit = getattr(self.engine, 'max_batch', None)
        batch = xs[0].shape[0]
        if limit is None or batch <= limit:
            return self.engine(xs)
        parts = []
        for start in range(0, batch, limit):
            piece = []
            for name, x in zip(self.engine.input_names, xs):
                if name.startswith('rope_'):
                    piece.append(x)
                elif name.startswith('kv_'):
                    piece.append(x[:, start:start + limit].contiguous())
                else:
                    piece.append(x[start:start + limit].contiguous())
            # The engine reuses its output buffer per shape; copy each slice out.
            parts.append(self.engine(tuple(piece)).clone())
        return torch.cat(parts)


def compare(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    assert a.numel() == b.numel(), (a.shape, b.shape)
    e = a-b
    return {'finite': bool(torch.isfinite(b).all()),
            'max_abs': e.abs().max().item(),
            'rmse': e.square().mean().sqrt().item(),
            'snr_db': (10*torch.log10(a.square().sum()/e.square().sum().clamp_min(1e-30))).item(),
            'cosine': torch.nn.functional.cosine_similarity(a, b, dim=0).item()}


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repeat', type=int, default=10)
    ap.add_argument('--graph', action='store_true')
    ap.add_argument('--label', default='stream')
    ap.add_argument('--compact', action='store_true')
    ap.add_argument('--codec', action='store_true')
    args = ap.parse_args()
    eng = Engine(LAB / ('dit_compact.plan' if args.compact else 'dit_bf16.plan'), graph=args.graph)
    xs = torch.load(LAB / 'inputs_b3.pt', weights_only=True)
    expected = torch.load(LAB / 'export_expected.pt', weights_only=True)
    got = eng(xs)
    torch.cuda.synchronize()
    micro = compare(expected, got)
    print('DIT_ERROR', micro, flush=True)
    assert micro['finite'], 'Non-finite engine output'
    rt = runtime()
    original = rt.model.forward_with_encoded_conditions
    original_decode = rt.codec.decode_latent
    codec_engine = Engine(LAB/'codec_bf16.plan', graph=args.graph, input_names=['latent'], output_name='audio') if args.codec else None
    adapter = Adapter(rt.model, eng, compact=args.compact)
    records = []
    refs = {}
    for mode in ['torch', 'trt']:
        rt.model.forward_with_encoded_conditions = original if mode == 'torch' else adapter
        rt.codec.decode_latent = original_decode if mode == 'torch' or codec_engine is None else lambda z: codec_engine((z.contiguous(),))
        for _ in range(3):
            rt.synthesize(request(seed=9000), log_fn=None)
        for i in range(args.repeat):
            req = request(seed=1001+i)
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = rt.synthesize(req, log_fn=None)
            torch.cuda.synchronize()
            wall = time.perf_counter()-start
            wave = result.audio.detach().cpu().clone()
            path = LAB / f'{args.label}_{mode}_{i:02}.wav'
            save_wav(str(path), wave, result.sample_rate)
            row = {'mode': mode, 'seed': req.seed, 'wall_s': wall,
                   'total_s': result.total_to_decode, 'stages': dict(result.stage_timings),
                   'audio_s': wave.shape[-1]/result.sample_rate, 'wav': str(path),
                   'wav_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            if mode == 'torch':
                refs[i] = wave
            else:
                row['error'] = compare(refs[i], wave)
                assert row['error']['finite']
            records.append(row)
            (LAB / f'comparison_{args.label}.json').write_text(json.dumps({'micro': micro, 'records': records}, indent=2))
            print(mode, i, round(wall, 5), row.get('error', ''), flush=True)
    summary = {mode: {'wall_median_s': statistics.median(r['wall_s'] for r in records if r['mode']==mode),
                      'stages_median_s': {k: statistics.median(r['stages'][k] for r in records if r['mode']==mode) for k in records[0]['stages']}}
               for mode in ['torch', 'trt']}
    summary['speedup'] = summary['torch']['wall_median_s']/summary['trt']['wall_median_s']
    summary['engine_calls'] = adapter.calls
    (LAB / f'comparison_{args.label}.json').write_text(json.dumps({'micro': micro, 'summary': summary, 'records': records}, indent=2))
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        main()
