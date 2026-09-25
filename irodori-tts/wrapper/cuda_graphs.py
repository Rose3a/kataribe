"""CUDA Graph replay for the fixed-shape condition encoders.

Per request, the text encoder (a Hugging Face backbone) and the 12-layer
context KV projection launch hundreds of small kernels; on Windows the launch
overhead, not the GPU, dominates them.  The tokenizer pads to a fixed length,
so the shapes repeat and one captured graph per shape replays the same
kernels with the same inputs: the outputs are bit-identical to eager mode.

A call that synchronises with the host (``.item()``, ``nonzero``, a pageable
copy) cannot be captured; the failed capture is cleaned up and that call then
always runs eagerly.  CPU inputs and autograd also run eagerly.

Capture-aware library code takes its capture branch while recording: the
Hugging Face mask helper skips its ``padding_mask.all()`` shortcut and always
passes the explicit mask, which eager mode drops only when a text fills all
256 tokens.  Both are exact attention masks; every shorter text is identical.
"""
from __future__ import annotations

from collections import OrderedDict
import sys

import torch


def _flatten(value, out):
    """Tensors of a (nested) tuple/list output, in order."""
    if isinstance(value, torch.Tensor):
        out.append(value)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _flatten(item, out)
    elif value is not None:
        raise TypeError(f'unsupported graph output {type(value).__name__}')
    return out


def _rebuild(template, tensors):
    if isinstance(template, torch.Tensor):
        return next(tensors)
    if isinstance(template, (tuple, list)):
        return type(template)(_rebuild(item, tensors) for item in template)
    return template


class GraphedCall:
    """Replay ``fn(*args)`` from a CUDA graph keyed on the tensor signature.

    Non-tensor arguments are part of the key (by identity), tensors are copied
    into static inputs.  Returned tensors are clones, so results of earlier
    calls stay valid while the graph is replayed again.
    """

    def __init__(self, fn, name, max_graphs=4, log=None):
        self.fn, self.name, self.max_graphs = fn, name, max_graphs
        self.graphs = OrderedDict()
        self.disabled = False
        self.log = log or (lambda message: print(message, file=sys.stderr, flush=True))
        self.replays = self.captures = self.eager = 0

    @staticmethod
    def _key(names, values):
        return names, tuple((tuple(v.shape), v.dtype, v.device) if isinstance(v, torch.Tensor)
                            else ('obj', id(v)) for v in values)

    def __call__(self, *args, **kwargs):
        names = tuple(sorted(kwargs))
        values = (*args, *(kwargs[name] for name in names))
        tensors = [v for v in values if isinstance(v, torch.Tensor)]
        if (self.disabled or not tensors or not all(t.is_cuda for t in tensors)
                or torch.is_grad_enabled()):
            self.eager += 1
            return self.fn(*args, **kwargs)
        key = self._key(names, values)
        entry = self.graphs.get(key)
        if entry is None:
            try:
                entry = self._capture(len(args), names, values)
            except Exception as exc:  # noqa: BLE001 - fall back to eager for good
                self.disabled = True
                self.graphs.clear()
                self.log(f'[cuda-graph] {self.name}: capture failed, running eagerly: {exc}')
                self.eager += 1
                return self.fn(*args, **kwargs)
            self.graphs[key] = entry
            if len(self.graphs) > self.max_graphs:
                self.graphs.popitem(last=False)
        else:
            self.graphs.move_to_end(key)
        static_values, graph, template, outputs = entry
        for dst, src in zip(static_values, values):
            if isinstance(dst, torch.Tensor):
                dst.copy_(src)
        graph.replay()
        self.replays += 1
        return _rebuild(template, iter([t.clone() for t in outputs]))

    def _capture(self, positional, names, values):
        static = tuple(v.clone() if isinstance(v, torch.Tensor) else v for v in values)

        def run():
            return self.fn(*static[:positional], **dict(zip(names, static[positional:])))

        stream = torch.cuda.current_stream()
        side = torch.cuda.Stream()
        side.wait_stream(stream)
        with torch.cuda.stream(side):
            # Warm up outside the capture: lazy initialisation, cuBLAS handles.
            for _ in range(2):
                run()
        stream.wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                template = run()
        except Exception:
            _recover_from_failed_capture(stream)
            raise
        self.captures += 1
        return static, graph, template, _flatten(template, [])


def _recover_from_failed_capture(stream):
    """Undo what a failed ``torch.cuda.graph`` leaves behind.

    A host sync inside the capture (``.item()``, ``bool(tensor)``) invalidates
    it, and the context manager then exits without restoring the current
    stream or unregistering the capture from the CUDA RNG: every later CUDA
    call fails.  Restoring the stream and completing one trivial capture puts
    both back (random numbers stay reproducible).
    """
    torch.cuda.set_stream(stream)
    # The invalidation is still pending as CUDA's last error; the next launch
    # reports it once (and clears it).
    for attempt in range(3):
        try:
            probe = torch.zeros(1, device=stream.device)
            break
        except RuntimeError:
            if attempt == 2:
                raise
    reset = torch.cuda.CUDAGraph()
    with torch.cuda.graph(reset):
        probe.add_(1)
    torch.cuda.set_stream(stream)
    torch.cuda.synchronize()


def install(model, log=None):
    """Graph the text/caption encoders and the context KV projection of ``model``.

    Returns the installed wrappers (for statistics and tests).
    """
    wrappers = {}
    for attr in ('text_encoder', 'caption_encoder'):
        module = getattr(model, attr, None)
        if module is None:
            continue
        # Instance attribute: nn.Module.__call__ dispatches to it.
        wrapper = GraphedCall(module.forward, attr, log=log)
        module.forward = wrapper
        wrappers[attr] = wrapper
    build_kv = GraphedCall(model.build_context_kv_cache, 'context_kv', log=log)
    model.build_context_kv_cache = build_kv
    wrappers['context_kv'] = build_kv
    return wrappers
