"""Keep the resident TensorRT engine's VRAM to what it actually uses.

The PyTorch model is loaded whole, but with the TensorRT backend most of it
never runs on the GPU:

- DiT blocks: TensorRT runs them; PyTorch only projects the context K/V
  (``wk_*``/``wv_*``/``k_norm``).  The rest duplicates the plan's weights.
- codec decoder: replaced by the TensorRT codec; kept only as a fallback.
- speaker encoder: used only for user-supplied reference audio.

Unused weights move to CPU RAM (nothing is dropped, nothing is recomputed in
another precision, so outputs stay bit-identical).  The two occasional users
move their module back to the GPU for the call.  Cached allocator blocks are
returned to the driver once the engine has been idle for a few seconds, so a
game started next to the editor can use them; back-to-back synthesis keeps
the cache and its speed.
"""
from __future__ import annotations

import sys
import threading
import time

import torch

# Used by build_context_kv_cache (JointAttention.project_context_kv).
KV_PROJECTION = frozenset({"wk_text", "wv_text", "wk_speaker", "wv_speaker",
                           "wk_caption", "wv_caption", "k_norm"})
# Top-level DiT modules that only the TensorRT plan runs.
PLAN_ONLY = ("cond_module", "delta_cond_module", "in_proj", "out_norm", "out_proj")


def _size(module):
    return sum(p.numel() * p.element_size() for p in module.parameters())


def offload_plan_weights(model):
    """Move DiT weights that the TensorRT plan already holds to CPU RAM."""
    moved = 0
    first = next(model.parameters())  # model.device follows the first parameter
    for block in model.blocks:
        for name, child in block.named_children():
            children = ([c for n, c in child.named_children() if n not in KV_PROJECTION]
                        if name == "attention" else [child])
            for module in children:
                moved += _size(module)
                module.to("cpu")
    for name in PLAN_ONLY:
        module = getattr(model, name, None)
        if module is not None and all(p is not first for p in module.parameters()):
            moved += _size(module)
            module.to("cpu")
    return moved


class OnDemand:
    """Keep ``module`` on the CPU and move it to ``device`` around ``fn`` calls."""

    def __init__(self, module, fn, device):
        self.module, self.fn, self.device = module, fn, device

    def __call__(self, *args, **kwargs):
        self.module.to(self.device)
        try:
            return self.fn(*args, **kwargs)
        finally:
            self.module.to("cpu")


def park_on_demand(module, device):
    """Move ``module`` to CPU; calling it brings it to ``device`` for the call."""
    moved = _size(module)
    # Instance attribute: nn.Module.__call__ dispatches to it.
    module.forward = OnDemand(module, module.forward, device)
    module.to("cpu")
    return moved


class IdleRelease:
    """Return cached CUDA blocks to the driver after ``delay`` idle seconds."""

    def __init__(self, lock, delay=5.0, log=None):
        self.lock, self.delay = lock, delay
        self.log = log or (lambda message: print(message, file=sys.stderr, flush=True))
        self.last = time.monotonic()
        self.pending = False
        self.releases = 0
        self._guard = threading.Lock()

    def touch(self):
        """Call after each synthesis; (re)arms the idle timer."""
        with self._guard:
            self.last = time.monotonic()
            if self.pending:
                return
            self.pending = True
        timer = threading.Timer(self.delay, self._fire)
        timer.daemon = True
        timer.start()

    def _fire(self):
        wait = self.last + self.delay - time.monotonic()
        if wait > 0.05:  # used again meanwhile: wait for the new idle period
            timer = threading.Timer(wait, self._fire)
            timer.daemon = True
            timer.start()
            return
        with self._guard:
            self.pending = False
        # Never release while a synthesis is running.
        if not self.lock.acquire(timeout=0):
            self.touch()
            return
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            self.releases += 1
        finally:
            self.lock.release()


def install(backend, log=None):
    """Apply to a TrtBackend.  Returns the bytes moved off the GPU."""
    runtime = backend.runtime
    model = runtime.model
    device = next(model.parameters()).device
    moved = offload_plan_weights(model)
    if getattr(model, "speaker_encoder", None) is not None:
        moved += park_on_demand(model.speaker_encoder, device)
    if backend.codec is not None:
        # The TensorRT codec's PyTorch fallback brings the decoder back.
        decoder = runtime.codec.model.decoder
        moved += _size(decoder)
        decoder.to("cpu")
        backend.codec.fallback = OnDemand(decoder, backend.codec.fallback, device)
    repack(model, runtime.codec.model)
    return moved


def repack(*modules):
    """Re-allocate the GPU-resident weights contiguously.

    Loading interleaves weights with temporaries, and the offload above leaves
    holes; a segment is only returned to the driver when it is entirely free.
    Round-tripping the remaining weights through the CPU (same values) lets
    ``empty_cache`` return every loading-time segment.
    """
    tensors = {}
    for module in modules:
        for tensor in (*module.parameters(), *module.buffers()):
            if tensor.is_cuda:
                tensors.setdefault(tensor.data_ptr(), []).append(tensor)
    # Only whole-storage tensors; a view would be split from its base.
    for group in tensors.values():
        data = group[0].data
        if data.storage_offset() or data.untyped_storage().nbytes() != data.nbytes:
            torch.cuda.empty_cache()
            return
    host = {ptr: group[0].data.to("cpu") for ptr, group in tensors.items()}
    for group in tensors.values():
        for tensor in group:
            tensor.data = torch.empty(0, device=tensor.device, dtype=tensor.dtype)
    torch.cuda.empty_cache()
    for ptr, group in tensors.items():
        data = host[ptr].to(group[0].device)
        for tensor in group:  # tied tensors stay tied
            tensor.data = data
    torch.cuda.empty_cache()
