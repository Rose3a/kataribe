"""Keep the resident TensorRT engine's VRAM to what it actually uses.

The PyTorch model is loaded whole, but with the TensorRT backend most of it
never runs on the GPU:

- DiT blocks: TensorRT runs them; PyTorch only projects the context K/V
  (``wk_*``/``wv_*``/``k_norm``).  The rest duplicates the plan's weights.
- codec decoder: replaced by the TensorRT codec; kept only as a fallback.
- speaker encoder: used only for user-supplied reference audio.

Unused weights move to CPU RAM (nothing is dropped, nothing is recomputed in
another precision, so outputs stay bit-identical).  torchao INT4 weights (v4-Large
INT4) move the same way; they are left out of ``repack`` (their data_ptr() is 0).  The two occasional users
move their module back to the GPU for the call.  Cached allocator blocks are
returned to the driver once the engine has been idle for a few seconds, so a
game started next to the editor can use them; back-to-back synthesis keeps
the cache and its speed.
"""
from __future__ import annotations
from pathlib import Path

import os
import sys
import threading
import time

import torch

# Used by build_context_kv_cache (JointAttention.project_context_kv).
KV_PROJECTION = frozenset({"wk_text", "wv_text", "wk_speaker", "wv_speaker",
                           "wk_caption", "wv_caption", "k_norm",
                           # blockwise の prefix の K/V（tts_cli._PrefixToCaption で plan に渡す）
                           "wk_prefix", "wv_prefix"})
# Top-level DiT modules that only the TensorRT plan runs.
PLAN_ONLY = ("cond_module", "delta_cond_module", "in_proj", "out_norm", "out_proj")


def _is_plain(tensor):
    """素の torch.Tensor か。torchao の量子化テンソルなどのサブクラスは data_ptr() が 0 になる。"""
    return type(tensor.data) is torch.Tensor


def has_quantized_weights_outside_plan(model):
    """TensorRT の plan だけが使う DiT の Linear 以外に、torchao の量子化テンソルがあるか。

    W8A8 の Blockwise Small は DiT の Linear だけが INT8 で、PyTorch 側は素の BF16 のまま
    （CUDA Graph を使っても VRAM は増えない）。v4-Large INT4 はテキストエンコーダなども量子化されている。
    """
    for block in model.blocks:
        plan_only = {id(p) for name, child in block.named_children()
                     for sub, mod in ([(n, c) for n, c in child.named_children() if n not in KV_PROJECTION]
                                      if name == "attention" else [("", child)])
                     for p in mod.parameters()}
        if any(not _is_plain(p) and id(p) not in plan_only for p in block.parameters()):
            return True
    blocks = {id(p) for p in model.blocks.parameters()}
    return any(not _is_plain(p) for p in model.parameters() if id(p) not in blocks)


def _param_bytes(param):
    if _is_plain(param):
        return param.numel() * param.element_size()
    # torchao INT4/INT8 など。見かけの要素数ではなく、実際に持っているテンソルの大きさで数える
    data = param.data
    try:
        names, _ = data.__tensor_flatten__()
    except (AttributeError, TypeError):
        return 0
    return sum(getattr(data, name).numel() * getattr(data, name).element_size() for name in names)


def _move(module, device):
    """module.to(device)。torchao の量子化テンソルは inference_mode の中では .to() できないので外す。

    合成は inference_mode で走り、その中で参照音声用の speaker_encoder や codec を GPU に戻す。
    素の Tensor は inference_mode(False) の中で動かしても結果は同じ。
    """
    with torch.inference_mode(False):
        return module.to(device)


def _size(module):
    return sum(_param_bytes(p) for p in module.parameters())


def _discard(module):
    """module の重みを捨てる（形は残すので、呼ぶと空のテンソルで失敗する）。捨てられなければ CPU へ。"""
    for param in module.parameters():
        try:
            param.data = torch.empty(0, dtype=param.dtype, device="cpu")
        except (RuntimeError, TypeError):  # torchao の量子化テンソルなど
            _move(module, "cpu")
            return


def offload_plan_weights(model, discard=True):
    """DiT weights that the TensorRT plan already holds leave the GPU.

    discard=True（既定）: 捨てる。TRT の合成は plan だけが DiT を計算し、PyTorch の DiT 本体は
    呼ばれない（plan に入れられない入力は Adapter が ValueError にする）ので、CPU の RAM にも残さない
    （約 0.6GB）。discard=False: 従来どおり CPU の RAM に退避する（IRODORI_TRT_KEEP_PLAN_WEIGHTS=1）。
    """
    release = _discard if discard else (lambda module: _move(module, "cpu"))
    moved = 0
    first = next(model.parameters())  # model.device follows the first parameter
    for block in model.blocks:
        for name, child in block.named_children():
            children = ([c for n, c in child.named_children() if n not in KV_PROJECTION]
                        if name == "attention" else [child])
            for module in children:
                moved += _size(module)
                release(module)
    for name in PLAN_ONLY:
        module = getattr(model, name, None)
        if module is not None and all(p is not first for p in module.parameters()):
            moved += _size(module)
            release(module)
    return moved


class OnDemand:
    """Keep ``module`` on the CPU and move it to ``device`` around ``fn`` calls."""

    def __init__(self, module, fn, device):
        self.module, self.fn, self.device = module, fn, device

    def __call__(self, *args, **kwargs):
        _move(self.module, self.device)
        try:
            return self.fn(*args, **kwargs)
        finally:
            _move(self.module, "cpu")


def park_on_demand(module, device):
    """Move ``module`` to CPU; calling it brings it to ``device`` for the call."""
    moved = _size(module)
    # Instance attribute: nn.Module.__call__ dispatches to it.
    module.forward = OnDemand(module, module.forward, device)
    _move(module, "cpu")
    return moved


class _Swapped:
    """module の重みを CPU（pinned）に置き、使うときだけ GPU に載せる。

    OnDemand の module.to() は戻すたびに CPU 側を作り直すが、こちらは CPU のコピーを持ち続け、
    戻すときは param.data を差し替えるだけにする（GPU→CPU のコピーが要らない）。
    """

    def __init__(self, modules, device):
        self.device = device
        self.params = [p for m in modules for p in m.parameters() if _is_plain(p)]
        self.host = []
        for p in self.params:
            data = p.data.cpu()
            try:
                data = data.pin_memory()
            except RuntimeError:
                pass
            self.host.append(data)
            p.data = data
        self.bytes = sum(h.numel() * h.element_size() for h in self.host)

    def __enter__(self):
        for p, h in zip(self.params, self.host):
            p.data = h.to(self.device, non_blocking=True)

    def __exit__(self, *exc):
        for p, h in zip(self.params, self.host):
            p.data = h


def park_prefix_encoder(model, device):
    """blockwise の prefix エンコーダ（115MB）を、prefix を encode するときだけ GPU に載せる。

    入力が直前の行の生成結果なので出力はキャッシュできない。続き生成の行だけ載せ替えが入る。
    """
    encoder = getattr(model, "latent_prefix_encoder", None)
    if encoder is None:
        return 0
    swap = _Swapped([encoder], device)
    original = encoder.forward

    def forward(*args, **kwargs):
        with swap:
            return original(*args, **kwargs)

    encoder.forward = forward
    return swap.bytes


def park_kv_projection(model, device):
    """文脈 K/V の射影（TRT 経路の PyTorch 側に残る DiT の wk/wv/k_norm、約 150MB）を、
    build_context_kv_cache のあいだだけ GPU に載せる。毎行 1 回呼ばれる。"""
    modules = [m for block in model.blocks for name, m in block.attention.named_children()
               if name in KV_PROJECTION]
    swap = _Swapped(modules, device)
    original = model.build_context_kv_cache

    def build_context_kv_cache(*args, **kwargs):
        with swap:
            return original(*args, **kwargs)

    model.build_context_kv_cache = build_context_kv_cache
    return swap.bytes


def park_text_backbone(model, device):
    """テキストエンコーダの本体（ModernBERT 310M、bf16 で約 0.6GB）を CPU の fp32 で動かす。

    省VRAM 用。プロジェクタ以降（小さい）は GPU のまま。トークナイザは 256 トークン固定で
    詰めるので、CPU では有効な長さまで切って計算し、元の長さにゼロで戻す（右詰めの padding は
    有効な位置に影響しないので同じ値）。fp32 なので GPU の bf16 とはビット一致しない
    （bf16 の CPU は fp32 の 4〜8 倍遅い）。短い 1 行で GPU より +70〜90ms 前後、長文ほど差が広がる。
    返り値は GPU から外した bytes（無ければ 0）。
    """
    wrapper = getattr(model, "pretrained_text_backbone", None)
    if wrapper is None:
        return 0
    moved = _size(wrapper)
    # model.device / model.dtype は最初のパラメータで決まり、本体が最初に登録されている。
    # CPU の fp32 に替えると全体が fp32 と見なされるので、登録順を最後に回す。
    model._modules["pretrained_text_backbone"] = model._modules.pop("pretrained_text_backbone")
    with torch.inference_mode(False):
        wrapper.to("cpu", torch.float32)
        if os.environ.get("IRODORI_TEXT_ON_CPU") == "int8":  # 実験: Linear だけ動的 INT8
            torch.ao.quantization.quantize_dynamic(wrapper.backbone, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
    original = wrapper.forward

    def forward(input_ids, mask):
        length = max(int(mask.sum(dim=-1).max().item()), 1)
        ids_cpu, mask_cpu = input_ids[:, :length].cpu(), mask[:, :length].cpu()
        with torch.inference_mode():
            state = original(ids_cpu, mask_cpu)
        state = state.to(device=device, non_blocking=False)
        pad = input_ids.shape[1] - length
        if pad:
            state = torch.nn.functional.pad(state, (0, 0, 0, pad))
        return state

    wrapper.forward = forward
    return moved


def make_unloadable(eng, plan_path, make_context):
    """TensorRT の Engine / CodecEngine を、unload() で plan ごと解放し、次に使うとき読み直せるようにする。

    run_engine.py は plan のキャッシュの identity に入るので触らず、インスタンスの class を差し替える。
    engine / context は遅延読み込みのプロパティにして、解放後に触っても自動で読み直す。
    返り値の unload() は、使っていないときに呼ぶ（合成中は IdleRelease が busy で止める）。
    """
    cls = type(eng)

    def getter(name):
        def get(self):
            if self.__dict__.get("_" + name) is None:
                self._reload()
            return self.__dict__["_" + name]
        return get

    def reload(self):
        t0 = time.perf_counter()
        self.__dict__["_engine"] = self.runtime.deserialize_cuda_engine(Path(plan_path).read_bytes())
        self.__dict__["_context"] = make_context(self.__dict__["_engine"])
        self.reloads = getattr(self, "reloads", 0) + 1
        self.reload_ms = (time.perf_counter() - t0) * 1000

    sub = type("Unloadable" + cls.__name__, (cls,), {
        "engine": property(getter("engine"), lambda self, v: self.__dict__.__setitem__("_engine", v)),
        "context": property(getter("context"), lambda self, v: self.__dict__.__setitem__("_context", v)),
        "_reload": reload,
    })
    eng.__dict__["_engine"] = eng.__dict__.pop("engine")
    eng.__dict__["_context"] = eng.__dict__.pop("context")
    eng.__class__ = sub

    def unload():
        if eng.__dict__.get("_engine") is None:
            return 0
        eng.__dict__["_context"] = None
        eng.__dict__["_engine"] = None
        if hasattr(eng, "outputs"):
            eng.outputs = {}
        return 1

    return unload


class IdleRelease:
    """Return cached CUDA blocks to the driver after ``delay`` idle seconds."""

    def __init__(self, lock, delay=5.0, log=None, unload=None, unload_after=None):
        self.lock, self.delay = lock, delay
        # 待機が unload_after 秒続いたら unload() を呼ぶ（TensorRT の plan の解放など）
        self.unload, self.unload_after, self._unload_timer = unload, unload_after, None
        self.log = log or (lambda message: print(message, file=sys.stderr, flush=True))
        self.last = time.monotonic()
        self.pending = False
        self.releases = 0
        self._guard = threading.Lock()

    def _unload_now(self):
        if not self.lock.acquire(timeout=0):
            return
        try:
            torch.cuda.synchronize()
            self.unload()
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        finally:
            self.lock.release()

    def touch(self):
        """Call after each synthesis; (re)arms the idle timer."""
        if self.unload is not None and self.unload_after:
            with self._guard:
                if self._unload_timer is not None:
                    self._unload_timer.cancel()
                self._unload_timer = threading.Timer(self.unload_after, self._unload_now)
                self._unload_timer.daemon = True
                self._unload_timer.start()
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


def text_on_cpu_enabled(low_vram):
    """省VRAM のときテキストエンコーダを CPU に置く。IRODORI_TEXT_ON_CPU=0/1 で上書き。"""
    return os.environ.get("IRODORI_TEXT_ON_CPU", "1" if low_vram else "0") in ("1", "int8")


# 省VRAM で、待機がこの秒数続いたら TensorRT の plan（DiT・codec）を解放する。次の合成の最初の 1 行が
# 約 1 秒遅れる（plan の読み直し）。編集中の短い間隔の合成では解放されず、席を離れたりゲームを
# 始めたりしたときに VRAM が約 1.2GB → 0.3GB になる。IRODORI_UNLOAD_AFTER で上書き（0 で無効）。
UNLOAD_AFTER_SECONDS = 30.0


def low_vram_flag(name, low_vram):
    """省VRAM のとき既定でオンの部品。環境変数 name=0/1 で上書き。"""
    return os.environ.get(name, "1" if low_vram else "0") == "1"


def unload_after_seconds(low_vram):
    return float(os.environ.get("IRODORI_UNLOAD_AFTER", str(UNLOAD_AFTER_SECONDS) if low_vram else "0"))


def install(backend, log=None, text_on_cpu=False, prefix_on_demand=False, kv_on_demand=False,
            unloadable_plans=False):
    """Apply to a TrtBackend.  Returns the bytes moved off the GPU."""
    runtime = backend.runtime
    model = runtime.model
    device = next(model.parameters()).device
    moved = offload_plan_weights(model, discard=os.environ.get("IRODORI_TRT_KEEP_PLAN_WEIGHTS") != "1")
    if text_on_cpu:
        moved += park_text_backbone(model, device)
    if prefix_on_demand:
        moved += park_prefix_encoder(model, device)
    if kv_on_demand:
        moved += park_kv_projection(model, device)
    if getattr(model, "speaker_encoder", None) is not None:
        moved += park_on_demand(model.speaker_encoder, device)
    if backend.codec is not None:
        # The TensorRT codec's PyTorch fallback brings the decoder back.
        decoder = runtime.codec.model.decoder
        moved += _size(decoder)
        _move(decoder, "cpu")
        backend.codec.fallback = OnDemand(decoder, backend.codec.fallback, device)
    if unloadable_plans:
        import tensorrt as trt
        unloaders = [make_unloadable(backend.engine, backend.plan_path,
                                     lambda e: e.create_execution_context())]
        if backend.codec is not None:
            codec_engine = backend.codec.engine
            unloaders.append(make_unloadable(
                codec_engine, backend.codec_plan_path,
                lambda e: e.create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED)))
        backend.unload_plans = lambda: [u() for u in unloaders]
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
            # 量子化テンソルは data_ptr() が全部 0 で1つにまとめられてしまうので、詰め直さない
            if tensor.is_cuda and _is_plain(tensor):
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
