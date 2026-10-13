"""参照音声の latent のキャッシュ。

参照音声を付けたセリフは、合成のたびに同じ音声を読み込み・リサンプル・codec でエンコード
し直していた。codec のエンコードは決定的（codec_deterministic_encode=True）なので、
同じ音声と同じ設定なら結果も同じになる。音声の中身のハッシュをキーにして、
エンコード結果（patchify 済み）を CPU に少しだけ持っておく。

runtime 側（runtime/trt-lab/repo/irodori_tts）は編集すると TensorRT plan のキャッシュが
無効になるので、ここでインスタンスのメソッドを包む。
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from pathlib import Path

# 1件は数十〜数百 KB（30 秒の参照で約 100 KB）。行ごとに参照音声を替えても足りる数。
MAX_ENTRIES = 16


def _reference_key(runtime, req, batch_size):
    """キャッシュできる要求（参照音声1つだけ）ならキー、それ以外は None。"""
    if req.no_ref or req.ref_wav is None or req.ref_wavs or req.ref_latent is not None or req.ref_latents:
        return None
    try:
        content = hashlib.sha256(Path(req.ref_wav).read_bytes()).hexdigest()
    except OSError:
        return None
    max_ref_seconds = (runtime.default_max_ref_seconds if req.max_ref_seconds is None
                       else float(req.max_ref_seconds))
    # 拡張子で読み込み方（デコーダ）が変わりうるので含める。
    return (content, Path(req.ref_wav).suffix.lower(), max_ref_seconds,
            req.ref_normalize_db, bool(req.ref_ensure_max), int(batch_size))


def install(runtime, max_entries: int = MAX_ENTRIES):
    """runtime._load_reference_latent をキャッシュ付きに包む。二重には包まない。"""
    original = runtime._load_reference_latent
    if getattr(original, "_reference_cache", None) is not None:
        return original._reference_cache
    cache: OrderedDict = OrderedDict()
    lock = threading.Lock()
    stats = {"hits": 0, "misses": 0}

    def cached(*, req, batch_size, messages):
        key = _reference_key(runtime, req, batch_size)
        if key is None:
            return original(req=req, batch_size=batch_size, messages=messages)
        with lock:
            hit = cache.get(key)
            if hit is not None:
                cache.move_to_end(key)
                stats["hits"] += 1
        if hit is not None:
            latent, mask, notes = hit
            messages.extend(notes)
            device = runtime.model_device
            return latent.to(device=device), mask.to(device=device)
        notes: list[str] = []
        latent, mask = original(req=req, batch_size=batch_size, messages=notes)
        messages.extend(notes)
        if latent is not None and mask is not None:
            with lock:
                stats["misses"] += 1
                cache[key] = (latent.detach().to("cpu"), mask.detach().to("cpu"), list(notes))
                while len(cache) > max_entries:
                    cache.popitem(last=False)
        return latent, mask

    cached._reference_cache = {"cache": cache, "stats": stats}
    runtime._load_reference_latent = cached
    return cached._reference_cache
