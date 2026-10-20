"""前のセリフから続けて生成する（blockwise）ための、前のセリフの latent の受け渡し。

blockwise 対応モデル（Irodori-TTS-v4.1-Small-Blockwise など、use_latent_prefix のもの）は、
前の文の末尾の latent を prefix にして、その続きとして次の文を作る。エディタは前のセリフの
音声（このエンジンが返した WAV）を ``irodori_context_audio`` で送ってくる。

- このエンジンが作った音声なら、合成時に覚えておいた latent をそのまま使う（音声の PCM の
  SHA-256 で引く）。デコード→エンコードの往復が無いので、続きの継ぎ目がいちばん自然になる。
- 覚えていない音声（エンジンを再起動した後など）は codec でエンコードし直す。中身を確かめられない
  音声で声を寄せることになるので、参照音声と同じく出力に透かしを入れる。
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import io
import threading
from collections import OrderedDict

import torch

# 30 秒の latent でも 750 × 32 の float32（約 96KB）。数十件なら数 MB で済む。
DEFAULT_LIMIT = 64
# prefix に使うのは末尾の 10 秒（学習と同じ）。エンコードし直すときも、その少し前からで足りる。
ENCODE_TAIL_SECONDS = 10.5
MAX_CONTEXT_BYTES = 10 * 1024 * 1024


def decode_data_url(value, label: str) -> bytes:
    """{"dataUrl": "data:audio/...;base64,..."} の中身を取り出す（参照音声と同じ形）。"""
    data_url = str((value or {}).get("dataUrl", "")) if isinstance(value, dict) else ""
    if not data_url.startswith("data:audio/") or ";base64," not in data_url:
        raise ValueError(f"{label} must be a base64 audio data URL")
    encoded = data_url.split(",", 1)[1]
    if len(encoded) > MAX_CONTEXT_BYTES * 4 // 3 + 4:
        raise ValueError(f"{label} is too large")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(f"invalid {label} data URL") from exc
    if len(raw) > MAX_CONTEXT_BYTES:
        raise ValueError(f"{label} is too large")
    return raw


def _read_pcm16(wav: bytes):
    """WAV を (int16 の1ch サンプル, サンプルレート) にする。読めなければ None。"""
    import soundfile as sf
    try:
        audio, rate = sf.read(io.BytesIO(wav), dtype="int16", always_2d=True)
    except Exception:  # noqa: BLE001 - WAV 以外（エディタが手を加えた音声など）はエンコードに回す
        return None
    return audio[:, 0].copy(), int(rate)


def pcm_digest_of_samples(samples: bytes) -> str:
    return hashlib.sha256(samples).hexdigest()


def pcm_digest(wav: bytes) -> str | None:
    """WAV の PCM（1ch の int16）の SHA-256。ヘッダの違い（ストリーミングで組み直した WAV）は無視する。"""
    pcm = _read_pcm16(wav)
    return None if pcm is None else pcm_digest_of_samples(pcm[0].tobytes())


def trailing_silence_seconds(wav: bytes, floor_db: float = -40.0) -> float:
    """前のセリフの末尾の無音の長さ（秒）。続きの行との間を測るのに使う（runtime の
    leading_silence_seconds と同じく、10ms ごとの RMS が最大より floor_db 下回る間を無音とみなす）。"""
    import numpy as np
    pcm = _read_pcm16(wav)
    if pcm is None:
        return 0.0
    samples, rate = pcm
    window = max(1, int(rate * 0.01))
    count = len(samples) // window
    if count == 0:
        return 0.0
    frames = samples[: count * window].astype(np.float32).reshape(count, window)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    loud = np.nonzero(rms > rms.max() * 10.0 ** (floor_db / 20.0))[0]
    return (count - 1 - int(loud[-1]) if loud.size else count) * window / float(rate)


class ContextLatentStore:
    """このエンジンが返した音声の latent を、PCM の SHA-256 で少しのあいだ覚えておく。"""

    def __init__(self, limit: int = DEFAULT_LIMIT):
        self.limit = limit
        self._items: OrderedDict[str, tuple[torch.Tensor, bool]] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, digest: str | None, latent: torch.Tensor | None, watermarked: bool) -> None:
        if digest is None or latent is None:
            return
        with self._lock:
            self._items.pop(digest, None)
            self._items[digest] = (latent.detach().float().cpu(), bool(watermarked))
            while len(self._items) > self.limit:
                self._items.popitem(last=False)

    def alias(self, digest: str, other: str) -> None:
        """同じ音声の別の PCM（ストリーミングで送った分など）でも引けるようにする。"""
        if digest == other:
            return
        with self._lock:
            value = self._items.get(digest)
        if value is not None:
            self.put(other, *value)

    def get(self, digest: str | None):
        if digest is None:
            return None
        with self._lock:
            value = self._items.get(digest)
            if value is not None:
                self._items.move_to_end(digest)
            return value

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


def context_latent(store: ContextLatentStore, raw: bytes, runtime, log=print):
    """前のセリフの音声から prefix 用の latent を得る。返り値: (latent (フレーム数, D), 透かしが要るか)。"""
    pcm = _read_pcm16(raw)
    if pcm is not None:
        hit = store.get(pcm_digest_of_samples(pcm[0].tobytes()))
        if hit is not None:
            log("[blockwise] continuing from the remembered latent of the previous line")
            return hit
    # 覚えていない音声: 末尾だけを codec でエンコードし直す（このエンジンの出力か確かめられないので透かし付き）
    if pcm is not None:
        samples, rate = pcm
        wav = torch.from_numpy(samples.astype("float32") / 32768.0)
    else:
        import soundfile as sf
        try:
            audio, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        except Exception as exc:  # noqa: BLE001
            raise ValueError("前のセリフの音声を読めませんでした") from exc
        wav = torch.from_numpy(audio.mean(axis=1))
    tail = int(ENCODE_TAIL_SECONDS * rate)
    wav = wav[-tail:].view(1, 1, -1)
    if wav.shape[-1] == 0:
        return None
    codec = runtime.codec
    with torch.inference_mode():
        latent = codec.encode_waveform(wav, sample_rate=rate, normalize_db=None, ensure_max=False)
    log("[blockwise] previous line was not generated by this engine run; encoded it again (watermarked)")
    return latent[0].float().cpu(), True
