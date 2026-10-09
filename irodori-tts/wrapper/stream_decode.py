"""Windowed codec decoding for streaming playback.

The DiT has to finish before any sample exists, but the codec (latent ->
waveform) is convolutional, so it can decode the latent in windows.  The first
window is small so playback starts early; every window is decoded with a few
extra frames on each side that are thrown away, which makes the result equal
to decoding in one piece (bit-exact for the TensorRT codec at MARGIN = 10).

``StreamingDecode`` wraps ``runtime.codec.decode_latent``.  With no sink set it
is a pass-through, so the normal path is untouched.  With a sink set it decodes
window by window and hands every window except the last to the sink; the last
one is held back because the runtime may still trim the tail of the audio, and
the caller sends the authoritative remainder when synthesis returns.
"""
from __future__ import annotations

import contextlib
import contextvars

import torch

FIRST_FRAMES = 12            # 0.48 s at 25 latent frames/s
NEXT_FRAMES = (24, 48)       # then 24, then 48 per window
MARGIN = 10                  # context frames decoded and discarded on each side


_SINK = contextvars.ContextVar("irodori_stream_sink", default=None)


@contextlib.contextmanager
def stream_to(sink):
    """Within this block (this thread), synthesis streams its first windows to ``sink``."""
    token = _SINK.set(sink)
    try:
        yield
    finally:
        _SINK.reset(token)


def current_sink():
    return _SINK.get()


def windows(total: int, first: int = FIRST_FRAMES, sizes=NEXT_FRAMES):
    """(start, end) latent-frame ranges that tile ``total`` frames."""
    start, index = 0, 0
    while start < total:
        size = first if index == 0 else sizes[min(index - 1, len(sizes) - 1)]
        yield start, min(start + size, total)
        start += size
        index += 1


def to_pcm16(audio: torch.Tensor) -> bytes:
    """(…, samples) float waveform -> little-endian int16 bytes."""
    clipped = audio.detach().float().reshape(-1).clamp(-1.0, 1.0)
    return clipped.mul(32767.0).round().to(torch.int16).cpu().numpy().tobytes()


class StreamingDecode:
    def __init__(self, inner, hop: int, sample_rate: int):
        self.inner, self.hop, self.sample_rate = inner, int(hop), int(sample_rate)
        self.sink = None        # callable(pcm16 bytes, sample_rate) or None
        self.sent_samples = 0   # samples already handed to the sink

    @classmethod
    def install(cls, runtime):
        codec = runtime.codec
        wrapper = cls(codec.decode_latent, codec.model.hop_length, codec.sample_rate)
        codec.decode_latent = wrapper
        return wrapper

    def begin(self, sink):
        self.sink, self.sent_samples = sink, 0

    def end(self):
        self.sink = None

    def __call__(self, latent):
        sink = self.sink
        total = latent.shape[1]
        if sink is None or latent.shape[0] != 1 or total <= FIRST_FRAMES + MARGIN:
            return self.inner(latent)
        hop, pieces = self.hop, []
        for start, end in windows(total):
            lo, hi = max(0, start - MARGIN), min(total, end + MARGIN)
            audio = self.inner(latent[:, lo:hi])
            piece = audio[..., (start - lo) * hop:(start - lo) * hop + (end - start) * hop]
            pieces.append(piece)
            if end < total:
                data = to_pcm16(piece)
                sink(data, self.sample_rate)
                self.sent_samples += len(data) // 2
        return torch.cat(pieces, dim=-1)
