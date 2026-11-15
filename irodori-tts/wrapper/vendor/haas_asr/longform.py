"""Long-form transcription: cut audio at quiet points, transcribe the chunks in batches.

Chunks default to 30 s because no training utterance was longer. A fixed cut
tends to land mid-word and damage the text on both sides, so each cut is
moved to the quietest short window near its target position.
"""

from __future__ import annotations

import numpy as np

from haas_asr.audio import SAMPLE_RATE
from haas_asr.transcription import transcribe
from haas_asr.vocabulary import Vocabulary

CHUNK_SECONDS = 30.0
SEARCH_SECONDS = 2.0
WINDOW_SECONDS = 0.1


def split_at_quiet_points(
    audio: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    chunk_seconds: float = CHUNK_SECONDS,
    search_seconds: float = SEARCH_SECONDS,
    window_seconds: float = WINDOW_SECONDS,
) -> list[tuple[int, int]]:
    """Return ``(start, end)`` sample ranges that cover `audio` without gaps.

    Each cut is placed at the centre of the lowest-energy window within
    ``search_seconds`` of its target, and never earlier than half a chunk
    after the previous cut, so no chunk is longer than
    ``chunk_seconds + search_seconds``.
    """
    step = int(chunk_seconds * sample_rate)
    search = int(search_seconds * sample_rate)
    window = int(window_seconds * sample_rate)

    bounds = []
    start = 0
    while audio.size - start > step:
        target = start + step
        low = max(start + step // 2, target - search)
        high = min(audio.size - window, target + search)
        cut = target
        if high > low:
            cut = low + _quietest_window(audio[low : high + window], window) + window // 2
        bounds.append((start, cut))
        start = cut
    if start < audio.size:
        bounds.append((start, audio.size))
    return bounds


def _quietest_window(audio: np.ndarray, window: int) -> int:
    """Offset of the `window`-sample span with the least energy."""
    energy = np.cumsum(np.concatenate([[0.0], audio.astype(np.float64) ** 2]))
    return int(np.argmin(energy[window:] - energy[:-window]))


def transcribe_longform(
    model,
    audio: np.ndarray,
    *,
    batch_size: int = 64,
    separator: str = "",
    vocabulary: Vocabulary | None = None,
    guard: bool = True,
    **split_kwargs,
) -> str:
    """Transcribe audio of any length and join the chunk texts.

    `audio` is mono float32 at the model's sample rate (see
    ``haas_asr.audio.load_audio``); `split_kwargs` go to
    ``split_at_quiet_points``. Japanese text has no word spaces, hence the
    empty default `separator`. `vocabulary` and `guard` are as in
    ``haas_asr.transcribe``.

    Changing `batch_size` can flip a few near-tie tokens (different GPU
    kernels for different shapes), so fix it when comparing runs.
    """
    chunks = [audio[start:end] for start, end in split_at_quiet_points(audio, **split_kwargs)]
    texts = transcribe(model, chunks, batch_size=batch_size, vocabulary=vocabulary, guard=guard)
    return separator.join(texts)
