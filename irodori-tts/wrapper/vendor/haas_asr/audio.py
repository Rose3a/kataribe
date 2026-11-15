"""Audio loading."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import librosa
import numpy as np
import soundfile as sf

# The model's input rate.
SAMPLE_RATE = 16_000


def load_audio(path: str | Path | BinaryIO, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Read an audio file (a path or an open binary file) as mono float32 at `sample_rate`."""
    audio, file_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if file_rate != sample_rate:
        audio = librosa.resample(audio, orig_sr=file_rate, target_sr=sample_rate)
    return np.ascontiguousarray(audio, dtype=np.float32)
