"""Transcribe short audio, optionally with a custom vocabulary."""

from __future__ import annotations

import numpy as np

from haas_asr import boosting
from haas_asr.vocabulary import Vocabulary


def transcribe(
    model,
    audios: list[np.ndarray],
    *,
    batch_size: int = 64,
    vocabulary: Vocabulary | None = None,
    guard: bool = True,
) -> list[str]:
    """The texts of `audios`, each up to about 30 s of mono float32 at the model's sample
    rate (see ``haas_asr.audio.load_audio``).

    With `vocabulary`, its words are boosted while decoding and written in their written
    form. `guard` (the default) also decodes without the words and keeps a boosted change
    only where it includes a whole listed word (see ``haas_asr.boosting``).
    """
    audios = list(audios)
    if vocabulary:
        return boosting.transcribe(model, audios, vocabulary, batch_size=batch_size, guard=guard)
    hypotheses = model.transcribe(audios, batch_size=batch_size, verbose=False)
    return [h.text.strip() for h in hypotheses]
