"""Compose a preview speaker embedding from per-token speaker contributions."""
from __future__ import annotations

import math

import torch


def compose_speaker_mix(tokens: list, cassette) -> torch.Tensor:
    """Return a 16x768 embedding; an unassigned token remains exactly zero.

    Contribution values determine the ratio among speakers. The largest value
    controls the token's overall gain, so adding another speaker never doubles
    the embedding magnitude by accident.
    """
    if not isinstance(tokens, list) or len(tokens) != 16:
        raise ValueError("tokens must contain exactly 16 entries")
    output = torch.zeros((16, 768), dtype=torch.float32)
    for token_index, entries in enumerate(tokens):
        if not isinstance(entries, list) or len(entries) > 32:
            raise ValueError(f"token {token_index + 1} must contain a speaker list (max 32)")
        if not entries:
            continue
        total = 0.0
        gain = 0.0
        weighted = torch.zeros(768, dtype=torch.float32)
        seen = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"token {token_index + 1} has an invalid contribution")
            name = entry.get("speaker")
            strength = entry.get("strength")
            if not isinstance(name, str) or not cassette.has(name) or name in seen:
                raise ValueError(f"token {token_index + 1} has an invalid or duplicate speaker")
            seen.add(name)
            if isinstance(strength, bool) or not isinstance(strength, (int, float)):
                raise ValueError("strength must be a number between 0 and 1")
            strength = float(strength)
            if not math.isfinite(strength) or not 0 <= strength <= 1:
                raise ValueError("strength must be a number between 0 and 1")
            if strength == 0:
                continue
            source = cassette.get(name).float()
            if source.ndim == 3 and source.shape[0] == 1:
                source = source[0]
            if source.shape != output.shape or not torch.isfinite(source).all():
                raise ValueError(f"speaker {name} must have a finite 16x768 embedding")
            weighted += source[token_index] * strength
            total += strength
            gain = max(gain, strength)
        if total:
            output[token_index] = weighted * (gain / total)
    return output
