import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wrapper"))
from speaker_mix import compose_speaker_mix  # noqa: E402


class FakeCassette:
    def __init__(self):
        self.values = {
            "a": torch.full((16, 768), 2.0),
            "b": torch.full((16, 768), 6.0),
        }

    def has(self, name):
        return name in self.values

    def get(self, name):
        return self.values[name]


class SpeakerMixTest(unittest.TestCase):
    def test_zero_baseline_and_local_overlap(self):
        tokens = [[] for _ in range(16)]
        tokens[3] = [
            {"speaker": "a", "strength": 0.75},
            {"speaker": "b", "strength": 0.25},
        ]
        mixed = compose_speaker_mix(tokens, FakeCassette())
        self.assertEqual(tuple(mixed.shape), (16, 768))
        self.assertTrue(torch.equal(mixed[0], torch.zeros(768)))
        self.assertTrue(torch.allclose(mixed[3], torch.full((768,), 2.25)))

    def test_invalid_strength_is_rejected(self):
        tokens = [[] for _ in range(16)]
        tokens[0] = [{"speaker": "a", "strength": float("nan")}]
        with self.assertRaises(ValueError):
            compose_speaker_mix(tokens, FakeCassette())


if __name__ == "__main__":
    unittest.main()
