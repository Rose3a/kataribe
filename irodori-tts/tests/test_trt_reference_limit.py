import math
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wrapper"))

from tts_cli import _cap_reference_seconds, _plan_max_ref_seconds  # noqa: E402


class _Profile:
    def __init__(self, high):
        self.high = high

    def get_tensor_profile_shape(self, name, index):
        assert name == "speaker_mask"
        return [(1, 1), (1, 16), (3, self.high)]


def _runtime(default=30.0):
    return types.SimpleNamespace(
        model_cfg=types.SimpleNamespace(speaker_patch_size=4, latent_patch_size=1),
        codec=types.SimpleNamespace(model=types.SimpleNamespace(hop_length=1920), sample_rate=48000),
        default_max_ref_seconds=default)


class TrtReferenceLimitTest(unittest.TestCase):
    def test_limit_fits_the_plan_including_the_extra_speaker_token(self):
        limit = _plan_max_ref_seconds(types.SimpleNamespace(engine=_Profile(192)), _runtime())
        # runtime と同じ計算で、切ったあとのトークン数（フレーム/4 + 1）が 192 に収まる
        frames = math.ceil(limit * 48000 / 1920)
        self.assertLessEqual(math.ceil(frames / 4) + 1, 192)
        self.assertGreater(limit, 29.5)

    def test_unknown_plan_does_not_cut(self):
        self.assertIsNone(_plan_max_ref_seconds(types.SimpleNamespace(), _runtime()))

    def test_cap_lowers_only_longer_limits(self):
        request = types.SimpleNamespace(max_ref_seconds=None)
        _cap_reference_seconds(request, _runtime(30.0), 10.0)
        self.assertEqual(request.max_ref_seconds, 10.0)
        request = types.SimpleNamespace(max_ref_seconds=5.0)
        _cap_reference_seconds(request, _runtime(30.0), 10.0)
        self.assertEqual(request.max_ref_seconds, 5.0)
        request = types.SimpleNamespace(max_ref_seconds=None)
        _cap_reference_seconds(request, _runtime(0.0), 10.0)  # 0 は「切らない」なので上限を入れる
        self.assertEqual(request.max_ref_seconds, 10.0)
        request = types.SimpleNamespace(max_ref_seconds=None)
        _cap_reference_seconds(request, _runtime(30.0), None)
        self.assertIsNone(request.max_ref_seconds)


if __name__ == "__main__":
    unittest.main()
