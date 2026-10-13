import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wrapper"))

import reference_cache  # noqa: E402


def _request(path, **overrides):
    values = dict(no_ref=False, ref_wav=str(path), ref_wavs=None, ref_latent=None, ref_latents=None,
                  max_ref_seconds=None, ref_normalize_db=-16.0, ref_ensure_max=True)
    values.update(overrides)
    return types.SimpleNamespace(**values)


class _FakeRuntime:
    default_max_ref_seconds = 30.0
    model_device = "cpu"

    def __init__(self):
        self.encodes = 0

    def _load_reference_latent(self, *, req, batch_size, messages):
        self.encodes += 1
        messages.append("info: encoded")
        value = float(len(Path(req.ref_wav).read_bytes()))
        return torch.full((batch_size, 3, 4), value), torch.ones((batch_size, 3), dtype=torch.bool)


class ReferenceCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, data):
        path = self.folder / name
        path.write_bytes(data)
        return path

    def test_same_audio_is_encoded_once_even_from_a_new_temp_file(self):
        runtime = _FakeRuntime()
        reference_cache.install(runtime)
        first_messages, second_messages = [], []
        first = runtime._load_reference_latent(
            req=_request(self.write("a.wav", b"abcd")), batch_size=1, messages=first_messages)
        # エンジンは毎回別の一時ファイルに書き出すので、中身で一致させる。
        second = runtime._load_reference_latent(
            req=_request(self.write("b.wav", b"abcd")), batch_size=1, messages=second_messages)
        self.assertEqual(runtime.encodes, 1)
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertEqual(second_messages, ["info: encoded"])

    def test_different_audio_or_settings_are_encoded_again(self):
        runtime = _FakeRuntime()
        reference_cache.install(runtime)
        path = self.write("a.wav", b"abcd")
        runtime._load_reference_latent(req=_request(path), batch_size=1, messages=[])
        runtime._load_reference_latent(req=_request(self.write("c.wav", b"xyz")), batch_size=1, messages=[])
        runtime._load_reference_latent(req=_request(path, max_ref_seconds=5.0), batch_size=1, messages=[])
        runtime._load_reference_latent(req=_request(path), batch_size=2, messages=[])
        self.assertEqual(runtime.encodes, 4)

    def test_no_reference_is_not_cached(self):
        runtime = _FakeRuntime()
        calls = []
        runtime._load_reference_latent = lambda **kw: calls.append(kw) or (None, None)
        reference_cache.install(runtime)
        runtime._load_reference_latent(req=_request(None, no_ref=True, ref_wav=None), batch_size=1, messages=[])
        runtime._load_reference_latent(req=_request(None, no_ref=True, ref_wav=None), batch_size=1, messages=[])
        self.assertEqual(len(calls), 2)

    def test_oldest_entries_are_dropped(self):
        runtime = _FakeRuntime()
        state = reference_cache.install(runtime, max_entries=2)
        for index in range(3):
            path = self.write(f"{index}.wav", bytes([index]) * (index + 1))
            runtime._load_reference_latent(req=_request(path), batch_size=1, messages=[])
        self.assertEqual(len(state["cache"]), 2)

    def test_install_twice_wraps_once(self):
        runtime = _FakeRuntime()
        first = reference_cache.install(runtime)
        self.assertIs(reference_cache.install(runtime), first)


if __name__ == "__main__":
    unittest.main()
