"""契約テスト: 話者埋め込みの次元がモデルと合わないときは分かるエラーにする。

話者ファイルは作ったモデル専用で、v4.1 Small 系は 768 次元、v4-Large は 1280 次元。
"""
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wrapper"))
from tts_cli import IrodoriTTS  # noqa: E402


class FakeCassette:
    def __init__(self):
        self.values = {"small": torch.ones((16, 768)), "large": torch.ones((16, 1280))}

    def get(self, name):
        return self.values[name]


def fake_tts(speaker_dim):
    tts = IrodoriTTS.__new__(IrodoriTTS)
    tts.cassette = FakeCassette()
    tts.default_speaker = "small"
    runtime = types.SimpleNamespace(model_cfg=types.SimpleNamespace(speaker_dim=speaker_dim))
    tts.backend = types.SimpleNamespace(
        runtime=runtime,
        synthesize=Mock(return_value={"backend": "fake", "wall_s": 0.0, "audio_s": 1.0}))
    return tts


class FakeDimCassette:
    speakers = ["small", "small.speaker", "large", "broken"]

    def dim_for(self, name):
        return {"small": 768, "large": 1280}.get(name)


class SpeakerDimTest(unittest.TestCase):
    def test_editor_lists_only_speakers_matching_the_model(self):
        from editor_engine import EditorAdapter
        cassette = FakeDimCassette()
        # 読めないファイルは一覧に残し、合成時にエラーで知らせる。
        self.assertEqual(EditorAdapter._usable_speakers(cassette, 1280), ["large", "broken"])
        self.assertEqual(EditorAdapter._usable_speakers(cassette, 768), ["small", "broken"])
        # モデルの次元が不明なら絞り込まない。
        self.assertEqual(EditorAdapter._usable_speakers(cassette, None),
                         ["small", "large", "broken"])

    def test_small_speaker_on_large_model_is_rejected_with_guidance(self):
        tts = fake_tts(1280)
        with self.assertRaisesRegex(ValueError, "話者「small」.*768 次元.*1280 次元"):
            tts.synthesize(text="テスト", speaker="small", out_wav=Mock())
        tts.backend.synthesize.assert_not_called()

    def test_additional_speaker_is_checked_too(self):
        tts = fake_tts(1280)
        with self.assertRaisesRegex(ValueError, "話者「small」"):
            tts.synthesize(text="テスト", speaker="large", out_wav=Mock(),
                           additional_speakers=[("small", 0.5)])

    def test_matching_width_and_no_speaker_are_accepted(self):
        tts = fake_tts(1280)
        tts.synthesize(text="テスト", speaker="large", out_wav=Mock())
        tts.synthesize(text="テスト", speaker="", out_wav=Mock())
        self.assertEqual(tts.backend.synthesize.call_count, 2)
        self.assertEqual(fake_tts(768).speaker_dim(), 768)


if __name__ == "__main__":
    unittest.main()
