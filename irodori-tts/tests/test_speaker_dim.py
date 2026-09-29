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

    def test_speaker_refresh_does_not_wait_for_model_loading(self):
        """モデル読み込み中でも話者一覧を返す（待たせるとエディタがタイムアウトし、
        前のモデル用の話者が一覧に残ったままになる）。"""
        import threading
        from unittest.mock import patch
        import editor_engine
        from editor_engine import EditorAdapter

        adapter = EditorAdapter.__new__(EditorAdapter)
        adapter.operation_lock = threading.RLock()
        adapter.speaker_lock = threading.Lock()
        adapter.state_lock = threading.RLock()
        adapter.delegate = None
        adapter._model_speaker_dim = lambda: 1280
        loading = threading.Event()
        release = threading.Event()

        def hold_lock():
            with adapter.operation_lock:
                loading.set()
                release.wait(10)

        holder = threading.Thread(target=hold_lock)
        holder.start()
        loading.wait(5)
        try:
            class ListedCassette(FakeDimCassette):
                dirs = []

                def path_for(self, name):
                    return None

                def folder_for(self, name):
                    return None

            with patch.object(editor_engine, "SpeakerCassette", lambda dirs, **_: ListedCassette()), \
                 patch.object(editor_engine, "speaker_catalog", lambda *a: []), \
                 patch.object(editor_engine, "_fallback_icon", lambda: None):
                done = threading.Thread(target=adapter.refresh)
                done.start()
                done.join(5)
            self.assertFalse(done.is_alive(), "refresh waited for operation_lock")
            self.assertEqual(sorted(adapter.id_to_name.values()), ["", "broken", "large"])
        finally:
            release.set()
            holder.join()

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
