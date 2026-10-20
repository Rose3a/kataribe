"""契約テスト: セリフ設定の「基本の話者＋追加話者」を保存した話者は、合成時と同じ埋め込みになる。"""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wrapper"))
import editor_engine  # noqa: E402
from tts_cli import IrodoriTTS, SpeakerCassette, mix_line_speakers  # noqa: E402


def embedding(seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((1, 16, 768), generator=generator).to(torch.bfloat16)


class LineMixSaveTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        from safetensors.torch import save_file
        for index, name in enumerate(("ドラキナ", "hana", "verina")):
            save_file({"speaker_embedding": embedding(index)},
                      str(self.root / f"{name}.speaker.safetensors"))

    def tearDown(self):
        self.tmp.cleanup()

    def cassette(self):
        return SpeakerCassette([self.root], preferred_dim=768)

    def synthesized_embedding(self, speaker, strength, additional):
        tts = IrodoriTTS.__new__(IrodoriTTS)
        tts.cassette = self.cassette()
        tts.default_speaker = speaker
        runtime = types.SimpleNamespace(model_cfg=types.SimpleNamespace(speaker_dim=768))
        tts.backend = types.SimpleNamespace(
            runtime=runtime,
            synthesize=Mock(return_value={"backend": "fake", "wall_s": 0.0, "audio_s": 1.0}))
        tts.synthesize(text="テスト", speaker=speaker, out_wav=Mock(),
                       speaker_strength=strength, additional_speakers=additional)
        return tts.backend.synthesize.call_args.args[0].ref_embed

    def adapter(self):
        adapter = editor_engine.EditorAdapter.__new__(editor_engine.EditorAdapter)
        adapter.id_to_name = {0: "", 1: "ドラキナ", 2: "hana", 3: "verina"}
        adapter._model_speaker_dim = lambda: 768
        adapter.refresh = lambda: None
        return adapter

    def test_mix_matches_what_synthesis_uses(self):
        additional = [("hana", 0.7), ("verina", 0.61)]
        expected = self.synthesized_embedding("ドラキナ", 0.56, additional)
        mixed = mix_line_speakers(self.cassette(), "ドラキナ", 0.56, additional, 768)
        self.assertEqual(mixed.dtype, expected.dtype)
        self.assertTrue(torch.equal(mixed, expected))

    def test_saved_speaker_sounds_the_same_at_strength_one(self):
        additional = [("hana", 0.7), ("verina", 0.61)]
        expected = self.synthesized_embedding("ドラキナ", 0.56, additional)
        with patch.object(editor_engine, "SPEAKER_DIR", self.root), \
             patch.object(editor_engine, "resolve_embed_dirs", lambda: [self.root]):
            result = self.adapter().line_mix_save({
                "name": "お気に入り", "speakerStyleId": 1, "speakerStrength": 0.56,
                "additionalSpeakers": [{"styleId": 2, "strength": 0.7},
                                       {"styleId": 3, "strength": 0.61}]})
        self.assertEqual(result["name"], "お気に入り")
        folder = self.root / editor_engine.LINE_MIX_FOLDER
        self.assertTrue((folder / "お気に入り.768.speaker.safetensors").exists())
        recipe = json.loads((folder / "お気に入り.line-mix.json").read_text(encoding="utf-8"))
        self.assertEqual(recipe["speaker"], "ドラキナ")
        self.assertEqual(recipe["additionalSpeakers"][1], {"speaker": "verina", "strength": 0.61})
        # 保存した話者を強度1・追加話者なしで使うと、合成に渡る値が1ビットも変わらない
        again = self.synthesized_embedding("お気に入り", 1.0, [])
        self.assertTrue(torch.equal(again.reshape(expected.shape), expected))

    def test_bad_names_and_duplicates_are_rejected(self):
        payload = {"speakerStyleId": 1, "speakerStrength": 0.5,
                   "additionalSpeakers": [{"styleId": 2, "strength": 0.5}]}
        with patch.object(editor_engine, "SPEAKER_DIR", self.root), \
             patch.object(editor_engine, "resolve_embed_dirs", lambda: [self.root]):
            adapter = self.adapter()
            for bad in ("", "a/b", "x.768", "CON", "あ" * 33):
                with self.assertRaises(ValueError):
                    adapter.line_mix_save({**payload, "name": bad})
            with self.assertRaisesRegex(ValueError, "もうあります"):
                adapter.line_mix_save({**payload, "name": "hana"})
            with self.assertRaises(ValueError):
                adapter.line_mix_save({**payload, "name": "ok", "additionalSpeakers": []})
            with self.assertRaises(ValueError):
                adapter.line_mix_save({**payload, "name": "ok", "additionalSpeakers": [
                    {"styleId": 1, "strength": 0.5}]})


if __name__ == "__main__":
    unittest.main()
