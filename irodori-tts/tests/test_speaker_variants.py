"""契約テスト: 1人の話者がモデルの次元ごとに埋め込みを持てる。

``tsukuyomi.speaker.safetensors``（768, v4.1 Small）と
``tsukuyomi.1280.speaker.safetensors``（1280, v4-Large）は同じ話者
``tsukuyomi`` として扱い、ID・画像・クレジットを共有する。
"""
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wrapper"))
from editor_engine import EditorAdapter  # noqa: E402
from speaker_catalog import credit_for, portrait_for, speaker_stem  # noqa: E402
from tts_cli import SpeakerCassette  # noqa: E402


def _embedding(path, dim):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"speaker_embedding": torch.full((16, dim), float(dim))}, str(path))


class SpeakerVariantTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        folder = self.root / "tsukuyomi"
        self.small = folder / "tsukuyomi.speaker.safetensors"
        self.large = folder / "tsukuyomi.1280.speaker.safetensors"
        _embedding(self.small, 768)
        _embedding(self.large, 1280)
        _embedding(self.root / "only_small" / "only_small.speaker.safetensors", 768)
        (folder / "tsukuyomi.png").write_bytes(b"\x89PNG\r\n\x1a\nportrait")
        (folder / "credit.txt").write_text("つくよみちゃん", encoding="utf-8")

    def test_width_tag_is_not_part_of_the_speaker_name(self):
        self.assertEqual(speaker_stem(self.large), "tsukuyomi")
        self.assertEqual(speaker_stem(self.small), "tsukuyomi")
        cassette = SpeakerCassette([self.root])
        self.assertEqual(sorted(cassette.speakers), ["only_small", "tsukuyomi"])

    def test_variant_follows_the_model_width(self):
        cassette = SpeakerCassette([self.root])
        # 幅が分からないときは元の（タグなしの）ファイル。
        self.assertEqual(cassette.path_for("tsukuyomi"), self.small)
        self.assertEqual(cassette.get("tsukuyomi", 1280).shape[-1], 1280)
        self.assertEqual(cassette.get("tsukuyomi", 768).shape[-1], 768)
        cassette.preferred_dim = 1280
        self.assertEqual(cassette.path_for("tsukuyomi"), self.large)
        self.assertEqual(cassette.dim_for("tsukuyomi"), 1280)
        self.assertEqual(cassette.folder_for("tsukuyomi"), "tsukuyomi")

    def test_large_model_lists_only_speakers_with_a_matching_variant(self):
        large = SpeakerCassette([self.root], preferred_dim=1280)
        self.assertEqual(EditorAdapter._usable_speakers(large, 1280), ["tsukuyomi"])
        small = SpeakerCassette([self.root], preferred_dim=768)
        self.assertEqual(sorted(EditorAdapter._usable_speakers(small, 768)),
                         ["only_small", "tsukuyomi"])

    def test_large_variant_shares_images_and_credit(self):
        cassette = SpeakerCassette([self.root], preferred_dim=1280)
        path = cassette.path_for("tsukuyomi")
        self.assertEqual(path, self.large)
        self.assertIsNotNone(portrait_for(path))
        self.assertEqual(credit_for(path), credit_for(self.small))


if __name__ == "__main__":
    unittest.main()
