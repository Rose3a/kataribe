import base64
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wrapper"))

from speaker_catalog import FALLBACK_ICON_PATH, credit_for, display_name_for, policy_for, portrait_for, speaker_catalog, thumbnail_for
from tts_cli import SpeakerCassette


class SpeakerCatalogTests(unittest.TestCase):
    def test_speaker_folder_is_relative_to_its_configured_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "direct.speaker.safetensors").write_bytes(b"embedding")
            grouped = (
                root
                / "idolmaster"
                / "idol"
                / "idol.speaker.safetensors"
            )
            grouped.parent.mkdir(parents=True)
            grouped.write_bytes(b"embedding")

            cassette = SpeakerCassette([root])

            self.assertIsNone(cassette.folder_for("direct"))
            self.assertEqual(cassette.folder_for("idol"), "idolmaster")

    def test_display_name_keeps_internal_identifier_out_of_the_ui(self):
        self.assertEqual(display_name_for("tsukuyomi"), "つくよみちゃん")
        self.assertEqual(display_name_for("external"), "external")

    def test_existing_raster_sidecar_wins_over_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            embedding = root / "speaker.safetensors"
            raster = root / "speaker.png"
            embedding.write_bytes(b"embedding")
            raster.write_bytes(b"png-payload")

            self.assertEqual(thumbnail_for(embedding), ("image/png", base64.b64encode(b"png-payload").decode("ascii")))
            self.assertEqual(portrait_for(embedding), ("image/png", base64.b64encode(b"png-payload").decode("ascii")))

    def test_icon_directory_wins_over_raster_sidecar(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            embedding = root / "speaker.safetensors"
            embedding.write_bytes(b"embedding")
            (root / "icon").mkdir()
            (root / "icon" / "speaker.png").write_bytes(b"icon-payload")
            (root / "speaker.png").write_bytes(b"sidecar-payload")

            self.assertEqual(
                thumbnail_for(embedding),
                ("image/png", base64.b64encode(b"icon-payload").decode("ascii")),
            )

    def test_icon_file_next_to_embedding_wins_over_speaker_image(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            embedding = root / "tsukuyomi.speaker.safetensors"
            embedding.write_bytes(b"embedding")
            (root / "icon.png").write_bytes(b"icon-payload")
            (root / "tsukuyomi.png").write_bytes(b"speaker-payload")

            self.assertEqual(
                thumbnail_for(embedding),
                ("image/png", base64.b64encode(b"icon-payload").decode("ascii")),
            )

    def test_credit_json_and_txt_are_supported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            embedding = root / "speaker.safetensors"
            embedding.write_bytes(b"embedding")
            (root / "credits.json").write_text(
                '{"credit": "イラスト素材：花兎*様"}', encoding="utf-8"
            )
            self.assertEqual(credit_for(embedding), "イラスト素材：花兎*様")

            (root / "credits.json").unlink()
            (root / "credit.txt").write_text("イラスト素材：花兎*様\n", encoding="utf-8")
            self.assertEqual(policy_for(embedding), "イラスト素材：花兎*様")
            self.assertIsNone(credit_for(embedding))

    def test_full_notice_wins_and_is_scoped_to_speaker_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tsukuyomi = root / "tsukuyomi"
            tsukuyomi.mkdir()
            other = root / "other"
            other.mkdir()
            notice = "音声データのクレジット\n\n【禁止事項】\n■禁止事項の本文"
            (tsukuyomi / "credits.json").write_text(
                '{"credit": "イラストのみ"}', encoding="utf-8"
            )
            (tsukuyomi / "credit.txt").write_text(notice, encoding="utf-8")
            self.assertEqual(policy_for(tsukuyomi / "tsukuyomi.speaker.safetensors"), notice)
            self.assertEqual(credit_for(tsukuyomi / "tsukuyomi.speaker.safetensors"), "イラストのみ")
            self.assertIsNone(policy_for(other / "other.speaker.safetensors"))
            (tsukuyomi / "credit.txt").write_text("\n", encoding="utf-8")
            self.assertEqual(policy_for(tsukuyomi / "tsukuyomi.speaker.safetensors"), "イラストのみ")

    def test_recursive_external_speaker_uses_default_icon_and_portrait(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            embedding = root / "nested" / "external.speaker.safetensors"
            embedding.parent.mkdir()
            embedding.write_bytes(b"embedding")
            expected = base64.b64encode(FALLBACK_ICON_PATH.read_bytes()).decode("ascii")

            catalog = dict(speaker_catalog([root]))

            self.assertEqual(catalog["external"], ("image/png", expected))
            self.assertEqual(portrait_for(embedding), ("image/png", expected))


if __name__ == "__main__":
    unittest.main()
