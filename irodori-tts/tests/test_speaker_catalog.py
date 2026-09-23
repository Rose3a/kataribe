import base64
import hashlib
import json
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from urllib.request import Request, urlopen
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wrapper"))

from speaker_catalog import FALLBACK_ICON_PATH, credit_for, display_name_for, policy_for, portrait_for, speaker_catalog, thumbnail_for
from tts_cli import SpeakerCassette
from voicevox_engine import Handler, _speaker_list_payload, _speaker_resource_index, _speaker_table


class SpeakerCatalogTests(unittest.TestCase):
    def test_no_speaker_choice_does_not_look_for_embedding_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "wakamo.speaker.safetensors").write_bytes(b"embedding")
            cassette = SpeakerCassette([root])
            tts = SimpleNamespace(
                embed_dirs=[root],
                cassette=cassette,
                speakers=lambda: cassette.speakers,
            )

            speakers, id_to_name = _speaker_table(tts)

            self.assertEqual(id_to_name[1], "話者なし")
            self.assertEqual(id_to_name[2], "wakamo")
            self.assertEqual(len(speakers), 2)

    def test_speaker_info_returns_loopback_resource_urls(self):
        image_bytes = b"\x89PNG\r\n\x1a\nresource"
        image = base64.b64encode(image_bytes).decode("ascii")
        adapter = SimpleNamespace(speakers_json=[{
            "name": "speaker",
            "speaker_uuid": "speaker-uuid",
            "styles": [{"name": "ノーマル", "id": 1, "type": "talk"}],
            "version": "0.2.0",
            "icon": image,
            "portrait": image,
        }])

        class TestHandler(Handler):
            pass

        TestHandler.adapter = adapter
        server = ThreadingHTTPServer(("127.0.0.1", 0), TestHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address
        base_url = f"http://{host}:{port}"
        try:
            with urlopen(f"{base_url}/speakers") as response:
                speakers = json.load(response)
            self.assertNotIn("icon", speakers[0])
            self.assertNotIn("portrait", speakers[0])

            with urlopen(
                f"{base_url}/speaker_info?speaker_uuid=speaker-uuid&resource_format=url"
            ) as response:
                speaker_info = json.load(response)
            resource_url = speaker_info["style_infos"][0]["icon"]
            self.assertTrue(resource_url.startswith(f"{base_url}/irodori/resource/"))
            with urlopen(resource_url) as response:
                self.assertEqual(response.read(), image_bytes)
                self.assertEqual(response.headers["Content-Type"], "image/png")
                self.assertIn("immutable", response.headers["Cache-Control"])

            origin = "http://127.0.0.1:5173"
            session_request = Request(
                f"{base_url}/irodori/session", headers={"Origin": origin}
            )
            with urlopen(session_request) as response:
                token = json.load(response)["token"]
            bundle_request = Request(
                f"{base_url}/irodori/speaker_bundle",
                headers={"Origin": origin, "X-Irodori-Session": token},
            )
            with urlopen(bundle_request) as response:
                bundle = json.load(response)
            self.assertEqual(bundle["speakers"], speakers)
            self.assertEqual(bundle["speaker_infos"]["speaker-uuid"], speaker_info)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_resource_lookup_deduplicates_matching_image_data(self):
        image = base64.b64encode(b"\x89PNG\r\n\x1a\nresource").decode("ascii")
        handler = Handler.__new__(Handler)
        handler.adapter = SimpleNamespace(speakers_json=[{
            "icon": image,
            "portrait": image,
            "mouth_parts": {"a": image},
        }])

        digest = hashlib.sha256(image.encode("ascii")).hexdigest()

        self.assertEqual(handler._resource_value(digest), image)
        self.assertEqual(handler._image_mime(base64.b64decode(image)), "image/png")

    def test_resource_index_keeps_one_entry_for_reused_fallback(self):
        image = base64.b64encode(b"\x89PNG\r\n\x1a\nfallback").decode("ascii")

        resource_index = _speaker_resource_index([
            {"icon": image, "portrait": image},
            {"icon": image, "portrait": image},
        ])

        self.assertEqual(len(resource_index), 1)

    def test_speaker_list_omits_duplicated_image_payloads(self):
        speaker = {
            "name": "speaker",
            "speaker_uuid": "speaker-uuid",
            "styles": [{"name": "ノーマル", "id": 1, "type": "talk"}],
            "version": "0.2.0",
            "irodori_folder": "series",
            "icon": "large-base64-icon",
            "portrait": "large-base64-portrait",
            "mouth_open": "large-base64-mouth",
        }

        self.assertEqual(
            _speaker_list_payload([speaker]),
            [{
                "name": "speaker",
                "speaker_uuid": "speaker-uuid",
                "styles": [{"name": "ノーマル", "id": 1, "type": "talk"}],
                "version": "0.2.0",
                "irodori_folder": "series",
            }],
        )

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
            self.assertIsNone(credit_for(embedding))

            (root / "credits.json").unlink()
            (root / "credit.txt").write_text("イラスト素材：花兎*様\n", encoding="utf-8")
            self.assertEqual(policy_for(embedding), "イラスト素材：花兎*様")
            self.assertEqual(credit_for(embedding), "イラスト素材：花兎*様")

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
            self.assertEqual(credit_for(tsukuyomi / "tsukuyomi.speaker.safetensors"), notice)
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
