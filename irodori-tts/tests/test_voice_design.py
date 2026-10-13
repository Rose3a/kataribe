import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wrapper"))

import voice_design  # noqa: E402
from tts_cli import SpeakerCassette  # noqa: E402

WAV = b"RIFF0000WAVEfake"


def _take(store, **params):
    defaults = dict(text="こんにちは", caption="落ち着いた女性", seed=1, steps=8, model="m")
    defaults.update(params)
    return store.get(store.put(WAV, defaults))


class VoiceLibraryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.library = voice_design.VoiceLibrary(self.root)
        self.takes = voice_design.TakeStore()
        self.encoded = []

    def encode(self, voice_file):
        self.encoded.append(voice_file)
        voice_design.latent_path(voice_file, voice_design.DEFAULT_CODEC_REPO).write_bytes(b"latent")

    def test_create_lists_and_keeps_the_recipe(self):
        voice_id = self.library.create(_take(self.takes), " ナレーター ", encode=self.encode)
        self.assertRegex(voice_id, voice_design.VOICE_ID)
        [voice] = self.library.list()
        self.assertEqual((voice["id"], voice["name"], voice["caption"], voice["seed"]),
                         (voice_id, "ナレーター", "落ち着いた女性", 1))
        self.assertEqual(self.library.audio(voice_id), WAV)
        self.assertEqual(len(self.encoded), 1)

    def test_only_takes_generated_by_the_engine_can_be_saved(self):
        with self.assertRaises(ValueError):
            self.takes.get("not-a-take")

    def test_duplicate_names_are_rejected(self):
        self.library.create(_take(self.takes), "A", encode=self.encode)
        with self.assertRaises(ValueError):
            self.library.create(_take(self.takes), "A", encode=self.encode)

    def test_failed_encoding_leaves_nothing_behind(self):
        def broken(_):
            raise RuntimeError("gpu error")
        with self.assertRaises(RuntimeError):
            self.library.create(_take(self.takes), "A", encode=broken)
        self.assertEqual(list((self.root / voice_design.VOICE_FOLDER).iterdir()), [])

    def test_rename_and_replace_keep_the_id(self):
        voice_id = self.library.create(_take(self.takes), "A", encode=self.encode)
        self.library.update(voice_id, name="B")
        replacement = self.takes.get(self.takes.put(b"RIFFnew", dict(caption="明るい少年", seed=9)))
        self.library.update(voice_id, take=replacement, encode=self.encode)
        [voice] = self.library.list()
        self.assertEqual((voice["id"], voice["name"], voice["caption"], voice["seed"]),
                         (voice_id, "B", "明るい少年", 9))
        self.assertEqual(self.library.audio(voice_id), b"RIFFnew")
        self.assertEqual(len(self.encoded), 2)

    def test_delete_and_bad_ids(self):
        voice_id = self.library.create(_take(self.takes), "A", encode=self.encode)
        with self.assertRaises(ValueError):
            self.library.delete("../tsukuyomi")
        self.library.delete(voice_id)
        self.assertEqual(self.library.list(), [])

    def test_take_key_matches_the_same_line_settings(self):
        query = {"irodori_text": "こんにちは", "irodori_caption": "落ち着いた男性の声", "irodori_seed": 1}
        same = dict(reversed(list(query.items())))
        self.assertEqual(voice_design.take_key(query, 0), voice_design.take_key(same, 0))
        self.assertNotEqual(voice_design.take_key(query, 0), voice_design.take_key(query, 5))
        self.assertNotEqual(voice_design.take_key(query, 0),
                            voice_design.take_key({**query, "irodori_seed": 2}, 0))

    def test_take_store_keeps_the_latest_take_per_key(self):
        store = voice_design.TakeStore()
        store.put(b"old", {}, key="k")
        store.put(b"new", {}, key="k")
        self.assertEqual(store.get("k")["wav"], b"new")

    def test_take_store_forgets_old_takes(self):
        store = voice_design.TakeStore(limit=2)
        first = store.put(WAV, {})
        store.put(WAV, {})
        store.put(WAV, {})
        with self.assertRaises(ValueError):
            store.get(first)


class VoiceCassetteTest(unittest.TestCase):
    def test_saved_voice_is_a_speaker_without_an_embedding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = voice_design.VoiceLibrary(root)
            takes = voice_design.TakeStore()
            voice_id = library.create(_take(takes), "ナレーター")
            cassette = SpeakerCassette([root])
            self.assertIn(voice_id, cassette.speakers)
            self.assertTrue(cassette.is_voice(voice_id))
            self.assertEqual(cassette.display_name(voice_id), "ナレーター")
            self.assertIsNone(cassette.dim_for(voice_id))  # どのモデルでも使える
            self.assertEqual(cassette.folder_for(voice_id), voice_design.VOICE_FOLDER)
            with self.assertRaises(ValueError):
                cassette.get(voice_id)  # 追加話者・マージには使えない
            library.update(voice_id, name="語り手")
            self.assertEqual(SpeakerCassette([root]).display_name(voice_id), "語り手")
            meta = json.loads((root / voice_design.VOICE_FOLDER / voice_id / f"{voice_id}.voice.json")
                              .read_text(encoding="utf-8"))
            self.assertEqual(meta["version"], 1)


if __name__ == "__main__":
    unittest.main()
