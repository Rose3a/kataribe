import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from unittest.mock import Mock, patch
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wrapper"))
from reading_dictionary import ReadingDictionary, make_word


class DictionaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "dictionary.json"
        self.dictionary = ReadingDictionary(self.path)

    def test_width_and_english_are_preserved(self):
        self.assertEqual(self.dictionary.convert("Ｈｅｌｌｏ！　world（カタカナ）＋１２３"),
                         "Hello! world(カタカナ)+123")
        self.assertEqual(self.dictionary.convert("A DMM i love you"),
                         "A DMM i love you")
        self.assertEqual(self.dictionary.convert("Irodori github gradio"),
                         "Irodori github gradio")
        self.assertEqual(self.dictionary.convert("zzzxxyy"), "zzzxxyy")

    def test_apostrophe_and_all_caps_tokens_do_not_raise(self):
        for text in ["ROCK'N ROLL", "O'BRIEN さん", "ROCK’N", "DON'T STOP"]:
            self.assertEqual(self.dictionary.convert(text), text)

    def test_user_entry_uses_longest_match(self):
        d = self.dictionary
        d.put(make_word("python3", "パイソンスリー"))
        d.put(make_word("githubactions", "ギットハブアクションズ"))
        # The longer user entry must win.
        self.assertEqual(d.convert("python3"), "パイソンスリー")
        self.assertEqual(d.convert("githubactions"), "ギットハブアクションズ")
        # English remains unchanged unless an explicit user entry matches.
        self.assertEqual(d.convert("python"), "python")
        self.assertEqual(d.convert("github"), "github")

    def test_override_longest_priority_and_no_cascade(self):
        d = self.dictionary
        d.put(make_word("hello", "トクシュ"))
        d.put(make_word("hello world", "ながい"))
        d.put(make_word("トクシュ", "ベツ"))
        d.put(make_word("HELLO", "ユウセン", priority=10))
        self.assertEqual(d.convert("Ｈｅｌｌｏ world! hello"), "ナガイ! ユウセン")
        self.assertEqual(ReadingDictionary(self.path).convert("hello"), "ユウセン")

    def test_update_delete_and_atomic_import(self):
        d = self.dictionary
        key = d.put(make_word("box", "ボックス"))
        d.put(make_word("box", "ハコ"), key)
        self.assertEqual(d.convert("box"), "ハコ")
        with self.assertRaises(ValueError):
            d.import_words({key: make_word("box", "ヨミ"), "invalid": {}}, True)
        self.assertEqual(d.convert("box"), "ハコ")
        d.import_words({key: make_word("box", "ヨミ")}, False)
        self.assertEqual(d.convert("box"), "ハコ")
        d.delete(key)
        self.assertEqual(ReadingDictionary(self.path).snapshot(), {})

    def test_invalid_words(self):
        for args in [("", "ヨミ"), ("a", "abc"), ("a", "ア", 2), ("a", "ア", 0, 11)]:
            with self.assertRaises(ValueError):
                make_word(*args)


class DictionaryApiTests(DictionaryTests):
    def test_synthesis_uses_latest_dictionary_and_checks_expanded_length(self):
        import voicevox_engine as engine
        adapter = engine.VoicevoxAdapter.__new__(engine.VoicevoxAdapter)
        adapter.id_to_name = {0: "話者なし"}
        adapter.lock = threading.Lock()
        adapter.progress_callback = None
        adapter.tts = Mock()
        adapter.tts.synthesize.side_effect = lambda **kwargs: kwargs["out_wav"].write(b"wav")
        with patch.object(engine, "READING_DICTIONARY", self.dictionary):
            query = engine._query("BOX！")
            key = self.dictionary.put(make_word("BOX", "ハコ"))
            self.assertEqual(adapter.synthesize(query, 0), b"wav")
            self.assertEqual(adapter.tts.synthesize.call_args.kwargs["text"], "ハコ!")
            self.assertEqual(query["irodori_text"], "BOX！")
            self.dictionary.put(make_word("BOX", "ア" * 257), key)
            with self.assertRaises(ValueError):
                adapter.synthesize(query, 0)

    def test_crud_and_query_without_model(self):
        import voicevox_engine as engine
        server = ThreadingHTTPServer(("127.0.0.1", 0), engine.Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        def request(method, path, body=None, authorized=True):
            conn = HTTPConnection("127.0.0.1", server.server_port)
            headers = {"Origin": "http://localhost:5173", "Content-Type": "application/json"}
            if authorized:
                headers["X-Irodori-Session"] = engine.SESSION_TOKEN
            conn.request(method, path, body, headers)
            response = conn.getresponse()
            data = response.read()
            conn.close()
            return response.status, json.loads(data) if data else None

        with patch.object(engine, "READING_DICTIONARY", self.dictionary):
            params = urlencode(dict(surface="BOX", pronunciation="ハコ", accent_type=0))
            self.assertEqual(request("POST", "/user_dict_word?" + params, authorized=False)[0], 403)
            status, key = request("POST", "/user_dict_word?" + params)
            self.assertEqual(status, 200)
            self.assertIn(key, request("GET", "/user_dict")[1])
            query = request("POST", "/audio_query?" + urlencode(dict(text="BOX！")))[1]
            self.assertEqual(query["kana"], "ハコ!")
            self.assertEqual(query["irodori_text"], "BOX！")
            self.assertEqual(request("PUT", f"/user_dict_word/{key}?" + params)[0], 204)
            self.assertEqual(request("POST", "/import_user_dict?override=true",
                                     json.dumps(self.dictionary.snapshot()))[0], 204)
            self.assertEqual(request("DELETE", f"/user_dict_word/{key}")[0], 204)
            self.assertEqual(request("GET", "/user_dict")[1], {})


if __name__ == "__main__":
    unittest.main()
