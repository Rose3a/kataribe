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

    def test_width_and_english_are_preserved_when_english_reading_is_off(self):
        convert = lambda text: self.dictionary.convert(text, english="off")
        self.assertEqual(convert("Ｈｅｌｌｏ！　world（カタカナ）＋１２３"),
                         "Hello! world(カタカナ)+123")
        self.assertEqual(convert("A DMM i love you"), "A DMM i love you")
        self.assertEqual(convert("Irodori github gradio"), "Irodori github gradio")
        self.assertEqual(convert("zzzxxyy"), "zzzxxyy")

    def test_english_is_read_as_katakana_by_default(self):
        d = self.dictionary
        self.assertEqual(d.convert("Ｈｅｌｌｏ！　world（カタカナ）＋１２３"),
                         "ハロー!ワールド(カタカナ)+123")
        self.assertEqual(d.convert("A DMM i love you"), "エーディーエムエムアイラブユー")
        self.assertEqual(d.convert("Irodori github gradio"), "イロドリギットハブグラディオ")
        self.assertEqual(d.convert("zzzxxyy"), "ゼットゼットゼットエックスエックスワイワイ")

    def test_apostrophe_and_all_caps_tokens_do_not_raise(self):
        for text in ["ROCK'N ROLL", "O'BRIEN さん", "ROCK’N", "DON'T STOP"]:
            self.assertEqual(self.dictionary.convert(text, english="off"), text)
            self.assertNotRegex(self.dictionary.convert(text), "[A-Za-z]")

    def test_user_entry_wins_over_english_reading(self):
        d = self.dictionary
        d.put(make_word("Zelda", "ぜるだ"))
        d.put(make_word("server", "サーバ"))
        self.assertEqual(d.convert("The Legend of Zelda の server"),
                         "ザレジェンドオブゼルダのサーバ")

    def test_hiragana_style_converts_all_katakana(self):
        d = self.dictionary
        d.put(make_word("ゼルダの伝説", "ぜるだのでんせつ"))
        self.assertEqual(d.convert("レンタルサーバーでゼルダの伝説を server", kana_style="hiragana"),
                         "れんたるさーばーでぜるだのでんせつをさーばー")
        self.assertEqual(d.convert("ｶﾀｶﾅ", kana_style="hiragana"), "かたかな")
        with self.assertRaises(ValueError):
            d.convert("text", kana_style="romaji")

    def test_english_reading_can_be_hiragana_only_for_english(self):
        d = self.dictionary
        d.put(make_word("Zelda", "ゼルダ"))
        # 英語の読みだけひらがなにし、元のカタカナ語と辞書の読みはそのまま。
        self.assertEqual(d.convert("I love レンタルサーバー and Zelda", english="hiragana"),
                         "あいらぶレンタルサーバーあんどゼルダ")
        self.assertEqual(d.convert("I love you", english="off"), "I love you")
        with self.assertRaises(ValueError):
            d.convert("text", english=True)

    def test_user_entry_uses_longest_match(self):
        d = self.dictionary
        d.put(make_word("python3", "パイソンスリー"))
        d.put(make_word("githubactions", "ギットハブアクションズ"))
        # The longer user entry must win.
        self.assertEqual(d.convert("python3"), "パイソンスリー")
        self.assertEqual(d.convert("githubactions"), "ギットハブアクションズ")
        # Without English reading, English remains unless a user entry matches.
        self.assertEqual(d.convert("python", english="off"), "python")
        self.assertEqual(d.convert("github", english="off"), "github")
        self.assertEqual(d.convert("python"), "パイソン")

    def test_override_longest_priority_and_no_cascade(self):
        d = self.dictionary
        d.put(make_word("hello", "トクシュ"))
        d.put(make_word("hello world", "ながい"))
        d.put(make_word("トクシュ", "ベツ"))
        d.put(make_word("HELLO", "ユウセン", priority=10))
        self.assertEqual(d.convert("Ｈｅｌｌｏ world! hello"), "ナガイ!ユウセン")
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
            query = request("POST", "/audio_query?" + urlencode(dict(text="BOX！", speaker=0)))[1]
            self.assertEqual(query["kana"], "ハコ!")
            self.assertEqual(query["irodori_text"], "BOX！")
            self.assertEqual(request("PUT", f"/user_dict_word/{key}?" + params)[0], 204)
            self.assertEqual(request("POST", "/import_user_dict?override=true",
                                     json.dumps(self.dictionary.snapshot()))[0], 204)
            self.assertEqual(request("DELETE", f"/user_dict_word/{key}")[0], 204)
            self.assertEqual(request("GET", "/user_dict")[1], {})


if __name__ == "__main__":
    unittest.main()
