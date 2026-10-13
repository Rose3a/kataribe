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
                         "ハロー! ワールド(カタカナ)+123")
        self.assertEqual(d.convert("A DMM i love you"), "エー ディーエムエム アイ ラブ ユー")
        self.assertEqual(d.convert("Irodori github gradio"), "イロドリ ギットハブ グラディオ")
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
                         "ザ レジェンド オブ ゼルダ の サーバ")

    def test_hiragana_style_converts_all_katakana(self):
        d = self.dictionary
        d.put(make_word("ゼルダの伝説", "ぜるだのでんせつ"))
        self.assertEqual(d.convert("レンタルサーバーでゼルダの伝説を server", kana_style="hiragana"),
                         "れんたるさーばーでぜるだのでんせつを さーばー")
        self.assertEqual(d.convert("ｶﾀｶﾅ", kana_style="hiragana"), "かたかな")
        with self.assertRaises(ValueError):
            d.convert("text", kana_style="romaji")

    def test_english_reading_can_be_hiragana_only_for_english(self):
        d = self.dictionary
        d.put(make_word("Zelda", "ゼルダ"))
        # 英語の読みだけひらがなにし、元のカタカナ語と辞書の読みはそのまま。
        self.assertEqual(d.convert("I love レンタルサーバー and Zelda", english="hiragana"),
                         "あい らぶ レンタルサーバー あんど ゼルダ")
        self.assertEqual(d.convert("I love you", english="off"), "I love you")
        with self.assertRaises(ValueError):
            d.convert("text", english=True)

    def test_join_spacing_also_joins_dictionary_readings(self):
        d = self.dictionary
        d.put(make_word("Zelda", "ゼルダ"))
        self.assertEqual(d.convert("The Legend of Zelda の server", spacing="join"),
                         "ザレジェンドオブゼルダのサーバー")
        # 英語を変換しないときは空白も触らない。
        self.assertEqual(d.convert("I love Zelda", english="off", spacing="join"),
                         "I love ゼルダ")
        with self.assertRaises(ValueError):
            d.convert("text", spacing="none")

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
        d.put(make_word("hello", "ユウセン", priority=10))
        self.assertEqual(d.convert("ｈｅｌｌｏ world! hello"), "ナガイ! ユウセン")
        self.assertEqual(ReadingDictionary(self.path).convert("hello"), "ユウセン")

    def test_english_entries_respect_case_and_word_boundaries(self):
        d = self.dictionary
        d.put(make_word("AI", "エーアイ"))
        d.put(make_word("it", "イット"))
        # 英字の語は前後が英数字なら当てない（MAIL の中の AI）
        self.assertEqual(d.convert("MAILとAI", english="off"), "MAILとエーアイ")
        # 大文字小文字を区別する。小文字の登録は文頭の大文字始まりにも当たる
        self.assertEqual(d.convert("IT企業", english="off"), "IT企業")
        self.assertEqual(d.convert("It is", english="off"), "イット is")

    def test_entries_with_symbols_match_cleaned_text(self):
        from text_cleanup import clean_pasted_text
        d = self.dictionary
        d.put(make_word("HUNTER×HUNTER", "ハンターハンター"))
        self.assertEqual(d.convert(clean_pasted_text("HUNTER×HUNTERを見た")), "ハンターハンターを見た")

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
        for args in [("", "ヨミ"), ("a", "abc"), ("a", "ア", 2), ("a", "ア", 0, 11),
                     ("a", "浦和|レッズ")]:
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


class BuiltinReadingsTests(unittest.TestCase):
    """同梱の読み辞書（別枠）。ユーザー辞書が優先し、英語の読みより先に当たる。"""

    def setUp(self):
        from reading_dictionary import BuiltinReadings
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        path = Path(self.tmp.name) / "builtin.json"
        path.write_text(json.dumps({"entries": [
            {"surface": "太秦", "reading": "ウズマサ"}, {"surface": "一期一会", "reading": "イチゴイチエ"},
            {"surface": "IT", "reading": "アイティー"}, {"surface": "5G", "reading": "ファイブジー"},
            {"surface": "Wi-Fi", "reading": "ワイファイ"}],
            "units": [{"surface": "km", "reading": "キロメートル"}, {"surface": "m", "reading": "メートル"},
                      {"surface": "℃", "reading": "ド"}]}, ensure_ascii=False), encoding="utf-8")
        self.builtin = BuiltinReadings(path)
        self.dictionary = ReadingDictionary(Path(self.tmp.name) / "user.json", builtin=self.builtin)

    def test_entries_and_units(self):
        convert = self.dictionary.convert
        self.assertEqual(convert("京都の太秦で一期一会"), "京都のウズマサでイチゴイチエ")
        self.assertEqual(convert("10kmと5m、35℃"), "10キロメートルと5メートル、35ド")
        self.assertEqual(convert("5GとWi-Fi"), "ファイブジーとワイファイ")
        self.assertEqual(convert("IT企業"), "アイティー企業")

    def test_boundaries(self):
        convert = self.dictionary.convert
        # 英字は大文字小文字を区別し、前後が英数字なら当てない。単位は数字の直後だけ
        self.assertEqual(convert("it", english="off"), "it")
        self.assertEqual(convert("ITEM", english="off"), "ITEM")
        self.assertEqual(convert("5GHz", english="off"), "5GHz")
        self.assertEqual(convert("km と m", english="off"), "km と m")
        # 漢字の語は前後が漢字なら当てない
        self.assertEqual(convert("太秦駅"), "太秦駅")

    def test_user_dictionary_wins(self):
        self.dictionary.put(make_word("太秦", "タイシン"))
        self.assertEqual(self.dictionary.convert("太秦"), "タイシン")

    def test_reload_on_change_and_missing_file(self):
        path = self.builtin.path
        data = json.loads(path.read_text(encoding="utf-8"))
        data["entries"].append({"surface": "山車", "reading": "ダシ"})
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        import os
        os.utime(path, (path.stat().st_atime, path.stat().st_mtime + 5))
        self.assertEqual(self.dictionary.convert("山車"), "ダシ")
        path.unlink()
        self.assertEqual(self.dictionary.convert("山車"), "山車")

    def test_shipped_file_loads(self):
        from reading_dictionary import BUILTIN_READINGS
        self.assertEqual(BUILTIN_READINGS.apply("10kmの一期一会と5G"), "10キロメートルのいちごいちえとファイブジー")
