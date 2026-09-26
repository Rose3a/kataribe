from pathlib import Path
import re
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wrapper"))
import english_reading as er
from reading_dictionary import ReadingDictionary


class EnglishReadingTests(unittest.TestCase):
    def test_bundled_dictionary_is_loaded(self):
        self.assertTrue(er.DATA_PATH.exists())
        self.assertGreater(len(er.dictionary()), 100000)

    def test_common_words(self):
        expected = {
            "server": "サーバー", "rental": "レンタル", "computer": "コンピューター",
            "data": "データ", "game": "ゲーム", "world": "ワールド", "station": "ステーション",
            "question": "クエスチョン", "english": "イングリッシュ", "cat": "キャット",
            "stop": "ストップ", "book": "ブック", "bed": "ベッド", "kids": "キッズ",
            "cats": "キャッツ", "push": "プッシュ", "water": "ウォーター", "talk": "トーク",
            "office": "オフィス", "car": "カー", "care": "ケア", "here": "ヒア",
            "table": "テーブル", "doctor": "ドクター", "text": "テキスト", "market": "マーケット",
            "seven": "セブン", "garden": "ガーデン", "support": "サポート", "music": "ミュージック",
            "quick": "クイック", "love": "ラブ", "player": "プレイヤー", "power": "パワー",
            "fire": "ファイヤー", "happy": "ハッピー", "fashion": "ファッション",
            "software": "ソフトウェア", "different": "ディファレント", "beautiful": "ビューティフル",
            "piano": "ピアノ", "university": "ユニバーシティー", "menu": "メニュー",
            "productivity": "プロダクティビティー", "rainbow": "レインボー", "channel": "チャンネル", "thank": "サンク", "legend": "レジェンド", "zelda": "ゼルダ",
        }
        for word, kana in expected.items():
            with self.subTest(word=word):
                self.assertEqual(er.word_to_kana(word), kana)

    def test_acronyms_are_spelled(self):
        self.assertEqual(er.word_to_kana("API"), "エーピーアイ")
        self.assertEqual(er.word_to_kana("usb"), "ユーエスビー")  # 辞書でも1文字ずつ読む語
        self.assertEqual(er.word_to_kana("HTML"), "エイチティーエムエル")  # 母音がない
        self.assertEqual(er.word_to_kana("DMM"), "ディーエムエム")  # 辞書にない
        self.assertEqual(er.word_to_kana("IT"), "アイティー")  # 大文字2文字は略語
        self.assertEqual(er.word_to_kana("NO"), "ノー")  # ただし普通の単語は読む
        self.assertEqual(er.word_to_kana("BOX"), "ボックス")
        self.assertEqual(er.word_to_kana("A"), "エー")
        self.assertEqual(er.word_to_kana("a"), "ア")

    def test_split_words_romaji_and_fallbacks(self):
        self.assertEqual(er.word_to_kana("GitHubActions"), "ギットハブアクションズ")
        self.assertEqual(er.word_to_kana("HTMLParser"), "エイチティーエムエルパーサー")
        self.assertEqual(er.word_to_kana("Irodori"), "イロドリ")
        self.assertEqual(er.word_to_kana("kataribe"), "カタリベ")
        self.assertEqual(er.word_to_kana("Wi-Fi"), "ワイファイ")
        self.assertEqual(er.word_to_kana("gradio"), "グラディオ")
        self.assertEqual(er.word_to_kana("gpu"), "ジーピーユー")
        self.assertEqual(er.word_to_kana("zzzxxyy"), "ゼットゼットゼットエックスエックスワイワイ")

    def test_sentences_keep_spacing_digits_and_japanese(self):
        self.assertEqual(er.convert_english("I love you."), "アイラブユー.")
        self.assertEqual(er.convert_english("The Legend of Zelda"), "ザレジェンドオブゼルダ")
        self.assertEqual(er.convert_english("レンタルサーバーはAWSで、python3を使う"),
                         "レンタルサーバーはエーダブリューエスで、パイソン3を使う")
        self.assertEqual(er.convert_english("I'm fine, thank you."), "アイムファイン,サンクユー.")

    def test_long_english_text(self):
        text = ("Welcome to our channel! Today, I'm going to show you how to set up a rental "
                "server and deploy your first web application in less than ten minutes.")
        kana = er.convert_english(text)
        self.assertNotRegex(kana, "[A-Za-z]")
        self.assertLessEqual(len(kana), len(text))
        self.assertTrue(kana.startswith("ウェルカムトゥーアワーチャンネル!トゥデイ,アイム"))

    def test_every_dictionary_word_becomes_katakana(self):
        bad = [w for w in er.dictionary() if not re.fullmatch(r"[ァ-ヴー]+", er.word_to_kana(w))]
        self.assertEqual(bad, [])

    def test_conversion_is_fast(self):
        er.dictionary()
        er.word_to_kana.cache_clear()
        text = "Today we test English words like server, rental, GitHub and API. " * 20
        started = time.perf_counter()
        er.convert_english(text)
        self.assertLess(time.perf_counter() - started, 0.5)

    def test_dictionary_loads_once_across_threads(self):
        with patch.object(er, "_words", None):
            results = []
            threads = [threading.Thread(target=lambda: results.append(er.dictionary()))
                       for _ in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertTrue(all(result is results[0] for result in results))

    def test_to_hiragana(self):
        self.assertEqual(er.to_hiragana("レンタルサーバー、ヴ、ｶﾀｶﾅ、漢字"),
                         "れんたるさーばー、ゔ、かたかな、漢字")


class ReadingOptionTests(unittest.TestCase):
    def test_adapter_passes_reading_options_to_dictionary(self):
        import voicevox_engine as engine
        adapter = engine.VoicevoxAdapter.__new__(engine.VoicevoxAdapter)
        adapter.id_to_name = {0: "話者なし"}
        adapter.lock = threading.Lock()
        adapter.progress_callback = None
        adapter.tts = Mock()
        adapter.tts.synthesize.side_effect = lambda **kwargs: kwargs["out_wav"].write(b"wav")
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        dictionary = ReadingDictionary(Path(tmp.name) / "dictionary.json")
        patcher = patch.object(engine, "READING_DICTIONARY", dictionary)
        patcher.start()
        self.addCleanup(patcher.stop)
        query = engine._query("server とサーバー")
        adapter.synthesize(dict(query), 0)
        self.assertEqual(adapter.tts.synthesize.call_args.kwargs["text"], "サーバーとサーバー")
        adapter.synthesize(dict(query, irodori_kana_style="hiragana"), 0)
        self.assertEqual(adapter.tts.synthesize.call_args.kwargs["text"], "さーばーとさーばー")
        adapter.synthesize(dict(query, irodori_english_reading="off"), 0)
        self.assertEqual(adapter.tts.synthesize.call_args.kwargs["text"], "server とサーバー")
        adapter.synthesize(dict(query, irodori_english_reading="hiragana"), 0)
        self.assertEqual(adapter.tts.synthesize.call_args.kwargs["text"], "さーばーとサーバー")
        for bad in (dict(irodori_kana_style="romaji"), dict(irodori_english_reading=True)):
            with self.assertRaises(ValueError):
                adapter.synthesize(dict(query, **bad), 0)

    def test_query_validation_accepts_reading_fields(self):
        import voicevox_engine as engine
        base = engine._query("text")
        for english in ("off", "katakana", "hiragana"):
            engine._validate_query(dict(base, irodori_english_reading=english,
                                        irodori_kana_style="hiragana"))
        for bad in (dict(irodori_english_reading=False), dict(irodori_kana_style="romaji")):
            with self.assertRaises(engine.RequestValidationError):
                engine._validate_query(dict(base, **bad))

    def test_editor_settings_no_longer_hold_reading_options(self):
        import editor_engine
        # 読み方はセリフごとの設定。共通設定には持たず、開発版の保存値は読み込み時に捨てる。
        base = editor_engine.EditorAdapter.DEFAULT_SETTINGS
        self.assertNotIn("english_reading", base)
        self.assertNotIn("kana_style", base)
        source = Path(editor_engine.__file__).read_text(encoding="utf-8")
        self.assertIn('saved.pop("english_reading", None)', source)
        self.assertNotIn("irodori_english_reading=self.settings", source)

if __name__ == "__main__":
    unittest.main()
