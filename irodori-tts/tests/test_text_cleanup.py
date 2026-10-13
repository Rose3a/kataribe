"""コピーした文の前処理（text_cleanup.py）のテスト。

改行・見えない文字・飾り記号はバイトのトークンや学習で見ていない記号として入り、ない音の元になる。
意味のある記号は読みにする。検証は work/line_breaks/ と work/special_chars/。
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wrapper"))
from text_cleanup import clean_pasted_text as clean  # noqa: E402


class LineBreakTest(unittest.TestCase):
    def test_runs_of_breaks_become_one_pause(self):
        self.assertEqual(clean("場合は\r\n\r\n障害年金"), "場合は、障害年金")
        self.assertEqual(clean("あ\n \n\t\n　\nい"), "あ、い")
        self.assertEqual(clean("あ\u000cい う"), "あ、い、う")

    def test_break_after_punctuation_is_removed(self):
        self.assertEqual(clean("あります。\r\n\r\n精神"), "あります。精神")
        self.assertEqual(clean("本当？\nうん"), "本当?うん")

    def test_outer_breaks_and_spaces(self):
        self.assertEqual(clean("\r\n\r\nこんにちは\r\n"), "こんにちは")
        self.assertEqual(clean("あ  \n  い"), "あ、い")
        self.assertEqual(clean("改行 なし"), "改行 なし")


class InvisibleAndHtmlTest(unittest.TestCase):
    def test_invisible_characters_are_removed(self):
        self.assertEqual(clean("今日は​‪晴れ‬﻿"), "今日は晴れ")

    def test_split_mark_is_kept(self):
        self.assertEqual(clean("浦和⁣レッズ"), "浦和⁣レッズ")

    def test_html(self):
        self.assertEqual(clean("行1<br>行2<BR/>行3"), "行1、行2、行3")
        self.assertEqual(clean("A&amp;B&nbsp;C"), "A&B C")
        self.assertEqual(clean("Q&Aです"), "Q&Aです")


class SymbolTest(unittest.TestCase):
    def test_line_head_bullets_are_removed(self):
        self.assertEqual(clean("■注意事項\n・持ち物\n→次へ"), "注意事項、持ち物、次へ")
        self.assertEqual(clean("※1 税込み価格です。"), "税込み価格です。")

    def test_marks_with_meaning_are_kept(self):
        self.assertEqual(clean("○×クイズ"), "○×クイズ")
        self.assertEqual(clean("今日も頑張ろう♪"), "今日も頑張ろう♪")
        self.assertEqual(clean("1〜3人"), "1〜3人")

    def test_readings(self):
        cases = {
            "料金は¥1,000です。": "料金は1,000円です。",
            "￥５００": "500円",
            "送料は$100です。": "送料は100ドルです。",
            "3×4は12": "3かける4は12",
            "10÷2": "10わる2",
            "±5%": "プラスマイナス5%",
            "売上≒100万円": "売上約100万円",
            "A≠B": "AノットイコールB",
            "会議は12:30から": "会議は12時30分から",
            "10:00開始": "10時開始",
            "16:9の画面": "16:9の画面",
            "2024/10/11まで": "2024年10月11日まで",
            "番号No.1": "番号ナンバー1",
            "ver.2": "バージョン2",
            "100万円〜のプラン": "100万円からのプラン",
            "①申請": "1申請",
            "A→B→C": "A、B、C",
            "https://example.com を見て": "example.com を見て",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(clean(source), expected)

    def test_large_quantities_are_grouped_by_man(self):
        cases = {
            "料金は1234567円です。": "料金は123万4567円です。",
            "参加者は100000人": "参加者は10万人",
            "人口は2,086,901人です": "人口は208万6901人です",
            "¥1,234,567": "123万4567円",
            "総額123456789円": "総額1億2345万6789円",
            "負債3000500000円": "負債30億50万円",
            "1,234,567": "123万4567",
            "12345円": "1万2345円",
            "1234円": "1234円",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(clean(source), expected)

    def test_identifiers_are_not_grouped(self):
        for text in ["注文番号は20240917です。", "電話は0312345678", "ID:1234567", "1234567.5円",
                     "100000", "10000000m", "2024年"]:
            with self.subTest(text=text):
                self.assertEqual(clean(text), text)

    def test_long_version_is_read_digit_by_digit(self):
        self.assertEqual(clean("ver.200009"), "バージョンニーゼロゼロゼロゼロキュー")
        self.assertEqual(clean("ver.12"), "バージョン12")

    def test_postal_code(self):
        self.assertEqual(clean("〒100-0001 東京都"), "郵便番号、イチゼロゼロのゼロゼロゼロイチ、東京都")


class ReadingFixTest(unittest.TestCase):
    def test_ruby_keeps_only_the_reading(self):
        self.assertEqual(clean("漢字《かんじ》を読む"), "かんじを読む")
        self.assertEqual(clean("｜東京《とうきょう》タワー"), "とうきょうタワー")
        self.assertEqual(clean("|魔法《まほう》少女"), "まほう少女")

    def test_emoji_joiner_is_kept_only_between_emoji(self):
        sigh = "😮‍💨"
        self.assertEqual(clean(sigh + "はぁ"), sigh + "はぁ")
        self.assertEqual(clean("a‍b"), "ab")

    def test_compatibility_characters(self):
        self.assertEqual(clean("№1"), "ナンバー1")
        self.assertEqual(clean("第Ⅲ章"), "第3章")
        self.assertEqual(clean("エリザベスⅡ世"), "エリザベス2世")
        self.assertEqual(clean("ドラクエⅢ"), "ドラクエスリー")
        self.assertEqual(clean("5ℓ"), "5l")
        self.assertEqual(clean("Windows™"), "Windows")
        self.assertEqual(clean("…まあね"), "…まあね")

    def test_negative_numbers_and_ranges(self):
        self.assertEqual(clean("気温は−3℃"), "気温はマイナス3°C")
        self.assertEqual(clean("最低気温-5℃"), "最低気温マイナス5°C")
        self.assertEqual(clean("差は -2 です"), "差は マイナス2 です")
        self.assertEqual(clean("10時-12時"), "10時から12時")
        self.assertEqual(clean("5-10分"), "5から10分")
        self.assertEqual(clean("03-1234-5678"), "03-1234-5678")
        self.assertEqual(clean("COVID-19"), "COVID-19")

    def test_abbreviations_and_ordinals(self):
        self.assertEqual(clean("1st stage"), "ファースト stage")
        self.assertEqual(clean("2ndシーズン"), "セカンドシーズン")
        self.assertEqual(clean("25th"), "25番目")
        self.assertEqual(clean("Dr. Smith"), "ドクター Smith")
        self.assertEqual(clean("p.12を参照"), "12ページを参照")
        self.assertEqual(clean("vol.2"), "ボリューム2")
        self.assertEqual(clean("AM9:00に集合"), "午前9時に集合")
        self.assertEqual(clean("9:30 PMまで"), "午後9時30分まで")

    def test_title_cross_and_laughter(self):
        self.assertEqual(clean("HUNTER×HUNTERを見た"), "HUNTER HUNTERを見た")
        self.assertEqual(clean("3×4"), "3かける4")
        self.assertEqual(clean("草www"), "草")
        self.assertEqual(clean("それなｗ"), "それな")
        self.assertEqual(clean("www.example.com"), "www.example.com")


if __name__ == "__main__":
    unittest.main()
