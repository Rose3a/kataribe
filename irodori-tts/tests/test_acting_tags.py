"""他の TTS の演技タグを Irodori-TTS の絵文字にする（acting_tags.py）テスト。

そのままだと英単語のカナ化で [laughing] → [ラフィング] と読み上げる（合成と kana-whisper で確認済み）。
"""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wrapper"))
import editor_engine  # noqa: E402
from acting_tags import lookup, replace_acting_tags  # noqa: E402
from reading_dictionary import ReadingDictionary  # noqa: E402
from text_cleanup import clean_pasted_text as clean  # noqa: E402
from token_split import active_threshold, find_tokenizer_json, split_token_ids  # noqa: E402

SIGH = "😮‍💨"
SMALL_TOKENIZER = "sbintuitions/modernbert-ja-310m"


class TableTest(unittest.TestCase):
    def test_main_tags(self):
        cases = {
            "[laughing]": "🤭", "<laugh>": "🤭", "(laughs)": "🤭", "[chuckles]": "🤭", "<giggle>": "🤭",
            "[laughter]": "🤭", "(笑)": "🤭", "（笑）": "🤭",
            "<breath>": SIGH, "[sighs]": SIGH, "(sighing)": SIGH, "[exhales]": SIGH, "（ため息）": SIGH,
            "<heavy breath>": "🌬️", "(panting)": "🌬️", "[gasps]": "😮", "(groans)": "🥵",
            "[sobbing]": "😭", "[crying]": "😭", "<scream>": "😱", "[whispers]": "👂", "<whispering>": "👂",
            "[yawning]": "🥱", "[snoring]": "😴", "(coughs)": "🤧", "(clears throat)": "🤧",
            "<throat-clearing>": "🤧", "(sneezes)": "🤧", "<tsk>": "😒", "[gulps]": "🥤", "[lipsmack]": "💋",
            "(humming)": "🎵", "[sarcastic]": "😏", "[excited]": "😆", "(happy)": "😆", "[angry]": "😠",
            "[curious]": "🤔", "[mischievously]": "😏", "[whispers sweetly]": "👂", "[crying loudly]": "😭😭",
            "[laughs harder]": "🤭🤭", "[starts laughing]": "🤭", "<strong>": "💪",
        }
        for tag, emoji in cases.items():
            with self.subTest(tag=tag):
                self.assertEqual(replace_acting_tags(tag), emoji)

    def test_pauses(self):
        self.assertEqual(clean("えっと[short pause]そうね。"), "えっと⏸️そうね。")
        self.assertEqual(clean("えっと<short pause>そうね。"), "えっと⏸️そうね。")
        self.assertEqual(clean("えっと[medium pause]そうね。"), "えっと⏸️⏸️そうね。")
        self.assertEqual(clean("えっと<long pause>そうね。"), "えっと⏸️⏸️⏸️そうね。")
        self.assertEqual(clean("えっと[pause]そうね。"), "えっと⏸️そうね。")
        self.assertEqual(clean("えっと[long-break]そうね。"), "えっと⏸️⏸️⏸️そうね。")
        self.assertEqual(clean("えっと（間）そうね。"), "えっと⏸️そうね。")

    def test_break_time_becomes_number_of_pauses(self):
        self.assertEqual(clean('待って<break time="300ms"/>ね'), "待って⏸️ね")
        self.assertEqual(clean('待って<break time="1.2s" />ね'), "待って⏸️⏸️⏸️ね")
        self.assertEqual(clean("待って<break time='10s'/>ね"), "待って⏸️⏸️⏸️⏸️⏸️ね")
        self.assertEqual(clean("待って<break/>ね"), "待って⏸️ね")

    def test_other_forms(self):
        self.assertEqual(clean("*laughs* そう"), "🤭そう")
        self.assertEqual(clean("<|laughter|>ね"), "🤭ね")
        self.assertEqual(clean("うん|mm-hmm|それで"), "うん👌それで")
        self.assertEqual(clean("［笑］ね"), "🤭ね")
        self.assertEqual(clean("＜ため息＞ね"), SIGH + "ね")

    def test_range_tags_keep_the_body(self):
        self.assertEqual(clean("<laughter>そうね</laughter>、本当に"), "🤭そうね、本当に")
        self.assertEqual(clean("<strong>絶対</strong>に"), "💪絶対に")
        self.assertEqual(clean("[laughing]そうね[/laughing]"), "🤭そうね")

    def test_spaces_around_tags(self):
        self.assertEqual(clean("そうなんだ <breath> でも、ね。"), f"そうなんだ{SIGH}でも、ね。")
        self.assertEqual(clean("[laughing] そうなんだ。"), "🤭そうなんだ。")
        self.assertEqual(clean("Hello [laughs] world"), "Hello 🤭 world")


class DropAndKeepTest(unittest.TestCase):
    def test_unknown_square_and_angle_tags_are_removed(self):
        self.assertEqual(clean("[unknown thing]です"), "です")
        self.assertEqual(clean("<low, gravelly voice>です"), "です")
        self.assertEqual(clean("[1]注釈"), "注釈")

    def test_sound_effects_songs_accents_and_speakers_are_removed(self):
        self.assertEqual(clean("[gunshot]うわっ"), "うわっ")
        self.assertEqual(clean("[audience laughing]あはは"), "あはは")
        self.assertEqual(clean("(applause)ありがとう"), "ありがとう")
        self.assertEqual(clean("[sings]ラララ"), "ラララ")
        self.assertEqual(clean("[strong French accent] Bonjour"), "Bonjour")
        self.assertEqual(clean("[S1] こんにちは [S2] やあ"), "こんにちは やあ")

    def test_parentheses_not_in_the_table_are_kept(self):
        for text in ("(注)これは", "(仮)タイトル", "(sad story)を読む", "田中(32)は", "(see below)", "5*3*2"):
            self.assertEqual(clean(text), clean(text, acting_tags=False), text)
        self.assertEqual(clean("(仮)タイトル"), "(仮)タイトル")

    def test_html_breaks_still_become_pauses(self):
        self.assertEqual(clean("行1<br>行2<BR/>行3"), "行1、行2、行3")

    def test_lookup_returns_none_for_unknown(self):
        self.assertIsNone(lookup("注"))
        self.assertIsNone(lookup("see below", loose=True))
        self.assertEqual(lookup("gunshot"), "")


class LaughWTest(unittest.TestCase):
    """笑いの w は文を笑いながら言う印なので、文の頭に 😏 を置く（w 1つで3個、増えるごとに1個、10個まで）。"""

    def test_w_moves_to_sentence_head(self):
        self.assertEqual(clean("それはないw"), "😏😏😏それはない!")
        self.assertEqual(clean("え、うわｗｗｗｗ、それってまじか"), "😏" * 6 + "え、うわ!それってまじか")
        self.assertEqual(clean("マジかwww。次行こう"), "😏" * 5 + "マジか!次行こう")
        self.assertEqual(clean("いいよ。それはないww"), "いいよ。😏😏😏😏それはない!")

    def test_mark_after_w(self):
        # 後ろの ?！ はそのまま、行頭の www には ! を置かない
        self.assertEqual(clean("マジかwww？"), "😏" * 5 + "マジか?")
        self.assertEqual(clean("まじかよwww！"), "😏" * 5 + "まじかよ!")
        self.assertEqual(clean("wwwで草"), "😏" * 5 + "で草")

    def test_count_is_capped_at_ten(self):
        self.assertEqual(clean("草" + "w" * 8), "😏" * 10 + "草!")
        self.assertEqual(clean("草" + "w" * 30), "😏" * 10 + "草!")

    def test_w_only_becomes_laugh(self):
        self.assertEqual(clean("wwwww"), "🤭")

    def test_english_and_urls_are_untouched(self):
        for text in ("wow", "www.example.com", "Hello world"):
            self.assertEqual(clean(text), clean(text, acting_tags=False))

    def test_off_just_drops_w(self):
        self.assertEqual(clean("それはないw", acting_tags=False), "それはない")


class SwitchTest(unittest.TestCase):
    def test_off_changes_nothing_about_tags(self):
        for text in ("[laughing] そうなんだ。", "(笑)そうだね", "<laugh>あはは", "えっと[short pause]そうね"):
            self.assertNotIn("🤭", clean(text, acting_tags=False))
            self.assertNotIn("⏸", clean(text, acting_tags=False))
        self.assertEqual(clean("[laughing] そうなんだ。", acting_tags=False), "[laughing] そうなんだ。")

    def test_text_without_tags_is_unchanged(self):
        for text in ("今日はいい天気ですね。", "Q&Aです", "1/2の確率", "【重要】お知らせ"):
            self.assertEqual(clean(text), clean(text, acting_tags=False))

    def test_editor_setting(self):
        adapter = editor_engine.EditorAdapter()
        adapter.config_path = Path(tempfile.mkdtemp()) / "editor-settings.json"
        adapter.settings = dict(adapter.DEFAULT_SETTINGS, backend="cpu")
        self.assertTrue(adapter.DEFAULT_SETTINGS["acting_tags"])
        self.assertTrue(adapter.status()["settings"]["acting_tags"])
        adapter.configure({"acting_tags": False})
        self.assertFalse(adapter.settings["acting_tags"])
        with self.assertRaises(ValueError):
            adapter.configure({"acting_tags": "off"})


class LaterStepsTest(unittest.TestCase):
    """絵文字が英単語のカナ化（読み辞書）と語彙分割で壊れないこと。"""

    def test_reading_dictionary_keeps_emoji(self):
        dictionary = ReadingDictionary(Path(tempfile.mkdtemp()) / "user_dictionary.json")
        for text in ("[laughing] Hello world", "(sighs) OK、もういいよ", "えっと<long pause>そうね",
                     "[whispers] ここだけの話"):
            cleaned = clean(text)
            for english in ("katakana", "hiragana", "off"):
                converted = dictionary.convert(cleaned, english=english)
                for emoji in ("🤭", SIGH, "⏸️", "👂"):
                    self.assertEqual(cleaned.count(emoji), converted.count(emoji), (text, english))
        self.assertEqual(dictionary.convert(clean("[laughing] Hello")), "🤭ハロー")

    @unittest.skipUnless(find_tokenizer_json(SMALL_TOKENIZER), "modernbert-ja の tokenizer.json が無い")
    def test_token_split_keeps_emoji(self):
        from transformers import AutoTokenizer
        hf = AutoTokenizer.from_pretrained(str(find_tokenizer_json(SMALL_TOKENIZER).parent))
        for text in (f"{SIGH}もういいよ。", "えっと⏸️⏸️⏸️そうね。", "もう、🤭そんなに笑わないで。", "👂ここだけの話。"):
            plain = hf(text, add_special_tokens=False)["input_ids"]
            for bunsetsu in (False, True):
                with active_threshold(-13.0, bunsetsu=bunsetsu):
                    ids = split_token_ids(hf, text)
                self.assertEqual(hf.decode(ids).replace(" ", ""), hf.decode(plain).replace(" ", ""), text)
                self.assertNotIn(hf.unk_token_id, ids, text)


if __name__ == "__main__":
    unittest.main()
