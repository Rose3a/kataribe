"""文節の頭で区切る（bunsetsu.py）と、語彙分割への組み込みの契約テスト。

fugashi と unidic-lite が無ければ飛ばす。modernbert-ja の tokenizer.json があれば実物でも確かめる。
"""
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wrapper"))
import bunsetsu  # noqa: E402
import token_split  # noqa: E402
from reading_dictionary import ReadingDictionary  # noqa: E402
from token_split import (SPLIT_MARK, active_threshold, describe_tokens, find_tokenizer_json,  # noqa: E402
                         split_token_ids)

SMALL_TOKENIZER = "sbintuitions/modernbert-ja-310m"
LARGE_TOKENIZER = "google/t5gemma-2-1b-1b"


def shown(text):
    return text.replace(SPLIT_MARK, "|")


@unittest.skipUnless(bunsetsu.available(), "fugashi / unidic-lite が無い")
class MarkTests(unittest.TestCase):
    def test_cuts_only_at_the_head_of_each_bunsetsu(self):
        self.assertEqual(shown(bunsetsu.mark("国境の長いトンネルを抜けると雪国であった。")),
                         "国境の|長いトンネルを|抜けると|雪国で|あった。")
        self.assertEqual(shown(bunsetsu.mark("どちらに乗ったらいいか教えて下さい。")),
                         "どちらに|乗ったら|いいか|教えて|下さい。")

    def test_compounds_and_inflections_are_not_cut(self):
        # 全部の形態素の境界で切ると 中|小|企業・遊|ん|だ になる
        for text in ("中小企業", "遊んだ", "資生堂", "心筋梗塞", "セブンイレブン"):
            self.assertEqual(bunsetsu.mark(text), text)

    def test_existing_marks_and_spaces_are_kept(self):
        self.assertEqual(shown(bunsetsu.mark(f"浦和{SPLIT_MARK}レッズが勝った")), "浦和|レッズが|勝った")
        self.assertEqual(shown(bunsetsu.mark("Hello world を見た")), "Hello world を|見た")
        self.assertEqual(bunsetsu.mark(""), "")

    def test_emoji_stay_together(self):
        text = "あははっ🤭、それ本当？😮‍💨"
        self.assertEqual(bunsetsu.mark(text).replace(SPLIT_MARK, ""), text)
        self.assertNotIn(SPLIT_MARK + "‍", bunsetsu.mark(text))


@unittest.skipUnless(bunsetsu.available() and find_tokenizer_json(SMALL_TOKENIZER),
                     "fugashi か modernbert-ja の tokenizer.json が無い")
class EncodingTests(unittest.TestCase):
    def setUp(self):
        from transformers import AutoTokenizer
        self.hf = AutoTokenizer.from_pretrained(str(find_tokenizer_json(SMALL_TOKENIZER).parent))

    def pieces(self, ids):
        return "|".join(self.hf.convert_ids_to_tokens(ids))

    def test_bunsetsu_with_threshold(self):
        text = "どちらに乗ったらいいか教えて下さい。"
        with active_threshold(-13.0):
            self.assertEqual(self.pieces(split_token_ids(self.hf, text)), "どちら|に乗|ったら|いいか|教えて|下さい|。")
        with active_threshold(-13.0, bunsetsu=True):
            self.assertEqual(self.pieces(split_token_ids(self.hf, text)), "どちら|に|乗った|ら|いいか|教えて|下さい|。")
        # 外れたら元に戻る
        self.assertFalse(token_split._BUNSETSU.get())

    def test_bunsetsu_alone_without_threshold(self):
        with active_threshold(None, bunsetsu=True):
            self.assertEqual(self.pieces(split_token_ids(self.hf, "駅はありますか")), "駅|は|ありますか")

    def test_tokenize_view(self):
        view = describe_tokens(["駅はありますか"], threshold=-13.0, bunsetsu=True)
        self.assertTrue(view["bunsetsu"])
        result = view["results"][0]
        self.assertEqual([t["text"] for t in result["tokens"]], ["駅", "は", "ありますか"])
        # 文節の区切りは、手で入れた区切り（split）としては見せない
        self.assertFalse(any(t.get("split") for t in result["tokens"]))
        self.assertEqual([t["text"] for t in result["original"]], ["駅", "はありますか"])
        self.assertFalse(describe_tokens(["駅はありますか"], threshold=-13.0)["bunsetsu"])


class SynthesisTests(unittest.TestCase):
    """合成のあいだだけ、全体の設定とセリフの語彙分割に従って文節の区切りが有効になる。"""

    def setUp(self):
        import voicevox_engine as engine
        self.engine = engine
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patcher = patch.object(engine, "READING_DICTIONARY",
                               ReadingDictionary(Path(tmp.name) / "user_dictionary.json"))
        patcher.start()
        self.addCleanup(patcher.stop)

    def active(self, bunsetsu_on, repo=SMALL_TOKENIZER, **extra):
        adapter = self.engine.VoicevoxAdapter.__new__(self.engine.VoicevoxAdapter)
        adapter.id_to_name = {0: "話者なし"}
        adapter.lock = threading.Lock()
        adapter.progress_callback = None
        adapter.tts = Mock()
        seen = []
        adapter.tts.synthesize.side_effect = lambda **kw: (seen.append((kw["text"], token_split._BUNSETSU.get())),
                                                         kw["out_wav"].write(b"wav"))
        adapter.token_split_threshold = "-13"
        adapter.token_split_bunsetsu = bunsetsu_on
        adapter.text_tokenizer_repo = repo
        adapter.synthesize({**self.engine._query("駅はありますか"), **extra}, 0)
        text, active = seen[-1]
        self.assertEqual(text, "駅はありますか")  # 文字列は書き換えない
        return active

    def test_setting_and_line_and_model(self):
        self.assertTrue(self.active(True))
        self.assertFalse(self.active(False))
        self.assertFalse(self.active(True, irodori_token_split="off"))
        self.assertFalse(self.active(True, repo=LARGE_TOKENIZER))
        self.assertFalse(token_split._BUNSETSU.get())

    def test_old_adapter_without_the_attribute(self):
        adapter = self.engine.VoicevoxAdapter.__new__(self.engine.VoicevoxAdapter)
        self.assertFalse(adapter.split_bunsetsu())


class EditorSettingTests(unittest.TestCase):
    def test_validated_and_off_by_default(self):
        from editor_engine import EditorAdapter
        self.assertIs(EditorAdapter.DEFAULT_SETTINGS["token_split_bunsetsu"], False)
        adapter = EditorAdapter.__new__(EditorAdapter)
        with patch.object(EditorAdapter, "model_info", return_value={}, create=True):
            settings = dict(EditorAdapter.DEFAULT_SETTINGS)
            adapter.validate({**settings, "token_split_bunsetsu": True})
            with self.assertRaisesRegex(ValueError, "token_split_bunsetsu"):
                adapter.validate({**settings, "token_split_bunsetsu": "on"})


if __name__ == "__main__":
    unittest.main()
