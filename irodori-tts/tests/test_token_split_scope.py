"""語彙分割辞書の対象（Small 系のみ / Large にも）の設定の契約テスト。

辞書は modernbert-ja のトークンの出現度で作ってある。Large（T5Gemma 2 のトークナイザ）は日本語の
語句を細かく割るので、既定（small）では辞書と同じトークナイザのモデルにだけ当てる。
all にすると Large にも当てる。実モデルは読まない。
"""
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wrapper"))
import token_split  # noqa: E402
from reading_dictionary import ReadingDictionary  # noqa: E402
from token_split import (SPLIT_MARK, TOKEN_SPLIT_SCOPES, TokenSplitDictionary,  # noqa: E402
                         describe_tokens, token_split_active)

SMALL_TOKENIZER = "sbintuitions/modernbert-ja-310m"
LARGE_TOKENIZER = "google/t5gemma-2-1b-1b"
VOCAB = ["浦和レッズ"]


def fake_offsets(text):
    out, i = [], 0
    while i < len(text):
        size = next((len(w) for w in VOCAB if text.startswith(w, i)), 1)
        out.append((i, i + size))
        i += size
    return out


class ScopeDecisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        path = Path(self.temp.name) / "dictionary.json"
        path.write_text(json.dumps({"version": 1, "tokenizer": SMALL_TOKENIZER, "entries": [
            {"surface": "浦和レッズ", "text": "浦和|レッズ"}]}, ensure_ascii=False), encoding="utf-8")
        self.dictionary = TokenSplitDictionary(path, tokenizer=fake_offsets)

    def test_dictionary_reports_its_tokenizer(self):
        self.assertEqual(self.dictionary.tokenizer_repo(), SMALL_TOKENIZER)

    def test_small_scope_applies_only_to_the_dictionary_tokenizer(self):
        self.assertTrue(token_split_active("small", SMALL_TOKENIZER, self.dictionary))
        self.assertFalse(token_split_active("small", LARGE_TOKENIZER, self.dictionary))

    def test_all_scope_applies_to_every_model(self):
        self.assertTrue(token_split_active("all", SMALL_TOKENIZER, self.dictionary))
        self.assertTrue(token_split_active("all", LARGE_TOKENIZER, self.dictionary))

    def test_none_scope_applies_to_no_model(self):
        for repo in (SMALL_TOKENIZER, LARGE_TOKENIZER, None, ""):
            self.assertFalse(token_split_active("none", repo, self.dictionary))

    def test_unknown_model_tokenizer_keeps_the_old_behaviour(self):
        # モデルが読み込まれる前や古い経路ではトークナイザが分からない。従来どおり当てる。
        self.assertTrue(token_split_active("small", None, self.dictionary))
        self.assertTrue(token_split_active("small", "", self.dictionary))

    def test_invalid_scope_is_rejected(self):
        with self.assertRaises(ValueError):
            token_split_active("large", SMALL_TOKENIZER, self.dictionary)
        self.assertEqual(TOKEN_SPLIT_SCOPES, ("small", "all", "none"))


class SynthesisScopeTests(unittest.TestCase):
    def setUp(self):
        import voicevox_engine as engine
        self.engine = engine
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        path = root / "dictionary.json"
        path.write_text(json.dumps({"version": 1, "tokenizer": SMALL_TOKENIZER, "entries": [
            {"surface": "浦和レッズ", "text": "浦和|レッズ"}]}, ensure_ascii=False), encoding="utf-8")
        self.split = TokenSplitDictionary(path, tokenizer=fake_offsets)
        self.reading = ReadingDictionary(root / "user_dictionary.json")
        for target, name in ((engine, "READING_DICTIONARY"),):
            patcher = patch.object(target, name, self.reading)
            patcher.start()
            self.addCleanup(patcher.stop)
        for module in (token_split, sys.modules["reading_dictionary"], engine):
            patcher = patch.object(module, "TOKEN_SPLIT_DICTIONARY", self.split)
            patcher.start()
            self.addCleanup(patcher.stop)

    def adapter(self, scope, repo):
        adapter = self.engine.VoicevoxAdapter.__new__(self.engine.VoicevoxAdapter)
        adapter.id_to_name = {0: "話者なし"}
        adapter.lock = threading.Lock()
        adapter.progress_callback = None
        adapter.tts = Mock()
        adapter.tts.synthesize.side_effect = lambda **kwargs: kwargs["out_wav"].write(b"wav")
        adapter.token_split_scope = scope
        adapter.text_tokenizer_repo = repo
        return adapter

    def spoken(self, adapter, **extra):
        query = {**self.engine._query("浦和レッズが勝った"), **extra}
        adapter.synthesize(query, 0)
        return adapter.tts.synthesize.call_args.kwargs["text"]

    def test_small_scope_splits_for_the_small_family(self):
        self.assertEqual(self.spoken(self.adapter("small", SMALL_TOKENIZER)),
                         "浦和" + SPLIT_MARK + "レッズ" + SPLIT_MARK + "が勝った")

    def test_small_scope_leaves_large_untouched(self):
        self.assertEqual(self.spoken(self.adapter("small", LARGE_TOKENIZER)), "浦和レッズが勝った")

    def test_none_scope_never_splits(self):
        for repo in (SMALL_TOKENIZER, LARGE_TOKENIZER):
            self.assertEqual(self.spoken(self.adapter("none", repo)), "浦和レッズが勝った")

    def test_all_scope_splits_for_large_too(self):
        self.assertEqual(self.spoken(self.adapter("all", LARGE_TOKENIZER)),
                         "浦和" + SPLIT_MARK + "レッズ" + SPLIT_MARK + "が勝った")

    def test_line_setting_off_wins_in_every_scope(self):
        for scope, repo in (("small", SMALL_TOKENIZER), ("all", SMALL_TOKENIZER), ("all", LARGE_TOKENIZER)):
            self.assertEqual(self.spoken(self.adapter(scope, repo), irodori_token_split="off"),
                             "浦和レッズが勝った")

    def test_adapter_without_scope_attributes_keeps_the_old_behaviour(self):
        adapter = self.adapter("small", SMALL_TOKENIZER)
        del adapter.token_split_scope, adapter.text_tokenizer_repo
        self.assertIn(SPLIT_MARK, self.spoken(adapter))

    def test_state_reports_whether_the_dictionary_applies(self):
        state = self.adapter("small", LARGE_TOKENIZER).token_split_state()
        self.assertEqual(state, {"scope": "small", "active": False, "modelTokenizer": LARGE_TOKENIZER,
                                 "dictionaryTokenizer": SMALL_TOKENIZER})
        self.assertTrue(self.adapter("small", LARGE_TOKENIZER).token_split_state("all")["active"])


class RareFlagTests(unittest.TestCase):
    """「出現度の低い側」の判定は modernbert-ja（Unigram）のもの。BPE のトークナイザには付けない。"""

    @staticmethod
    def tokenizer(unigram):
        class Backend:
            def encode(self, text, add_special_tokens=False):
                return type("Enc", (), {"ids": [65000], "offsets": [(0, len(text))]})()

            def to_str(self):
                model = ({"type": "Unigram", "vocab": [["a", -1.0]] * 70000} if unigram
                         else {"type": "BPE", "vocab": {}})
                return json.dumps({"model": model})

        hf = type("HF", (), {"backend_tokenizer": Backend()})()
        return type("Tok", (), {"tokenizer": hf})()

    def first_token(self, unigram):
        token_split._SCORES.clear()
        try:
            return describe_tokens(["ab"], self.tokenizer(unigram))["results"][0]["tokens"][0]
        finally:
            token_split._SCORES.clear()

    def test_rare_flag_only_for_tokenizers_with_unigram_scores(self):
        self.assertTrue(self.first_token(unigram=True)["rare"])
        self.assertFalse(self.first_token(unigram=False)["rare"])


if __name__ == "__main__":
    unittest.main()
