"""語彙分割（4文字以上で出現度が境目以下のトークンを避けてエンコードする）の契約テスト。

小さな Unigram トークナイザで境目の挙動を確かめる。modernbert-ja の tokenizer.json が
キャッシュにあれば、実物でも確かめる。モデルは読まない。
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
from token_split import (DEFAULT_TOKEN_SPLIT_THRESHOLD, SPLIT_MARK, TOKEN_SPLIT_THRESHOLDS,  # noqa: E402
                         active_threshold, describe_tokens, find_tokenizer_json,
                         install_split_encoding, low_score_tokenizer, split_token_ids,
                         threshold_value, token_split_applies)

SMALL_TOKENIZER = "sbintuitions/modernbert-ja-310m"
LARGE_TOKENIZER = "google/t5gemma-2-1b-1b"
# (トークン, 出現度)。浦和レッズ は珍しい5文字、ください はよく出る4文字、レッズ は3文字。
VOCAB = [("<unk>", 0.0), ("浦和レッズ", -14.3), ("ください", -9.3), ("レッズ", -14.0),
         ("浦和", -11.0), ("レッ", -12.0), ("が", -5.0), ("勝った", -10.0)]


def tiny_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import Unigram
    chars = sorted({c for piece, _ in VOCAB[1:] for c in piece} - {p for p, _ in VOCAB})
    vocab = VOCAB + [(c, -16.0) for c in chars]
    return Tokenizer(Unigram(vocab, unk_id=0, byte_fallback=False))


def pieces(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False).tokens


class HF:
    """PreTrainedTokenizerFast の代わり（backend_tokenizer と encode だけ）。"""

    def __init__(self, backend):
        self.backend_tokenizer = backend

    def encode(self, text, add_special_tokens=False):
        return self.backend_tokenizer.encode(text, add_special_tokens=add_special_tokens).ids


class FakeTextTokenizer:
    add_bos = True
    bos_token_id = 2
    pad_token_id = 0

    def __init__(self, backend):
        self.tokenizer = HF(backend)
        self.calls = 0

    def encode(self, text, add_bos=None):
        import torch
        return torch.tensor([2, *self.tokenizer.encode(text)])

    def batch_encode(self, texts, max_length=None):
        self.calls += 1
        return "original", max_length


class LowScoreTokenizerTests(unittest.TestCase):
    def setUp(self):
        self.tok = tiny_tokenizer()

    def test_settings(self):
        self.assertEqual(TOKEN_SPLIT_THRESHOLDS, ("-10", "-11", "-12", "-13", "none"))
        self.assertEqual(DEFAULT_TOKEN_SPLIT_THRESHOLD, "-13")
        self.assertEqual(threshold_value("-13"), -13.0)
        self.assertIsNone(threshold_value("none"))
        with self.assertRaises(ValueError):
            threshold_value("-14")

    def test_rare_long_tokens_are_split_and_common_ones_kept(self):
        self.assertEqual(pieces(self.tok, "浦和レッズがください"), ["浦和レッズ", "が", "ください"])
        low = low_score_tokenizer(self.tok, -13.0)
        # 3文字の「レッズ」は対象外なので残る
        self.assertEqual(pieces(low, "浦和レッズがください"), ["浦和", "レッズ", "が", "ください"])

    def test_raising_the_threshold_splits_more_words(self):
        low = low_score_tokenizer(self.tok, -9.0)
        self.assertEqual(pieces(low, "ください"), list("ください"))

    def test_three_character_tokens_are_never_split(self):
        low = low_score_tokenizer(self.tok, -10.0)
        self.assertEqual(pieces(low, "レッズ"), ["レッズ"])

    def test_original_tokenizer_is_untouched_and_copies_are_cached(self):
        low = low_score_tokenizer(self.tok, -13.0)
        self.assertIs(low, low_score_tokenizer(self.tok, -13.0))
        self.assertIsNot(low, low_score_tokenizer(self.tok, -12.0))
        self.assertEqual(pieces(self.tok, "浦和レッズ"), ["浦和レッズ"])

    def test_non_unigram_tokenizers_are_left_alone(self):
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        self.assertIsNone(low_score_tokenizer(Tokenizer(BPE()), -13.0))

    def test_only_the_modernbert_family_gets_it(self):
        self.assertTrue(token_split_applies(SMALL_TOKENIZER))
        self.assertFalse(token_split_applies(LARGE_TOKENIZER))
        self.assertFalse(token_split_applies("llm-jp/llm-jp-3-1.8b"))
        # 読み込み前・古い経路ではトークナイザが分からない。当てる側に倒す
        self.assertTrue(token_split_applies(None))


@unittest.skipIf(find_tokenizer_json(SMALL_TOKENIZER) is None, "modernbert-ja の tokenizer.json が無い")
class RealTokenizerTests(unittest.TestCase):
    def test_unreadable_names_split_and_common_words_stay(self):
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(find_tokenizer_json(SMALL_TOKENIZER)))
        low = low_score_tokenizer(tok, -13.0)
        self.assertEqual(pieces(tok, "浦和レッズ"), ["浦和レッズ"])
        self.assertEqual(pieces(low, "浦和レッズ"), ["浦和", "レッ", "ズ"])
        self.assertEqual(pieces(low, "ゼルダの伝説"), ["ゼルダ", "の", "伝説"])
        for common in ("しました", "ください", "について"):
            self.assertEqual(pieces(low, common), [common])
        # パディングは写しに持ち込まない
        self.assertNotIn("<pad>", pieces(low, "浦和レッズ"))


class SplitEncodingTests(unittest.TestCase):
    def setUp(self):
        self.text_tokenizer = FakeTextTokenizer(tiny_tokenizer())
        install_split_encoding(self.text_tokenizer)
        install_split_encoding(self.text_tokenizer)  # 二重に包まない
        vocab = self.text_tokenizer.tokenizer.backend_tokenizer.get_vocab()
        self.id = vocab.__getitem__

    def test_plain_text_keeps_the_original_path(self):
        self.assertEqual(self.text_tokenizer.batch_encode(["浦和レッズ"], max_length=4), ("original", 4))
        self.assertEqual(self.text_tokenizer.calls, 1)

    def test_active_threshold_encodes_with_the_copy(self):
        with active_threshold(-13.0):
            ids, mask = self.text_tokenizer.batch_encode(["浦和レッズが"], max_length=8)
        expected = [2] + [self.id(p) for p in ("浦和", "レッズ", "が")] + [0, 0, 0, 0]
        self.assertEqual(ids.tolist(), [expected])
        self.assertEqual(mask[0].tolist(), [True] * 4 + [False] * 4)
        self.assertEqual(self.text_tokenizer.calls, 0)
        # 抜けたら元に戻る
        self.assertEqual(self.text_tokenizer.batch_encode(["浦和レッズ"]), ("original", None))

    def test_none_threshold_means_no_split(self):
        with active_threshold(None):
            self.assertEqual(self.text_tokenizer.batch_encode(["浦和レッズ"]), ("original", None))

    def test_split_mark_still_splits(self):
        hf = self.text_tokenizer.tokenizer
        self.assertEqual(split_token_ids(hf, f"浦和{SPLIT_MARK}レッズ"),
                         [self.id("浦和"), self.id("レッズ")])
        ids, mask = self.text_tokenizer.batch_encode([f"浦和{SPLIT_MARK}レッズ"], max_length=2)
        self.assertEqual(ids.tolist(), [[2, self.id("浦和")]])
        self.assertTrue(mask.all())

    def test_runtime_normalization_keeps_the_mark(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runtime" / "trt-lab" / "repo"))
        from irodori_tts.text_normalization import normalize_text
        self.assertIn(SPLIT_MARK, normalize_text(f"「浦和{SPLIT_MARK}レッズ」").strip())


class SynthesisTests(unittest.TestCase):
    """合成のあいだだけ、全体の境目とセリフの設定に従って語彙分割が有効になる。文字列は変えない。"""

    def setUp(self):
        import voicevox_engine as engine
        self.engine = engine
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patcher = patch.object(engine, "READING_DICTIONARY",
                               ReadingDictionary(Path(tmp.name) / "user_dictionary.json"))
        patcher.start()
        self.addCleanup(patcher.stop)

    def adapter(self, threshold, repo):
        adapter = self.engine.VoicevoxAdapter.__new__(self.engine.VoicevoxAdapter)
        adapter.id_to_name = {0: "話者なし"}
        adapter.lock = threading.Lock()
        adapter.progress_callback = None
        adapter.tts = Mock()
        adapter.seen = []

        def synthesize(**kwargs):
            adapter.seen.append((kwargs["text"], token_split._ACTIVE.get()))
            kwargs["out_wav"].write(b"wav")
        adapter.tts.synthesize.side_effect = synthesize
        adapter.token_split_threshold = threshold
        adapter.text_tokenizer_repo = repo
        return adapter

    def active(self, adapter, **extra):
        adapter.synthesize({**self.engine._query("浦和レッズが勝った"), **extra}, 0)
        text, threshold = adapter.seen[-1]
        self.assertEqual(text, "浦和レッズが勝った")
        return threshold

    def test_default_threshold_for_the_small_family(self):
        self.assertEqual(self.active(self.adapter("-13", SMALL_TOKENIZER)), -13.0)
        self.assertEqual(self.active(self.adapter("-10", SMALL_TOKENIZER)), -10.0)
        self.assertIsNone(token_split._ACTIVE.get())

    def test_other_tokenizers_and_none_never_split(self):
        self.assertIsNone(self.active(self.adapter("-13", LARGE_TOKENIZER)))
        self.assertIsNone(self.active(self.adapter("none", SMALL_TOKENIZER)))

    def test_line_setting_off_wins(self):
        self.assertIsNone(self.active(self.adapter("-13", SMALL_TOKENIZER), irodori_token_split="off"))
        with self.assertRaises(ValueError):
            self.active(self.adapter("-13", SMALL_TOKENIZER), irodori_token_split="maybe")

    def test_adapter_without_attributes_uses_the_default(self):
        adapter = self.adapter("-13", SMALL_TOKENIZER)
        del adapter.token_split_threshold, adapter.text_tokenizer_repo
        self.assertEqual(self.active(adapter), -13.0)

    def test_kana_is_not_rewritten(self):
        query = self.engine._query("浦和レッズです")
        self.assertEqual((query["kana"], query["irodori_text"]), ("浦和レッズです", "浦和レッズです"))


class TokenizeApiTests(unittest.TestCase):
    """/irodori/tokenize（モデルなし）。辞書の API はもう無い。"""

    def test_tokenize_and_removed_dictionary_api(self):
        from http.client import HTTPConnection
        from http.server import ThreadingHTTPServer
        import voicevox_engine as engine

        server = ThreadingHTTPServer(("127.0.0.1", 0), engine.Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        def post(path, body):
            conn = HTTPConnection("127.0.0.1", server.server_port)
            headers = {"Origin": "http://localhost:5173", "Content-Type": "application/json",
                       "X-Irodori-Session": engine.SESSION_TOKEN}
            conn.request("POST", path, json.dumps(body, ensure_ascii=False).encode("utf-8"), headers)
            response = conn.getresponse()
            data = response.read()
            conn.close()
            return response.status, json.loads(data) if data else None

        self.assertEqual(post("/irodori/token_split/list", {})[0], 404)
        self.assertEqual(post("/irodori/tokenize", {"texts": "x"})[0], 400)
        if find_tokenizer_json(SMALL_TOKENIZER) is None:
            return
        status, view = post("/irodori/tokenize", {"texts": ["浦和レッズ", "浦和|レッズ"]})
        self.assertEqual((status, view["source"], view["threshold"], view["applies"]),
                         (200, "fallback", -13.0, True))
        first, second = view["results"]
        self.assertEqual([t["text"] for t in first["original"]], ["浦和レッズ"])
        self.assertTrue(first["original"][0]["low"])
        self.assertEqual([t["text"] for t in first["tokens"]], ["浦和", "レッ", "ズ"])
        self.assertEqual([t.get("text", "|") for t in second["tokens"]][:2], ["浦和", "|"])


class DescribeTokensTests(unittest.TestCase):
    @staticmethod
    def tokenizer(unigram):
        class Backend:
            def encode(self, text, add_special_tokens=False):
                return type("Enc", (), {"ids": [1], "offsets": [(0, len(text))]})()

            def to_str(self):
                model = ({"type": "Unigram", "vocab": [["<unk>", 0.0], ["浦和レッズ", -14.3]]} if unigram
                         else {"type": "BPE", "vocab": {}})
                return json.dumps({"model": model})

        hf = type("HF", (), {"backend_tokenizer": Backend()})()
        return type("Tok", (), {"tokenizer": hf})()

    def view(self, unigram, repo=SMALL_TOKENIZER):
        token_split._SCORES.clear()
        try:
            return describe_tokens(["浦和レッズ"], self.tokenizer(unigram), -13.0, repo)
        finally:
            token_split._SCORES.clear()

    def test_bpe_and_other_tokenizers_are_not_split(self):
        self.assertFalse(self.view(unigram=False)["applies"])
        self.assertFalse(self.view(unigram=True, repo=LARGE_TOKENIZER)["applies"])
        result = self.view(unigram=False)["results"][0]
        self.assertEqual(result["tokens"], result["original"])
        self.assertFalse(result["original"][0]["low"])


if __name__ == "__main__":
    unittest.main()
