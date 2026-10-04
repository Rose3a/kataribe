import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wrapper"))
from reading_dictionary import ReadingDictionary  # noqa: E402
from token_split import (DICTIONARY_PATH, SPLIT_MARK, ZERO_WIDTH_SPACE,  # noqa: E402
                         TokenSplitDictionary, find_tokenizer_json, install_split_encoding,
                         split_token_ids)

VOCAB = ["浦和レッズ", "ゼルダの伝説", "テキストエディタ", "ワールド", "ワール", "ゼルダ",
         "うらわれっず"]


def fake_offsets(text):
    """VOCAB の最長一致、それ以外は1文字ずつ（トークナイザの代わり）。"""
    out, i = [], 0
    while i < len(text):
        size = next((len(w) for w in sorted(VOCAB, key=len, reverse=True)
                     if text.startswith(w, i)), 1)
        out.append((i, i + size))
        i += size
    return out


def write_dictionary(path, entries):
    path.write_text(json.dumps({"version": 1, "tokenizer": "sbintuitions/modernbert-ja-310m",
                                "entries": [{"surface": surface, "text": text}
                                            for surface, text in entries.items()]},
                               ensure_ascii=False), encoding="utf-8")


class FakeHF:
    """1文字1トークン（ID は文字コード）。ただし「浦和レッズ」だけはまとまりの1トークン。"""

    def encode(self, text, add_special_tokens=False):
        ids, rest = [], text
        while rest:
            if rest.startswith("浦和レッズ"):
                ids.append(1)
                rest = rest[5:]
            else:
                ids.append(ord(rest[0]))
                rest = rest[1:]
        return ids


class FakeTextTokenizer:
    add_bos = True
    bos_token_id = 2
    pad_token_id = 0

    def __init__(self):
        self.tokenizer = FakeHF()
        self.calls = 0

    def encode(self, text, add_bos=None):
        import torch
        return torch.tensor([2, *self.tokenizer.encode(text)])

    def batch_encode(self, texts, max_length=None):
        self.calls += 1
        return "original", max_length


class TokenSplitDictionaryTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "token_split_dictionary.json"

    def test_notation_and_token_positions(self):
        write_dictionary(self.path, {"浦和レッズ": "浦和|レッ|ズ", "ゼルダの伝説": "ゼルダの 伝説",
                                     "ゼルダ": "ぜるだ", "テキストエディタ": "テキスト[ZW]エディタ",
                                     "ワール": "ワー|ル"})
        d = TokenSplitDictionary(self.path, tokenizer=fake_offsets)
        self.assertEqual(d.apply("浦和レッズとゼルダの伝説"),
                         f"浦和{SPLIT_MARK}レッ{SPLIT_MARK}ズ{SPLIT_MARK}と{SPLIT_MARK}ゼルダの 伝説")
        self.assertEqual(d.apply("テキストエディタ"), f"テキスト{ZERO_WIDTH_SPACE}エディタ")
        self.assertEqual(d.apply("ゼルダ"), "ぜるだ")
        # 「ワール」はトークンとして出たときだけ。「ワールド」の中では当てない
        self.assertEqual(d.apply("ワールドとワール"), f"ワールドと{SPLIT_MARK}ワー{SPLIT_MARK}ル")

    def test_phrases_that_are_not_one_token_match_as_text(self):
        write_dictionary(self.path, {"浦和レッズ戦": "浦和|レッズ|戦", "ワール": "ワー|ル"})
        d = TokenSplitDictionary(self.path, tokenizer=fake_offsets)
        self.assertEqual(d.apply("浦和レッズ戦とワールドとワール"),
                         f"浦和{SPLIT_MARK}レッズ{SPLIT_MARK}戦{SPLIT_MARK}とワールドと{SPLIT_MARK}ワー{SPLIT_MARK}ル")

    def test_user_entries_are_a_separate_file_and_win(self):
        write_dictionary(self.path, {"浦和レッズ": "浦和|レッ|ズ", "ゼルダの伝説": "ゼルダ|の|伝説"})
        user = self.path.parent / "token_split_user.json"
        d = TokenSplitDictionary(self.path, tokenizer=fake_offsets, user_path=user)
        d.put_user("浦和レッズ", "浦和|れっず", note="試し")
        d.put_user("自治スレ", "じち\u2063スレ")  # 実際の区切り文字でも | として保存する
        d.put_user("ゼルダの伝説", "ゼルダの伝説")  # 単語と同じにすれば自動の登録を止められる
        self.assertEqual(d.apply("浦和レッズとゼルダの伝説"),
                         f"浦和{SPLIT_MARK}れっず{SPLIT_MARK}とゼルダの伝説")
        self.assertEqual(d.apply("自治スレ"), f"じち{SPLIT_MARK}スレ")
        saved = {e["surface"]: e for e in json.loads(user.read_text(encoding="utf-8"))["entries"]}
        self.assertEqual(saved["自治スレ"]["text"], "じち|スレ")
        self.assertEqual(saved["浦和レッズ"]["note"], "試し")
        # メモを省いて書き換えだけ直すと、メモは残る
        d.put_user("浦和レッズ", "浦和|レッズ")
        self.assertEqual({e["surface"]: e["note"] for e in d.user_entries()}["浦和レッズ"], "試し")
        self.assertTrue(d.delete_user("浦和レッズ"))
        self.assertFalse(d.delete_user("浦和レッズ"))
        self.assertEqual(d.apply("浦和レッズ"), f"浦和{SPLIT_MARK}レッ{SPLIT_MARK}ズ")
        # 自動生成のファイルには触らない
        self.assertEqual(len(d.auto_entries()), 2)
        for bad in [("", "x"), ("x", " "), ("x", "a\nb")]:
            with self.assertRaises(ValueError):
                d.put_user(*bad)

    def test_reading_dictionary_hits_are_not_split_again(self):
        write_dictionary(self.path, {"浦和レッズ": "浦和|レッ|ズ"})
        dictionary = TokenSplitDictionary(self.path, tokenizer=fake_offsets)
        with patch("reading_dictionary.TOKEN_SPLIT_DICTIONARY", dictionary):
            from reading_dictionary import make_word
            reading = ReadingDictionary(Path(self.path.parent) / "user.json")
            reading.put(make_word("浦和レッズ戦", "ウラワレッズセン"))
            self.assertEqual(reading.convert("浦和レッズ戦と浦和レッズ"),
                             f"ウラワレッズセンと{SPLIT_MARK}浦和{SPLIT_MARK}レッ{SPLIT_MARK}ズ")

    def test_boundaries_keep_neighbours_apart(self):
        # 元は「友達」「がシェアした投稿」と切れていた。置き換え後も「友達が」にくっつけない
        write_dictionary(self.path, {"がシェアした投稿": "|が|シェア|した|投稿", "浦和レッズ": "浦和レッズ"})
        VOCAB.extend(["がシェアした投稿", "友達"])
        self.addCleanup(lambda: [VOCAB.remove(w) for w in ("がシェアした投稿", "友達")])
        d = TokenSplitDictionary(self.path, tokenizer=fake_offsets)
        self.assertEqual(d.apply("友達がシェアした投稿を見た"),
                         f"友達{SPLIT_MARK}が{SPLIT_MARK}シェア{SPLIT_MARK}した{SPLIT_MARK}投稿{SPLIT_MARK}を見た")
        # 文頭・文末には入れない。書き換えが語句と同じなら何もしない
        self.assertEqual(d.apply("がシェアした投稿"),
                         f"が{SPLIT_MARK}シェア{SPLIT_MARK}した{SPLIT_MARK}投稿")
        self.assertEqual(d.apply("浦和レッズの試合"), "浦和レッズの試合")

    def test_without_a_tokenizer_nothing_is_rewritten(self):
        write_dictionary(self.path, {"浦和レッズ": "浦和|レッズ"})
        self.assertEqual(TokenSplitDictionary(self.path, tokenizer=False).apply("浦和レッズ"),
                         "浦和レッズ")

    @unittest.skipIf(find_tokenizer_json("sbintuitions/modernbert-ja-310m") is None,
                     "modernbert-ja tokenizer is not cached")
    def test_real_tokenizer_keeps_common_words_intact(self):
        write_dictionary(self.path, {"浦和レッズ": "浦和|レッ|ズ", "ワール": "ワー|ル",
                                     "ミング": "ミン|グ"})
        d = TokenSplitDictionary(self.path)
        self.assertEqual(d.apply("浦和レッズのワールドカップ、タイミング"),
                         f"浦和{SPLIT_MARK}レッ{SPLIT_MARK}ズ{SPLIT_MARK}のワールドカップ、タイミング")

    @unittest.skipIf(not DICTIONARY_PATH.exists(), "no shipped dictionary")
    def test_shipped_splits_keep_the_text(self):
        data = json.loads(DICTIONARY_PATH.read_text(encoding="utf-8"))
        for entry in data["entries"]:
            if entry["method"] in ("split", "split_fine", "zw"):
                self.assertEqual(entry["text"].replace("|", "").replace("[ZW]", ""),
                                 entry["surface"])

    def test_missing_file_is_a_no_op_and_reload_on_change(self):
        d = TokenSplitDictionary(self.path, tokenizer=fake_offsets)
        self.assertEqual(d.apply("浦和レッズ"), "浦和レッズ")
        write_dictionary(self.path, {"浦和レッズ": "浦和|レッズ"})
        self.assertEqual(d.apply("浦和レッズ"), f"浦和{SPLIT_MARK}レッズ")

    def test_reading_dictionary_applies_it_last_and_can_turn_it_off(self):
        write_dictionary(self.path, {"浦和レッズ": "浦和|レッズ", "うらわれっず": "うらわ|れっず"})
        dictionary = TokenSplitDictionary(self.path, tokenizer=fake_offsets)
        with patch("reading_dictionary.TOKEN_SPLIT_DICTIONARY", dictionary):
            reading = ReadingDictionary(Path(self.path.parent) / "user.json")
            self.assertEqual(reading.convert("浦和レッズ"), f"浦和{SPLIT_MARK}レッズ")
            self.assertEqual(reading.convert("浦和レッズ", token_split="off"), "浦和レッズ")
            # ひらがな化した後の文字列に当てる
            self.assertEqual(reading.convert("ウラワレッズ", kana_style="hiragana"),
                             f"うらわ{SPLIT_MARK}れっず")
            with self.assertRaises(ValueError):
                reading.convert("x", token_split="maybe")

    def test_runtime_normalization_keeps_the_mark(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runtime" / "trt-lab" / "repo"))
        from irodori_tts.text_normalization import normalize_text
        self.assertIn(SPLIT_MARK, normalize_text(f"「浦和{SPLIT_MARK}レッズ」").strip())


class TokenSplitApiTests(unittest.TestCase):
    """辞書画面が使う /irodori/token_split/* と /irodori/tokenize（モデルなし）。"""

    def test_list_put_delete_and_tokenize(self):
        import threading
        from http.client import HTTPConnection
        from http.server import ThreadingHTTPServer
        import voicevox_engine as engine

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        auto = Path(tmp.name) / "auto.json"
        write_dictionary(auto, {"浦和レッズ": "浦和|レッ|ズ"})
        dictionary = TokenSplitDictionary(auto, tokenizer=fake_offsets,
                                          user_path=Path(tmp.name) / "user.json")
        server = ThreadingHTTPServer(("127.0.0.1", 0), engine.Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        def post(path, body, authorized=True):
            conn = HTTPConnection("127.0.0.1", server.server_port)
            headers = {"Origin": "http://localhost:5173", "Content-Type": "application/json"}
            if authorized:
                headers["X-Irodori-Session"] = engine.SESSION_TOKEN
            conn.request("POST", path, json.dumps(body, ensure_ascii=False).encode("utf-8"), headers)
            response = conn.getresponse()
            data = response.read()
            conn.close()
            return response.status, json.loads(data) if data else None

        with patch.object(engine, "TOKEN_SPLIT_DICTIONARY", dictionary):
            self.assertEqual(post("/irodori/token_split/list", {}, authorized=False)[0], 403)
            status, listed = post("/irodori/token_split/list", {})
            self.assertEqual((status, listed["user"], len(listed["auto"])), (200, [], 1))
            status, saved = post("/irodori/token_split/put",
                                 {"surface": "自治スレ", "text": "じち|スレ", "note": "試し"})
            self.assertEqual((status, saved["text"], saved["note"]), (200, "じち|スレ", "試し"))
            self.assertEqual(post("/irodori/token_split/list", {})[1]["user"][0]["surface"], "自治スレ")
            self.assertEqual(post("/irodori/token_split/put", {"surface": "", "text": "x"})[0], 400)
            self.assertEqual(post("/irodori/token_split/delete", {"surface": "自治スレ"}),
                             (200, {"deleted": True}))
            self.assertEqual(post("/irodori/token_split/list", {})[1]["user"], [])
            self.assertEqual(post("/irodori/tokenize", {"texts": "x"})[0], 400)
            if find_tokenizer_json("sbintuitions/modernbert-ja-310m") is not None:
                status, view = post("/irodori/tokenize", {"texts": ["浦和レッズ", "浦和|レッズ"]})
                self.assertEqual((status, view["source"]), (200, "fallback"))
                first, second = view["results"]
                self.assertEqual([t["text"] for t in first["tokens"]], ["浦和レッズ"])
                self.assertTrue(first["tokens"][0]["rare"])
                self.assertEqual([t.get("text", "|") for t in second["tokens"]][:2], ["浦和", "|"])


class AudioQueryTests(unittest.TestCase):
    def test_kana_is_not_split_until_synthesis(self):
        # エディタは irodori_text を送らず kana を送り返すことがある。kana に区切りが
        # 入っていると、セリフの設定で「使わない」にしても外せない。
        import voicevox_engine as engine
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        auto = Path(tmp.name) / "auto.json"
        write_dictionary(auto, {"浦和レッズ": "浦和|レッ|ズ"})
        dictionary = TokenSplitDictionary(auto, tokenizer=fake_offsets)
        with patch("reading_dictionary.TOKEN_SPLIT_DICTIONARY", dictionary):
            query = engine._query("浦和レッズです")
            self.assertEqual((query["kana"], query["irodori_text"]), ("浦和レッズです", "浦和レッズです"))
            convert = engine.READING_DICTIONARY.convert
            self.assertEqual(convert(query["kana"], **engine._reading_options({})),
                             f"浦和{SPLIT_MARK}レッ{SPLIT_MARK}ズ{SPLIT_MARK}です")
            self.assertEqual(convert(query["kana"], **engine._reading_options(
                {"irodori_token_split": "off"})), "浦和レッズです")


class SplitEncodingTests(unittest.TestCase):
    def test_split_ids_avoid_the_merged_token(self):
        hf = FakeHF()
        self.assertEqual(hf.encode("浦和レッズ"), [1])
        self.assertEqual(split_token_ids(hf, f"浦和{SPLIT_MARK}レッズ"),
                         [ord(c) for c in "浦和レッズ"])

    def test_batch_encode_pads_and_masks_like_the_runtime(self):
        tok = FakeTextTokenizer()
        install_split_encoding(tok)
        install_split_encoding(tok)  # 二重に包まない
        self.assertEqual(tok.batch_encode(["浦和レッズ"], max_length=4), ("original", 4))
        ids, mask = tok.batch_encode([f"浦和{SPLIT_MARK}レッズ"] * 2, max_length=8)
        expected = [2] + [ord(c) for c in "浦和レッズ"] + [0, 0]
        self.assertEqual(ids.tolist(), [expected, expected])
        self.assertEqual(mask[0].tolist(), [True] * 6 + [False] * 2)
        ids, mask = tok.batch_encode([f"浦和{SPLIT_MARK}レッズ"], max_length=3)
        self.assertEqual(ids.tolist(), [[2, ord("浦"), ord("和")]])
        self.assertTrue(mask.all())
        ids, _ = tok.batch_encode([f"浦和{SPLIT_MARK}レッズ"])
        self.assertEqual(ids.shape[1], 6)
        self.assertEqual(tok.calls, 1)


if __name__ == "__main__":
    unittest.main()
