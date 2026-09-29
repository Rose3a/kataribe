"""契約テスト: 合成前にコンソールへ出すトークンの分け方の表示。

ランタイムと同じ正規化を通してから分け、半角スペースは ␣、ゼロ幅スペースは [ZW] で見せる。
"""
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "irodori-tts" / "wrapper"))
sys.path.insert(0, str(ROOT / "runtime" / "trt-lab" / "repo"))
from tts_cli import token_view  # noqa: E402


class FakeTokenizer:
    """文字ごとに分け、空白は SentencePiece と同じ ▁ にする。"""

    def __init__(self):
        self.seen = []

    def encode(self, text, add_special_tokens=True):
        self.seen.append(text)
        return list(range(len(text)))

    def convert_ids_to_tokens(self, ids):
        text = self.seen[-1]
        return ["▁" if text[i] == " " else text[i] for i in ids]


def runtime_with(tokenizer):
    return types.SimpleNamespace(tokenizer=types.SimpleNamespace(tokenizer=tokenizer))


class TokenViewTest(unittest.TestCase):
    def test_marks_spaces_and_counts_tokens(self):
        view = token_view(runtime_with(FakeTokenizer()), "ゼ ル​ダ")
        self.assertEqual(view, "tokens (5): ゼ|␣|ル|[ZW]|ダ")

    def test_uses_runtime_normalization(self):
        tokenizer = FakeTokenizer()
        token_view(runtime_with(tokenizer), "  ねぇ..  ")
        # normalize_text が「..」を「…」に変え、前後の空白を落としたものを分ける
        self.assertEqual(tokenizer.seen[-1], "ねぇ…")

    def test_backend_without_tokenizer_shows_nothing(self):
        self.assertIsNone(token_view(types.SimpleNamespace(), "テスト"))
        self.assertIsNone(token_view(None, "テスト"))


if __name__ == "__main__":
    unittest.main()
