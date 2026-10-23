"""文節の頭で区切る: 付属語（助詞・助動詞）のあとに自立語が来る位置に、見えない区切りを入れる。

トークナイザ（modernbert-ja）は「に乗|ったら」「を抜け|ると」「はありますか」のように、助詞を
次の語にくっつけて1トークンにすることがある。MeCab（fugashi + unidic-lite）で文節の頭を探し、
そこで分けてエンコードする（token_split.split_token_ids が U+2063 の位置で分ける）。
すべての形態素の境界で切ると、遊|ん|だ・中|小|企業・資|生|堂 のように細かすぎて読みが崩れる
ので、文節の頭だけにする。

検証（work/mecab_split/）: Blockwise w8a8・16ステップ・300文で、語彙分割 -13 と併用すると
対象語の NG 率 0.074 → 0.067（発話単位で 直った 15・壊れた 4。上手＝ウワテ・寒気など文脈で読みが
決まる語が直った）。発話の長さ・300ms 以上の無音はほぼ変わらない。境目 -13 のすぐ上に残る
長いトークン（セブンイレブン -12.48）は、前後の分け方が変わると崩れることがある。

fugashi か辞書が無ければ何もしない（available() が False）。
"""
from __future__ import annotations

import threading

SPLIT_MARK = "⁣"  # INVISIBLE SEPARATOR（token_split.SPLIT_MARK と同じ）
# 付属語。このあとに文節の頭（下の NOT_HEAD 以外）が来たら区切る。
FUZOKU = {"助詞", "助動詞"}
# 文節の頭にならない品詞（付属語・接尾辞・記号・空白）
NOT_HEAD = {"助詞", "助動詞", "接尾辞", "補助記号", "空白"}

_LOCK = threading.Lock()
_TAGGER = None
_ERROR: str | None = None


def _tagger():
    global _TAGGER, _ERROR
    with _LOCK:
        if _TAGGER is None and _ERROR is None:
            try:
                import fugashi
                _TAGGER = fugashi.Tagger()  # unidic-lite
            except Exception as exc:  # noqa: BLE001 - 無ければ区切らない
                _ERROR = f"{type(exc).__name__}: {exc}"
        return _TAGGER


def available() -> bool:
    """文節の区切りが使えるか（fugashi と unidic-lite が入っているか）。"""
    return _tagger() is not None


def unavailable_reason() -> str | None:
    _tagger()
    return _ERROR


def boundaries(text: str) -> list[int]:
    """文節の頭の位置（文字の添字）。先頭は含めない。"""
    tagger = _tagger()
    if tagger is None or not text:
        return []
    with _LOCK:  # MeCab の Tagger はスレッドをまたいで同時に使わない
        words = [(w.surface, w.feature.pos1) for w in tagger(text)]
    spans, pos = [], 0
    for surface, pos1 in words:
        start = text.find(surface, pos)  # MeCab は空白を落とすので、元の文字列に当て直す
        if start < 0:
            continue
        spans.append((start, pos1))
        pos = start + len(surface)
    return [b[0] for a, b in zip(spans, spans[1:]) if a[1] in FUZOKU and b[1] not in NOT_HEAD]


def mark(text: str) -> str:
    """文節の頭に見えない区切りを入れた文。すでにある区切りはそのまま残す。"""
    if not text:
        return text
    out = []
    for part in text.split(SPLIT_MARK):
        cuts = [b for b in boundaries(part) if 0 < b < len(part)]
        pieces, last = [], 0
        for b in cuts:
            pieces.append(part[last:b])
            last = b
        pieces.append(part[last:])
        out.append(SPLIT_MARK.join(pieces))
    return SPLIT_MARK.join(out)
