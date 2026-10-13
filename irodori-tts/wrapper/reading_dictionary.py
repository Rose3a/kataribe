"""Persistent pronunciation overrides and Japanese speech preprocessing."""
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import unicodedata
import uuid

from english_reading import convert_english, to_hiragana

KANA_STYLES = ("katakana", "hiragana")
# 英単語・英文の読み: 変換しない / カタカナにする / ひらがなにする
ENGLISH_READINGS = ("off", "katakana", "hiragana")
# 変換した英語の前後の空白: 残して語ごとに区切る / 詰めてつなげて読む
ENGLISH_SPACINGS = ("keep", "join")


def normalize_width(text):
    # Keep Japanese punctuation and full-width kana intact.
    return text.translate({**{i: i - 0xFEE0 for i in range(0xFF01, 0xFF5F)}, 0x3000: 0x20})


def make_word(surface, pronunciation, accent_type=0, priority=5):
    if not isinstance(surface, str) or not surface.strip():
        raise ValueError("単語は必須です")
    if not isinstance(pronunciation, str):
        raise ValueError("読みはカタカナで入力してください")
    pronunciation = unicodedata.normalize("NFKC", pronunciation)
    pronunciation = ''.join(chr(ord(c) + 0x60) if 'ぁ' <= c <= 'ゖ' else c for c in pronunciation)
    if not re.fullmatch(r"[ァ-ヴー]+", pronunciation):
        raise ValueError("読みはひらがな・カタカナで入力してください")
    count = len(re.findall(r"[ァ-ヴー][ァィゥェォャュョヮ]?", pronunciation))
    accent_type, priority = int(accent_type), int(priority)
    if not 0 <= priority <= 10 or not 0 <= accent_type <= count:
        raise ValueError("優先度またはアクセントの範囲が不正です")
    return dict(surface=normalize_width(surface), pronunciation=pronunciation,
                yomi=pronunciation, accent_type=accent_type, priority=priority,
                mora_count=count, part_of_speech="名詞", part_of_speech_detail_1="固有名詞",
                part_of_speech_detail_2="一般", part_of_speech_detail_3="*",
                inflectional_type="*", inflectional_form="*", stem="*",
                accent_associative_rule="*")


BUILTIN_READINGS_PATH = Path(__file__).resolve().parent / "data" / "builtin_readings.json"
_KANJI = "一-龯々〆ヶ"


class BuiltinReadings:
    """kataribe に同梱する読み辞書（ユーザー辞書とは別枠）。

    data/builtin_readings.json の
      entries: 読みが1通りに決まる語（四字熟語・難読語・地名・英字略語など）。表記 → 読み
      units  : 数字の直後（空白1つまで）にあるときだけ読む単位（km → キロメートル）
    読みは英字と単位がカタカナ、漢字の語がひらがな（データ側でそうしてある）。
    ユーザー辞書に同じ語があればそちらが優先する（ユーザー辞書が先に当たり、残りにだけ当てる）。
    英字の語は前後が英数字でないとき、漢字の語は前後が漢字でないときだけ当てる
    （「AI」が「MAIL」の中に、「太秦」が長い漢字語の中に当たらないように）。
    ファイルを書き換えたら、エンジンを再起動せずに次の変換から拾う。
    """

    def __init__(self, path=BUILTIN_READINGS_PATH):
        self.path = Path(path)
        self.lock = threading.Lock()
        self._mtime = None
        self._words = None   # (正規表現, {表記: 読み})
        self._units = None

    def _load(self):
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            mtime = None
        if mtime == self._mtime:
            return
        data = json.loads(self.path.read_text(encoding="utf-8")) if mtime is not None else {}
        # 英字の語は大文字小文字を区別する（英文の it・am が IT・AM の読みにならないように）
        words = {}
        for entry in data.get("entries", []):
            surface, reading = normalize_width(entry["surface"]), entry["reading"]
            words.setdefault(surface, reading)
        units = {normalize_width(e["surface"]): e["reading"] for e in data.get("units", [])}

        # 前後の条件は置き換えのときに確かめる（語ごとに先読み・後読みを付けると、
        # 数千語の正規表現のコンパイルに数秒かかる）
        ordered = sorted(words, key=len, reverse=True)
        self._words = (re.compile("|".join(map(re.escape, ordered))), words) if words else None
        ordered = sorted(units, key=len, reverse=True)
        self._units = (re.compile(r"(?<=[0-9]) ?(" + "|".join(map(re.escape, ordered)) + r")(?![A-Za-z])"),
                       units) if units else None
        self._mtime = mtime

    def apply(self, text):
        with self.lock:
            self._load()
            words, units = self._words, self._units
        if words is not None:
            pattern, table = words
            text = pattern.sub(lambda m: table[m[0]] if _bounded(m) else m[0], text)
        if units is not None:
            pattern, table = units
            text = pattern.sub(lambda m: table[m[1]], text)
        return text


_ASCII_WORD = re.compile(r"[A-Za-z0-9.+/&' -]+")
_KANJI_CHAR = re.compile(rf"[{_KANJI}]")
_ALNUM_CHAR = re.compile(r"[A-Za-z0-9]")


def _bounded(match):
    """英字の語は前後が英数字でないとき、漢字の語は前後が漢字でないときだけ当てる。"""
    text, word, start, end = match.string, match[0], match.start(), match.end()
    before, after = text[start - 1:start], text[end:end + 1]
    if _ASCII_WORD.fullmatch(word):
        return not (_ALNUM_CHAR.match(before) or _ALNUM_CHAR.match(after))
    if _KANJI_CHAR.match(word[0]) and _KANJI_CHAR.match(before):
        return False
    return not (_KANJI_CHAR.match(word[-1]) and _KANJI_CHAR.match(after))


class ReadingDictionary:
    def __init__(self, path, builtin=None):
        self.path = Path(path)
        # 同梱の読み辞書（別枠）。ユーザー辞書に当たらなかった部分に、英語の読みより先に当てる
        self.builtin = builtin
        self.lock = threading.RLock()
        self.words = {}
        if self.path.exists():
            self.words = self._validate(json.loads(self.path.read_text(encoding="utf-8")))

    @staticmethod
    def _validate(words):
        if not isinstance(words, dict):
            raise ValueError("辞書はオブジェクトで指定してください")
        result = {}
        for key, word in words.items():
            uuid.UUID(key)
            if not isinstance(word, dict):
                raise ValueError("辞書の単語が不正です")
            result[key] = make_word(word.get("surface"), word.get("pronunciation"),
                                    word.get("accent_type", 0), word.get("priority", 5))
        return result

    def snapshot(self):
        with self.lock:
            return {key: dict(value) for key, value in self.words.items()}

    def _save(self, words):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(words, stream, ensure_ascii=False, indent=2)
            os.replace(name, self.path)
            self.words = words
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def put(self, word, key=None):
        with self.lock:
            if key is not None and key not in self.words:
                raise KeyError("単語が見つかりません")
            key = key or str(uuid.uuid4())
            self._save({**self.words, **self._validate({key: word})})
            return key

    def delete(self, key):
        with self.lock:
            words = self.snapshot()
            del words[key]
            self._save(words)

    def import_words(self, words, override):
        incoming = self._validate(words)
        with self.lock:
            self._save({**self.words, **incoming} if override else {**incoming, **self.words})

    def convert(self, text, english="katakana", kana_style="katakana", spacing="keep"):
        """ユーザー辞書を当て、残りに同梱の読み辞書を当て、残った英語をカナ読みにする。

        english は英語の読み方（off なら英字はそのまま残す、hiragana なら英語の
        読みだけひらがなにする）。kana_style="hiragana" なら、辞書と英語の読みを
        含む文中のカタカナをすべてひらがなにする。spacing="join" なら、変換した
        英語とユーザー辞書の読みの前後の空白を詰める（英語を変換するときだけ）。
        語彙分割（token_split.py）は文字列を書き換えず、エンコード時に当てる。
        """
        if english not in ENGLISH_READINGS:
            raise ValueError(f"english must be one of {ENGLISH_READINGS}")
        if kana_style not in KANA_STYLES:
            raise ValueError(f"kana_style must be one of {KANA_STYLES}")
        if spacing not in ENGLISH_SPACINGS:
            raise ValueError(f"spacing must be one of {ENGLISH_SPACINGS}")
        join = english != "off" and spacing == "join"

        def rest(part):
            if self.builtin is not None:
                part = self.builtin.apply(part)
            if english == "off":
                return part
            return convert_english(part, hiragana=english == "hiragana", join=join)

        out = []
        for part, _from_user in self._apply_words(normalize_width(text), rest, join=join):
            if kana_style == "hiragana":
                part = to_hiragana(part)
            out.append(part)
        return "".join(out)

    def _apply_words(self, text, rest, join=False):
        """(文字列, ユーザー辞書の読みか) の列を返す。

        ユーザー辞書に当たらなかった部分だけを rest で変換する（辞書の読みは再変換しない）。
        """
        # 本文は合成の前に text_cleanup で整える（HUNTER×HUNTER → HUNTERHUNTER、¥100 → 100円）。
        # 登録した表記も同じように整えてから比べないと、記号を含む語が当たらない。
        from text_cleanup import clean_pasted_text

        words = sorted(self.snapshot().values(),
                       key=lambda w: (-len(w["surface"]), -w["priority"]))
        readings = {}
        for word in words:
            surface = normalize_width(word["surface"])
            surface = clean_pasted_text(surface) or surface
            # 英字は大文字小文字を区別する（it を登録しても IT企業 には当てない）。
            # 小文字だけで登録した語は、文頭の大文字始まり（It）にも当てる。
            variants = [surface]
            if surface.isascii() and surface == surface.lower() and surface[:1].isalpha():
                variants.append(surface[:1].upper() + surface[1:])
            for variant in variants:
                # If the same surface is registered more than once, keep the
                # highest-priority pronunciation (the list is already priority-sorted).
                readings.setdefault(variant, word["pronunciation"])
        if not readings:
            return [(rest(text), False)]
        # Longest key first: Python's alternation keeps the first match, so a
        # shorter user entry would otherwise swallow a longer one.
        ordered = sorted(readings, key=len, reverse=True)
        pattern = '(' + '|'.join(re.escape(k) for k in ordered) + ')'
        if join:
            pattern = r'[ 	　]*' + pattern + r'[ 	　]*'
        pattern = re.compile(pattern)
        result, start = [], 0
        for match in pattern.finditer(text):
            word = match[1]
            # 英字の語は前後が英数字でないときだけ（AI を登録しても MAIL には当てない）
            if _ASCII_WORD.fullmatch(word) and (
                    _ALNUM_CHAR.match(text[match.start(1) - 1:match.start(1)])
                    or _ALNUM_CHAR.match(text[match.end(1):match.end(1) + 1])):
                continue
            result.extend(((rest(text[start:match.start()]), False),
                           (readings[word], True)))
            start = match.end()
        return result + [(rest(text[start:]), False)]


BUILTIN_READINGS = BuiltinReadings()
READING_DICTIONARY = ReadingDictionary(Path(__file__).resolve().parents[2] / "user_dictionary.json",
                                       builtin=BUILTIN_READINGS)
