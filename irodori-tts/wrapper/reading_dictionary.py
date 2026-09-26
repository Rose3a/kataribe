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


class ReadingDictionary:
    def __init__(self, path):
        self.path = Path(path)
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
        """ユーザー辞書を当て、残った英語をカナ読みにする。

        english は英語の読み方（off なら英字はそのまま残す、hiragana なら英語の
        読みだけひらがなにする）。kana_style="hiragana" なら、辞書と英語の読みを
        含む文中のカタカナをすべてひらがなにする。spacing="join" なら、変換した
        英語とユーザー辞書の読みの前後の空白を詰める（英語を変換するときだけ）。
        """
        if english not in ENGLISH_READINGS:
            raise ValueError(f"english must be one of {ENGLISH_READINGS}")
        if kana_style not in KANA_STYLES:
            raise ValueError(f"kana_style must be one of {KANA_STYLES}")
        if spacing not in ENGLISH_SPACINGS:
            raise ValueError(f"spacing must be one of {ENGLISH_SPACINGS}")
        join = english != "off" and spacing == "join"

        def rest(part):
            if english == "off":
                return part
            return convert_english(part, hiragana=english == "hiragana", join=join)

        text = self._apply_words(normalize_width(text), rest, join=join)
        return to_hiragana(text) if kana_style == "hiragana" else text

    def _apply_words(self, text, rest, join=False):
        # ユーザー辞書に当たらなかった部分だけを rest で変換する（辞書の読みは再変換しない）。
        words = sorted(self.snapshot().values(),
                       key=lambda w: (-len(w["surface"]), -w["priority"]))
        readings = {}
        user_surfaces = set()
        for word in words:
            surface = normalize_width(word["surface"]).lower()
            # If the same surface is registered more than once, keep the
            # highest-priority pronunciation (the list is already priority-sorted).
            if surface not in user_surfaces:
                readings[surface] = word["pronunciation"]
                user_surfaces.add(surface)
        if not readings:
            return rest(text)
        # Longest key first: Python's alternation keeps the first match, so a
        # shorter user entry would otherwise swallow a longer one.
        ordered = sorted(readings, key=len, reverse=True)
        pattern = '(' + '|'.join(re.escape(k) for k in ordered) + ')'
        if join:
            pattern = r'[ 	　]*' + pattern + r'[ 	　]*'
        pattern = re.compile(pattern, re.IGNORECASE | re.ASCII)
        result, start = [], 0
        for match in pattern.finditer(text):
            result.extend((rest(text[start:match.start()]), readings[match[1].lower()]))
            start = match.end()
        return ''.join(result) + rest(text[start:])


READING_DICTIONARY = ReadingDictionary(Path(__file__).resolve().parents[2] / "user_dictionary.json")
