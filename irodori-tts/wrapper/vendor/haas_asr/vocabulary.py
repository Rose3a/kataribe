"""Custom vocabulary: words the model should recognise, and how to write them.

A word has a written form (``Haas``, ``Third Intelligence``) and optional readings, the
spellings Haas may write for it (ハース, サードインテリジェンス). While decoding, the
word's spellings are boosted (``haas_asr.boosting``); in the transcript, wherever one of
them is found, the written form replaces it.

Matching ignores what Haas's output varies in: hiragana and katakana, full-width
letters and digits, letter case, and spaces between words or where the script
changes. Boosting needs exact token sequences, so a word is boosted in the spellings
Haas can write: each reading as given and, for a reading in kana, in katakana too (Haas
writes names and loanwords in katakana), and the written form, all in lower case (Haas
writes no capital letters).
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass


class Vocabulary:
    """Words to boost while decoding, each with its written form.

    `words` maps each written form to its readings (a list, or one reading as a
    string), or is a list of written forms without readings::

        Vocabulary({"Haas": ["ハース"], "Third Intelligence": ["サードインテリジェンス"]})

    `alpha` is the boosting strength. Larger values fix more missed words but also
    write listed words where they were not said more often; the default, 2, balances
    the two.
    """

    def __init__(self, words: Mapping[str, Iterable[str] | str] | Iterable[str], *, alpha=2.0):
        if not isinstance(words, Mapping):
            words = dict.fromkeys(words, ())
        self.words: dict[str, tuple[str, ...]] = {}
        for text, readings in words.items():
            text = _clean(text)
            if not text:
                raise ValueError("a word's written form is empty")
            if isinstance(readings, str):
                readings = [readings]
            kept = dict.fromkeys(r for r in map(_clean, readings) if r and r != text)
            self.words[text] = tuple(kept)
        self.alpha = alpha
        self.spellings = tuple(
            dict.fromkeys(
                s for text, readings in self.words.items() for s in spellings(text, readings)
            )
        )
        forms = {
            (form.chars, form.write): form
            for text, readings in self.words.items()
            for spelling in (*readings, text)
            if (form := _form(spelling, text)).chars
        }
        # By first character, so a position is tried only against the forms that can start
        # there; longest first, so a word inside a longer one is not matched alone.
        self._forms: dict[str, list[_Form]] = {}
        for form in sorted(forms.values(), key=lambda f: -len(f.chars)):
            self._forms.setdefault(form.chars[0], []).append(form)

    @classmethod
    def parse(cls, entries: Iterable[str], **kwargs) -> Vocabulary:
        """From ``"written form=reading,reading"`` strings, as on a command line:
        ``Vocabulary.parse(["Haas=ハース,ハウス", "AIモデル"])``."""
        words: dict[str, list[str]] = {}
        for entry in entries:
            text, _, readings = entry.partition("=")
            words.setdefault(text.strip(), []).extend(readings.split(","))
        return cls(words, **kwargs)

    def __bool__(self) -> bool:
        return bool(self.words)

    def __repr__(self) -> str:
        return f"Vocabulary({self.words!r}, alpha={self.alpha!r})"

    def find(self, text: str) -> list[tuple[int, int, str]]:
        """Where `text` (as Haas wrote it) holds the words: ``(start, end, written form)``."""
        folded = [_fold(c) for c in text]
        found, i = [], 0
        while i < len(text):
            for form in self._forms.get(folded[i], ()):
                end = _match(text, folded, i, form)
                if end > i:
                    found.append((i, end, form.write))
                    i = end
                    break
            else:
                i += 1
        return found

    def format(self, text: str) -> str:
        """`text` with every word found in it in its written form."""
        out, last = [], 0
        for start, end, write in self.find(text):
            out += [text[last:start], write]
            last = end
        return "".join(out) + text[last:]


def spellings(text: str, readings: Iterable[str] = ()) -> list[str]:
    """The spellings a word is boosted in (see the module's docstring)."""
    out = []
    for reading in readings:
        kana = all("぀" <= c <= "ヿ" for c in reading)
        for variant in (reading, _katakana(reading) if kana else reading):
            if variant.lower() not in out:
                out.append(variant.lower())
    if text.lower() not in out:
        out.append(text.lower())
    return out


def _clean(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())


def _katakana(text: str) -> str:
    """Hiragana as katakana (other characters unchanged)."""
    return "".join(chr(ord(c) + 0x60) if 0x3041 <= ord(c) <= 0x3096 else c for c in text)


def _fold(c: str) -> str:
    """One character as matching compares it: katakana, half-width, lower case. Always one
    character, so positions in folded text are positions in the original."""
    o = ord(c)
    if 0x3041 <= o <= 0x3096:  # hiragana -> katakana
        return chr(o + 0x60)
    if 0xFF01 <= o <= 0xFF5E:  # full-width ASCII -> ASCII
        c = chr(o - 0xFEE0)
    if c == "　":
        return " "
    lower = c.lower()
    return lower if len(lower) == 1 else c


def _script(c: str) -> str:
    if "a" <= c.lower() <= "z":
        return "latin"
    if c.isdigit():
        return "digit"
    if "゠" <= c <= "ヿ" or "぀" <= c <= "ゟ":
        return "kana"
    if "㐀" <= c <= "鿿" or c == "々":
        return "kanji"
    return "other " + c


def _latin(c: str | None) -> bool:
    return c is not None and _script(c) == "latin"


@dataclass(frozen=True)
class _Form:
    chars: str  # folded, without spaces
    gaps: frozenset[int]  # where a transcript may have spaces between them
    write: str


def _form(spelling: str, write: str) -> _Form:
    chars, gaps = [], set()
    for c in spelling:
        if c.isspace():
            if chars:
                gaps.add(len(chars))
            continue
        f = _fold(c)
        if chars and _script(chars[-1]) != _script(f):
            gaps.add(len(chars))
        chars.append(f)
    return _Form("".join(chars), frozenset(gaps), write)


def _match(text: str, folded: list[str], i: int, form: _Form) -> int:
    """Where `form` ends if it is read at text[i]; -1 if it is not there. A form starting
    or ending in a Latin letter is not matched inside a longer Latin word."""
    if _latin(form.chars[0]) and i > 0 and _latin(folded[i - 1]):
        return -1
    j = i
    for k, c in enumerate(form.chars):
        if k in form.gaps:
            while j < len(text) and text[j].isspace():
                j += 1
        if j >= len(text) or folded[j] != c:
            return -1
        j += 1
    if _latin(form.chars[-1]) and j < len(text) and _latin(folded[j]):
        return -1
    return j
