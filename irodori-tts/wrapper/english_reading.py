"""英単語・英文をカタカナ読みに変換する（CMUdict と規則だけで、モデルは使わない）。

Irodori はテキストをそのまま読むため、覚えていない英単語は崩れる。合成前に
一般的なカタカナ英語へ置き換えて、日本語話者が読む程度の読みにそろえる。

優先順:
1. 慣用読みの表（LOANWORDS）
2. 大文字の略語（API、DMM）はアルファベット読み
3. CMUdict の発音記号を規則でカタカナにする
4. キャメルケース・ローマ字・複合語は分けて読む
5. どれでもなければ綴りをローマ字風に読む（母音がなければ1文字ずつ）
"""
import gzip
from functools import lru_cache
from pathlib import Path
import re
import threading
import unicodedata

DATA_PATH = Path(__file__).resolve().with_name("data") / "cmudict.txt.gz"

_load_lock = threading.Lock()
_words = None


def dictionary():
    """CMUdict（見出し語 -> 発音記号）。初回の英単語で1回だけ読む。"""
    global _words
    if _words is None:
        with _load_lock:
            if _words is None:
                words = {}
                try:
                    with gzip.open(DATA_PATH, "rt", encoding="utf-8") as stream:
                        for line in stream:
                            word, _, phones = line.rstrip("\n").partition(" ")
                            words[word] = phones
                except OSError as exc:
                    # 配布物に辞書が入っていないと、英単語が綴り読みに落ちる。
                    print(f"[irodori] english reading dictionary unavailable: {exc}", flush=True)
                _words = words
    return _words


# 発音記号からだと慣用とずれる語。見出しは小文字。
LOANWORDS = {
    "the": "ザ", "of": "オブ", "your": "ユア", "was": "ワズ", "from": "フロム", "this": "ディス",
    "don't": "ドント", "hello": "ハロー", "sorry": "ソーリー", "again": "アゲイン", "year": "イヤー",
    "welcome": "ウェルカム", "new": "ニュー", "news": "ニュース", "zero": "ゼロ", "ok": "オーケー",
    "okay": "オーケー", "ai": "エーアイ", "vs": "バーサス", "etc": "エトセトラ",
    "idea": "アイデア", "orange": "オレンジ", "baby": "ベイビー", "lady": "レディ",
    "body": "ボディ", "city": "シティ", "london": "ロンドン", "front": "フロント", "money": "マネー",
    "monday": "マンデー", "tuesday": "チューズデー", "wednesday": "ウェンズデー",
    "thursday": "サーズデー", "friday": "フライデー", "saturday": "サタデー", "sunday": "サンデー",
    "today": "トゥデイ", "video": "ビデオ", "radio": "ラジオ", "studio": "スタジオ",
    "audio": "オーディオ", "coffee": "コーヒー", "cake": "ケーキ", "chicken": "チキン",
    "kitchen": "キッチン", "apple": "アップル", "team": "チーム", "ticket": "チケット",
    "language": "ランゲージ", "image": "イメージ", "message": "メッセージ", "network": "ネットワーク",
    "upload": "アップロード", "email": "イーメール", "blog": "ブログ", "pro": "プロ",
    "program": "プログラム", "programming": "プログラミング", "window": "ウィンドウ",
    "windows": "ウィンドウズ", "phone": "フォン", "smartphone": "スマートフォン",
    "iphone": "アイフォン", "youtube": "ユーチューブ", "twitter": "ツイッター",
    "github": "ギットハブ", "wi": "ワイ", "pokemon": "ポケモン", "anime": "アニメ",
    "karaoke": "カラオケ", "sushi": "スシ", "tokyo": "トウキョウ", "kyoto": "キョウト",
    "osaka": "オオサカ", "soccer": "サッカー", "nintendo": "ニンテンドー", "text": "テキスト", "channel": "チャンネル",
}

# 大文字だけの2文字語でも、略語ではなく単語として読むもの。
CAPITAL_WORDS = {"NO", "GO", "OH", "SO", "MY", "HI", "BY", "UP", "ON", "IN", "TO", "DO",
                 "ME", "WE", "HE", "BE", "OF", "OR", "IF", "AM", "AN", "AS", "AT", "IS"}

LETTERS = dict(zip(
    "abcdefghijklmnopqrstuvwxyz",
    ["エー", "ビー", "シー", "ディー", "イー", "エフ", "ジー", "エイチ", "アイ", "ジェー",
     "ケー", "エル", "エム", "エヌ", "オー", "ピー", "キュー", "アール", "エス", "ティー",
     "ユー", "ブイ", "ダブリュー", "エックス", "ワイ", "ゼット"]))

# アルファベットの名前の発音。CMUdict の略語（api、usb）を見分けるのに使う。
LETTER_PHONES = dict(zip("abcdefghijklmnopqrstuvwxyz", [
    "EY", "B IY", "S IY", "D IY", "IY", "EH F", "JH IY", "EY CH", "AY", "JH EY", "K EY",
    "EH L", "EH M", "EH N", "OW", "P IY", "K Y UW", "AA R", "EH S", "T IY", "Y UW", "V IY",
    "D AH B AH L Y UW", "EH K S", "W AY", "Z IY"]))

_ROWS = {
    "": "ア イ ウ エ オ", "K": "カ キ ク ケ コ", "G": "ガ ギ グ ゲ ゴ", "S": "サ シ ス セ ソ",
    "Z": "ザ ジ ズ ゼ ゾ", "T": "タ ティ トゥ テ ト", "D": "ダ ディ ドゥ デ ド",
    "N": "ナ ニ ヌ ネ ノ", "HH": "ハ ヒ フ ヘ ホ", "F": "ファ フィ フ フェ フォ",
    "V": "バ ビ ブ ベ ボ", "B": "バ ビ ブ ベ ボ", "P": "パ ピ プ ペ ポ", "M": "マ ミ ム メ モ",
    "Y": "ヤ イ ユ イエ ヨ", "R": "ラ リ ル レ ロ", "L": "ラ リ ル レ ロ",
    "W": "ワ ウィ ウ ウェ ウォ", "CH": "チャ チ チュ チェ チョ", "JH": "ジャ ジ ジュ ジェ ジョ",
    "SH": "シャ シ シュ シェ ショ", "ZH": "ジャ ジ ジュ ジェ ジョ", "TH": "サ シ ス セ ソ",
    "DH": "ザ ジ ズ ゼ ゾ",
}
ROWS = {key: dict(zip("aiueo", value.split())) for key, value in _ROWS.items()}
# 子音だけのとき（後ろに母音がない）の読み。
ALONE = {"K": "ク", "G": "グ", "S": "ス", "Z": "ズ", "T": "ト", "D": "ド", "N": "ン",
         "HH": "フ", "F": "フ", "V": "ブ", "B": "ブ", "P": "プ", "M": "ム", "Y": "イ",
         "R": "ル", "L": "ル", "W": "ウ", "CH": "チ", "JH": "ジ", "SH": "シュ", "ZH": "ジュ",
         "TH": "ス", "DH": "ズ", "NG": "ング"}
# 子音＋Y＋母音（computer の ピュー）。
PALATAL = {"K": "キ", "G": "ギ", "P": "ピ", "B": "ビ", "M": "ミ", "N": "ニ", "HH": "ヒ",
           "F": "フ", "V": "ビ", "R": "リ", "L": "リ", "D": "デ", "T": "テ", "S": "シ", "Z": "ジ"}
SMALL_Y = {"a": "ャ", "u": "ュ", "o": "ョ"}
# 子音＋W＋母音（quick の クイ）。
AFTER_W = {"a": "ワ", "i": "イ", "u": "ウ", "e": "エ", "o": "オ"}
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
SHORT = {"AE", "EH", "IH", "AH", "UH", "AA", "AO"}
# 短母音の直後で語末に来ると促音が入る子音（cat -> キャット、bed -> ベッド）。
GEMINATE = {"P", "T", "K", "D", "G", "CH", "JH", "SH"}
# 語中でも、綴りが重なっていれば促音にする（happy -> ハッピー、soccer -> サッカー）。
# SH は sh / ss の綴りだけ（fashion -> ファッション、national -> ナショナル）。
DOUBLED = {"P": "pp", "K": "cc|kk", "G": "gg", "CH": "tch", "SH": "sh|ss"}


def _spelling_groups(word):
    """綴りの母音のまとまり。発音記号の母音と順に対応させて、あいまい母音の色を決める。"""
    groups = [m.group() for m in re.finditer(r"[aeiou]+(?:y(?![aeiou]))?|y(?![aeiou])", word)]
    if len(groups) > 1:
        if re.search(r"[^aeiou]le$", word):
            groups[-1] = ""  # table の -le は母音を読まない
        elif re.search(r"[^aeiouy]e$", word) or re.search(r"[^aeiousxz]es$|[^aeioutd]ed$", word):
            groups.pop()  # 語末の黙字 e
    return groups


def _group_vowel(group):
    if group == "":
        return None
    if group in ("ai", "ay"):
        return "e"
    last = group[-1]
    return {"u": "a", "y": "i"}.get(last, last)


def _resolve_vowels(word, phones):
    """母音ごとに (a/i/u/e/o, 後ろに付ける ー/イ/ウ) を決める。None は読まない。"""
    groups = _spelling_groups(word)
    indexes = [i for i, p in enumerate(phones) if p[0] in VOWELS]
    count = len(indexes)
    result = {}
    for order, i in enumerate(indexes):
        base, stress = phones[i]
        if groups:
            if count == len(groups):
                group = groups[order]
            else:
                group = groups[min(len(groups) - 1,
                                   round(order * (len(groups) - 1) / max(count - 1, 1)))]
        else:
            group = "a"
        after = phones[i + 1][0] if i + 1 < len(phones) else None
        after2 = phones[i + 2][0] if i + 2 < len(phones) else None
        before = phones[i - 1][0] if i > 0 else None
        final = i == len(phones) - 1
        # 後ろに母音が続くか、語末の R（here -> ヒア）なら長音にしない。
        open_next = after in VOWELS or (after == "R" and after2 not in VOWELS)
        if base == "AA":
            vowel = "o" if "o" in group or before == "W" else "a"
            value = (vowel, "")
        elif base == "AE":
            value = ("a", "")
        elif base == "AH":
            if stress:
                value = ("a", "")
            else:
                vowel = _group_vowel(group)
                if vowel == "a" and group == "u" and before == "F" and after == "L":
                    vowel = "u"  # beautiful -> ビューティフル
                elif word.endswith("dom") and i == len(phones) - 2:
                    vowel = "a"  # kingdom -> キングダム
                if vowel == "e" and after == "N" and i + 2 == len(phones) \
                        and before not in ("D", "T"):
                    vowel = None  # seven -> セブン、open -> オープン
                value = (vowel, "")
        elif base == "AO":
            long = (after in ("L", "R") or after is None or before == "W"
                    or re.search(r"aw|au|al|ough|oa", word))
            value = ("o", "ー" if long else "")
        elif base == "AW":
            value = ("a", "" if after == "ER" else "ウ")  # power -> パワー
        elif base == "AY":
            value = ("a", "イ")
        elif base == "EH":
            value = ("e", "")
        elif base == "ER":
            value = ("a", "ー")
        elif base == "EY":
            if after == "N" and after2 == "JH":
                tail = ""
            elif final or after in VOWELS or after in ("Z", "D") and i + 2 == len(phones) \
                    or after == "N" and after2 not in VOWELS:
                tail = "イ"
            else:
                tail = "ー"
            value = ("e", tail)
        elif base == "IH":
            value = ("e" if not stress and group == "e" else "i", "")
        elif base == "IY":
            value = ("i", "" if open_next or not stress and not final else "ー")
        elif base == "OW":
            # piano -> ピアノ、window -> ウィンドー
            # 語中の第1強勢以外の OW も伸ばさない（productivity -> プロダクティビティー）。
            short = stress != 1 and (not final or not word.endswith("ow"))
            value = ("o", "" if short else "ー")
        elif base == "OY":
            value = ("o", "イ")
        elif base == "UH":
            value = ("u", "")
        else:  # UW
            # 語中の第1強勢以外は伸ばさない（university -> ユニバーシティー）。
            value = ("u", "" if open_next or stress != 1 and not final else "ー")
        result[i] = value
    return result


def _syllable(consonant, vowel, base):
    if base == "AE" and consonant in ("K", "G"):
        return {"K": "キャ", "G": "ギャ"}[consonant]  # cat -> キャット
    return ROWS[consonant][vowel]


def phones_to_kana(word, pronunciation):
    phones = []
    for p in pronunciation.split():
        base, stress = p.rstrip("012"), int(p[-1]) if p[-1].isdigit() else 0
        if phones and phones[-1][0] == "ER" and base in VOWELS:
            # different の ER＋母音は「ア＋ラ行」で読む（ディファレント）。
            phones[-1:] = [("AH", 1), ("R", 0)]
        phones.append((base, stress))
    vowels = _resolve_vowels(word, phones)
    names = [p[0] for p in phones]
    n = len(phones)
    out = []

    def vowel_at(j):
        return j < n and names[j] in VOWELS and vowels[j][0] is not None

    i = 0
    while i < n:
        name = names[i]
        if name in VOWELS:
            vowel, tail = vowels[i]
            if vowel is not None:
                previous = names[i - 1] if i else None
                if name == "ER" and previous in ("AY", "EY", "OY", "IY"):
                    out.append("ヤー")  # player -> プレイヤー
                elif name == "ER" and previous in ("AW", "OW", "UW"):
                    out.append("ワー")  # power -> パワー
                else:
                    out.append(ROWS[""][vowel] + tail)
            i += 1
            continue
        nxt = names[i + 1] if i + 1 < n else None
        if name == "NG":
            out.append("ン")
            if vowel_at(i + 1):
                out.append(ROWS["G"][vowels[i + 1][0]] + vowels[i + 1][1])
                i += 2
                continue
            if nxt not in ("K", "G"):
                out.append("グ")
            i += 1
            continue
        if nxt == "Y" and vowel_at(i + 2) and name in PALATAL \
                and vowels[i + 2][0] in SMALL_Y:
            vowel, tail = vowels[i + 2]
            out.append(PALATAL[name] + SMALL_Y[vowel] + tail)
            i += 3
            continue
        if nxt == "W" and vowel_at(i + 2) and (name in ("K", "G", "S", "TH")
                                               or name == "T" and i == 0):
            vowel, tail = vowels[i + 2]
            out.append(("トゥ" if name == "T" else ALONE[name]) + AFTER_W[vowel] + tail)
            i += 3
            continue
        previous = names[i - 1] if i else None
        if (i + 1 < n and previous in SHORT and phones[i - 1][1] == 1 and not vowels[i - 1][1]
                and DOUBLED.get(name) and re.search(DOUBLED[name], word)):
            out.append("ッ")  # happy -> ハッピー、fashion -> ファッション
        if vowel_at(i + 1):
            vowel, tail = vowels[i + 1]
            out.append(_syllable(name, vowel, names[i + 1]) + tail)
            i += 2
            continue
        previous_vowel = previous in VOWELS and vowels[i - 1][0] is not None
        if name == "R" and previous_vowel:
            # 語末・子音前の R: car -> カー、care -> ケア、for -> フォー
            if previous == "AA":
                out.append("ー")
            elif previous not in ("AO", "ER", "OW"):
                out.append("ア")
            i += 1
            continue
        rest = names[i + 1:]
        if (name in GEMINATE and previous in SHORT and previous_vowel and not vowels[i - 1][1]
                and (not rest or rest in (["S"], ["Z"]) or rest == ["T"] and word.endswith("ed"))):
            out.append("ッ")
        if name == "T" and nxt == "S" and not vowel_at(i + 2):
            out.append("ツ")
            i += 2
            continue
        if name == "D" and nxt == "Z" and not vowel_at(i + 2):
            out.append("ズ")
            i += 2
            continue
        if name in ("M", "N") and nxt in ("P", "B", "M"):
            out.append("ン")
        else:
            out.append(ALONE[name])
        i += 1
    return re.sub(r"ー+", "ー", "".join(out))


# ---- ローマ字（Irodori、kataribe など英語辞書にない日本語） ----
_ROMAJI = {}
for _row, _kana in {
    "": "ア イ ウ エ オ", "k": "カ キ ク ケ コ", "s": "サ シ ス セ ソ", "t": "タ チ ツ テ ト",
    "n": "ナ ニ ヌ ネ ノ", "h": "ハ ヒ フ ヘ ホ", "m": "マ ミ ム メ モ", "y": "ヤ - ユ - ヨ",
    "r": "ラ リ ル レ ロ", "w": "ワ - - - ヲ", "g": "ガ ギ グ ゲ ゴ", "z": "ザ ジ ズ ゼ ゾ",
    "d": "ダ ヂ ヅ デ ド", "b": "バ ビ ブ ベ ボ", "p": "パ ピ プ ペ ポ", "f": "- - フ - -",
    "j": "ジャ ジ ジュ ジェ ジョ", "sh": "シャ シ シュ シェ ショ", "ch": "チャ チ チュ チェ チョ",
    "ts": "- - ツ - -",
}.items():
    for _v, _k in zip("aiueo", _kana.split()):
        if _k != "-":
            _ROMAJI[_row + _v] = _k
for _c, _i in {"k": "キ", "g": "ギ", "n": "ニ", "h": "ヒ", "m": "ミ", "r": "リ",
               "b": "ビ", "p": "ピ"}.items():
    for _v, _s in SMALL_Y.items():
        _ROMAJI[_c + "y" + _v] = _i + _s
_ROMAJI.update({"si": "シ", "ti": "チ", "tu": "ツ", "hu": "フ", "zi": "ジ", "di": "ヂ",
                "du": "ヅ", "sya": "シャ", "syu": "シュ", "syo": "ショ", "tya": "チャ",
                "tyu": "チュ", "tyo": "チョ", "zya": "ジャ", "zyu": "ジュ", "zyo": "ジョ",
                "jya": "ジャ", "jyu": "ジュ", "jyo": "ジョ"})
# ローマ字にない綴り向けの英語寄りの読み（ローマ字風の読みで使う）。
_LOOSE = {"ti": "ティ", "di": "ディ", "tu": "トゥ", "du": "ドゥ", "fa": "ファ", "fi": "フィ",
          "fe": "フェ", "fo": "フォ", "wi": "ウィ", "we": "ウェ", "wo": "ウォ", "va": "バ",
          "vi": "ビ", "vu": "ブ", "ve": "ベ", "vo": "ボ", "la": "ラ", "li": "リ", "lu": "ル",
          "le": "レ", "lo": "ロ", "ca": "カ", "ci": "シ", "cu": "ク", "ce": "セ", "co": "コ",
          "qa": "クア", "qi": "クイ", "qu": "ク", "qe": "クエ", "qo": "クオ", "xa": "クサ",
          "xi": "クシ", "xu": "クス", "xe": "クセ", "xo": "クソ", "tha": "サ", "thi": "シ",
          "thu": "ス", "the": "セ", "tho": "ソ", "pha": "ファ", "phi": "フィ", "phu": "フ",
          "phe": "フェ", "pho": "フォ", "ye": "イエ", "yi": "イ", "wu": "ウ", "ja": "ジャ"}
_LOOSE_ALONE = {"b": "ブ", "c": "ク", "d": "ド", "f": "フ", "g": "グ", "h": "", "j": "ジ",
                "k": "ク", "l": "ル", "m": "ム", "n": "ン", "p": "プ", "q": "ク", "r": "ル",
                "s": "ス", "t": "ト", "v": "ブ", "w": "ウ", "x": "クス", "z": "ズ", "y": "イ"}


def romaji_to_kana(word, loose=False):
    """ローマ字をカタカナにする。loose=False で読めない綴りがあれば None。"""
    word = word.lower()
    table = {**_ROMAJI, **_LOOSE} if loose else _ROMAJI
    out, i = [], 0
    while i < len(word):
        for size in (3, 2, 1):
            piece = word[i:i + size]
            if piece in table:
                out.append(table[piece])
                i += size
                break
        else:
            c = word[i]
            nxt = word[i + 1] if i + 1 < len(word) else ""
            if c == "n" and nxt not in "aiueoy":
                out.append("ン")
                if nxt == "'":
                    i += 1
            elif c == nxt and c not in "aiueon":
                out.append("ッ")
            elif c == "m" and nxt in "bpm" and not loose:
                out.append("ン")
            elif loose and c in _LOOSE_ALONE:
                if c == "c" and nxt == "k":
                    out.append("ッ")
                else:
                    out.append(_LOOSE_ALONE[c])
            elif loose:
                pass
            else:
                return None
            i += 1
    return "".join(out) or None


def spell(word):
    return "".join(LETTERS[c] for c in word.lower() if c in LETTERS)


def _split_compound(word, words):
    """複合語を辞書にある語（3文字以上、最大3語）に分ける。"""
    n = len(word)
    best = {0: []}
    for end in range(3, n + 1):
        for start in range(0, end - 2):
            part = word[start:end]
            if start in best and part in words and len(best[start]) < 3:
                candidate = best[start] + [part]
                if end not in best or len(candidate) < len(best[end]):
                    best[end] = candidate
    return best.get(n)


def _cmu(lower, words):
    phones = words.get(lower)
    if phones is None:
        return None
    plain = re.sub(r"[0-2]", "", phones).split()
    if len(lower) > 1 and plain == " ".join(LETTER_PHONES.get(c, "?") for c in lower).split():
        return spell(lower)  # 辞書でも1文字ずつ読む略語（usb -> ユーエスビー）
    return phones_to_kana(lower, phones)


@lru_cache(maxsize=8192)
def word_to_kana(word):
    """英字（' と - を含む）1語をカタカナにする。"""
    word = word.replace("’", "'")
    if "-" in word:
        return "".join(word_to_kana(part) for part in word.split("-") if part)
    lower = word.lower()
    words = dictionary()
    if lower in LOANWORDS and not (word.isupper() and len(word) <= 2
                                   and word not in CAPITAL_WORDS):
        return LOANWORDS[lower]
    if len(word) == 1:
        return "アイ" if lower == "i" else ("ア" if word == "a" else LETTERS[lower])
    if word.isupper() and "'" not in word:
        if len(word) <= 2 and word not in CAPITAL_WORDS:
            return spell(word)
        if lower not in LOANWORDS and (lower not in words or not re.search(r"[AEIOUY]", word)):
            return spell(word)  # 辞書にない略語、母音のない略語（HTML、GCP）
    kana = _cmu(lower, words)
    if kana:
        return kana
    parts = re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|'[a-z]+", word)
    if len(parts) > 1:  # GitHubActions、iPhone、ROCK'N
        return "".join(word_to_kana(p.lstrip("'")) for p in parts if p.lstrip("'"))
    if "'" in lower:
        return "".join(word_to_kana(p) for p in lower.split("'") if p)
    kana = romaji_to_kana(lower)
    if kana:
        return kana
    parts = _split_compound(lower, words)
    if parts:
        return "".join(word_to_kana(p) for p in parts)
    if re.search(r"[aeiou]", lower) and len(lower) >= 5:
        return romaji_to_kana(lower, loose=True)
    return spell(word)


WORD = re.compile(r"[A-Za-z]+(?:[-'’][A-Za-z]+)*")
# 変換した語の前後の空白（改行は除く）。カナの間に空白があると Irodori は
# 語ごとに区切って読むので、join=True なら詰めてつなげて読ませる。
WORD_WITH_SPACES = re.compile(r"[ \t　]*(" + WORD.pattern + r")[ \t　]*")


def convert_english(text, hiragana=False, join=False):
    """文中の英字の並びをカナ読みにする。数字や記号、空白、元の日本語はそのまま残す。

    join=True なら変換した語の前後の空白を詰める（アイ ラブ ユー → アイラブユー）。
    """
    pattern, group = (WORD_WITH_SPACES, 1) if join else (WORD, 0)
    if hiragana:
        return pattern.sub(lambda m: to_hiragana(word_to_kana(m.group(group))), text)
    return pattern.sub(lambda m: word_to_kana(m.group(group)), text)


_KATAKANA = {code: code - 0x60 for code in range(0x30A1, 0x30F7)}
_HALF_WIDTH_KANA = re.compile(r"[ｦ-ﾟ]+")


def to_hiragana(text):
    """カタカナをひらがなにする（長音記号はそのまま）。"""
    text = _HALF_WIDTH_KANA.sub(lambda m: unicodedata.normalize("NFKC", m.group()), text)
    return text.translate(_KATAKANA)
