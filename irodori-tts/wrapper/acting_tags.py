"""他の TTS の演技タグ（[laughing]・<sigh>・(laughs) など）を、Irodori-TTS の絵文字に置き換える。

Gemini（<laugh>・<short pause>、旧版は [laughing]）、ElevenLabs v3（[whispers]）、Fish Audio
（S1 は (laughing)、S2 は [laughing]）、CosyVoice（[breath]・<laughter>…</laughter>）、Dia（(sighs)）、
Bark（[laughter]）の書き方を受け付ける。そのままだと英単語のカナ化で「ラフィング」と読み上げる。
Irodori-TTS v4 系は絵文字で演技を指示できる（models/EMOJI_ANNOTATIONS.md）。

- [] と <> は、表にないタグも消す（読み上げてしまうよりよい）。閉じタグ（</laughter>）も消す。
- () と （） と *x* は、日本語の補足（（注）・(仮)）とぶつかるので、表に当たったときだけ置き換える。
- Gemini の相槌 |mm-hmm| は 👌。
- 効果音・歌・なまり・話者タグ（[S1]）は表せないので消す。
- 間は ⏸️ の数で長さを表す（⏸️ 1つで約0.4秒、⏸️⏸️⏸️ で約1.1秒。work の実測）。
text_cleanup.clean_pasted_text の最初の方（英単語のカナ化・NFKC より前）で当てる。
"""
from __future__ import annotations

import re

LAUGH, SIGH, PANT, GASP, MOAN, CRY, SCREAM = "🤭", "😮‍💨", "🌬️", "😮", "🥵", "😭", "😱"
WHISPER, YAWN, SLEEP, COUGH, TSK, GULP, LIPS, HUM, NOD = "👂", "🥱", "😴", "🤧", "😒", "🥤", "💋", "🎵", "👌"
PAUSE = "⏸️"
DROP = ""  # 表せないので消す

# 英語の語（原形）→ 絵文字。複数形・ing・ed・ly は _stems でそろえる
_WORDS = {
    **dict.fromkeys(["laugh", "laughter", "chuckle", "giggle", "snicker", "cackle", "snort", "wheeze"], LAUGH),
    **dict.fromkeys(["breath", "breathe", "exhale", "sigh", "phew", "pff"], SIGH),
    **dict.fromkeys(["pant", "huff"], PANT),
    **dict.fromkeys(["gasp", "inhale"], GASP),
    **dict.fromkeys(["moan", "groan", "grunt", "grr", "growl", "argh"], MOAN),
    **dict.fromkeys(["cry", "sob", "whimper", "weep", "sad", "sorrowful"], CRY),
    **dict.fromkeys(["scream", "shriek", "shout", "yell"], SCREAM),
    **dict.fromkeys(["whisper", "murmur", "hush"], WHISPER),
    "yawn": YAWN, "snore": SLEEP,
    **dict.fromkeys(["cough", "sneeze", "sniff", "sniffle"], COUGH),
    **dict.fromkeys(["tsk", "tut", "cluck", "clucking"], TSK),
    **dict.fromkeys(["swallow", "gulp"], GULP),
    "lipsmack": LIPS, "hum": HUM, "humming": HUM, "pause": PAUSE, "break": PAUSE, "silence": PAUSE + PAUSE,
    # 感情・話し方
    **dict.fromkeys(["happy", "joyful", "excited", "delighted", "cheer"], "😆"),
    **dict.fromkeys(["cheerful", "glad", "warm", "friendly"], "😊"),
    **dict.fromkeys(["angry", "mad", "annoyed", "frustrated", "furious", "irritated", "sulky"], "😠"),
    **dict.fromkeys(["surprised", "shocked", "amazed", "astonished"], "😲"),
    **dict.fromkeys(["nervous", "anxious", "scared", "afraid", "fearful", "panicked", "stutter"], "😰"),
    **dict.fromkeys(["worried", "concerned"], "😟"),
    **dict.fromkeys(["shy", "embarrassed", "bashful", "flustered"], "🫣"),
    **dict.fromkeys(["sarcastic", "mischievous", "teasing", "playful", "flirtatious", "coy", "smirk"], "😏"),
    **dict.fromkeys(["curious", "confused", "questioning", "wondering", "puzzled"], "🤔"),
    **dict.fromkeys(["tender", "gentle", "soft", "softly", "gently", "kind", "loving", "sweet", "sweetly"], "🫶"),
    **dict.fromkeys(["sleepy", "tired", "drowsy", "exhausted"], "😪"),
    **dict.fromkeys(["hurried", "rushed", "fast", "quickly", "rapid"], "⏩"),
    **dict.fromkeys(["slow", "slowly"], "🐢"),
    **dict.fromkeys(["confident", "proud", "smug"], "😎"),
    **dict.fromkeys(["relieved", "calm", "content", "satisfied"], "😌"),
    **dict.fromkeys(["pleading", "begging", "desperate"], "🙏"),
    **dict.fromkeys(["exasperated", "disgusted", "unimpressed"], "🙄"),
    **dict.fromkeys(["pained", "painful", "agonized", "suffering"], "😖"),
    **dict.fromkeys(["trembling", "timid", "shaky", "uncertain", "hesitant"], "🥺"),
    **dict.fromkeys(["emphasis", "emphatic", "strong", "strongly", "forceful", "determined"], "💪"),
    "drunk": "🥴", "narration": "📖", "narrator": "📖", "echo": "📢", "muffled": "🤐", "phone": "📞",
    # 表せないもの（効果音・歌・なまり・話者タグ）
    **dict.fromkeys(["gunshot", "explosion", "applause", "clap", "clapping", "beep", "music", "noise", "fart",
                     "burp", "sing", "song", "woo", "accent", "hiss", "hissing", "mumble", "whistle", "mn",
                     "audience", "crowd", "background", "vocalized", "door", "footstep", "knock", "s1", "s2",
                     "s3", "s4", "speaker", "sound", "effect", "sfx"], DROP),
}

# 語の組み合わせで意味が決まるもの（語ごとに見るより先に当てる）
_PHRASES = {
    "short pause": PAUSE, "medium pause": PAUSE * 2, "long pause": PAUSE * 3, "long break": PAUSE * 3,
    "short break": PAUSE, "beat": PAUSE,
    "heavy breath": PANT, "heavy breathing": PANT, "breathing heavily": PANT, "out of breath": PANT,
    "quick breath": SIGH, "deep breath": SIGH,
    "clear throat": COUGH, "clears throat": COUGH, "clearing throat": COUGH, "throat clearing": COUGH,
    "laughs harder": LAUGH * 2, "laughing hard": LAUGH * 2, "crying loudly": CRY * 2,
    "in a hurry tone": "⏩", "in a hurry": "⏩", "soft tone": "🫶", "sleep talking": SLEEP, "lip smack": LIPS,
    "mm hmm": NOD, "uh huh": NOD, "mhm": NOD, "backchannel": NOD,
    "audience laughing": DROP, "crowd laughing": DROP, "background laughter": DROP,
}

# 日本語のタグ（括弧の中身と完全一致）
_JAPANESE = {
    **dict.fromkeys(["笑", "笑い", "笑う", "笑いながら", "くすくす", "クスクス", "微笑", "苦笑", "含み笑い", "失笑"], LAUGH),
    **dict.fromkeys(["爆笑", "大笑い", "大爆笑"], LAUGH * 2),
    **dict.fromkeys(["息", "吐息", "溜息", "ため息", "溜め息", "深呼吸", "嘆息"], SIGH),
    **dict.fromkeys(["息切れ", "荒い息", "息遣い", "呼吸", "はぁはぁ", "ハァハァ"], PANT),
    **dict.fromkeys(["息をのむ", "息を呑む", "息を飲む"], GASP),
    **dict.fromkeys(["喘ぎ", "あえぎ", "うめき", "呻き", "唸り", "うなり"], MOAN),
    **dict.fromkeys(["泣", "泣き", "泣く", "泣きながら", "嗚咽", "すすり泣き", "号泣", "涙"], CRY),
    **dict.fromkeys(["悲鳴", "叫び", "叫ぶ", "絶叫"], SCREAM),
    **dict.fromkeys(["囁き", "ささやき", "囁く", "ささやく", "囁き声", "小声", "ひそひそ", "耳元で"], WHISPER),
    **dict.fromkeys(["あくび", "欠伸"], YAWN),
    **dict.fromkeys(["いびき", "寝言", "寝息"], SLEEP),
    **dict.fromkeys(["咳", "せき", "咳払い", "咳き込む", "くしゃみ", "鼻をすする"], COUGH),
    "舌打ち": TSK,
    **dict.fromkeys(["ごくり", "唾を飲む", "唾を飲み込む", "生唾"], GULP),
    "リップノイズ": LIPS, "鼻歌": HUM,
    **dict.fromkeys(["相槌", "あいづち", "頷き", "うなずき"], NOD),
    **dict.fromkeys(["間", "沈黙", "無言", "一拍", "ため"], PAUSE),
    **dict.fromkeys(["長い間", "長い沈黙"], PAUSE * 3),
    **dict.fromkeys(["怒", "怒り", "怒って", "怒りながら", "不満げに", "拗ねて"], "😠"),
    **dict.fromkeys(["驚", "驚き", "驚いて"], "😲"),
    **dict.fromkeys(["喜", "喜び", "嬉しそうに", "喜んで"], "😆"),
    **dict.fromkeys(["楽しげに", "楽しそうに"], "😊"),
    **dict.fromkeys(["悲しそうに", "悲しげに"], CRY),
    **dict.fromkeys(["照", "照れ", "照れながら", "恥ずかしそうに"], "🫣"),
    **dict.fromkeys(["呆れ", "呆れて"], "🙄"),
    **dict.fromkeys(["焦", "汗", "慌てて", "動揺", "緊張"], "😰"),
    **dict.fromkeys(["優しく"], "🫶"),
    **dict.fromkeys(["眠そうに", "気だるげに"], "😪"),
    **dict.fromkeys(["早口"], "⏩"),
    **dict.fromkeys(["ゆっくり"], "🐢"),
    **dict.fromkeys(["震え声", "震えながら"], "🥺"),
    **dict.fromkeys(["からかうように", "甘えるように"], "😏"),
    **dict.fromkeys(["得意げに", "自慢げに"], "😎"),
    **dict.fromkeys(["安堵", "ほっとして"], "😌"),
    **dict.fromkeys(["心配そうに"], "😟"),
    **dict.fromkeys(["ナレーション", "モノローグ", "独白"], "📖"),
    **dict.fromkeys(["電話越し"], "📞"),
    **dict.fromkeys(["効果音", "SE", "BGM", "歌", "拍手"], DROP),
}

# 後で text_cleanup が改行にする HTML のタグは残す
_HTML_KEEP = frozenset(["br", "p"])

_SECONDS_PER_PAUSE = 0.4
_MAX_PAUSES = 5

_CONTENT = r"[^\[\]<>()（）［］＜＞|*\n]{1,40}"
_BREAK = re.compile(
    r"[<＜]\s*break\b(?:[^>＞]*?\btime\s*=\s*[\"']?\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|s)?\s*[\"']?)?[^>＞]*[>＞]",
    re.IGNORECASE)
# <x>・</x>・<x/>・<|x|>・[x]
_STRICT = re.compile(rf"[<＜]\|?\s*(/)?\s*({_CONTENT}?)\s*/?\s*\|?[>＞]|[\[［]\s*({_CONTENT})\s*[\]］]")
# (x)・（x）・*x*・|x|
_LOOSE = re.compile(rf"[(（]\s*({_CONTENT})\s*[)）]|\*({_CONTENT})\*|(?<![|｜])\|([A-Za-z][A-Za-z \-']{{0,24}})\|")
_WORD = re.compile(r"[a-z0-9]+")


def _stems(word: str):
    yield word
    for suffix, repl in (("ally", ""), ("ily", "y"), ("ly", ""), ("ing", ""), ("ing", "e"), ("ed", ""), ("ed", "e"),
                         ("es", ""), ("s", ""), ("ies", "y"), ("ied", "y")):
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            stem = word[: -len(suffix)] + repl
            yield stem
            if suffix in ("ing", "ed") and len(stem) > 2 and stem[-1] == stem[-2]:
                yield stem[:-1]  # sobbing → sob


def _lookup_word(word: str) -> str | None:
    for stem in _stems(word):
        if stem in _WORDS:
            return _WORDS[stem]
    return None


def lookup(content: str, loose: bool = False) -> str | None:
    """タグの中身 → 絵文字（消すなら ""）。表になければ None。

    loose（() と *x*）は、英語なら1語か決まった言い回しだけ当てる（(sad story) のような補足を拾わない）。
    """
    text = content.strip().strip(".,!?:;。、")
    if not text:
        return None
    if text in _JAPANESE:
        return _JAPANESE[text]
    lowered = text.lower()
    words = _WORD.findall(lowered.replace("_", " ").replace("-", " "))
    if not words:
        return None
    phrase = " ".join(words)
    if phrase in _PHRASES:
        return _PHRASES[phrase]
    if loose and len(words) > 1:
        return None
    found = [_lookup_word(w) for w in words]
    if DROP in found:  # [audience laughing]・[strong French accent] は表せない
        return DROP
    for value in found:
        if value:
            return value  # [whispers sweetly] → 囁き、[crying loudly] → 泣き
    return None


def _pauses(seconds: float) -> str:
    return PAUSE * max(1, min(_MAX_PAUSES, round(seconds / _SECONDS_PER_PAUSE)))


def _break(match: re.Match) -> str:
    if not match[1]:
        return PAUSE
    seconds = float(match[1]) / (1000 if (match[2] or "").lower() == "ms" else 1)
    return _pauses(seconds)


def _strict(match: re.Match) -> str:
    if match[2] is not None and match[0][:1] in "<＜":
        name = match[2].strip().lower()
        if name in _HTML_KEEP:
            return match[0]
        if match[1]:  # 閉じタグ（範囲の終わり）
            return ""
        return lookup(match[2]) or ""
    if match[3].lstrip().startswith("/"):  # [/laugh]
        return ""
    return lookup(match[3]) or ""


def _loose(match: re.Match) -> str:
    content = match[1] or match[2]
    if content is not None:
        found = lookup(content, loose=True)
        return match[0] if found is None else found
    found = lookup(match[3])
    return NOD if found is None else found  # Gemini の |reaction| は相槌


def _join(text: str) -> str:
    """置き換えた所の前後の空白を詰める（英語の語どうしの間だけ1つ残す）。"""
    def squeeze(match: re.Match) -> str:
        before = text[match.start() - 1] if match.start() else ""
        after = text[match.end()] if match.end() < len(text) else ""
        body = match[1]
        if before.isascii() and before.isalnum() and after.isascii() and after.isalnum():
            return f" {body} " if body else " "
        if not body and before and after and match[0] != match[0].strip():
            return " "  # 消したタグの両側の語を1つにつなげない（[S1] こんにちは [S2] やあ）
        return body
    return _MARKED.sub(squeeze, text)


_MARK_OPEN, _MARK_CLOSE = "", ""
_MARKED = re.compile(rf"[ \t　]*{_MARK_OPEN}([^{_MARK_CLOSE}]*){_MARK_CLOSE}[ \t　]*")


def replace_acting_tags(text: str) -> str:
    if not any(c in text for c in "[]<>()（）［］＜＞|*"):
        return text
    changed = False

    def mark(fn):
        def wrapped(match: re.Match) -> str:
            nonlocal changed
            out = fn(match)
            if out == match[0]:
                return out
            changed = True
            return f"{_MARK_OPEN}{out}{_MARK_CLOSE}"
        return wrapped

    text = _BREAK.sub(mark(_break), text)
    text = _STRICT.sub(mark(_strict), text)
    text = _LOOSE.sub(mark(_loose), text)
    return _join(text) if changed else text


# ネットスラングの笑いの w（それはないw・マジかwww）。笑い声を足すのではなく、その文を
# 笑いながら・小ばかにするように言う印なので、文の頭に 😏 を置く（3つ重ねると笑いながら
# 話す調子になる。エディタで試して確認）。w が1つなら 😏 3つ、w が増えるごとに1つ足し、
# 10 個まで（10 個並べても破綻しなかった）。w そのものは読まない（ダブリューの連打になる）。
# w の位置には ! を置く。😏 だけだと「マジかwww」が笑わずに言うだけになるが、! で勢いが
# 付いて笑っている感じが強くなる（エディタで確認）。w の直後の 、。 は ! に吸収する。
# 日本語の直後か、行頭・空白の後の 2つ以上の連なりだけを拾う（wow・www.example.com は触らない）。
LAUGH_W, _LAUGH_W_MIN, _LAUGH_W_MAX = "😏", 3, 10
_LAUGH_W_RUN = re.compile(
    r"(?:(?<=[^\x00-\x7f])(?P<after_word>[wW]+)|(?<![^\s])[wW]{2,})(?![A-Za-z0-9.])(?P<mark>[、。，,!！?？]?)"
)


def _laugh_w_mark(match: re.Match) -> str:
    # 行頭の www（wwwで草）は言葉の後ではないので ! を置かない。後ろに !? があればそれを残す
    if match["mark"] in ("", "、", "。", "，", ","):
        return "!" if match["after_word"] else ""
    return match["mark"]
_SENTENCE_END = re.compile(r"[。！？!?\n]")


def replace_laugh_w(text: str) -> str:
    if "w" not in text and "W" not in text:
        return text
    out: list[str] = []
    pos = 0
    for sentence_end in [*(m.end() for m in _SENTENCE_END.finditer(text)), len(text)]:
        sentence = text[pos:sentence_end]
        pos = sentence_end
        runs = [len(m[0]) - len(m["mark"]) for m in _LAUGH_W_RUN.finditer(sentence)]
        if not runs:
            out.append(sentence)
            continue
        stripped = _LAUGH_W_RUN.sub(_laugh_w_mark, sentence)
        if not stripped.strip():  # w だけの文（wwwww）は話す中身がないので笑い声にする
            out.append(stripped + LAUGH)
            continue
        head = len(stripped) - len(stripped.lstrip())
        count = min(_LAUGH_W_MIN + max(runs) - 1, _LAUGH_W_MAX)
        out.append(stripped[:head] + LAUGH_W * count + stripped[head:])
    return "".join(out)
