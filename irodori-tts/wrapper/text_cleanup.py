"""コピーした文に紛れ込む特殊文字を、合成の前に整える。

Web ページ・Word・PDF からコピーしたセリフには、改行・見えない文字・HTML の実体参照・
箇条書きの記号などが混ざる。これらはトークナイザの語彙になくバイトのトークンで入ったり、
学習でほとんど見ていなかったりして、ない音（「モイス」「パ」など）や読み違いの元になる。
検証は work/line_breaks/ と work/special_chars/（合成して kana-whisper で書き起こし）。

方針:
- 意味のある記号は読みにする（¥1,000 → 1,000円、3×4 → 3かける4、12:30 → 12時30分 など）。
- 意味のない記号（見えない文字・行頭の箇条書き・飾り）は消す。
- 改行は、連続をまとめて「、」の間にする（直前が句読点なら消すだけ）。
- モデルがもともと読める記号（〜 の範囲、1/2、♪、【】、Q&A など）は触らない。
読み辞書（reading_dictionary）より前に当てる。
"""
from __future__ import annotations

import html
import re
import unicodedata

from reading_dictionary import normalize_width

# 見えない文字（方向制御・ゼロ幅・BOM・ソフトハイフンなど）。U+2063 は語彙分割の区切りなので残す。
# ZWJ（U+200D）は絵文字をつなぐ役（😮‍💨 は「吐息」で、😮「息をのむ」と💨は別物）なので、
# 絵文字に挟まれたものだけ残す。トークナイザにも ZWJ のトークンがあり、上流も消さない。
_INVISIBLE = re.compile(
    "[\u00ad\u034f\u061c\u180e\u200b\u200c\u200e\u200f\u202a-\u202e\u2060-\u2062\u2064\u2066-\u2069\ufeff]")
_PICTOGRAPH = "\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\u2190-\u21FF\u2300-\u23FF\u3297\u3299"
_STRAY_ZWJ = re.compile(
    f"(?<![{_PICTOGRAPH}\ufe0f\U0001F3FB-\U0001F3FF])\u200d|\u200d(?![{_PICTOGRAPH}])")

# ルビ（青空文庫・小説投稿サイトの書き方）。読みだけを残す。
#   ｜漢字《かんじ》・|漢字《かんじ》 … 親文字を｜で区切る
#   漢字《かんじ》                     … 直前の漢字の続きが親文字
# 上流の正規化は《》だけを消すので、そのままだと「漢字かんじ」と二重に読む。
_RUBY_BAR = re.compile(r"[|｜]([^|｜《》\n]{1,30})《([^《》\n]{1,40})》")
_RUBY_KANJI = re.compile(r"[一-龯々〆ヶ]+《([^《》\n]{1,40})》")

# NFKC で ASCII になる互換文字のうち、そのままだと読みを外すもの（№ → No、™ → TM）は先に読みにする。
_COMPAT_FIRST = {"№": "ナンバー", "℡": "テル", "™": "", "®": "", "©": "", "℠": ""}
_COMPAT_FIRST_RE = re.compile("[" + "".join(_COMPAT_FIRST) + "]")
# ローマ数字（Ⅰ〜Ⅻ・ⅰ〜ⅻ）。第Ⅲ章・エリザベスⅡ世のように前後に「第」や助数詞があれば数字、
# それ以外（作品名のⅢ・Ⅶ）は英語の数の読みにする。NFKC だと III になり「アイアイアイ」と読む。
_ROMAN = {chr(0x2160 + i): i + 1 for i in range(12)}
_ROMAN.update({chr(0x2170 + i): i + 1 for i in range(12)})
_ROMAN_ENGLISH = ["ワン", "ツー", "スリー", "フォー", "ファイブ", "シックス", "セブン", "エイト",
                  "ナイン", "テン", "イレブン", "トゥエルブ"]
_ROMAN_RE = re.compile("(第)?([" + "".join(_ROMAN) + "])([章巻部期世代編条話幕回号]|番目)?")
# 「…」は NFKC で「...」になるので守る（上流が最後に「…」へ戻すが、途中の規則が誤解しないように）
_KEEP_ELLIPSIS = str.maketrans({"…": "\ue000", "‥": "\ue001"})
_RESTORE_ELLIPSIS = str.maketrans({"\ue000": "…", "\ue001": "‥"})
# 改行の仲間（垂直タブ・改ページ・NEL・行区切り・段落区切り）
_OTHER_BREAKS = re.compile("[\u000b\u000c\u0085\u2028\u2029]")
_HTML_BREAK = re.compile(r"<\s*(?:br|/?p)\s*/?\s*>", re.IGNORECASE)
_HTML_ENTITY = re.compile(r"&(?:[A-Za-z]{2,8}|#[0-9]{1,6}|#x[0-9A-Fa-f]{1,5});")

_LINE_BREAKS = re.compile(r"[ \t\u3000]*(?:\r\n|[\r\n])[\s\u3000]*")
_PAUSE_BEFORE = frozenset("。、，,．.！!？?…‥:：")

# 行頭の箇条書き・飾り。行頭（改行の直後・文頭）にあるときだけ消す。
_BULLETS = "■□◆◇●○◎★☆✓✔✅▶►▷▸•∙‧▪・※→⇒➡➔*"
_LINE_HEAD = re.compile(rf"(^|\n)[ \t\u3000]*(?:[{re.escape(_BULLETS)}](?![×✕])[ \t\u3000]*)+")
# 文中でも意味を持たない飾り（○● は「○×」のように読む場面があるので残す）。※1 のような注の番号も消す
_DECORATION = re.compile("※[0-9]*|[■□◆◇★✓✔✅▶►▷▸•∙‧▪]")
_ARROWS = re.compile("[→⇒➡➔]")

_CIRCLED = {chr(0x2460 + i): str(i + 1) for i in range(20)}          # ①〜⑳
_CIRCLED.update({chr(0x3251 + i): str(i + 21) for i in range(15)})  # ㉑〜㉟
_CIRCLED.update({chr(0x32B1 + i): str(i + 36) for i in range(15)})  # ㊱〜㊿
_CIRCLED.update({chr(0x2776 + i): str(i + 1) for i in range(10)})   # ❶〜❿
_CIRCLED_RE = re.compile("[" + "".join(_CIRCLED) + "]")

# 序数（1st アルバム → ファーストアルバム）。21st 以降は「21番目」
_ORDINALS = ["ファースト", "セカンド", "サード", "フォース", "フィフス", "シックスス", "セブンス",
             "エイス", "ナインス", "テンス", "イレブンス", "トゥエルフス", "サーティーンス",
             "フォーティーンス", "フィフティーンス", "シックスティーンス", "セブンティーンス",
             "エイティーンス", "ナインティーンス", "トゥエンティース"]


def _ordinal(match: re.Match) -> str:
    n = int(match[1])
    return _ORDINALS[n - 1] if 1 <= n <= len(_ORDINALS) else f"{n}番目"


def _roman(match: re.Match) -> str:
    prefix, numeral, counter = match[1] or "", match[2], match[3] or ""
    n = _ROMAN[numeral]
    if prefix or counter:
        return f"{prefix}{n}{counter}"
    return _ROMAN_ENGLISH[n - 1]


_DIGIT_READINGS = dict(zip("0123456789", ["ゼロ", "イチ", "ニー", "サン", "ヨン", "ゴー", "ロク", "ナナ", "ハチ", "キュー"]))


def _digits(number: str) -> str:
    """郵便番号のように1桁ずつ読む数。"""
    return "".join(_DIGIT_READINGS[d] for d in number)


def _man(number: str) -> str:
    """4桁ずつ万・億・兆で区切る（1234567 → 123万4567、100000 → 10万）。

    モデルは4桁までなら確実に読むが、6桁以上で途中に0がある数や7桁以上の数は崩れる
    （100000人 → イチマンニン）。区切ると全部読めた（work/numbers/）。
    """
    n = int(number.replace(",", ""))
    out = ""
    for unit, name in ((10 ** 12, "兆"), (10 ** 8, "億"), (10 ** 4, "万")):
        if n // unit % 10000:
            out += f"{n // unit % 10000}{name}"
    return out + (str(n % 10000) if n % 10000 else "")


# 数量として読む数のあとに来る単位。番号（注文番号 20240917 など）は1桁ずつ読めるので、
# 単位が続くか、カンマで3桁区切りになっている数だけを区切る。英字の単位（km など）は
# 読み辞書が数字の直後でだけ読むので、ここには入れない（「1000万km」だと読まれなくなる）。
_COUNTERS = ("円|ドル|ユーロ|ポンド|元|ウォン|人|名|個|件|本|枚|台|回|冊|歳|倍|点|票|戸|世帯|匹|頭|羽|"
             "社|店|校|軒|部|通|株|口|か所|カ所|ヶ所|箇所|坪|平米|字|文字|行|ページ|歩|位|日|時間|分|秒|"
             "年間|か月|カ月|ヶ月|週間|万|億|兆|%|パーセント|ポイント|メートル|キロ|グラム|トン|リットル|"
             "カロリー|バイト")
_QUANTITY = re.compile(
    rf"(?<![0-9A-Za-z.,\-])([1-9][0-9]{{4,15}}|[1-9][0-9]{{0,2}}(?:,[0-9]{{3}})+)(?![0-9]|[.,\-][0-9])"
    rf"(?=\s?(?:{_COUNTERS})|(?<=[0-9],[0-9]{{3}}))")


_NUM = r"[0-9][0-9,]*(?:\.[0-9]+)?"
_RULES = [
    # 通貨: 記号を数の後ろの読みに
    (re.compile(rf"[¥￥]\s?({_NUM})(万|億|兆)?"), lambda m: f"{m[1]}{m[2] or ''}円"),
    (re.compile(rf"\$\s?({_NUM})(万|億|兆)?"), lambda m: f"{m[1]}{m[2] or ''}ドル"),
    (re.compile(rf"€\s?({_NUM})(万|億|兆)?"), lambda m: f"{m[1]}{m[2] or ''}ユーロ"),
    (re.compile(rf"£\s?({_NUM})(万|億|兆)?"), lambda m: f"{m[1]}{m[2] or ''}ポンド"),
    # 時刻（12:30 → 12時30分、9:05 → 9時5分）。比率の 16:9 などは分が2桁でないので当たらない
    (re.compile(r"(?<![0-9:])([01]?[0-9]|2[0-4]):([0-5][0-9])(?![0-9:])"),
     lambda m: f"{int(m[1])}時" + ("" if m[2] == "00" else f"{int(m[2])}分")),
    # 日付（2024/10/11・2024-10-11 → 2024年10月11日）。年が4桁のものだけ
    (re.compile(r"(?<![0-9])((?:19|20)[0-9]{2})[/\-.]([01]?[0-9])[/\-.]([0-3]?[0-9])(?![0-9])"),
     lambda m: f"{m[1]}年{int(m[2])}月{int(m[3])}日"),
    # 郵便番号（〒100-0001 → 郵便番号 1 0 0 の 0 0 0 1）
    (re.compile(r"〒\s?([0-9]{3})-?([0-9]{4})"), lambda m: f"郵便番号、{_digits(m[1])}の{_digits(m[2])}、"),
    (re.compile("〒"), lambda m: "郵便番号"),
    # 番号・版
    (re.compile(r"(?<![A-Za-z])(?:No|NO|no)\.\s?(?=[0-9])"), lambda m: "ナンバー"),
    # 版の番号が3桁以上（ver.200009）なら1桁ずつ。数字のままだと「ニセンゼロキュー」と崩れる
    (re.compile(r"(?<![A-Za-z])(?:ver|Ver|VER)\.\s?([0-9]{3,})(?![0-9.])"), lambda m: "バージョン" + _digits(m[1])),
    (re.compile(r"(?<![A-Za-z])(?:ver|Ver|VER)\.\s?(?=[0-9])"), lambda m: "バージョン"),
    # 午前・午後（AM9:00・9:00 PM は、上の時刻の規則で「9時」になっている）
    (re.compile(r"(?<![A-Za-z])(?:AM|am|A\.M\.|a\.m\.)\s?(?=[0-9]{1,2}時)"), lambda m: "午前"),
    (re.compile(r"(?<![A-Za-z])(?:PM|pm|P\.M\.|p\.m\.)\s?(?=[0-9]{1,2}時)"), lambda m: "午後"),
    (re.compile(r"([0-9]{1,2}時(?:[0-9]{1,2}分)?)\s?(?:AM|am|A\.M\.|a\.m\.)(?![A-Za-z])"),
     lambda m: f"午前{m[1]}"),
    (re.compile(r"([0-9]{1,2}時(?:[0-9]{1,2}分)?)\s?(?:PM|pm|P\.M\.|p\.m\.)(?![A-Za-z])"),
     lambda m: f"午後{m[1]}"),
    # 序数（1st・2nd・3rd・4th）
    (re.compile(r"(?<![0-9A-Za-z])([0-9]{1,3})(?:st|nd|rd|th|ST|ND|RD|TH)(?![A-Za-z])"), _ordinal),
    # 敬称・ページ・巻の略記（英語の辞書に当たると Dr. → ドライブ、vol → ブイオーエル になる）
    (re.compile(r"(?<![A-Za-z])Dr\.\s?(?=[A-Za-z])"), lambda m: "ドクター "),
    (re.compile(r"(?<![A-Za-z])Mr\.\s?(?=[A-Za-z])"), lambda m: "ミスター "),
    (re.compile(r"(?<![A-Za-z])Mrs\.\s?(?=[A-Za-z])"), lambda m: "ミセス "),
    (re.compile(r"(?<![A-Za-z])Ms\.\s?(?=[A-Za-z])"), lambda m: "ミズ "),
    (re.compile(r"(?<![A-Za-z])pp?\.\s?([0-9]+)(?:\s?[-〜~]\s?([0-9]+))?"),
     lambda m: f"{m[1]}ページ" + (f"から{m[2]}ページ" if m[2] else "")),
    (re.compile(r"(?<![A-Za-z])(?:vol|Vol|VOL)\.\s?(?=[0-9])"), lambda m: "ボリューム"),
    (re.compile(r"(?<![A-Za-z])(?:ch|Ch|CH)\.\s?(?=[0-9])"), lambda m: "チャプター"),
    # 負の数（−3℃ → マイナス3℃）。上流の正規化は U+2212 を消すので「3ド」になっていた。
    # 数・単位のあとのハイフンは範囲（10時-12時 → 10時から12時、5-10分 → 5から10分）
    (re.compile(r"(?<=[0-9][時分秒人個件円日月年歳点名回本枚])\s?[-−]\s?(?=[0-9])"), lambda m: "から"),
    (re.compile(r"(?<![0-9\-])([0-9]{1,4})\s?[-−]\s?([0-9]{1,4})(?=\s?(?:[分秒時人個件円日月年歳点名回本枚%]|時間|ページ))"),
     lambda m: f"{m[1]}から{m[2]}"),
    # 温度の負の数は前の文字によらない（最低気温-5℃）
    (re.compile(r"(?<![0-9A-Za-z.])[-−](?=[0-9.]+\s?(?:°C|°F|度))"), lambda m: "マイナス"),
    # U+2212（−）は数字の後でなければ負の記号。ASCII の - は文頭・空白・括弧・句読点・助詞の後だけ
    # （5-10 や 03-1234 は前が数字なので当たらない）
    (re.compile(r"(?<![0-9A-Za-z.\-−])−(?=[0-9])"), lambda m: "マイナス"),
    (re.compile(r"(?:^|(?<=[\s(（「『【\[=:：、。,はがもをにで]))-(?=[0-9])"), lambda m: "マイナス"),
    # 演算・比較。英字の語どうしをつなぐ ×（HUNTER×HUNTER、SPY×FAMILY）は作品名の飾りなので読まない
    # （詰めると1語に見えて綴りで読まれるので、空白で区切る）
    (re.compile(r"(?<=[A-Za-z]{2})\s?[×✕]\s?(?=[A-Za-z]{2})"), lambda m: " "),
    (re.compile(r"(?<![○◯〇])\s?[×✕]\s?"), lambda m: "かける"),
    (re.compile(r"\s?÷\s?"), lambda m: "わる"),
    (re.compile(r"±\s?"), lambda m: "プラスマイナス"),
    (re.compile(r"\s?≒\s?(?=[0-9])"), lambda m: "約"),
    (re.compile(r"\s?[≒≈]\s?"), lambda m: "ニアリーイコール"),
    (re.compile(r"\s?≠\s?"), lambda m: "ノットイコール"),
    (re.compile(r"\s?[≧≥]\s?"), lambda m: "ダイナリイコール"),
    (re.compile(r"\s?[≦≤]\s?"), lambda m: "ショウナリイコール"),
    # 「100万円〜」のように、数・単位の後ろで終わる 〜 は「から」（範囲の 1〜3 はモデルが読める）
    (re.compile(r"(?<=[0-9万億兆円人個件時分秒歳日月年点名])[〜～~](?![0-9〜～~])"), lambda m: "から"),
    # 笑いの w（草www、それなw）。日本語の直後の w の連なりは読まない（ダブリューの連打になる）
    (re.compile(r"(?<=[^\x00-\x7f])[wW]+(?![A-Za-z0-9])"), lambda m: ""),
    # URL の頭（https:// と www.）は読まない
    (re.compile(r"https?://(?:www\.)?"), lambda m: ""),
    # 5桁以上の数量は万・億で区切る（通貨の記号を読みにした後に当てる）
    (_QUANTITY, lambda m: _man(m[1]) if len(m[1].replace(",", "")) >= 5 else m[0]),
]


def _line_breaks_to_pauses(text: str) -> str:
    def replace(match: re.Match) -> str:
        before = text[match.start() - 1] if match.start() else ""
        return "" if not before or before in _PAUSE_BEFORE else "、"
    return _LINE_BREAKS.sub(replace, text)


def _compat(text: str) -> str:
    """\u4e92\u63db\u6587\u5b57\uff08\u2116\u30fb\u2162\u30fb\u2113\u30fb\u338f\u30fb\u534a\u89d2\u30ab\u30ca\u30fb\u00b2 \u306a\u3069\uff09\u3092\u3001\u8aad\u307f\u8f9e\u66f8\u3084\u898f\u5247\u304c\u5206\u304b\u308b\u5f62\u306b\u305d\u308d\u3048\u308b\u3002

    \u4e0a\u6d41\u306e\u6b63\u898f\u5316\u3082\u6700\u5f8c\u306b NFKC \u3092\u304b\u3051\u308b\u306e\u3067\u3001\u5148\u306b\u304b\u3051\u3066\u3082\u6700\u7d42\u306e\u6587\u5b57\u5217\u306f\u5909\u308f\u3089\u306a\u3044\u3002
    \u305f\u3060\u3057\u3001\u305d\u306e\u307e\u307e\u3060\u3068\u8aad\u307f\u3092\u5916\u3059\u3082\u306e\uff08\u2116 \u2192 No\u3001\u2162 \u2192 III\uff09\u306f\u5148\u306b\u8aad\u307f\u306b\u3059\u308b\u3002
    """
    text = _COMPAT_FIRST_RE.sub(lambda m: _COMPAT_FIRST[m[0]], text)
    text = _ROMAN_RE.sub(_roman, text)
    text = _CIRCLED_RE.sub(lambda m: _CIRCLED[m[0]], text)
    return unicodedata.normalize("NFKC", text.translate(_KEEP_ELLIPSIS)).translate(_RESTORE_ELLIPSIS)


def clean_pasted_text(text: str) -> str:
    text = _STRAY_ZWJ.sub("", _INVISIBLE.sub("", text))
    text = _RUBY_BAR.sub(r"\2", text)
    text = _RUBY_KANJI.sub(r"\1", text)
    text = normalize_width(_compat(text))
    text = _OTHER_BREAKS.sub("\n", text)
    text = _HTML_BREAK.sub("\n", text)
    if "&" in text:
        text = _HTML_ENTITY.sub(lambda m: html.unescape(m[0]), text).replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _DECORATION.sub("", text)
    text = _LINE_HEAD.sub(r"\1", text)
    for pattern, replace in _RULES:
        text = pattern.sub(replace, text)
    text = _ARROWS.sub("、", text)
    text = _line_breaks_to_pauses(text.strip())
    text = re.sub(r"、[ 、]*", "、", text)
    text = re.sub(r"(^|[。!?])、+", r"\1", text)
    return text.strip(" 、")
