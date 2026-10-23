"""語彙分割: 学習の少ない「まとまりトークン」を避けて、細かいトークンで読ませる。

トークナイザ（modernbert-ja）は「浦和レッズ」「ゼルダの伝説」のような語句を丸ごと1トークンに
する。こうした出現度の低いトークンは TTS の学習でほとんど見ていないため、別の語に化けたり
詰まったりする。そこで、4文字以上で出現度（Unigram の logp）が境目以下のトークンを使わない
ことにした（スコアを -1e9 にした写しのトークナイザでエンコードする）。トークナイザが
代わりに「浦和|レッ|ズ」のような学習済みのトークンへ分けるので、文字列は書き換えない。

境目より上でも言えない語は、FORCE_SPLITS に割り方を書いて、その場で割る（セブン|イレブン など）。
境目は全体の設定（token_split_threshold: -10 / -11 / -12 / -13、既定 -13）。0 に近いほど
よく出る語なので、境目を上げる（-10 に近づける）ほど分ける語が増える。-13 なら4文字以上の
日本語の語句の約2割（よく出る側）はそのまま残る。スコアの尺度は modernbert-ja のものなので、
このトークナイザを使うモデル（v4 Small 系）にだけ当てる。

検証（work/len_split/）: ふつうの文ではトークン +8%・音声の長さ +3%、文の読み誤りは増えない。
読めない語を含む文の OK率は 0.41 → 0.75。生成時間はほとんど変わらない（長さ予測がトークン数を
特徴量に使うので、分けすぎると発話が間延びする。-10 に近いほど伸びる）。

文中の見えない区切り（U+2063）では、その位置で分けてエンコードする（区切り用のトークンは
入れない）。全体の設定 token_split_bunsetsu がオンなら、文節の頭（助詞・助動詞のあとに自立語が
来る位置）でも分ける（bunsetsu.py。MeCab を使う）。
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import re
import threading
from pathlib import Path

SPLIT_MARK = "⁣"  # INVISIBLE SEPARATOR
ZERO_WIDTH_SPACE = "​"
# 語彙分割を 使う / 使わない（セリフごと）
TOKEN_SPLITS = ("on", "off")
# 分けるトークンの出現度の境目（全体の設定）。none ならどのセリフにも当てない。
TOKEN_SPLIT_THRESHOLDS = ("-10", "-11", "-12", "-13", "none")
DEFAULT_TOKEN_SPLIT_THRESHOLD = "-13"
# これより短いトークンは分けない（3文字以下のカタカナ語は分けると悪化しやすい）。
MIN_SPLIT_CHARS = 4
# 境目の尺度を決めたトークナイザ。これを使うモデル（v4 Small 系）にだけ当てる。
DEFAULT_TOKENIZER_REPO = "sbintuitions/modernbert-ja-310m"
# 境目以下でも分けないトークン。分けると読めなくなり、まとまりのままなら読める語
# （work/vocab_single: 語だけで 4回中ほぼ毎回言い間違え、分割なしでは言えた。耳でも確認）。
# 4字熟語が1字ずつに割れる・分け方で読みが変わる（瀬戸内|海 → せとうちうみ）など。
KEEP_TOKENS = frozenset({
    "心筋梗塞", "日常茶飯事", "瀬戸内海", "老若男女", "興味津々", "高田馬場", "アクリル板", "年モデル",
    "オフショア", "モスクワ", "テラス席", "缶バッジ", "ジュリー", "ッチャー", "ティーン", "トゥルー",
    "マリッジ", "リフィル", "リベット",
    # 分けても言えるが、1文字の片などに割れて抑揚が崩れるので、耳で聞いて「分けない」と決めた語
    # （work/split_tool の decisions.json）
    "デパート", "カリウム", "タピオカ", "ボーイズ", "フレックス", "フリース", "ファブリック",
    "ユナイテッド", "ブルース", "ミスター", "タイムズ", "ソケット", "コスチューム", "プレビュー",
    "リーズナブル", "プロダクション", "ファイト", "アバター", "産婦人科", "行方不明", "直射日光",
})
# 出現度が境目より上でも、この割り方で必ず分けるトークン（語 → 片）。片は語彙にあればそのトークン
# （出現度が低くても使う）、無ければその片を写しのトークナイザでエンコードした列になる。
# 境目のすぐ上（-13〜-11）には、ウェブの文章に多く音声の学習に少ない語が1トークンのまま残り、
# 文の中で言えない。調べて「分けずに言えず、この割り方なら言えた」語だけを入れる
# （work/near_threshold: 1,342語を文型で読ませ、NG の語を分けた形で読み直した。割り方は
# そのとき言えた形。セブンイレブンは work/mecab_split/seven.py、楽天市場などは compare2.py）。
# 境目を一律に上げると、もとから言えた語（フルート・トランク）まで フ|ルート・ト|ランク と割れて
# 抑揚が崩れるので、語を指定する。元の分け方の中で、このトークンだけをその場で割る（前後の区切りは
# 変えない）。
FORCE_SPLITS = {
    "セブンイレブン": ("セブン", "イレブン"),
    "楽天市場": ("楽天", "市場"),
    "外壁塗装": ("外壁", "塗装"),
    "イベルメクチン": ("イベルメ", "クチン"),
    "ミノキシジル": ("ミノ", "キシ", "ジル"),
    "送料無料": ("送料", "無料"),
    "ヴィンテージ": ("ヴィン", "テージ"),
    "点セット": ("点", "セット"),
    "フォルダ": ("フォル", "ダ"),
    "消費者金融": ("消費者", "金融"),
    "代表取締役": ("代表", "取締役"),
    "本セット": ("本", "セット"),
    "厚生労働省": ("厚生", "労働", "省"),
    "オンラインカジノ": ("オンライン", "カジノ"),
    "当サイト": ("当", "サイト"),
    "プロペシア": ("プロ", "ペ", "シア"),
    "リノベーション": ("リノベ", "ー", "ション"),
    "メール便": ("メール", "便"),
    "不動産投資": ("不動産", "投資"),
    "ティック": ("ティ", "ック"),
    "フジテレビ": ("フジ", "テレビ"),
    "東日本大震災": ("東日本", "大震災"),
    "新品未使用": ("新品", "未使用"),
    "ネクスガード": ("ネ", "クス", "ガード"),
    "全身脱毛": ("全身", "脱毛"),
    "債務整理": ("債務", "整理"),
    "ス感染症": ("ス", "感染症"),
    "代表取締役社長": ("代表", "取締役", "社長"),
    "枚セット": ("枚", "セット"),
    "フィナステリド": ("フィ", "ナス", "テリ", "ド"),
    "クチコミ": ("クチ", "コミ"),
    "ブロガー": ("ブロ", "ガー"),
    "ルイヴィトン": ("ルイ", "ヴィ", "トン"),
    "転職エージェント": ("転職", "エージェント"),
    "数量限定": ("数量", "限定"),
    "ホワイトニング": ("ホワイト", "ニング"),
    "無料ダウンロード": ("無料", "ダウンロード"),
    "スーパーコピー": ("スーパー", "コピー"),
    "一般社団法人": ("一般", "社", "団", "法人"),
    "日本語版": ("日本語", "版"),
    "ペット用品": ("ペット", "用品"),
    "投資信託": ("投資", "信託"),
    "脱毛サロン": ("脱毛", "サロン"),
    "東京都港区": ("東京都", "港区"),
    "注文住宅": ("注文", "住宅"),
    "マイナビ": ("マイ", "ナビ"),
    "行政書士": ("行政", "書", "士"),
    "無料相談": ("無料", "相談"),
    "市区町村": ("市", "区", "町", "村"),
    "ザガーロ": ("ザ", "ガー", "ロ"),
    "リハビリテーショ": ("リハビリ", "テーショ"),
    "株主優待": ("株主", "優待"),
    "オンラインストア": ("オンライン", "ストア"),
    "プレゼンテーショ": ("プレゼン", "テーショ"),
    "個人事業主": ("個人", "事業主"),
    "社会福祉法人": ("社会", "福祉", "法人"),
    "カルティエ": ("カル", "ティエ"),
    "イオンモール": ("イオン", "モール"),
    "ピーター": ("ピー", "ター"),
    "銀行振込": ("銀行", "振込"),
    "スタッフ一同": ("スタッフ", "一同"),
    "即日融資": ("即日", "融資"),
    "プラセンタ": ("プラ", "センタ"),
    "大学大学院": ("大学", "大学院"),
    "モンクレール": ("モン", "ク", "レール"),
    "賃貸物件": ("賃貸", "物件"),
    "協同組合": ("協同", "組合"),
    # 耳で聞いて決めた割り方（work/split_tool の decisions.json）
    "年以上前": ("年", "以上", "前"),
    "土日祝日": ("土日", "祝日"),
    "ウェイト": ("ウェ", "イト"),
}
# ひらがな（と長音符・踊り字）だけのトークンは分けない。字のとおりに読むので分けても読みは
# 変わらず、「か|ならず」「わ|からず」のように語の途中で切れて間や抑揚が崩れるだけ
# （work/len_split: 除外しても判定語の OK率は 0.75 → 0.76、音声の長さは +3% → +2.6%）。
_HIRAGANA_ONLY = re.compile(r"[ぁ-ゖゝゞー]+")
_LOWERED_SCORE = -1e9


def decode_notation(text: str) -> str:
    """表示用の表記（| と [ZW]）を実際に入れる文字へ。"""
    return text.replace("|", SPLIT_MARK).replace("[ZW]", ZERO_WIDTH_SPACE)


def encode_notation(text: str) -> str:
    return text.replace(SPLIT_MARK, "|").replace(ZERO_WIDTH_SPACE, "[ZW]")


def strip_marks(text: str) -> str:
    return text.replace(SPLIT_MARK, "")


def threshold_value(setting) -> float | None:
    """全体の設定（"-13" など）を数値へ。none なら None。"""
    if setting not in TOKEN_SPLIT_THRESHOLDS:
        raise ValueError(f"token_split_threshold must be one of {', '.join(TOKEN_SPLIT_THRESHOLDS)}")
    return None if setting == "none" else float(setting)


def token_split_applies(model_tokenizer_repo: str | None) -> bool:
    """このモデルに語彙分割を当てるか。

    境目の尺度は modernbert-ja の出現度なので、同じトークナイザのモデルだけ。モデルの
    トークナイザが分からないとき（読み込み前・古い経路）は、当てる側に倒す（エンコード時に
    Unigram でなければ何もしない）。
    """
    return not model_tokenizer_repo or str(model_tokenizer_repo) == DEFAULT_TOKENIZER_REPO


def _avoids(piece: str, score: float, threshold: float) -> bool:
    """写しのトークナイザで使わないトークンか（出現度が境目以下の長いトークン）。"""
    text = piece.replace("▁", "")
    return (score <= threshold and len(text) >= MIN_SPLIT_CHARS and text not in KEEP_TOKENS
            and text not in FORCE_SPLITS and not _HIRAGANA_ONLY.fullmatch(text))


def _splits(piece: str, score: float, threshold: float) -> bool:
    """このトークンを語彙分割で分けるか（使わないトークンと、割り方を指定したトークン）。"""
    return piece.replace("▁", "") in FORCE_SPLITS or _avoids(piece, score, threshold)


# ---------------------------------------------------------------- 写しのトークナイザ
_LOW_LOCK = threading.Lock()
# id(元のトークナイザ) -> (元のトークナイザ, {境目: 写し or None})。元を持っておき id の再利用を防ぐ。
_LOW: dict[int, tuple[object, dict[float, object]]] = {}
# id(写し) -> {割り方を指定したトークンの id: ((片のトークンの id, 文字数), ...)}
_FORCED: dict[int, dict[int, tuple[tuple[int, int], ...]]] = {}


def low_score_tokenizer(tokenizer, threshold: float):
    """4文字以上で出現度が threshold 以下のトークンを使わない、tokenizer の写し。

    KEEP_TOKENS と、ひらがなだけのトークンは除く。

    tokenizer は tokenizers.Tokenizer（HF の backend_tokenizer）。Unigram でなければ None。
    特殊トークンとバイトのトークンは出現度が 0 なので対象にならない。
    """
    key = id(tokenizer)
    with _LOW_LOCK:
        entry = _LOW.get(key)
        if entry is None or entry[0] is not tokenizer:
            entry = (tokenizer, {})
            _LOW[key] = entry
        cache = entry[1]
        if threshold not in cache:
            cache[threshold] = _build_low(tokenizer, threshold)
        return cache[threshold]


def _build_low(tokenizer, threshold: float):
    from tokenizers import Tokenizer
    data = json.loads(tokenizer.to_str())
    model = data.get("model", {})
    if model.get("type") != "Unigram" or not model.get("vocab"):
        return None
    for item in model["vocab"]:
        piece, score = item[0], float(item[1])
        if _avoids(piece, score, threshold):
            item[1] = _LOWERED_SCORE
    low = Tokenizer.from_str(json.dumps(data, ensure_ascii=False))
    # 写しは本文のトークン列だけを作る（固定長パディングや BOS は呼び出し側で付ける）
    low.no_padding()
    low.no_truncation()
    forced = {}
    for word, pieces in FORCE_SPLITS.items():
        word_id = low.token_to_id(word)
        if word_id is None:
            continue
        parts = []
        for piece in pieces:
            piece_id = low.token_to_id(piece)  # 語彙にある片は、出現度が低くてもそのまま使う
            if piece_id is not None:
                parts.append((piece_id, len(piece)))
                continue
            encoding = low.encode(piece, add_special_tokens=False)
            parts.extend((i, b - a) for i, (a, b) in zip(encoding.ids, encoding.offsets) if a != b)
        if sum(n for _, n in parts) == len(word):
            forced[word_id] = tuple(parts)
    _FORCED[id(low)] = forced
    return low


def _encode_low(low, text: str) -> list[tuple[int, int, int]]:
    """写しのトークナイザでエンコードし、割り方を指定したトークンをその場で片に割る。

    返り値は (id, 開始, 終わり)。前後のトークンの区切りは変えない。
    """
    forced = _FORCED.get(id(low), {})
    out = []
    encoding = low.encode(text, add_special_tokens=False)
    for token_id, (start, end) in zip(encoding.ids, encoding.offsets):
        pieces = forced.get(token_id)
        if not pieces or sum(n for _, n in pieces) != end - start:
            out.append((token_id, start, end))
            continue
        for piece_id, length in pieces:
            out.append((piece_id, start, start + length))
            start += length
    return out


# ---------------------------------------------------------------- 合成ごとの指定
# いま合成中のセリフに当てる境目（None なら分けない）。合成はリクエストのスレッドで
# 同期的に進むので、contextvar でエンコードまで届く。
_ACTIVE: contextvars.ContextVar[float | None] = contextvars.ContextVar("token_split_threshold",
                                                                      default=None)
# いま合成中のセリフを、文節の頭でも分けるか（全体の設定 token_split_bunsetsu）。
_BUNSETSU: contextvars.ContextVar[bool] = contextvars.ContextVar("token_split_bunsetsu", default=False)


@contextlib.contextmanager
def active_threshold(threshold: float | None, bunsetsu: bool = False):
    """この中でエンコードする文に、境目 threshold の語彙分割を当てる（None なら当てない）。

    bunsetsu なら、文節の頭でも分ける（境目とは別に効く）。
    """
    token = _ACTIVE.set(threshold)
    token_b = _BUNSETSU.set(bool(bunsetsu))
    try:
        yield
    finally:
        _BUNSETSU.reset(token_b)
        _ACTIVE.reset(token)


def _with_bunsetsu(text: str) -> str:
    """文節の区切りの指定があれば、文節の頭に見えない区切りを入れる。"""
    if not _BUNSETSU.get():
        return text
    import bunsetsu
    return bunsetsu.mark(text)


def _active_low(hf_tokenizer):
    """いまの指定で使う写しのトークナイザ。分けないなら None。"""
    threshold = _ACTIVE.get()
    backend = getattr(hf_tokenizer, "backend_tokenizer", None)
    if threshold is None or backend is None:
        return None
    return low_score_tokenizer(backend, threshold)


def split_token_ids(hf_tokenizer, text: str) -> list[int]:
    """見えない区切りの位置で分けてエンコードし、つなげた ID 列を返す。

    語彙分割の指定（active_threshold）があれば、写しのトークナイザでエンコードする。
    文節の区切りの指定があれば、文節の頭でも分ける。
    """
    return _encode_parts(hf_tokenizer, _with_bunsetsu(text))


def _encode_parts(hf_tokenizer, text: str) -> list[int]:
    low = _active_low(hf_tokenizer)
    ids: list[int] = []
    for part in text.split(SPLIT_MARK):
        if not part:
            continue
        if low is not None:
            ids.extend(token_id for token_id, _, _ in _encode_low(low, part))
        else:
            ids.extend(hf_tokenizer.encode(part, add_special_tokens=False))
    return ids


def split_token_groups(hf_tokenizer, text: str) -> list[int | None]:
    """split_token_ids と同じ並びで、各トークンが語彙分割で生まれたものなら、元のトークンごとの番号
    （「ゼルダの伝説」→ ゼルダ|の|伝説 の 3 つが同じ番号）。そうでなければ None。"""
    threshold = _ACTIVE.get()
    text = _with_bunsetsu(text)
    low = _active_low(hf_tokenizer)
    if low is None:
        return [None] * len(_encode_parts(hf_tokenizer, text))
    backend = hf_tokenizer.backend_tokenizer
    scores = _scores(backend)
    groups: list[int | None] = []
    count = 0
    for part in text.split(SPLIT_MARK):
        if not part:
            continue
        original = backend.encode(part, add_special_tokens=False)
        spans = [offset for token_id, offset in zip(original.ids, original.offsets)
                 if offset[0] != offset[1]
                 and _splits(part[offset[0]:offset[1]], scores.get(token_id, 0.0), threshold)]
        base = count
        count += len(spans)
        for _, start, end in _encode_low(low, part):
            found = [base + i for i, (a, b) in enumerate(spans) if a <= start and end <= b]
            groups.append(found[0] if found and start != end else None)
    return groups


def install_split_encoding(text_tokenizer) -> None:
    """PretrainedTextTokenizer の encode / batch_encode を語彙分割と見えない区切りに対応させる。

    区切りを含まず語彙分割の指定も無い文は元の処理のまま（固定長パディングなどの挙動も変えない）。
    """
    if text_tokenizer is None or getattr(text_tokenizer, "_split_encoding", False):
        return
    import torch

    original_encode = text_tokenizer.encode
    original_batch = text_tokenizer.batch_encode

    def plain(text):
        return (SPLIT_MARK not in text and not _BUNSETSU.get()
                and _active_low(text_tokenizer.tokenizer) is None)

    def encode(text, add_bos=None):
        if plain(text):
            return original_encode(text, add_bos=add_bos)
        ids = split_token_ids(text_tokenizer.tokenizer, text)
        if text_tokenizer.add_bos if add_bos is None else bool(add_bos):
            ids.insert(0, int(text_tokenizer.bos_token_id))
        return torch.tensor(ids, dtype=torch.long)

    def batch_encode(texts, max_length=None):
        texts = list(texts)
        if all(plain(t) for t in texts):
            return original_batch(texts, max_length=max_length)
        rows = [encode(t).tolist() for t in texts]
        if max_length is None:
            max_length = max(max(len(r), 1) for r in rows)
        if max_length <= 0:
            raise ValueError(f"max_length must be > 0, got {max_length}")
        batch = torch.full((len(rows), max_length), text_tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(rows), max_length), dtype=torch.bool)
        for index, row in enumerate(rows):
            row = row[:max_length]
            batch[index, :len(row)] = torch.tensor(row, dtype=torch.long)
            mask[index, :len(row)] = True
        return batch, mask

    text_tokenizer.encode = encode
    text_tokenizer.batch_encode = batch_encode
    text_tokenizer._split_encoding = True


# ---------------------------------------------------------------- 画面の表示用
def find_tokenizer_json(repo: str) -> Path | None:
    """モデルと同じトークナイザの tokenizer.json（Hugging Face のキャッシュ）を探す。"""
    try:
        from huggingface_hub import try_to_load_from_cache
        found = try_to_load_from_cache(repo, "tokenizer.json")
        if isinstance(found, str):
            return Path(found)
    except Exception:  # noqa: BLE001 - 見つからなければ下の既定の場所を見る
        pass
    hub = Path(__file__).resolve().parents[2] / ".cache" / "huggingface" / "hub"
    snapshots = sorted((hub / f"models--{repo.replace('/', '--')}" / "snapshots").glob("*/tokenizer.json"))
    return snapshots[-1] if snapshots else None


_SCORES: dict[int, dict[int, float]] = {}


def _scores(tokenizer) -> dict[int, float]:
    """Unigram の各トークンの出現度（logp）。Unigram 以外なら空。"""
    key = id(tokenizer)
    if key not in _SCORES:
        model = json.loads(tokenizer.to_str()).get("model", {})
        vocab = model.get("vocab") if model.get("type") == "Unigram" else None
        _SCORES[key] = {i: float(v[1]) for i, v in enumerate(vocab or [])}
    return _SCORES[key]


_FALLBACK = None


def describe_tokens(texts, text_tokenizer=None, threshold: float | None = None,
                    model_tokenizer_repo: str | None = None, bunsetsu: bool = False) -> dict:
    """文ごとに、モデルに渡るトークンの分け方を返す（/irodori/tokenize）。

    text_tokenizer は読み込み中のモデルの PretrainedTextTokenizer。無ければ（モデル未読み込み）
    既定のトークナイザで代わりに分ける。results の各要素は
      original: 語彙分割なしのトークン（low は語彙分割で避けるトークン）
      tokens  : 境目 threshold で語彙分割したトークン（分けないなら original と同じ）
    bunsetsu なら tokens は文節の頭でも分ける（その位置には split の印を入れない）。
    | と [ZW] の表記はそのまま区切りとして扱う。
    """
    global _FALLBACK
    hf = getattr(text_tokenizer, "tokenizer", None)
    tokenizer = getattr(hf, "backend_tokenizer", None)
    source = "model"
    if tokenizer is None:
        if _FALLBACK is None:
            from tokenizers import Tokenizer
            path = find_tokenizer_json(DEFAULT_TOKENIZER_REPO)
            _FALLBACK = Tokenizer.from_file(str(path)) if path else False
        tokenizer, source = _FALLBACK or None, "fallback"
        model_tokenizer_repo = DEFAULT_TOKENIZER_REPO
    if tokenizer is None:
        return {"available": False, "source": source, "threshold": threshold, "applies": False,
                "bunsetsu": False, "results": []}
    try:
        from irodori_tts.text_normalization import normalize_text
    except ImportError:
        import unicodedata

        def normalize_text(text):
            return unicodedata.normalize("NFKC", text)
    scores = _scores(tokenizer)
    low = None
    applies = token_split_applies(model_tokenizer_repo)
    if threshold is not None and applies:
        low = low_score_tokenizer(tokenizer, threshold)
    if bunsetsu:
        import bunsetsu as bunsetsu_module
        bunsetsu = applies and bunsetsu_module.available()

    def pieces(text, encoder, by_bunsetsu=False):
        tokens = []
        for index, part in enumerate(text.split(SPLIT_MARK)):
            if index:
                tokens.append({"split": True})
            subparts = bunsetsu_module.mark(part).split(SPLIT_MARK) if by_bunsetsu else [part]
            for sub in subparts:
                if not sub:
                    continue
                if encoder is low:
                    spans = _encode_low(low, sub)
                else:
                    encoding = encoder.encode(sub, add_special_tokens=False)
                    spans = [(i, a, b) for i, (a, b) in zip(encoding.ids, encoding.offsets)]
                for token_id, start, end in spans:
                    if start == end:
                        continue  # 読み込んだモデルのトークナイザは固定長にパディングすることがある
                    piece = sub[start:end]
                    score = scores.get(token_id, 0.0)
                    tokens.append({"text": piece, "id": token_id, "score": round(score, 2),
                                   "low": low is not None and _splits(piece, score, threshold)})
        return tokens

    results = []
    for text in texts:
        text = normalize_text(decode_notation(str(text))).strip()
        original = pieces(text, tokenizer)
        split = low is not None or bunsetsu
        results.append({"text": text, "original": original,
                        "tokens": pieces(text, low if low is not None else tokenizer, bunsetsu) if split else original})
    return {"available": True, "source": source, "threshold": threshold, "applies": low is not None,
            "bunsetsu": bool(bunsetsu), "results": results}
