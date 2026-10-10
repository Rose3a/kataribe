"""語彙分割: 学習の少ない「まとまりトークン」を避けて、細かいトークンで読ませる。

トークナイザ（modernbert-ja）は「浦和レッズ」「ゼルダの伝説」のような語句を丸ごと1トークンに
する。こうした出現度の低いトークンは TTS の学習でほとんど見ていないため、別の語に化けたり
詰まったりする。そこで、4文字以上で出現度（Unigram の logp）が境目以下のトークンを使わない
ことにした（スコアを -1e9 にした写しのトークナイザでエンコードする）。トークナイザが
代わりに「浦和|レッ|ズ」のような学習済みのトークンへ分けるので、文字列は書き換えない。

境目は全体の設定（token_split_threshold: -10 / -11 / -12 / -13、既定 -13）。0 に近いほど
よく出る語なので、境目を上げる（-10 に近づける）ほど分ける語が増える。-13 なら4文字以上の
日本語の語句の約2割（よく出る側）はそのまま残る。スコアの尺度は modernbert-ja のものなので、
このトークナイザを使うモデル（v4 Small 系）にだけ当てる。

検証（work/len_split/）: ふつうの文ではトークン +8%・音声の長さ +3%、文の読み誤りは増えない。
読めない語を含む文の OK率は 0.41 → 0.75。生成時間はほとんど変わらない（長さ予測がトークン数を
特徴量に使うので、分けすぎると発話が間延びする。-10 に近いほど伸びる）。

文中の見えない区切り（U+2063）では、その位置で分けてエンコードする（区切り用のトークンは
入れない）。
"""
from __future__ import annotations

import contextlib
import contextvars
import json
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


def _piece_chars(piece: str) -> int:
    return len(piece.replace("▁", ""))


# ---------------------------------------------------------------- 写しのトークナイザ
_LOW_LOCK = threading.Lock()
# id(元のトークナイザ) -> (元のトークナイザ, {境目: 写し or None})。元を持っておき id の再利用を防ぐ。
_LOW: dict[int, tuple[object, dict[float, object]]] = {}


def low_score_tokenizer(tokenizer, threshold: float):
    """4文字以上で出現度が threshold 以下のトークンを使わない、tokenizer の写し。

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
        if score <= threshold and _piece_chars(piece) >= MIN_SPLIT_CHARS:
            item[1] = _LOWERED_SCORE
    low = Tokenizer.from_str(json.dumps(data, ensure_ascii=False))
    # 写しは本文のトークン列だけを作る（固定長パディングや BOS は呼び出し側で付ける）
    low.no_padding()
    low.no_truncation()
    return low


# ---------------------------------------------------------------- 合成ごとの指定
# いま合成中のセリフに当てる境目（None なら分けない）。合成はリクエストのスレッドで
# 同期的に進むので、contextvar でエンコードまで届く。
_ACTIVE: contextvars.ContextVar[float | None] = contextvars.ContextVar("token_split_threshold",
                                                                      default=None)


@contextlib.contextmanager
def active_threshold(threshold: float | None):
    """この中でエンコードする文に、境目 threshold の語彙分割を当てる（None なら当てない）。"""
    token = _ACTIVE.set(threshold)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


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
    """
    low = _active_low(hf_tokenizer)
    ids: list[int] = []
    for part in text.split(SPLIT_MARK):
        if not part:
            continue
        if low is not None:
            ids.extend(low.encode(part, add_special_tokens=False).ids)
        else:
            ids.extend(hf_tokenizer.encode(part, add_special_tokens=False))
    return ids


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
        return SPLIT_MARK not in text and _active_low(text_tokenizer.tokenizer) is None

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
                    model_tokenizer_repo: str | None = None) -> dict:
    """文ごとに、モデルに渡るトークンの分け方を返す（/irodori/tokenize）。

    text_tokenizer は読み込み中のモデルの PretrainedTextTokenizer。無ければ（モデル未読み込み）
    既定のトークナイザで代わりに分ける。results の各要素は
      original: 語彙分割なしのトークン（low は語彙分割で避けるトークン）
      tokens  : 境目 threshold で語彙分割したトークン（分けないなら original と同じ）
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
                "results": []}
    try:
        from irodori_tts.text_normalization import normalize_text
    except ImportError:
        import unicodedata

        def normalize_text(text):
            return unicodedata.normalize("NFKC", text)
    scores = _scores(tokenizer)
    low = None
    if threshold is not None and token_split_applies(model_tokenizer_repo):
        low = low_score_tokenizer(tokenizer, threshold)

    def pieces(text, encoder):
        tokens = []
        for index, part in enumerate(text.split(SPLIT_MARK)):
            if index:
                tokens.append({"split": True})
            if not part:
                continue
            encoding = encoder.encode(part, add_special_tokens=False)
            for token_id, (start, end) in zip(encoding.ids, encoding.offsets):
                piece = part[start:end]
                score = scores.get(token_id, 0.0)
                tokens.append({"text": piece, "id": token_id, "score": round(score, 2),
                               "low": (low is not None and score <= threshold
                                       and _piece_chars(piece) >= MIN_SPLIT_CHARS)})
        return tokens

    results = []
    for text in texts:
        text = normalize_text(decode_notation(str(text))).strip()
        original = pieces(text, tokenizer)
        results.append({"text": text, "original": original,
                        "tokens": pieces(text, low) if low is not None else original})
    return {"available": True, "source": source, "threshold": threshold, "applies": low is not None,
            "results": results}
