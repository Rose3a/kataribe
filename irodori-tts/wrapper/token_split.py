"""語彙分割辞書: 学習データが少ない「まとまりトークン」を分けて読ませる。

トークナイザ（modernbert-ja）は「浦和レッズ」「ゼルダの伝説」「テキストエディタ」のような
語句を丸ごと1トークンにする。こうした出現度の低いトークンは TTS の学習でほとんど見ていない
ため、別の語に化けたり詰まったりする。辞書は tools/token_rescue.py が ASR で評価して作り、
語句を次のどれかに書き換える（data/token_split_dictionary.json の text 欄の書き方）。

- ``|``    見えない区切り。テキストには U+2063 を入れ、トークナイザの直前でその位置で
           分けてエンコードする。モデルには「浦和|レッズ」のような学習済みのトークンが
           区切り用のトークンなしで渡るので、空白と違って間が入らない。
- ``[ZW]`` ゼロ幅スペース（U+200B）。語彙にある1トークンとして渡る。
- 空白やひらがななど、それ以外の文字はそのまま入れる。
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from pathlib import Path

SPLIT_MARK = "⁣"  # INVISIBLE SEPARATOR
ZERO_WIDTH_SPACE = "​"
# 語彙分割辞書を 使う / 使わない
TOKEN_SPLITS = ("on", "off")
# 辞書をどのモデルに当てるか（全体の設定）。
#   small: 辞書を作ったトークナイザ（modernbert-ja）を使うモデル＝Small 系だけ。既定
#   all  : Large など別のトークナイザのモデルにも当てる
# 辞書は modernbert-ja のトークンの出現度で作ってあり、Large（T5Gemma 2 のトークナイザ）は日本語の
# 語句を細かく割るので、同じ語句が読めない問題はもともと起きにくい。
TOKEN_SPLIT_SCOPES = ("small", "all")
DEFAULT_TOKEN_SPLIT_SCOPE = "small"
DEFAULT_TOKENIZER_REPO = "sbintuitions/modernbert-ja-310m"
DICTIONARY_PATH = Path(__file__).resolve().parent / "data" / "token_split_dictionary.json"
# 自分で足した登録（読めない語の対策用）。読み方＆アクセント辞書（user_dictionary.json）とは別。
USER_DICTIONARY_PATH = Path(__file__).resolve().parents[2] / "token_split_user.json"


def decode_notation(text: str) -> str:
    """辞書の表記（| と [ZW]）を実際に入れる文字へ。"""
    return text.replace("|", SPLIT_MARK).replace("[ZW]", ZERO_WIDTH_SPACE)


def encode_notation(text: str) -> str:
    return text.replace(SPLIT_MARK, "|").replace(ZERO_WIDTH_SPACE, "[ZW]")


def strip_marks(text: str) -> str:
    return text.replace(SPLIT_MARK, "")


class TokenSplitDictionary:
    """トークン（語句）→ 書き換え後の文字列。

    1トークンの語句は、文字列一致ではなく、トークナイザが実際にその語句を1トークンとして
    出した位置にだけ当てる。「ワール」の登録が「ワールド」（別の、よく学習されたトークン）の
    中に当たって壊さないようにするため。1トークンにならない語句（手で足した「浦和レッズ戦」
    など）は文字列一致で当てる。トークナイザを読めないときは何もしない。
    """

    def __init__(self, path: Path = DICTIONARY_PATH, tokenizer=None, user_path: Path | None = None):
        self.path = Path(path)
        # 自分の登録。同じ語なら自動生成の登録より優先する（書き換えを単語と同じにすれば止められる）
        self.user_path = Path(user_path) if user_path else None
        self.lock = threading.RLock()
        self._mtime = None
        self.entries: dict[str, str] = {}
        self._phrases = None  # 1トークンにならない語句を長い順に並べた正規表現
        self.repo = None
        # テスト用: text -> [(start, end), ...] を返す関数
        self._tokenize = tokenizer

    def _load(self):
        # ツールで作り直したファイルを、エンジンを再起動せずに拾う。
        paths = [self.path] + ([self.user_path] if self.user_path else [])
        mtime = tuple(p.stat().st_mtime if p.exists() else None for p in paths)
        if mtime == self._mtime:
            return
        entries, repo = {}, DEFAULT_TOKENIZER_REPO
        for path, stamp in zip(paths, mtime):
            if stamp is None:
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            repo = data.get("tokenizer") or repo
            for entry in data.get("entries", []):
                surface, text = entry.get("surface"), entry.get("text")
                if isinstance(surface, str) and surface and isinstance(text, str):
                    entries[surface] = decode_notation(text)
        self.entries, self.repo = entries, repo
        self._phrases = None
        self._mtime = mtime

    def tokenizer_repo(self) -> str:
        """辞書を作ったトークナイザ（辞書ファイルの tokenizer 欄）。"""
        with self.lock:
            self._load()
            return self.repo or DEFAULT_TOKENIZER_REPO

    def _tokenizer(self):
        if self._tokenize is None and self.repo:
            path = find_tokenizer_json(self.repo)
            if path is None:
                self._tokenize = False
            else:
                from tokenizers import Tokenizer
                tokenizer = Tokenizer.from_file(str(path))
                self._tokenize = lambda text: tokenizer.encode(text, add_special_tokens=False).offsets
        return self._tokenize or None

    def apply(self, text: str) -> str:
        with self.lock:
            self._load()
            entries = self.entries
            tokenize = self._tokenizer() if entries else None
            if tokenize is not None and self._phrases is None:
                phrases = [k for k in entries if tokenize(k) != [(0, len(k))]]
                self._phrases = (re.compile("|".join(map(re.escape, sorted(phrases, key=len, reverse=True))))
                                 if phrases else False)
            phrases = self._phrases
        if not entries or tokenize is None:
            return text
        parts = []
        if not phrases:
            parts = self._apply_tokens(text, tokenize, entries)
        else:
            last = 0
            for match in phrases.finditer(text):
                parts += self._apply_tokens(text[last:match.start()], tokenize, entries)
                parts.append((entries[match[0]], match[0]))
                last = match.end()
            parts += self._apply_tokens(text[last:], tokenize, entries)
        return _join_with_boundaries(parts)

    @staticmethod
    def _apply_tokens(text, tokenize, entries) -> list[tuple[str, str | None]]:
        """(文字列, 置き換えた語句 or None) の列。"""
        if not text:
            return []
        out, last = [], 0
        for start, end in tokenize(text):
            replacement = entries.get(text[start:end])
            if replacement is not None and start >= last:
                out += [(text[last:start], None), (replacement, text[start:end])]
                last = end
        return out + [(text[last:], None)]

    # ---- 自分の登録（辞書画面・tools/token_rescue.py から編集する）
    def user_entries(self) -> list[dict]:
        if self.user_path is None or not self.user_path.exists():
            return []
        return json.loads(self.user_path.read_text(encoding="utf-8")).get("entries", [])

    def auto_entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text(encoding="utf-8")).get("entries", [])

    def put_user(self, surface: str, text: str, note: str | None = None, **stats) -> dict:
        surface, text = _check_entry(surface, text)
        with self.lock:
            entries = [e for e in self.user_entries() if e.get("surface") != surface]
            previous = next((e for e in self.user_entries() if e.get("surface") == surface), {})
            entry = {"surface": surface, "text": text,
                     "note": previous.get("note", "") if note is None else str(note), **stats}
            self._save_user(entries + [entry])
            return entry

    def delete_user(self, surface: str) -> bool:
        with self.lock:
            entries = self.user_entries()
            kept = [e for e in entries if e.get("surface") != surface]
            self._save_user(kept)
            return len(kept) != len(entries)

    def _save_user(self, entries: list[dict]) -> None:
        if self.user_path is None:
            raise ValueError("自分の登録の保存先がありません")
        entries = sorted(entries, key=lambda e: e["surface"])
        lines = ",\n".join("  " + json.dumps(e, ensure_ascii=False) for e in entries)
        body = ('{\n "version": 1,\n "description": "語彙分割辞書の自分の登録。text の | は見えない区切り、'
                '[ZW] はゼロ幅スペース。同じ語なら自動生成の登録より優先",\n "entries": ['
                + ("\n" + lines + "\n " if entries else "") + "]\n}\n")
        self.user_path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=self.user_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(body)
            os.replace(name, self.user_path)
        finally:
            if os.path.exists(name):
                os.unlink(name)
        self._mtime = None


def _join_with_boundaries(parts) -> str:
    """置き換えた語句の前後に見えない区切りを入れてつなぐ。

    元のトークン化ではその語句の両端で切れていたので、置き換え後も前後の語と
    くっつかないようにする（「友達|がシェアした投稿」が「友達が|シェア…」にならない）。
    文頭・文末と、書き換えが語句と同じ（何もしない）登録には入れない。
    """
    out = ""
    pending = False  # 直前が置き換えた語句（後ろに区切りが要る）
    for part, surface in parts:
        if not part:
            continue
        if surface is not None and part != surface:
            part = part.strip(SPLIT_MARK)
            if out and not out.endswith(SPLIT_MARK):
                out += SPLIT_MARK
            out += part
            pending = True
            continue
        if pending and not part.startswith(SPLIT_MARK):
            out += SPLIT_MARK
        out += part
        pending = False
    return out


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


def _check_entry(surface, text) -> tuple[str, str]:
    import unicodedata
    if not isinstance(surface, str) or not surface.strip():
        raise ValueError("単語は必須です")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("書き換えは必須です")
    surface = unicodedata.normalize("NFKC", surface).strip()
    text = encode_notation(unicodedata.normalize("NFKC", text).strip())
    if any(unicodedata.category(c) == "Cc" for c in surface + text):
        raise ValueError("改行や制御文字は使えません")
    if len(surface) > 64 or len(text) > 128:
        raise ValueError("単語は64文字、書き換えは128文字までです")
    return surface, text


TOKEN_SPLIT_DICTIONARY = TokenSplitDictionary(user_path=USER_DICTIONARY_PATH)


def token_split_active(scope: str, model_tokenizer_repo: str | None, dictionary=None) -> bool:
    """語彙分割辞書をこのモデルに当てるか。

    scope が all なら常に当てる。small なら、モデルのトークナイザが辞書を作ったものと同じときだけ。
    モデルのトークナイザが分からないとき（読み込み前・古い経路）は、従来どおり当てる。
    """
    if scope not in TOKEN_SPLIT_SCOPES:
        raise ValueError(f"token_split_scope must be one of {', '.join(TOKEN_SPLIT_SCOPES)}")
    if scope == "all" or not model_tokenizer_repo:
        return True
    return str(model_tokenizer_repo) == (dictionary or TOKEN_SPLIT_DICTIONARY).tokenizer_repo()


# 出現度の低い側の複数文字トークン（tools/token_rescue.py の --rare-id と同じ境目）。
RARE_TOKEN_ID = 60000
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


def describe_tokens(texts, text_tokenizer=None) -> dict:
    """文ごとに、モデルに渡るトークンの分け方を返す（辞書画面の表示用）。

    text_tokenizer は読み込み中のモデルの PretrainedTextTokenizer。無ければ（モデル未読み込み）
    既定のトークナイザで代わりに分ける。| と [ZW] の表記はそのまま区切りとして扱う。
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
    if tokenizer is None:
        return {"available": False, "source": source, "results": []}
    try:
        from irodori_tts.text_normalization import normalize_text
    except ImportError:
        import unicodedata

        def normalize_text(text):
            return unicodedata.normalize("NFKC", text)
    scores = _scores(tokenizer)
    # 「出現度の低い側」の判定（ID の境目と Unigram の logp）は modernbert-ja のもの。BPE など
    # 出現度を持たないトークナイザ（Large）では、ID が大きくても珍しいトークンとは限らないので付けない。
    rare_applies = bool(scores)
    with TOKEN_SPLIT_DICTIONARY.lock:
        TOKEN_SPLIT_DICTIONARY._load()
        split_entries = set(TOKEN_SPLIT_DICTIONARY.entries)
    results = []
    for text in texts:
        text = normalize_text(decode_notation(str(text))).strip()
        tokens = []
        for index, part in enumerate(text.split(SPLIT_MARK)):
            if index:
                tokens.append({"split": True})
            if not part:
                continue
            encoding = tokenizer.encode(part, add_special_tokens=False)
            for token_id, (start, end) in zip(encoding.ids, encoding.offsets):
                piece = part[start:end]
                tokens.append({"text": piece, "id": token_id,
                               "score": round(scores.get(token_id, 0.0), 2),
                               "rare": rare_applies and token_id >= RARE_TOKEN_ID and len(piece) > 1,
                               "dictionary": piece in split_entries})
        results.append({"text": text, "tokens": tokens})
    return {"available": True, "source": source, "results": results}


def split_token_ids(hf_tokenizer, text: str) -> list[int]:
    """見えない区切りの位置で分けてエンコードし、つなげた ID 列を返す。"""
    ids: list[int] = []
    for part in text.split(SPLIT_MARK):
        if part:
            ids.extend(hf_tokenizer.encode(part, add_special_tokens=False))
    return ids


def install_split_encoding(text_tokenizer) -> None:
    """PretrainedTextTokenizer の encode / batch_encode を見えない区切りに対応させる。

    区切りを含まない文は元の処理のまま（固定長パディングなどの挙動も変えない）。
    """
    if text_tokenizer is None or getattr(text_tokenizer, "_split_encoding", False):
        return
    import torch

    original_encode = text_tokenizer.encode
    original_batch = text_tokenizer.batch_encode

    def body(text):
        return split_token_ids(text_tokenizer.tokenizer, text)

    def encode(text, add_bos=None):
        if SPLIT_MARK not in text:
            return original_encode(text, add_bos=add_bos)
        ids = body(text)
        if text_tokenizer.add_bos if add_bos is None else bool(add_bos):
            ids.insert(0, int(text_tokenizer.bos_token_id))
        return torch.tensor(ids, dtype=torch.long)

    def batch_encode(texts, max_length=None):
        texts = list(texts)
        if not any(SPLIT_MARK in t for t in texts):
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
