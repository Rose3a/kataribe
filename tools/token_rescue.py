"""学習の少ない「まとまりトークン」を見つけ、ASR で評価して語彙分割辞書を作る。

トークナイザは「浦和レッズ」「ゼルダの伝説」のような語句を丸ごと1トークンにすることがあり、
出現度の低いトークンほど TTS が読めない。ここでは語彙の1トークンを網羅的に読ませて ASR で
採点し、読めないトークンを分け方・表記を変えて読ませ直し、いちばん読めた書き換えを
irodori-tts/wrapper/data/token_split_dictionary.json に書く（エンジンが UI・API の両方で使う）。

    # 1) 語彙から候補を列挙し、書き換え案を出す（モデル不要・数秒）
    python tools\\token_rescue.py candidates
    # 2) 原文を合成して ASR で採点し、読めないトークンの CSV を作る（再開可）
    python tools\\token_rescue.py scan --limit 2000
    #    既に評価済みの CSV（トークン,token_id,OK率 …）を取り込むこともできる
    python tools\\token_rescue.py import "OK率50以下.csv"
    # 3) 読めなかったトークンを書き換え案ごとに合成・採点する（再開可）
    python tools\\token_rescue.py rescue --max-ok 0.5
    # 4) 採点結果から辞書を書き出す（自分の登録 token_split_user.json には触らない）
    python tools\\token_rescue.py build
    # 5) 載らなかった語を自分で詰める
    python tools\\token_rescue.py todo                     # 一覧は work\\token_rescue\\todo.csv
    python tools\\token_rescue.py try 自治スレ "じち|スレ" "ジチスレ" --add   # 採点して良ければ足す
    python tools\\token_rescue.py add 自治スレ "じち|スレ"   # 採点せずに足す
    python tools\\token_rescue.py remove 自治スレ

書き換え案（method）:
  split          見えない区切り（|）。そのトークンを使わない分け方でエンコードする。間は入らない
  split_fine     出現度の低いトークンも避けて、さらに細かく分ける
  zw             区切りにゼロ幅スペース（[ZW]、語彙にある1トークン）を入れる
  hiragana       カタカナをひらがなにする（まだ低頻度トークンになるなら見えない区切りも入れる）
  space_particle 助詞（の・を・に…）の後だけ空白、他は見えない区切り（「ゼルダの 伝説」）

採点は「台本の中の対象語」と ASR 書き起こしの最も近い部分との文字誤り率（カタカナは
ひらがなにそろえ、記号と長音符は無視）で、0 なら OK。あわせて ASR の時刻から対象語の
内側に入った無音（語中の間）を測り、「浦和 レッズ」のように語の途中で間が空く案は辞書に
採らない。
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

BOX = Path(__file__).resolve().parents[1]
WRAPPER = BOX / "irodori-tts" / "wrapper"
sys.path.insert(0, str(WRAPPER))

from token_split import (DICTIONARY_PATH, SPLIT_MARK, TOKEN_SPLIT_DICTIONARY,  # noqa: E402
                         USER_DICTIONARY_PATH, ZERO_WIDTH_SPACE, decode_notation,
                         encode_notation)

WORK = BOX / "work" / "token_rescue"
CANDIDATES_CSV = WORK / "candidates.csv"
SCAN_CSV = WORK / "scan.csv"
RESCUE_CSV = WORK / "rescue.csv"
# 句読点を挟まない台本にして、語の中・前後に入った間を測れるようにする
DEFAULT_CARRIERS = ("これは{X}です。", "{X}の話をしよう。")
DEFAULT_SEEDS = (1001, 2002, 3003, 4004)
PARTICLES = {"の", "を", "に", "が", "は", "と", "で", "へ", "や", "も", "から", "まで", "より"}
METHODS = ("split", "split_fine", "zw", "hiragana", "reading", "space_particle")
UNSCORED_METHODS = ("rule", "manual")
# 同点なら自然な順（間が入らない・表記を変えない）を選ぶ
METHOD_PENALTY = {"original": 0.0, "split": 0.0, "split_fine": 0.02, "zw": 0.03,
                  "hiragana": 0.05, "reading": 0.06, "space_particle": 0.1}

HIRA = re.compile(r"[ぁ-ゟ]")
KATA = re.compile(r"[ァ-ヿー]")
KANJI = re.compile(r"[一-鿿々〆ヶ]")
JAPANESE = re.compile(r"^[ぁ-ゟァ-ヿ一-鿿々〆ヶー]+$")


# ---------------------------------------------------------------- tokenizer
def default_tokenizer_json() -> Path:
    found = sorted(glob.glob(str(BOX / ".cache" / "huggingface" / "hub"
                                 / "models--sbintuitions--modernbert-ja-310m"
                                 / "snapshots" / "*" / "tokenizer.json")))
    if not found:
        raise SystemExit("tokenizer.json が見つかりません。--tokenizer で指定してください")
    return Path(found[-1])


class Vocab:
    """Unigram のスコア（出現度 logp）と、見えない区切りを考慮したエンコード。"""

    def __init__(self, path: Path):
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(path))
        model = json.loads(Path(path).read_text(encoding="utf-8"))["model"]
        self.pieces = [piece for piece, _ in model["vocab"]]
        self.score = {piece: score for piece, score in model["vocab"]}
        self.ids = {piece: index for index, piece in enumerate(self.pieces)}
        self.max_len = max(len(p) for p in self.pieces)
        self._rare = {}

    def rare(self, rare_id: int) -> set[str]:
        """出現度の低い側（rare_id 以降）の複数文字トークン。"""
        if rare_id not in self._rare:
            self._rare[rare_id] = {p for p in self.pieces[rare_id:] if len(p) > 1}
        return self._rare[rare_id]

    def encode(self, text: str) -> list[str]:
        """合成時と同じ: 見えない区切りで分けて、それぞれをエンコードした piece 列。"""
        pieces = []
        for part in text.split(SPLIT_MARK):
            if part:
                pieces.extend(self.tokenizer.encode(part, add_special_tokens=False).tokens)
        return pieces

    def best_split(self, text: str, banned) -> list[str] | None:
        """banned の piece を使わない、スコア最大の分け方（Unigram の Viterbi と同じ）。"""
        n = len(text)
        best = [float("-inf")] * (n + 1)
        back = [0] * (n + 1)
        best[0] = 0.0
        for end in range(1, n + 1):
            for start in range(max(0, end - self.max_len), end):
                piece = text[start:end]
                if piece in banned:
                    continue
                score = self.score.get(piece)
                if score is None:
                    if end - start > 1:
                        continue
                    score = -30.0  # 語彙に無い1文字（バイト分解になる）
                if best[start] + score > best[end]:
                    best[end], back[end] = best[start] + score, start
        if best[n] == float("-inf"):
            return None
        pieces, end = [], n
        while end > 0:
            pieces.append(text[back[end]:end])
            end = back[end]
        return pieces[::-1]


def kind_of(text: str) -> str:
    kinds = {name for name, pattern in (("hiragana", HIRA), ("katakana", KATA), ("kanji", KANJI))
             if pattern.search(text)}
    if kinds == {"katakana"} or kinds == {"hiragana"} or kinds == {"kanji"}:
        return kinds.pop()
    return "mixed"


def to_hiragana(text: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in text)


_KAKASI = None


def kakasi():
    """漢字の読み（pykakasi）。入っていなければ None（読みでの採点と reading 案を省く）。"""
    global _KAKASI
    if _KAKASI is None:
        try:
            import pykakasi
            _KAKASI = pykakasi.kakasi()
        except ImportError:
            _KAKASI = False
    return _KAKASI or None


def reading(text: str) -> str | None:
    converter = kakasi()
    return "".join(item["hira"] for item in converter.convert(text)) if converter else None


def kanji_to_kana(text: str) -> str | None:
    """漢字の部分だけひらがなの読みにする（カタカナ・ひらがなはそのまま）。"""
    converter = kakasi()
    if not converter:
        return None
    return "".join(item["hira"] if KANJI.search(item["orig"]) else item["orig"]
                   for item in converter.convert(text))


def rewrites(vocab: Vocab, surface: str, rare_id: int) -> dict[str, str]:
    """書き換え案（method → 実際に合成する文字列）。分けようがない案は出さない。"""
    rare = vocab.rare(rare_id)
    out: dict[str, str] = {}
    split = vocab.best_split(surface, {surface})
    if split and len(split) > 1:
        out["split"] = SPLIT_MARK.join(split)
        out["zw"] = ZERO_WIDTH_SPACE.join(split)
        if any(piece in PARTICLES for piece in split[:-1]):
            out["space_particle"] = "".join(
                piece + ((" " if piece in PARTICLES else SPLIT_MARK) if i < len(split) - 1 else "")
                for i, piece in enumerate(split))
    fine = vocab.best_split(surface, rare | {surface})
    if fine and len(fine) > 1 and fine != split:
        out["split_fine"] = SPLIT_MARK.join(fine)
    def avoid_rare(text):
        if any(p in rare for p in vocab.encode(text)):
            pieces = vocab.best_split(text, rare)
            return SPLIT_MARK.join(pieces) if pieces else text
        return text

    if KATA.search(surface):
        hira = avoid_rare(to_hiragana(surface))
        if hira != surface:
            out["hiragana"] = hira
    if KANJI.search(surface):
        kana = kanji_to_kana(surface)
        if kana and kana != surface:
            out["reading"] = avoid_rare(kana)
    return out


# ---------------------------------------------------------------- scoring
def normalize(text: str) -> str:
    from asr_timeline import normalize as asr_normalize
    return asr_normalize(text.replace(SPLIT_MARK, "").replace(ZERO_WIDTH_SPACE, ""))


def segment_match(target: str, hypothesis: str) -> tuple[float, int, int]:
    """target と hypothesis の中で最も近い部分文字列 hypothesis[a:b] を探す。

    (編集距離 / len(target), a, b) を返す。書き起こしの前後の余分（台本や言いよどみ）は数えない。
    """
    if not target:
        return 0.0, 0, 0
    width = len(hypothesis) + 1
    previous, origin = [0] * width, list(range(width))  # どこから始まってもよい
    for i, char in enumerate(target, 1):
        current, starts = [i] + [0] * len(hypothesis), [0] * width
        for j, other in enumerate(hypothesis, 1):
            current[j], starts[j] = min(
                (previous[j - 1] + (char != other), origin[j - 1]),
                (previous[j] + 1, origin[j]),
                (current[j - 1] + 1, starts[j - 1]))
        previous, origin = current, starts
    end = min(range(width), key=lambda j: previous[j])
    return previous[end] / len(target), origin[end], end


def silences(samples, rate: int = 16000, min_ms: int = 60) -> list[tuple[float, float]]:
    """発話の最初と最後の有声区間のあいだにある無音区間（秒、20ms 単位）。"""
    import numpy as np
    frame = rate // 50
    count = len(samples) // frame
    if count < 3:
        return []
    rms = np.sqrt(np.mean(samples[:count * frame].reshape(count, frame) ** 2, axis=1))
    voiced = rms > max(rms.max() * 0.06, 1e-4)
    index = np.flatnonzero(voiced)
    if index.size < 2:
        return []
    out, begin = [], None
    for k in range(index[0], index[-1] + 1):
        if not voiced[k] and begin is None:
            begin = k
        elif voiced[k] and begin is not None:
            if (k - begin) * 20 >= min_ms:
                out.append((begin * 0.02, k * 0.02))
            begin = None
    return out


def judge_audio(samples, target: str, asr: dict, surface_only: bool = False) -> dict:
    """対象語の採点と間の計測。

    error は「表記での一致」と「読みでの一致」の良い方（surface_only なら表記だけ）。
    語中の間は、ASR で対象語と対応した文字のあいだに入った無音のうち最長のもの。
    台本との境目（「これは｜浦和レッズ」）の間は数えない。
    """
    spaces = []
    chars, times = [], []
    for token, stamp in zip(asr["tokens"], asr["timestamps"]):
        for char in normalize(token):
            chars.append(char)
            times.append(stamp)
    spaces.append((normalize(target), chars, times))
    target_reading = None if surface_only else reading(normalize(target))
    if target_reading:
        chars, times = [], []
        for token, stamp in zip(asr["tokens"], asr["timestamps"]):
            for char in normalize(reading(normalize(token)) or ""):
                chars.append(char)
                times.append(stamp)
        spaces.append((normalize(target_reading), chars, times))
    best = None
    for reference, chars, times in spaces:
        error, a, b = segment_match(reference, "".join(chars))
        if best is None or error < best[0]:
            best = (error, a, b, chars, times)
    error, a, b, chars, times = best
    gaps = silences(samples)
    inner = 0
    for s, e in gaps:
        before = sum(1 for t in times if t < s)  # 無音より前に始まった文字の数
        if a < before < b:
            inner = max(inner, int(round((e - s) * 1000)))
    return {"error": error, "segment": "".join(chars[a:b]), "word_pause": inner,
            "pause": max((int(round((e - s) * 1000)) for s, e in gaps), default=0)}


class Judge:
    """エディタと同じ合成経路（VoicevoxAdapter.synthesize）＋ sherpa-onnx の ASR。"""

    def __init__(self, args):
        checkpoint = Path(args.checkpoint) if args.checkpoint else BOX / "models" / "model.safetensors"
        os.environ["IRODORI_CHECKPOINT"] = str(checkpoint)
        os.environ.setdefault("HF_HOME", str(BOX / ".cache" / "huggingface"))
        os.environ.setdefault("IRODORI_SHOW_TOKENS", "0")
        from voicevox_engine import VoicevoxAdapter
        from asr_timeline import AsrTimeline, decode_wav, ensure_asr_model
        state = ensure_asr_model(wait_seconds=600)
        if not state["ready"]:
            raise SystemExit(f"ASR モデルを用意できません: {state['error']}")
        self.decode_wav = decode_wav
        self.asr = AsrTimeline()
        started = time.perf_counter()
        plan = None
        if args.backend == "trt":
            from trt_cache import ensure_plan
            plan = ensure_plan(checkpoint, lambda *a: None, lambda m: print(m, flush=True))
        embed_dirs = [BOX / "speakers" / "tsukuyomi"] if args.speaker != "none" else []
        self.adapter = VoicevoxAdapter(args.backend, 0, embed_dirs, plan)
        name = "話者なし" if args.speaker == "none" else args.speaker
        self.speaker_id = self.adapter.name_to_id[name]
        self.carriers = args.carrier or list(DEFAULT_CARRIERS)
        self.seeds = args.seeds
        self.extra = json.loads(args.query) if args.query else {}
        self.save_wav = Path(args.save_wav) if args.save_wav else None
        print(f"model loaded in {time.perf_counter() - started:.1f}s "
              f"({checkpoint.name}, speaker={name}, backend={args.backend})", flush=True)

    def trial(self, index: int, target: str, text: str, surface_only: bool = False) -> dict:
        carrier = self.carriers[index % len(self.carriers)]
        query = {**self.extra, "irodori_text": carrier.replace("{X}", text),
                 "irodori_seed": self.seeds[index % len(self.seeds)],
                 # 書き換えは渡した文字列そのものを評価する（辞書を重ねない）
                 "irodori_token_split": "off"}
        wav = self.adapter.synthesize(query, self.speaker_id)
        if self.save_wav:
            self.save_wav.mkdir(parents=True, exist_ok=True)
            name = re.sub(r'[\\/:*?"<>|\s]', "_", encode_notation(text))
            (self.save_wav / f"{target}_{name}_{index}.wav").write_bytes(wav)
        samples = self.decode_wav(wav)
        asr = self.asr.transcribe(samples)
        result = judge_audio(samples, target, asr, surface_only)
        return {**result, "script": carrier.replace("{X}", target), "heard": asr["text"],
                "ok": result["error"] == 0}

    def evaluate(self, target: str, text: str, trials: int, screen: bool = False,
                 surface_only: bool = False) -> dict:
        """trials 回読ませて採点。screen なら1回目が OK のときそこで打ち切る。"""
        results = []
        for index in range(trials):
            results.append(self.trial(index, target, text, surface_only))
            if screen and index == 0 and results[0]["ok"]:
                break
        missed = [r for r in results if not r["ok"]]
        note = ""
        # 毎回同じ1〜2文字違いに聞き取られるのは、TTS ではなく ASR の表記ゆれのことが多い
        # （ハイブリット → ハイブリッド など）
        if (len(results) > 1 and missed and len({r["segment"] for r in missed}) == 1
                and all(r["error"] <= 0.34 for r in missed)):
            note = f"表記ゆれの疑い（{missed[0]['segment']}）"
        return {"ok_rate": round(sum(r["ok"] for r in results) / len(results), 4), "note": note,
                "error": round(statistics.mean(r["error"] for r in results), 4),
                "pause": int(statistics.median(r["pause"] for r in results)),
                # 読めた試行のうち、語中の間がいちばん長かったもの
                "word_pause": max((r["word_pause"] for r in results if r["ok"]), default=0),
                "trials": len(results), "script": results[0]["script"],
                "heard": [r["heard"] for r in results]}


# ---------------------------------------------------------------- csv helpers
def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class Appender:
    """1行ごとに書き込んで flush する（止めても続きから再開できる）。"""

    def __init__(self, path: Path, fields: list[str]):
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists() or path.stat().st_size == 0
        self.stream = open(path, "a", encoding="utf-8-sig" if new else "utf-8", newline="")
        self.writer = csv.DictWriter(self.stream, fieldnames=fields, extrasaction="ignore")
        if new:
            self.writer.writeheader()

    def write(self, row: dict):
        self.writer.writerow(row)
        self.stream.flush()

    def close(self):
        self.stream.close()


HEARD = [f"書き起こし{i}" for i in range(1, 5)]
SCAN_FIELDS = ["トークン", "token_id", "種類", "文字数", "出現度logp", "OK率", "区間誤り率",
               "最長の間ms", "語中の間ms", "試行数", "メモ", "台本", *HEARD]
CANDIDATE_FIELDS = ["トークン", "token_id", "種類", "文字数", "出現度logp", *METHODS]
RESCUE_FIELDS = ["トークン", "token_id", "種類", "method", "書き換え", "トークン列", "OK率",
                 "区間誤り率", "最長の間ms", "語中の間ms", "試行数", "メモ", *HEARD]


def heard_columns(result: dict) -> dict:
    return {name: text for name, text in zip(HEARD, result["heard"])}


# ---------------------------------------------------------------- commands
def cmd_candidates(args) -> int:
    vocab = Vocab(args.tokenizer)
    rows = []
    for token_id in range(len(vocab.pieces) - 1, -1, -1):  # 出現度の低い順
        piece = vocab.pieces[token_id]
        if len(piece) < args.min_chars or not JAPANESE.match(piece) or token_id < args.min_id:
            continue
        kind = kind_of(piece)
        if args.kinds and kind not in args.kinds:
            continue
        options = rewrites(vocab, piece, args.rare_id)
        rows.append({"トークン": piece, "token_id": token_id, "種類": kind, "文字数": len(piece),
                     "出現度logp": round(vocab.score[piece], 3),
                     **{m: encode_notation(t) for m, t in options.items()}})
    WORK.mkdir(parents=True, exist_ok=True)
    with open(CANDIDATES_CSV, "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CANDIDATE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    by_kind = {}
    for row in rows:
        by_kind[row["種類"]] = by_kind.get(row["種類"], 0) + 1
    print(f"{len(rows)} tokens -> {CANDIDATES_CSV}  {by_kind}")
    return 0


def cmd_scan(args) -> int:
    candidates = read_csv(CANDIDATES_CSV)
    if not candidates:
        raise SystemExit("先に candidates を実行してください")
    done = {row["token_id"] for row in read_csv(SCAN_CSV)}
    todo = [row for row in candidates if row["token_id"] not in done][:args.limit]
    print(f"scan: {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        return 0
    judge = Judge(args)
    out = Appender(SCAN_CSV, SCAN_FIELDS)
    started = time.perf_counter()
    try:
        for count, row in enumerate(todo, 1):
            result = judge.evaluate(row["トークン"], row["トークン"], args.trials,
                                    screen=not args.no_screen)
            out.write({**row, "OK率": result["ok_rate"], "区間誤り率": result["error"],
                       "最長の間ms": result["pause"], "語中の間ms": result["word_pause"],
                       "試行数": result["trials"], "メモ": result["note"],
                       "台本": result["script"], **heard_columns(result)})
            if count % 20 == 0 or count == len(todo):
                rate = (time.perf_counter() - started) / count
                print(f"  {count}/{len(todo)}  {rate:.2f}s/token  "
                      f"last={row['トークン']} ok={result['ok_rate']}", flush=True)
    finally:
        out.close()
    failing = sum(1 for row in read_csv(SCAN_CSV) if float(row["OK率"]) <= args.report_ok)
    print(f"OK率 {args.report_ok} 以下: {failing} tokens（{SCAN_CSV}）")
    return 0


def cmd_import(args) -> int:
    """既存の評価 CSV（トークン・token_id・OK率 の列があればよい）を scan.csv に足す。"""
    vocab = Vocab(args.tokenizer)
    done = {row["token_id"] for row in read_csv(SCAN_CSV)}
    out = Appender(SCAN_CSV, SCAN_FIELDS)
    added = skipped = 0
    try:
        for row in read_csv(Path(args.csv)):
            piece = row.get("トークン", "")
            token_id = str(vocab.ids.get(piece, row.get("token_id", "")))
            if not piece or token_id in done or piece not in vocab.ids:
                skipped += 1
                continue
            heard = [row.get(name, "") for name in HEARD]
            out.write({"トークン": piece, "token_id": token_id, "種類": kind_of(piece),
                       "文字数": len(piece), "出現度logp": round(vocab.score[piece], 3),
                       "OK率": row.get("OK率", ""), "区間誤り率": row.get("区間誤り率", ""),
                       "試行数": sum(1 for h in heard if h) or "", "台本": row.get("台本", ""),
                       **dict(zip(HEARD, heard))})
            done.add(token_id)
            added += 1
    finally:
        out.close()
    print(f"imported {added} rows (skipped {skipped}) -> {SCAN_CSV}")
    return 0


def cmd_rescue(args) -> int:
    vocab = Vocab(args.tokenizer)
    targets = [row for row in read_csv(SCAN_CSV) if row["OK率"] != ""
               and float(row["OK率"]) <= args.max_ok]
    targets.sort(key=lambda row: (float(row["OK率"]), -int(row["token_id"])))
    if args.only:
        targets = [row for row in targets if row["トークン"] in set(args.only)] or [
            {"トークン": t, "token_id": vocab.ids.get(t, ""), "種類": kind_of(t)} for t in args.only]
    done = {(row["token_id"], row["method"]) for row in read_csv(RESCUE_CSV)}
    pending = [row for row in targets
               if not all((str(row["token_id"]), m) in done for m in ("original", *args.methods))]
    pending = pending[:args.limit]
    print(f"rescue: {len(targets)} targets, {len(pending)} to go", flush=True)
    if not pending:
        return 0
    judge = Judge(args)
    out = Appender(RESCUE_CSV, RESCUE_FIELDS)
    started = time.perf_counter()
    try:
        for count, row in enumerate(pending, 1):
            surface, token_id = row["トークン"], str(row["token_id"])
            options = {"original": surface, **rewrites(vocab, surface, args.rare_id)}
            summary = []
            for method in ("original", *args.methods):
                text = options.get(method)
                if text is None or (token_id, method) in done:
                    continue
                # 読みに書き換えた案は、読み変換の誤りと自己一致しないよう表記だけで採点する
                result = judge.evaluate(surface, text, args.trials,
                                        surface_only=method == "reading")
                out.write({"トークン": surface, "token_id": token_id, "種類": kind_of(surface),
                           "method": method, "書き換え": encode_notation(text),
                           "トークン列": "|".join(vocab.encode(text)),
                           "OK率": result["ok_rate"], "区間誤り率": result["error"],
                           "最長の間ms": result["pause"], "語中の間ms": result["word_pause"],
                       "試行数": result["trials"], "メモ": result["note"],
                           **heard_columns(result)})
                summary.append(f"{method}={result['ok_rate']:.2f}")
                # 間の入らない分け方で全部読めたら、それ以上は試さない
                if method in ("split", "split_fine") and result["ok_rate"] == 1.0:
                    break
            rate = (time.perf_counter() - started) / count
            print(f"  {count}/{len(pending)} {surface}: {' '.join(summary)}  ({rate:.1f}s/token)",
                  flush=True)
    finally:
        out.close()
    return 0


def choose(rows: list[dict], args) -> dict | None:
    """1トークン分の採点結果から辞書に載せる書き換えを選ぶ（載せないなら None）。"""
    original = next((r for r in rows if r["method"] == "original"), None)
    if original is None:
        return None
    base_ok = float(original["OK率"])
    best, best_score = None, float("-inf")
    for row in rows:
        if row["method"] == "original" or row["method"] not in METHOD_PENALTY:
            continue
        ok = float(row["OK率"])
        # 語の途中に間が入る案は、読めても採らない（「浦和 レッズ」のような不自然な間）
        if int(row.get("語中の間ms") or 0) > args.max_word_pause:
            continue
        score = ok - METHOD_PENALTY[row["method"]] - 0.1 * float(row["区間誤り率"])
        if score > best_score:
            best, best_score = row, score
    if best is None:
        return None
    ok = float(best["OK率"])
    if ok < args.min_ok or ok - base_ok < args.min_gain:
        return None
    return {"surface": best["トークン"], "text": best["書き換え"], "method": best["method"],
            "token_id": int(best["token_id"]), "ok_before": base_ok, "ok_after": ok,
            "tokens": best["トークン列"]}


def load_dictionary() -> dict:
    if not DICTIONARY_PATH.exists():
        return {}
    return json.loads(DICTIONARY_PATH.read_text(encoding="utf-8"))


def save_dictionary(entries: list[dict], tokenizer: Path, criteria: dict | None = None) -> None:
    vocab = Vocab(tokenizer)
    entries = sorted(entries, key=lambda e: e["surface"])
    data = {
        "version": 1,
        "description": "語彙分割辞書。tools/token_rescue.py build が生成する。"
                       "text の | は見えない区切り、[ZW] はゼロ幅スペース。"
                       "method が rule（機械判定の分割）・manual（手で決めた分割）の行は ASR で採点して"
                       "いないもので、build でも消えない。"
                       "自分の登録は token_split_user.json（同じ語ならそちらが優先）",
        "tokenizer": Path(tokenizer).parent.parent.parent.name.replace("models--", "").replace("--", "/"),
        "vocab_size": len(vocab.pieces),
        "criteria": criteria if criteria is not None else load_dictionary().get("criteria", {}),
        "entries": [],
    }
    DICTIONARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    # 1語1行（2000語でも差分を追いやすく）
    head = json.dumps(data, ensure_ascii=False, indent=1)
    lines = ",\n".join("  " + json.dumps(entry, ensure_ascii=False) for entry in entries)
    if entries:
        head = head.replace('"entries": []', '"entries": [\n' + lines + "\n ]")
    DICTIONARY_PATH.write_text(head + "\n", encoding="utf-8")


def cmd_build(args) -> int:
    groups: dict[str, list[dict]] = {}
    for row in read_csv(RESCUE_CSV):
        groups.setdefault(row["token_id"], []).append(row)
    entries = [entry for rows in groups.values() if (entry := choose(rows, args))]
    # ASR で採点していない登録（rule: 機械判定の分割、manual: 手で決めた分割）は作り直さずに残す
    scored = {entry["surface"] for entry in entries}
    entries += [entry for entry in load_dictionary().get("entries", [])
                if entry.get("method") in UNSCORED_METHODS and entry["surface"] not in scored]
    save_dictionary(entries, args.tokenizer,
                    {"min_ok": args.min_ok, "min_gain": args.min_gain,
                     "max_word_pause_ms": args.max_word_pause})
    by_method = {}
    for entry in entries:
        by_method[entry["method"]] = by_method.get(entry["method"], 0) + 1
    print(f"{len(scored)} / {len(groups)} tokens（採点していない登録 {len(entries) - len(scored)} を含めて {len(entries)}）"
          f" -> {DICTIONARY_PATH}  {by_method}")
    return 0


def cmd_todo(args) -> int:
    """辞書に載らなかった語と、いちばん惜しかった書き換えを todo.csv に書く。"""
    listed = {e["surface"] for e in load_dictionary().get("entries", [])}
    groups: dict[str, list[dict]] = {}
    for row in read_csv(RESCUE_CSV):
        groups.setdefault(row["token_id"], []).append(row)
    rows = []
    for group in groups.values():
        original = next((r for r in group if r["method"] == "original"), None)
        if original is None or original["トークン"] in listed:
            continue
        tried = [r for r in group if r["method"] != "original"]
        best = max(tried, key=lambda r: (float(r["OK率"]), -METHOD_PENALTY.get(r["method"], 1)),
                   default=None)
        if float(original["OK率"]) >= args.min_ok:
            reason = "原文で読める"
        elif best is None:
            reason = "書き換え案なし"
        elif float(best["OK率"]) < args.min_ok:
            reason = "読めない"
        elif int(best.get("語中の間ms") or 0) > args.max_word_pause:
            reason = "語中に間"
        else:
            reason = "原文と差が小さい"
        rows.append({"トークン": original["トークン"], "token_id": original["token_id"],
                     "種類": original["種類"], "原文OK率": original["OK率"],
                     "原文メモ": original["メモ"], "理由": reason,
                     "最良の案": best["method"] if best else "",
                     "書き換え": best["書き換え"] if best else "",
                     "OK率": best["OK率"] if best else "",
                     "語中の間ms": best.get("語中の間ms", "") if best else "",
                     "原文の書き起こし": original["書き起こし1"],
                     **({name: best[name] for name in HEARD} if best else {})})
    rows.sort(key=lambda r: (r["理由"], float(r["OK率"] or 0), r["トークン"]))
    path = WORK / "todo.csv"
    fields = ["トークン", "token_id", "種類", "原文OK率", "原文メモ", "理由", "最良の案", "書き換え",
              "OK率", "語中の間ms", "原文の書き起こし", *HEARD]
    with open(path, "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    counts = {}
    for row in rows:
        counts[row["理由"]] = counts.get(row["理由"], 0) + 1
    print(f"{len(rows)} tokens -> {path}  {counts}")
    return 0


def add_manual(surface: str, text: str, tokenizer: Path, **stats) -> None:
    """自分の登録（token_split_user.json）に足す。辞書画面と同じファイル。"""
    vocab = Vocab(tokenizer)
    entry = TOKEN_SPLIT_DICTIONARY.put_user(surface, text, **stats)
    print(f"added: {surface} -> {entry['text']}  ({'|'.join(vocab.encode(decode_notation(text)))})"
          f"  -> {USER_DICTIONARY_PATH}")


def cmd_try(args) -> int:
    """自分で考えた書き換えを、原文と並べて ASR で採点する（--add でいちばん良い案を辞書へ）。"""
    vocab = Vocab(args.tokenizer)
    surface = args.surface
    texts = [("original", surface)] + [(f"try{i}", decode_notation(t))
                                       for i, t in enumerate(args.rewrites, 1)]
    judge = Judge(args)
    out = Appender(WORK / "try.csv", RESCUE_FIELDS)
    results = []
    try:
        for method, text in texts:
            # 漢字を含む語の表記を変えた案は、読み変換の誤りと自己一致しないよう表記だけで採点する
            surface_only = bool(KANJI.search(surface)) and normalize(text) != normalize(surface)
            result = judge.evaluate(surface, text, args.trials, surface_only=surface_only)
            results.append((method, text, result))
            out.write({"トークン": surface, "token_id": vocab.ids.get(surface, ""),
                       "種類": kind_of(surface), "method": method, "書き換え": encode_notation(text),
                       "トークン列": "|".join(vocab.encode(text)), "OK率": result["ok_rate"],
                       "区間誤り率": result["error"], "最長の間ms": result["pause"],
                       "語中の間ms": result["word_pause"], "試行数": result["trials"],
                       "メモ": result["note"], **heard_columns(result)})
            print(f"{encode_notation(text):24s} OK {result['ok_rate']:.2f}  語中の間 "
                  f"{result['word_pause']:4d}ms  {' / '.join(result['heard'])}", flush=True)
    finally:
        out.close()
    if args.add:
        base = results[0][2]["ok_rate"]
        usable = [(m, t, r) for m, t, r in results[1:] if r["word_pause"] <= args.max_word_pause]
        if not usable:
            print("語中の間が長い案しか無いので、辞書には足していません")
            return 1
        method, text, result = max(usable, key=lambda x: (x[2]["ok_rate"], -x[2]["error"]))
        if result["ok_rate"] <= base:
            print("原文より良い案が無いので、辞書には足していません")
            return 1
        add_manual(surface, text, args.tokenizer, ok_before=base, ok_after=result["ok_rate"])
    return 0


def cmd_add(args) -> int:
    """採点せずに辞書へ足す（または書き換える）。"""
    add_manual(args.surface, args.text, args.tokenizer)
    return 0


def cmd_remove(args) -> int:
    removed = sum(TOKEN_SPLIT_DICTIONARY.delete_user(surface) for surface in args.surfaces)
    print(f"removed {removed} entries from {USER_DICTIONARY_PATH}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", type=Path, default=None, help="tokenizer.json（既定は modernbert-ja-310m）")
    ap.add_argument("--rare-id", type=int, default=60000,
                    help="この ID 以降（出現度が低い側）の複数文字トークンを split_fine/hiragana で避ける")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("candidates", help="語彙から日本語トークンを列挙し、書き換え案を出す")
    p.add_argument("--min-chars", type=int, default=2)
    p.add_argument("--min-id", type=int, default=0, help="これより小さい ID（頻出）は除く")
    p.add_argument("--kinds", nargs="*", choices=("hiragana", "katakana", "kanji", "mixed"))

    def model_options(p, trials):
        p.add_argument("--checkpoint", help="既定は models/model.safetensors（エディタと同じ）")
        p.add_argument("--backend", default="cuda", choices=("cuda", "trt", "cpu", "radeon"))
        p.add_argument("--speaker", default="tsukuyomi", help="話者名。none で話者なし")
        p.add_argument("--carrier", action="append", help="台本。{X} に語が入る（複数可、試行ごとに交互）")
        p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
        p.add_argument("--trials", type=int, default=trials)
        p.add_argument("--query", help="音声クエリに足す irodori_* の JSON（例: '{\"irodori_steps\": 8}'）")
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--save-wav", help="合成した音声をこのフォルダに残す（聞き比べ用）")

    p = sub.add_parser("scan", help="原文を合成して ASR で採点（読めないトークンの CSV）")
    model_options(p, 4)
    p.add_argument("--no-screen", action="store_true", help="1回目で読めても全試行する")
    p.add_argument("--report-ok", type=float, default=0.5)

    p = sub.add_parser("import", help="既存の評価 CSV を scan.csv に取り込む")
    p.add_argument("csv")

    p = sub.add_parser("rescue", help="読めないトークンを書き換え案ごとに採点")
    model_options(p, 4)
    p.add_argument("--max-ok", type=float, default=0.5, help="原文の OK率 がこれ以下のトークンを対象にする")
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    p.add_argument("--only", nargs="+", help="このトークンだけ試す（scan に無くてもよい）")

    p = sub.add_parser("build", help="採点結果から辞書を書き出す")
    p.add_argument("--min-ok", type=float, default=0.75, help="書き換え後の OK率 の下限")
    p.add_argument("--min-gain", type=float, default=0.25, help="原文からの OK率 の改善幅の下限")
    p.add_argument("--max-word-pause", type=int, default=180,
                   help="語の途中に入ってよい無音の上限（ms）。促音の溜めは 100ms 前後")

    p = sub.add_parser("todo", help="辞書に載らなかった語の一覧（todo.csv）を書く")
    p.add_argument("--min-ok", type=float, default=0.75)
    p.add_argument("--max-word-pause", type=int, default=180)

    p = sub.add_parser("try", help="自分で考えた書き換えを ASR で採点する")
    model_options(p, 4)
    p.add_argument("surface", help="対象の語（例: 自治スレ）")
    p.add_argument("rewrites", nargs="+", help="書き換え案。| は見えない区切り、[ZW] はゼロ幅スペース")
    p.add_argument("--add", action="store_true", help="原文より良い案があれば、いちばん良いものを辞書に足す")
    p.add_argument("--max-word-pause", type=int, default=180)

    p = sub.add_parser("add", help="採点せずに自分の登録（token_split_user.json）へ足す")
    p.add_argument("surface")
    p.add_argument("text", help="書き換え。| は見えない区切り、[ZW] はゼロ幅スペース")

    p = sub.add_parser("remove", help="自分の登録から語を消す")
    p.add_argument("surfaces", nargs="+")

    args = ap.parse_args()
    args.tokenizer = args.tokenizer or default_tokenizer_json()
    return {"candidates": cmd_candidates, "scan": cmd_scan, "import": cmd_import,
            "rescue": cmd_rescue, "build": cmd_build, "todo": cmd_todo, "try": cmd_try,
            "add": cmd_add, "remove": cmd_remove}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
