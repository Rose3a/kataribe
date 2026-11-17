"""話者学習: 音声（とセリフのテキスト）から、16 トークンの話者埋め込み（.speaker.safetensors）を作る。

調べたことと数字は work/speaker_ft/NOTES.md（git 管理外）。要点:
- ベースモデル（v4.1 Small と同じ構造なら何でも）は全部凍結し、16 トークンだけを学習する。成果物は話者ファイル 1 個で、
  PyTorch でも TensorRT でもそのまま使える（plan の作り直し不要）。
- 初期値は話者エンコーダ（ゼロショット）の出力から作る（1 個目 = 平均トークン、残り 15 = 区間平均）。
  同じ声の合い具合に届く step 数がおよそ半分になる。
- 時間予測器（= 話速）が見る話者の情報は 1 個目のトークンだけ。speed=learn（既定）では時間予測の誤差も
  話者トークンに流して、キャラの話速を覚えさせる。speed=keep は 1 個目を初期値のまま固定（話速はゼロショット相当）。
  何もしない（元の話者学習）と、1 個目が RF の誤差だけで動いて話速が崩れる（ゆっくりのキャラが速くなる）。
- VRAM: 重みは bf16、テキストエンコーダは前計算して GPU から下ろし、勾配チェックポイントを使う。バッチ 16 でも約 2GB。
  マスク付き attention はアプリの cuDNN 優先だと逆伝播が 7 倍遅いので、学習では efficient attention にする。
- CPU でも動く（fp32。bf16 命令の無い CPU で bf16 は極端に遅い）。

流れ: 音声を集める → テキストが無いものを Haas-ja で書き起こす（haas_worker.py、別の venv）→ 前後の無音を詰めて
DACVAE で符号化 → 学習 → 話者フォルダに保存。進捗は標準出力に JSON 1 行ずつ（エンジンの speaker_train_job.py が読む）。

    python speaker_train.py --name <話者の名前> --data <音声のフォルダ> [--base <モデル>] [--speed learn|keep] [--steps 500]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOX_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3"}
FPS = 25.0
MAX_TEXT_LEN = 256
MIN_SECONDS = 1.0
MAX_FRAMES = 750  # 30 秒（ベースの学習と同じ上限）
TRIM_PAD_SECONDS = 0.05
DEFAULT_STEPS = 500
DEFAULT_LR = 0.01
DEFAULT_DUR_WEIGHT = 1.0
LR_SCHEDULES = ("constant", "cosine")
# 検証: 学習に使わない音声を少し取り分け、決まった t と雑音で loss を測る（学習の loss は t が毎回違うので揺れが大きい）。
# 本数が少ないときは取り分けない（学習の音声を減らさない）
DEFAULT_VAL_FRAC = 0.05
VAL_MIN_CLIPS = 4
VAL_MAX_CLIPS = 32  # 32 本 × 4 点で 1 回およそ 1 秒（GPU）。500 ステップ・20 回で学習時間の 1 割ほど
VAL_MIN_TOTAL = 24
VAL_T_POINTS = 4  # 学習の t の分布（logit-normal）を 4 等分した各区間の真ん中
KEEP_CHOICES = ("last", "best")
INIT_REF_SECONDS = 60.0
TOKENS = 16
SPEAKER_SUBDIR = "学習した話者"
TRAIN_META_SUFFIX = ".train.json"
CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"
# 付属テキストと書き起こしの読みがこれ以上違えば、付属テキストは音声と合っていないとみなして書き起こしを使う。
# 短い相槌（「はあ」「えぇ…」）は書き起こしの方が当てにならないので、付属テキストがこの文字数未満なら置き換えない
MISMATCH_THRESHOLD = 0.35
MISMATCH_MIN_CHARS = 6
GB = 2 ** 30

TAG_RE = re.compile(r"<[^>]{1,80}>")
BRANCH_RE = re.compile(r"\{[A-Za-z]#")
WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def emit(event: str, **payload) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def progress(stage: str, percent: float, message: str = "", **extra) -> None:
    emit("progress", stage=stage, percent=round(float(percent), 1), message=message, **extra)


class TrainError(RuntimeError):
    """利用者に見せるエラー（日本語の説明）。"""


# ---------------------------------------------------------------- 名前・パス

def validate_name(name: str) -> str:
    name = (name or "").strip()
    if (not 1 <= len(name) <= 32 or re.search(r'[\\/:*?"<>|.\x00-\x1f]', name)
            or name.upper().split(" ")[0] in WINDOWS_RESERVED):
        raise TrainError('名前は1〜32文字で、\\ / : * ? " < > | . は使えません')
    return name


# 学習の設定で選べる範囲（エディタの入力欄とエンジンの確かめにも使う）。(下限, 上限, 名前)
LIMITS = {
    "steps": (50, 5000, "ステップ数"),
    "lr": (1e-5, 1.0, "学習率"),
    "warmup": (0, 2000, "ウォームアップ"),
    "batch": (0, 64, "バッチ"),
    "dur_weight": (0.0, 10.0, "時間予測の重み"),
    "val_frac": (0.0, 0.3, "検証に回す割合"),
    "val_every": (0, 5000, "検証の間隔"),
    "seed": (0, 2 ** 31 - 1, "シード"),
}


def check_settings(args) -> None:
    """学習の設定が範囲内か。外れていれば TrainError（日本語の説明）。"""
    for key, (lo, hi, label) in LIMITS.items():
        value = getattr(args, key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise TrainError(f"{label}が数ではありません: {value!r}")
        if isinstance(lo, int) and not isinstance(value, int):
            raise TrainError(f"{label}は整数です: {value!r}")
        if not lo <= value <= hi:
            raise TrainError(f"{label}は {lo:g}〜{hi:g} です（{value:g}）")
    if args.warmup >= args.steps:
        raise TrainError(f"ウォームアップ（{args.warmup}）はステップ数（{args.steps}）より小さくしてください")
    if args.lr_schedule not in LR_SCHEDULES:
        raise TrainError(f"学習率の変え方は {' / '.join(LR_SCHEDULES)} です")
    if args.keep not in KEEP_CHOICES:
        raise TrainError(f"保存する話者は {' / '.join(KEEP_CHOICES)} です")


def speaker_root() -> Path:
    return Path(os.environ.get("IRODORI_EMBED_DIR", str(BOX_ROOT / "speakers")))


def output_paths(name: str) -> tuple[Path, Path]:
    folder = speaker_root() / SPEAKER_SUBDIR / name
    return folder / f"{name}.speaker.safetensors", folder / f"{name}{TRAIN_META_SUFFIX}"


def haas_python() -> Path | None:
    """Haas-ja 用の venv の python。IRODORI_HAAS_PYTHON か .local/venv-haas。"""
    env = os.environ.get("IRODORI_HAAS_PYTHON")
    candidates = [Path(env)] if env else []
    candidates.append(BOX_ROOT / ".local" / "venv-haas" / "Scripts" / "python.exe")
    candidates.append(BOX_ROOT / ".local" / "venv-haas" / "bin" / "python")
    return next((c for c in candidates if c.is_file()), None)


# ---------------------------------------------------------------- テキスト

def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp932"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def clean_text(text: str) -> tuple[str, bool]:
    """(整えた文, 信用できるか)。色などのタグを外し、改行はつなぐ。{M#…}{F#…} のような分岐の書式は信用しない。"""
    unreliable = bool(BRANCH_RE.search(text))
    text = TAG_RE.sub("", text)
    text = "".join(line.strip() for line in text.splitlines() if line.strip())
    text = re.sub(r"\s+", " ", text).strip()
    return text, not unreliable


def has_words(text: str) -> bool:
    return len(re.sub(r"[\W_…ー〜~]", "", text)) >= 2


_TAGGER = None


def reading_key(text: str) -> str:
    """比べるための読み。fugashi（unidic-lite）があれば読み（カナ）、無ければ表記をひらがなにそろえたもの。"""
    global _TAGGER
    text = unicodedata.normalize("NFKC", text)
    if _TAGGER is None:
        try:
            import fugashi
            _TAGGER = fugashi.Tagger()
        except Exception:  # noqa: BLE001 - 無ければ表記で比べる
            _TAGGER = False
    if _TAGGER:
        parts = []
        for w in _TAGGER(text):
            kana = getattr(w.feature, "kana", None)
            parts.append(kana if kana and kana != "*" else w.surface)
        text = "".join(parts)
    text = "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in text)
    return re.sub(r"[\W_ー]", "", text).lower()


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def mismatch(ref: str, hyp: str) -> float:
    a, b = reading_key(ref), reading_key(hyp)
    if not a:
        return 1.0
    return edit_distance(a, b) / len(a)


# ---------------------------------------------------------------- 音声を集める

def parse_list_files(folder: Path) -> dict[str, str]:
    """Style-Bert-VITS2 の esd.list のような「ファイル|...|テキスト」の一覧（*.list）を、ファイル名（拡張子なし）→ 文に。"""
    texts: dict[str, str] = {}
    for lst in folder.rglob("*.list"):
        for line in read_text(lst).splitlines():
            parts = line.strip().split("|")
            if len(parts) >= 2 and parts[0]:
                texts[Path(parts[0].replace("\\", "/")).stem] = parts[-1]
    return texts


def scan(folder: Path) -> list[dict]:
    import soundfile as sf

    if not folder.is_dir():
        raise TrainError(f"フォルダが見つかりません: {folder}")
    listed = parse_list_files(folder)
    items = []
    for path in sorted(p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_EXTS and p.is_file()):
        try:
            info = sf.info(str(path))
            seconds = float(info.frames) / float(info.samplerate)
        except Exception:  # noqa: BLE001 - 読めない音声は飛ばす
            items.append({"wav": str(path), "skip": "unreadable"})
            continue
        txt = path.with_suffix(".txt")
        raw = read_text(txt) if txt.is_file() else listed.get(path.stem)
        item = {"wav": str(path), "seconds": seconds, "text": None, "text_source": None}
        if raw is not None:
            text, reliable = clean_text(raw)
            if text and has_words(text):
                item.update(text=text, text_source="file" if reliable else "file_unreliable")
        items.append(item)
    if not items:
        raise TrainError(f"音声ファイル（{', '.join(sorted(AUDIO_EXTS))}）が見つかりません: {folder}")
    return items


# ---------------------------------------------------------------- 書き起こし（Haas-ja）

def transcribe(items: list[dict], device: str, work: Path, pct: tuple[float, float]) -> None:
    py = haas_python()
    if py is None:
        raise TrainError("テキストの無い音声があります。書き起こしに使う Haas-ja が入っていません。"
                         "bat\\haas_setup.bat を実行するか、音声と同じ名前の .txt を置いてください。")
    lst = work / "haas_in.jsonl"
    out = work / "haas_out.jsonl"
    lst.write_text("".join(json.dumps({"wav": it["wav"]}, ensure_ascii=False) + "\n" for it in items), encoding="utf-8")
    cmd = [str(py), str(HERE / "haas_worker.py"), "--list", str(lst), "--out", str(out), "--device", device]
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", env=env, cwd=str(HERE))
    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        try:
            msg = json.loads(line) if line.startswith("{") else None
        except ValueError:
            msg = None
        if msg is None:
            if line:
                tail = (tail + [line])[-20:]
            continue
        if msg.get("event") == "loading":
            progress("transcribe", pct[0], f"書き起こしのモデルを読み込んでいます（{msg.get('device')}）")
        elif msg.get("event") == "progress":
            frac = msg["done"] / max(msg["total"], 1)
            progress("transcribe", pct[0] + (pct[1] - pct[0]) * frac,
                     f"書き起こし {msg['done']}/{msg['total']}")
    if proc.wait() != 0 or not out.is_file():
        raise TrainError("書き起こし（Haas-ja）に失敗しました:\n" + "\n".join(tail[-8:]))
    hyp = {}
    for line in out.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            hyp[r["wav"]] = r.get("hyp") or ""
    for it in items:
        it["hyp"] = hyp.get(it["wav"], "")


# ---------------------------------------------------------------- 無音の詰めと符号化

def trim_silence(wav, sr: int, pad: float = TRIM_PAD_SECONDS, thresh_db: float = -40.0):
    import numpy as np

    mono = np.abs(wav if wav.ndim == 1 else wav.mean(axis=1))
    peak = float(mono.max()) if mono.size else 0.0
    hop = max(1, int(sr * 0.01))
    n = len(mono) // hop
    if peak <= 0 or n == 0:
        return wav
    env = mono[: n * hop].reshape(n, hop).max(axis=1)
    loud = np.flatnonzero(env >= peak * 10 ** (thresh_db / 20))
    if loud.size == 0:
        return wav
    start, end = int(loud[0] * hop), int(min(len(mono), (loud[-1] + 1) * hop))
    return wav[max(0, start - int(sr * pad)): min(len(wav), end + int(sr * pad))]


def file_key(path: str) -> str:
    st = os.stat(path)
    return hashlib.sha1(f"{Path(path).resolve()}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8")).hexdigest()[:20]


def encode(items: list[dict], device: str, work: Path, pct: tuple[float, float]) -> None:
    import soundfile as sf
    import torch
    from irodori_tts.codec import DACVAECodec

    lat_dir = work / "latents"
    lat_dir.mkdir(parents=True, exist_ok=True)
    todo = [it for it in items if not (lat_dir / f"{file_key(it['wav'])}.pt").is_file()]
    codec = None
    if todo:
        progress("encode", pct[0], "音声を符号化するモデルを読み込んでいます")
        codec = DACVAECodec.load(repo_id=CODEC_REPO, device=device)
    done = 0
    for it in items:
        path = lat_dir / f"{file_key(it['wav'])}.pt"
        if not path.is_file():
            wav, sr = sf.read(it["wav"], dtype="float32", always_2d=True)
            wav = trim_silence(wav, sr)
            x = torch.from_numpy(wav.mean(axis=1, keepdims=True).T.copy())
            with torch.no_grad():
                latent = codec.encode_waveform(x, sample_rate=sr)[0].cpu()
            torch.save(latent, path)
            it["frames"] = int(latent.shape[0])
            done += 1
            if done % 20 == 0 or done == len(todo):
                progress("encode", pct[0] + (pct[1] - pct[0]) * done / max(len(todo), 1),
                         f"音声の符号化 {done}/{len(todo)}")
        else:
            it["frames"] = int(torch.load(path, map_location="cpu", weights_only=True).shape[0])
        it["latent"] = str(path)
    del codec
    if device == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- モデル

def resolve_checkpoint(spec: str | None) -> Path:
    """ローカルの .safetensors か、Hugging Face の repo id（owner/repo[/subfolder]）。"""
    if not spec:
        spec = str(BOX_ROOT / "models" / "model.safetensors")
    path = Path(spec).expanduser()
    if path.is_file():
        return path.resolve()
    from irodori_tts import inference_runtime as ir
    try:
        os.environ.setdefault("HF_HUB_OFFLINE", "0")
        return Path(ir.download_hf_checkpoint(spec))
    except Exception as exc:  # noqa: BLE001
        raise TrainError(f"ベースモデルが見つかりません: {spec}（{exc}）") from exc


def checkpoint_config(path: Path) -> dict:
    import struct

    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        meta = json.loads(handle.read(header_len).decode("utf-8", "replace")).get("__metadata__") or {}
    config = json.loads(meta.get("config_json") or "{}")
    config["_quantized"] = bool(meta.get("irodori_quantization_json"))
    return config


def compatible(config: dict) -> str | None:
    """話者学習に使えないなら理由。v4.1 Small と同じ構造（RF・幅 1280・12 層・話者 768・token_sum の時間予測）だけ。"""
    if not config:
        return "設定（config_json）の無いモデルです"
    if config.get("_quantized"):
        return "量子化したモデルでは学習できません（bf16 / fp32 の版を選んでください）"
    if config.get("flow_parameterization") == "meanflow":
        return "MeanFlow のモデルでは学習できません"
    want = {"model_dim": 1280, "num_layers": 12, "speaker_dim": 768}
    for key, value in want.items():
        if config.get(key) != value:
            return f"v4.1 Small と構造が違います（{key}={config.get(key)}）"
    if not config.get("use_speaker_condition", True) or not config.get("use_duration_predictor", True):
        return "話者の条件か時間予測の無いモデルです"
    if not str(config.get("duration_architecture", "")).startswith("token_sum"):
        return "時間予測の形が v4.1 Small と違います"
    return None


def hf_hub_dirs() -> list[Path]:
    dirs = []
    for base in (os.environ.get("HF_HOME"), str(BOX_ROOT / ".cache" / "huggingface"),
                 str(Path.home() / ".cache" / "huggingface")):
        if base and (Path(base) / "hub").is_dir() and Path(base, "hub").resolve() not in dirs:
            dirs.append(Path(base, "hub").resolve())
    return dirs


def list_base_models(model_dir: Path | None = None) -> list[dict]:
    """ベースに選べるモデル（手元にある .safetensors のうち、v4.1 Small と同じ構造のもの）。
    {"path", "label", "source"}。HF のキャッシュは repo id（とサブフォルダ）で表す。"""
    found: list[dict] = []
    seen: set[str] = set()

    def add(path: Path, label: str, source: str) -> None:
        key = str(path.resolve()).lower()
        if key in seen or path.name.endswith(".speaker.safetensors"):
            return
        seen.add(key)
        try:
            why = compatible(checkpoint_config(path))
        except (OSError, ValueError):
            return
        if why is None:
            found.append({"path": str(path), "label": label, "source": source})

    model_dir = model_dir or (BOX_ROOT / "models")
    if model_dir.is_dir():
        for path in sorted(model_dir.rglob("*.safetensors")):
            add(path, str(path.relative_to(model_dir)).replace("\\", "/"), "local")
    labels: set[str] = set()  # 同じモデルが複数のキャッシュにあっても 1 つだけ
    for hub in hf_hub_dirs():
        for repo_dir in sorted(hub.glob("models--*")):
            repo = repo_dir.name[len("models--"):].replace("--", "/")
            ref = repo_dir / "refs" / "main"
            main = ref.read_text(encoding="utf-8").strip() if ref.is_file() else ""
            # 同じ repo の古いスナップショットは出さない（refs/main の版を先に、無ければ新しい順）
            paths = sorted(repo_dir.glob("snapshots/*/**/model.safetensors"),
                           key=lambda p: (p.relative_to(repo_dir).parts[1] != main, -p.stat().st_mtime))
            for path in paths:
                sub = path.parent.relative_to(next(p for p in path.parents if p.parent.name == "snapshots"))
                label = repo if str(sub) == "." else f"{repo}/{sub.as_posix()}"
                if label not in labels:
                    labels.add(label)
                    add(path, label, "huggingface")
    return found


def load_model(ckpt: Path, dtype):
    from tts_cli import _import_runtime
    _import_runtime()
    from irodori_tts import inference_runtime as ir
    from irodori_tts.config import ModelConfig, merge_dataclass_overrides
    from irodori_tts.model import TextToLatentRFDiT
    from irodori_tts.tokenizer import PretrainedTextTokenizer

    state, cfg_dict, _, te_cfg = ir._load_checkpoint_for_inference(ckpt, load_dtype=dtype)
    cfg = merge_dataclass_overrides(ModelConfig(), cfg_dict, section="checkpoint model_config")
    with ir._meta_parameters():
        model = TextToLatentRFDiT(cfg, pretrained_backbone_config=te_cfg,
                                  load_pretrained_backbone_weights=not cfg.use_pretrained_text_encoder)
    model.load_state_dict(state, assign=True)
    del state
    for p in model.parameters():
        p.requires_grad_(False)
    src, local = ir._resolve_tokenizer_source(ckpt, cfg.text_tokenizer_repo)
    tok = PretrainedTextTokenizer.from_pretrained(repo_id=src, add_bos=bool(cfg.text_add_bos), local_files_only=local,
                                                  revision=None if local else cfg.text_encoder_revision)
    return model, cfg, tok


def move(module, device, dtype=None) -> None:
    import torch

    if module is None:
        return
    with torch.no_grad():
        for p in module.parameters():
            p.data = p.data.to(device=device, dtype=dtype if (dtype is not None and p.is_floating_point()) else p.dtype)
        for m in module.modules():
            for k, b in m._buffers.items():
                if b is not None:
                    m._buffers[k] = b.to(device=device)


def use_efficient_attention() -> None:
    """マスク付き attention を efficient attention にする（cuDNN 優先のままだと学習の逆伝播が 7 倍遅い）。"""
    from torch.nn.attention import SDPBackend
    import irodori_tts.attention as attn_mod

    attn_mod._SDPA_PRIORITY = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


# ---------------------------------------------------------------- 学習

def lr_factor(step: int, total: int, schedule: str = "constant", warmup: int = 0) -> float:
    """step（1 始まり）での学習率の倍率。ウォームアップは 0 から直線で上げ、cosine は残りの区間で 0 近くまで下げる。"""
    if warmup > 0 and step <= warmup:
        return step / warmup
    if schedule == "cosine":
        span = max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, (step - warmup - 1) / span)))
    return 1.0


def val_count(total: int, frac: float) -> int:
    """検証に回す本数。割合 0 か、音声が少ないときは 0（取り分けない）。"""
    if frac <= 0 or total < VAL_MIN_TOTAL:
        return 0
    return int(min(VAL_MAX_CLIPS, max(VAL_MIN_CLIPS, round(total * frac)), total - 8))


def split_rows(total: int, frac: float, seed: int) -> tuple[list[int], list[int]]:
    """(学習の番号, 検証の番号)。同じシードなら同じ分け方。"""
    order = list(range(total))
    random.Random(seed + 1).shuffle(order)
    n = val_count(total, frac)
    return sorted(order[n:]), sorted(order[:n])


def auto_val_every(steps: int, cuda: bool) -> int:
    """検証の間隔の既定。GPU は全体で約 20 回、CPU は重いので約 10 回。"""
    target = steps / (20 if cuda else 10)
    return max(5, int(round(target / 5.0)) * 5)


def train(rows: list[dict], ckpt: Path, args, pct: tuple[float, float],
          val_ids: list[int] | None = None) -> tuple:
    import torch
    import torch.nn.functional as F
    from irodori_tts.duration import build_duration_features
    from irodori_tts.model import patch_sequence_with_mask

    dev = torch.device(args.device)
    cuda = dev.type == "cuda"
    dtype = torch.bfloat16 if cuda else torch.float32
    batch = args.batch or (16 if cuda else 4)
    ckpt_every = args.grad_ckpt if args.grad_ckpt is not None else (1 if cuda else 0)
    if cuda:
        use_efficient_attention()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    val_ids = sorted(val_ids or [])
    val_set = set(val_ids)
    train_ids = [i for i in range(len(rows)) if i not in val_set]
    stop_file = Path(args.stop_file) if getattr(args, "stop_file", None) else None

    progress("load", pct[0], "ベースモデルを読み込んでいます")
    lat = [torch.load(r["latent"], map_location="cpu", weights_only=True).float() for r in rows]
    model, cfg, tok = load_model(ckpt, dtype)
    model.train()

    # 初期値（ゼロショット）: 話者エンコーダだけ載せて計算し、下ろす（検証の音声は使わない）
    order = list(train_ids)
    random.Random(args.seed).shuffle(order)
    pieces, total = [], 0
    for i in order:
        pieces.append(lat[i])
        total += rows[i]["frames"]
        if total >= INIT_REF_SECONDS * FPS:
            break
    move(model.speaker_encoder, dev)
    move(model.speaker_norm, dev)
    ref = torch.cat(pieces)[None].to(dev, dtype)
    with torch.no_grad():
        rl, rm = patch_sequence_with_mask(seq=ref, mask=torch.ones(ref.shape[:2], dtype=torch.bool, device=dev),
                                          patch_size=cfg.speaker_patch_size)
        st = model.speaker_norm(model.speaker_encoder(rl, rm))
        st, _ = model._prepend_masked_mean_token(st, rm)
    st = st[0].float()
    init = torch.cat([st[:1], F.adaptive_avg_pool1d(st[1:].T[None], TOKENS - 1)[0].T]).cpu()
    move(model.speaker_encoder, "cpu")
    model.speaker_encoder = None
    del ref, rl, rm, st
    if cuda:
        torch.cuda.empty_cache()

    # テキストを前計算（推論と同じく 256 まで詰めて計算し、実際の長さだけ残す）して、modernbert は下ろす
    progress("load", pct[0] + 1, "セリフのテキストを前計算しています")
    texts = sorted({r["text"] for r in rows})
    cache: dict[str, torch.Tensor] = {}
    move(model.pretrained_text_backbone, dev)
    move(model.text_encoder, dev)
    move(model.text_norm, dev)
    model.pretrained_text_backbone.eval()
    with torch.no_grad():
        for k in range(0, len(texts), 32):
            part = texts[k:k + 32]
            ids, mask = tok.batch_encode(part, max_length=MAX_TEXT_LEN)
            ids, mask = ids.to(dev), mask.to(dev)
            ts = model.text_norm(model.text_encoder(model.pretrained_text_backbone, ids, mask))
            for j, t in enumerate(part):
                cache[t] = ts[j, : int(mask[j].sum())].to("cpu", torch.bfloat16 if cuda else torch.float32)
    move(model.pretrained_text_backbone, "cpu")
    model.pretrained_text_backbone = None
    for name, child in model.named_children():
        if child is not None and name != "speaker_encoder":
            move(child, dev)
    model.set_gradient_checkpointing(ckpt_every > 0, store_every=ckpt_every if ckpt_every > 1 else 0)
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)

    spk = torch.nn.Parameter(init.to(dev))
    init_dev = init.to(dev)
    init_norm = float(init_dev.norm().clamp_min(1e-8))
    if args.speed == "keep":
        keep_mask = torch.ones_like(spk)
        keep_mask[0] = 0
        spk.register_hook(lambda g: g * keep_mask)
    dur_weight = args.dur_weight if args.speed == "learn" else 0.0
    schedule = getattr(args, "lr_schedule", "constant")
    warmup = max(0, min(int(getattr(args, "warmup", 0)), args.steps - 1))
    opt = torch.optim.AdamW([spk], lr=args.lr * lr_factor(1, args.steps, schedule, warmup),
                            betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    cap_dim = cfg.caption_dim_resolved

    def batch_tensors(ids_):
        bsz = len(ids_)
        T = max(rows[i]["frames"] for i in ids_)
        x0 = torch.zeros((bsz, T, cfg.latent_dim))
        valid = torch.zeros((bsz, T), dtype=torch.bool)
        btexts = [rows[i]["text"] for i in ids_]
        L = max(cache[t].shape[0] for t in btexts)
        ts = torch.zeros((bsz, L, cache[btexts[0]].shape[-1]), dtype=cache[btexts[0]].dtype)
        tm = torch.zeros((bsz, L), dtype=torch.bool)
        for j, i in enumerate(ids_):
            n = rows[i]["frames"]
            x0[j, :n] = lat[i][:n]
            valid[j, :n] = True
            m = cache[btexts[j]].shape[0]
            ts[j, :m] = cache[btexts[j]]
            tm[j, :m] = True
        feats = build_duration_features(btexts, token_counts=tm.sum(dim=1), max_text_len=MAX_TEXT_LEN,
                                        has_speaker=torch.ones(bsz, dtype=torch.bool))
        frames = torch.tensor([rows[i]["frames"] for i in ids_], dtype=torch.float32)
        return (x0.to(dev), valid.to(dev), ts.to(dev, dtype), tm.to(dev), feats.to(dev), frames.to(dev))

    def conditions(bsz):
        st_ = spk.to(dtype)[None].expand(bsz, -1, -1)
        sm_ = torch.ones((bsz, st_.shape[1]), dtype=torch.bool, device=dev)
        cs_ = torch.zeros((bsz, 1, cap_dim), dtype=dtype, device=dev)
        cm_ = torch.zeros((bsz, 1), dtype=torch.bool, device=dev)
        return st_, sm_, cs_, cm_

    def predict_log_frames(ts, tm, feats, st_, sm_, cs_, cm_, has_speaker=True, detach=True):
        bsz = ts.shape[0]
        return model.predict_duration_log_frames(
            text_state=ts, text_mask=tm, speaker_state=st_, speaker_mask=sm_, caption_state=cs_, caption_mask=cm_,
            duration_features=feats, has_speaker=torch.full((bsz,), has_speaker, dtype=torch.bool, device=dev),
            has_caption=torch.zeros(bsz, dtype=torch.bool, device=dev), detach_condition=detach)

    def make_batches():
        idx = list(train_ids)
        rng.shuffle(idx)
        out, win = [], batch * 64
        for s in range(0, len(idx), win):
            chunk = sorted(idx[s:s + win], key=lambda i: rows[i]["frames"])
            out += [chunk[k:k + batch] for k in range(0, len(chunk), batch)]
        out = [b for b in out if len(b) == batch] or out
        rng.shuffle(out)
        return out

    # 検証: 音声ごとに決まった t（VAL_T_POINTS 個）と決まった雑音で loss を測る。雑音は最初に作って CPU に置く
    val_t = torch.sigmoid(torch.erfinv(2.0 * (torch.arange(VAL_T_POINTS) + 0.5) / VAL_T_POINTS - 1.0) * (2.0 ** 0.5)).tolist()
    gen = torch.Generator().manual_seed(args.seed + 7919)
    val_pairs = sorted(((i, k) for i in val_ids for k in range(VAL_T_POINTS)), key=lambda p: rows[p[0]]["frames"])
    val_noise = {p: torch.randn((rows[p[0]]["frames"], cfg.latent_dim), generator=gen) for p in val_pairs}
    val_every = int(getattr(args, "val_every", 0) or 0) or auto_val_every(args.steps, cuda)
    report_every = max(1, int(round(args.steps / 250)))

    def evaluate() -> dict:
        """検証の loss（RF）と時間予測の誤差（秒）。dropout を止めて測る。"""
        model.eval()
        total, count, errs = 0.0, 0, []
        with torch.no_grad():
            st_all = conditions(batch)
            for k in range(0, len(val_pairs), batch):
                chunk = val_pairs[k:k + batch]
                x0, valid, ts, tm, _, _ = batch_tensors([i for i, _ in chunk])
                noise = torch.zeros_like(x0)
                for j, p in enumerate(chunk):
                    nz = val_noise[p]
                    noise[j, :nz.shape[0]] = nz.to(dev)
                t = torch.tensor([val_t[q] for _, q in chunk], device=dev, dtype=torch.float32)
                x_t = ((1 - t[:, None, None]) * x0 + t[:, None, None] * noise).to(dtype)
                st_, sm_, cs_, cm_ = (c[: len(chunk)] for c in st_all)
                v = model.forward_with_encoded_conditions(x_t=x_t, t=t.to(dtype), text_state=ts, text_mask=tm,
                                                          speaker_state=st_, speaker_mask=sm_, caption_state=cs_,
                                                          caption_mask=cm_, latent_mask=valid)
                diff = ((v.float() - (noise - x0)) ** 2).mean(dim=-1)
                w = valid.float()
                total += float(((diff * w).sum(-1) / w.sum(-1).clamp_min(1)).sum())
                count += len(chunk)
            for k in range(0, len(val_ids), batch):
                ids_ = val_ids[k:k + batch]
                _, _, ts, tm, feats, frames = batch_tensors(ids_)
                st_, sm_, cs_, cm_ = (c[: len(ids_)] for c in st_all)
                logf = predict_log_frames(ts, tm, feats, st_, sm_, cs_, cm_)
                errs += (torch.expm1(logf.float()).clamp_min(0) - frames).abs().tolist()
        model.train()
        return {"loss": round(total / max(count, 1), 5), "dur_mae_sec": round(sum(errs) / max(len(errs), 1) / FPS, 3)}

    emit("config", batch=batch, grad_ckpt=ckpt_every, dtype=str(dtype).replace("torch.", ""),
         train_clips=len(train_ids), val_clips=len(val_ids), val_points=len(val_pairs), val_every=val_every if val_ids else 0,
         report_every=report_every, lr=args.lr, lr_schedule=schedule, warmup=warmup, steps=args.steps,
         dur_weight=dur_weight, params=int(spk.numel()), keep=getattr(args, "keep", "last"), seed=args.seed)

    history: dict[str, list] = {"train": [], "val": []}
    best = {"loss": math.inf, "step": 0, "spk": None}

    def record_val(step: int) -> None:
        point = {"step": step, **evaluate()}
        if point["loss"] < best["loss"]:
            best.update(loss=point["loss"], step=step, spk=spk.detach().clone())
            point["best"] = True
        history["val"].append(point)
        emit("metrics", kind="val", **point)

    t0 = time.time()
    if val_ids:
        progress("train", pct[0] + 2, "検証の基準（学習前のゼロショット）を測っています")
        record_val(0)
    batches: list[list[int]] = []
    hist = {"rf": 0.0, "dur": 0.0, "mae": 0.0, "grad": 0.0, "n": 0}
    step_losses: list[float] = []
    span = pct[1] - pct[0] - 2
    window_t = time.time()
    t_train = time.time()
    done_steps = 0
    stopped_at = None
    for step in range(1, args.steps + 1):
        if stop_file is not None and stop_file.exists():
            stopped_at = done_steps
            break
        lr_now = args.lr * lr_factor(step, args.steps, schedule, warmup)
        for group in opt.param_groups:
            group["lr"] = lr_now
        if not batches:
            batches = make_batches()
        x0, valid, ts, tm, feats, frames = batch_tensors(batches.pop())
        bsz = x0.shape[0]
        u = ((torch.arange(bsz, device=dev, dtype=torch.float32) + torch.rand(bsz, device=dev)) / bsz).clamp(1e-6, 1 - 1e-6)
        t = torch.sigmoid(torch.erfinv(2.0 * u - 1.0) * (2.0 ** 0.5))[torch.randperm(bsz, device=dev)].clamp(0.001, 0.999)
        noise = torch.randn_like(x0)
        x_t = ((1 - t[:, None, None]) * x0 + t[:, None, None] * noise).to(dtype)
        st_, sm_, cs_, cm_ = conditions(bsz)
        v = model.forward_with_encoded_conditions(x_t=x_t, t=t.to(dtype), text_state=ts, text_mask=tm,
                                                  speaker_state=st_, speaker_mask=sm_, caption_state=cs_,
                                                  caption_mask=cm_, latent_mask=valid)
        diff = ((v.float() - (noise - x0)) ** 2).mean(dim=-1)
        w = valid.float()
        rf = ((diff * w).sum(-1) / w.sum(-1).clamp_min(1)).mean()
        logf = predict_log_frames(ts, tm, feats, st_, sm_, cs_, cm_, detach=dur_weight <= 0)
        dur = F.huber_loss(logf.float(), torch.log1p(frames), delta=0.1)
        loss = rf + dur_weight * dur
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = float(spk.grad.detach().float().norm()) if spk.grad is not None else 0.0
        opt.step()
        done_steps = step
        rf_value = float(rf.detach())
        step_losses.append(rf_value)
        hist["rf"] += rf_value
        hist["dur"] += float(dur.detach())
        hist["mae"] += float((torch.expm1(logf.detach().float()).clamp_min(0) - frames).abs().mean())
        hist["grad"] += grad
        hist["n"] += 1
        if step % report_every == 0 or step == args.steps:
            now = time.time()
            n = max(hist["n"], 1)
            point = {"step": step, "loss": round(hist["rf"] / n, 5), "dur": round(hist["dur"] / n, 5),
                     "dur_mae_sec": round(hist["mae"] / n / FPS, 3), "lr": float(f"{lr_now:.6g}"),
                     "grad": round(hist["grad"] / n, 5),
                     "drift": round(float((spk.detach() - init_dev).norm()) / init_norm, 5),
                     "steps_per_sec": round(n / max(now - window_t, 1e-6), 3), "elapsed_sec": round(now - t0, 1)}
            history["train"].append(point)
            emit("metrics", kind="train", **point)
            window_t = now
            hist = {"rf": 0.0, "dur": 0.0, "mae": 0.0, "grad": 0.0, "n": 0}
        if val_ids and step % val_every == 0 and step != args.steps:
            t_eval = time.time()
            record_val(step)
            window_t += time.time() - t_eval  # 速さ（step/秒）に検証の時間を混ぜない
        if step % report_every == 0 or step == args.steps:
            el = time.time() - t_train
            eta = round(el / step * (args.steps - step))
            extra = {"step": step, "steps": args.steps, "eta_sec": eta, "rf": point["loss"],
                     "dur_mae_sec": point["dur_mae_sec"]}
            if cuda:
                extra["vram_gb"] = round(torch.cuda.max_memory_allocated(dev) / GB, 2)
            progress("train", pct[0] + 2 + span * step / args.steps,
                     f"学習 {step}/{args.steps}（残り約 {max(0, eta) // 60} 分 {max(0, eta) % 60} 秒）", **extra)

    if val_ids and (not history["val"] or history["val"][-1]["step"] != done_steps):
        progress("train", pct[1] - 1, "最後の検証をしています")
        record_val(done_steps)
    keep = getattr(args, "keep", "last")
    saved_step = done_steps
    if keep == "best" and best["spk"] is not None:
        with torch.no_grad():
            spk.copy_(best["spk"])
        saved_step = best["step"]

    # 話速の記録: 保存する話者で「実際 ÷ 予測」（補正の候補）と「この話者 ÷ 話者なし」（ベースと比べた遅さ）
    metrics = {}
    model.eval()
    with torch.no_grad():
        sample = list(train_ids)
        rng.shuffle(sample)
        ratios, paces = [], []
        for k in range(0, min(len(sample), 320), batch):
            ids_ = sample[k:k + batch]
            _, _, ts, tm, feats, frames = batch_tensors(ids_)
            st_, sm_, cs_, cm_ = conditions(len(ids_))
            learned = torch.expm1(predict_log_frames(ts, tm, feats, st_, sm_, cs_, cm_).float())
            none = torch.expm1(predict_log_frames(ts, tm, feats, None, None, cs_, cm_, has_speaker=False).float())
            ratios += (frames / learned.clamp_min(1)).tolist()
            paces += (learned / none.clamp_min(1)).tolist()
        ratios.sort()
        paces.sort()
        tail = step_losses[-max(1, len(step_losses) // 10):] if step_losses else []
        metrics = {"actual_over_predicted": round(ratios[len(ratios) // 2], 3),
                   "pace_vs_base": round(paces[len(paces) // 2], 3),
                   "train_seconds": round(time.time() - t0, 1),
                   "steps_done": done_steps, "saved_step": saved_step,
                   "final_train_loss": round(sum(tail) / len(tail), 5) if tail else None,
                   "drift": round(float((spk.detach() - init_dev).norm()) / init_norm, 5)}
        if stopped_at is not None:
            metrics["stopped_at"] = stopped_at
        if history["val"]:
            first, last = history["val"][0], history["val"][-1]
            saved = next((p for p in history["val"] if p["step"] == saved_step), last)
            metrics.update(val_loss_init=first["loss"], val_loss_final=last["loss"], val_loss_best=best["loss"],
                           best_step=best["step"], val_loss_saved=saved["loss"],
                           val_dur_mae_sec_init=first["dur_mae_sec"], val_dur_mae_sec_saved=saved["dur_mae_sec"])
        if cuda:
            metrics["peak_vram_gb"] = round(torch.cuda.max_memory_allocated(dev) / GB, 2)
    return spk.detach().float().cpu().contiguous(), metrics, {
        "batch": batch, "grad_ckpt": ckpt_every, "dtype": str(dtype).replace("torch.", ""),
        "train_clips": len(train_ids), "val_clips": len(val_ids), "val_every": val_every if val_ids else 0,
        "report_every": report_every, "history": history}


# ---------------------------------------------------------------- 本体

def run(args) -> dict:
    import torch

    name = validate_name(args.name)
    target, meta_path = output_paths(name)
    if target.exists() and not args.overwrite:
        raise TrainError(f"「{name}」という学習した話者はもうあります。別の名前にするか、上書きを選んでください")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise TrainError("CUDA の GPU が見つかりません。CPU を選んでください")
    check_settings(args)

    data_dir = Path(args.data).expanduser().resolve()
    ckpt = resolve_checkpoint(args.base)
    why = compatible(checkpoint_config(ckpt))
    if why:
        raise TrainError(f"このモデルは話者学習のベースにできません: {why}")

    work = Path(args.work_dir).resolve() if args.work_dir else (
        BOX_ROOT / ".cache" / "speaker-train" / hashlib.sha1(str(data_dir).lower().encode("utf-8")).hexdigest()[:16])
    work.mkdir(parents=True, exist_ok=True)

    progress("scan", 0, "音声を集めています")
    items = scan(data_dir)
    readable = [it for it in items if "skip" not in it]
    stats = {"files": len(items), "unreadable": len(items) - len(readable)}
    short = [it for it in readable if it["seconds"] < MIN_SECONDS]
    usable = [it for it in readable if it["seconds"] >= MIN_SECONDS]
    stats["too_short"] = len(short)

    need = [it for it in usable if it["text"] is None or it["text_source"] == "file_unreliable"]
    check = args.verify_text and haas_python() is not None
    targets = usable if check else need
    stats["transcribed"] = 0
    if targets:
        transcribe(targets, args.device, work, (2, 25))
        stats["transcribed"] = len(targets)
    replaced = 0
    for it in usable:
        hyp = (it.get("hyp") or "").strip()
        if it["text"] is None or it["text_source"] == "file_unreliable":
            it["text"], it["text_source"] = (hyp, "asr") if has_words(hyp) else (None, None)
        elif (check and hyp and len(reading_key(it["text"])) >= MISMATCH_MIN_CHARS
              and mismatch(it["text"], hyp) > MISMATCH_THRESHOLD):
            it["text"], it["text_source"] = hyp, "asr_replaced"
            replaced += 1
    stats["text_replaced_by_asr"] = replaced
    from tts_cli import _import_runtime
    _import_runtime()
    from irodori_tts.text_normalization import normalize_text
    for it in usable:
        if it["text"]:
            it["text"] = normalize_text(it["text"]).strip()
    with_text = [it for it in usable if it["text"] and has_words(it["text"])]
    stats["no_text"] = len(usable) - len(with_text)

    encode(with_text, args.device, work, (25 if targets else 5, 45))
    rows = [it for it in with_text if it["frames"] <= MAX_FRAMES]
    stats["too_long"] = len(with_text) - len(rows)
    stats["used"] = len(rows)
    stats["used_minutes"] = round(sum(r["frames"] for r in rows) / FPS / 60, 1)
    if len(rows) < 8:
        raise TrainError(f"学習に使える音声が {len(rows)} 本しかありません（1〜30 秒でテキストのあるもの）。"
                         "もっと音声を入れてください（合計 10 分以上がおすすめ）")
    train_ids, val_ids = split_rows(len(rows), args.val_frac, args.seed)
    stats["val"] = len(val_ids)
    val_set = set(val_ids)
    (work / "manifest.jsonl").write_text("".join(json.dumps(
        {**{k: r[k] for k in ("wav", "text", "text_source", "frames", "latent")}, "split": "val" if i in val_set else "train"},
        ensure_ascii=False) + "\n" for i, r in enumerate(rows)), encoding="utf-8")
    split_note = f"（うち検証 {len(val_ids)} 本）" if val_ids else ""
    progress("prepare", 45, f"{len(rows)} 本・{stats['used_minutes']} 分で学習します{split_note}", stats=stats)

    embedding, metrics, run_cfg = train(rows, ckpt, args, (45, 97), val_ids=val_ids)
    history = run_cfg.pop("history")

    progress("save", 98, "話者を保存しています")
    from safetensors.torch import save_file
    target.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "trainer": "kataribe speaker_train v2",
        "base_checkpoint": str(ckpt),
        "base_sha1_head": hashlib.sha1(ckpt.open("rb").read(1 << 20)).hexdigest()[:16],
        "speed": args.speed, "steps": str(metrics["saved_step"]), "tokens": str(TOKENS), "init": "zeroshot",
        "lr": str(args.lr), "lr_schedule": args.lr_schedule,
        "actual_over_predicted": str(metrics["actual_over_predicted"]),
        "pace_vs_base": str(metrics["pace_vs_base"]),
    }
    tmp = target.with_suffix(".tmp")
    save_file({"speaker_embedding": embedding}, str(tmp), metadata=meta)
    os.replace(tmp, target)
    settings = {"steps": args.steps, "lr": args.lr, "lr_schedule": args.lr_schedule, "warmup": args.warmup,
                "batch": args.batch, "dur_weight": args.dur_weight, "speed": args.speed, "val_frac": args.val_frac,
                "val_every": args.val_every, "keep": args.keep, "seed": args.seed, "device": args.device,
                "verify_text": bool(args.verify_text)}
    record = {
        "version": 2, "name": name, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_dir": str(data_dir), "base_checkpoint": str(ckpt), "speed": args.speed, "steps": args.steps,
        "lr": args.lr, "dur_weight": args.dur_weight if args.speed == "learn" else 0.0, "device": args.device,
        **run_cfg, "settings": settings, "data": stats, "metrics": metrics, "history": history,
        "text_sources": {s: sum(1 for r in rows if r["text_source"] == s) for s in {r["text_source"] for r in rows}},
    }
    meta_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    progress("done", 100, "完了しました")
    return {"name": name, "speaker_file": str(target), "record": str(meta_path), "data": stats, "metrics": metrics}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="音声から話者（16 トークンの話者埋め込み）を学習する")
    ap.add_argument("--name", required=True, help="話者の名前（speakers/学習した話者/<名前>/ に保存）")
    ap.add_argument("--data", required=True, help="音声のフォルダ（同じ名前の .txt があればセリフとして使う。無ければ Haas-ja で書き起こす）")
    ap.add_argument("--base", default=None, help="ベースモデル（.safetensors か HF の repo id）。既定 models/model.safetensors")
    ap.add_argument("--speed", choices=["learn", "keep"], default="learn",
                    help="learn = キャラの話速も学習する / keep = 話速は学習しない（ゼロショット相当）")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--batch", type=int, default=0, help="0 = 自動（GPU 16 / CPU 4）")
    ap.add_argument("--grad-ckpt", type=int, default=None, help="勾配チェックポイント（既定 GPU 1 / CPU 0）")
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--lr-schedule", choices=LR_SCHEDULES, default="constant",
                    help="学習率の変え方: constant = 一定 / cosine = 最後に向けて 0 近くまで下げる")
    ap.add_argument("--warmup", type=int, default=0, help="学習率を 0 から上げていくステップ数")
    ap.add_argument("--dur-weight", type=float, default=DEFAULT_DUR_WEIGHT)
    ap.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC,
                    help=f"検証に回す音声の割合（0 で検証しない。{VAL_MIN_TOTAL} 本未満のときも検証しない）")
    ap.add_argument("--val-every", type=int, default=0, help="検証の間隔（ステップ）。0 = 自動（GPU 約 20 回・CPU 約 10 回）")
    ap.add_argument("--keep", choices=KEEP_CHOICES, default="last",
                    help="保存する話者: last = 最後 / best = 検証 loss が一番低かったとき（検証が無ければ最後）")
    ap.add_argument("--stop-file", default=None, help="このファイルができたら、その時点の話者を保存して終える")
    ap.add_argument("--verify-text", type=int, default=1,
                    help="付属テキストも Haas-ja で確かめ、大きく違うものは書き起こしに置き換える（Haas が入っているときだけ）")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    try:
        result = run(args)
    except TrainError as exc:
        emit("error", message=str(exc))
        return 2
    except Exception as exc:  # noqa: BLE001 - エンジンに理由を返す
        import traceback
        emit("error", message=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-4000:])
        return 1
    emit("done", result=result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
