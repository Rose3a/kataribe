"""声づくり: キャプションで作った声を、話者として保存して使う。

流れ:
1. トーク画面のセリフで、キャプション（と話者なし・任意の話者）を指定して生成する。
   エンジンは、参照音声を使わない合成の結果を、合成条件をキーにして少しのあいだ覚えておく（TakeStore）。
2. 気に入ったら、そのセリフの設定パネルから名前を付けて保存する。保存は合成条件で照合した
   エンジンの合成結果からだけ作るので、外部の音声ファイルからは作れない
   （参照音声と違い、透かしを入れない前提のため）。
3. 保存した声は話者一覧に並び、合成では参照音声のエンコード結果（latent）を
   そのまま参照に使う（runtime の ref_latent）。毎回のエンコードは要らない。

保存先は speakers/声づくり/<ID>/ で、中身は
  <ID>.voice.json   … 表示名・キャプション・セリフ・シード・モデルなど（作り直し用の記録）
  <ID>.wav          … 保存したテイクの音声
  <ID>.<codec>.latent.pt … codec ごとのエンコード結果（無ければ wav から作る）
  <ID>.png          … アイコン（任意）
ID は保存時に作る固定の文字列で、話者 ID はこれから決まる。表示名を変えても、
声を差し替えても、その声を使っているセリフの話者は変わらない。
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path

VOICE_FOLDER = "声づくり"
VOICE_SUFFIX = ".voice.json"
VOICE_ID = re.compile(r"voice-[0-9a-f]{10}")
DEFAULT_CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"
MAX_NAME_CHARS = 32
MAX_CAPTION_CHARS = 500
MAX_ICON_BYTES = 2_000_000


def is_voice_file(path: Path) -> bool:
    return path.name.endswith(VOICE_SUFFIX)


def voice_stem(path: Path) -> str:
    return path.name[: -len(VOICE_SUFFIX)]


def read_meta(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("version") != 1:
        raise ValueError(f"unsupported voice file: {path}")
    return value


def display_name(path: Path) -> str:
    try:
        name = read_meta(path).get("name")
    except (OSError, ValueError):
        name = None
    return name if isinstance(name, str) and name.strip() else voice_stem(path)


def _codec_tag(codec_repo: str) -> str:
    return hashlib.sha1(codec_repo.encode("utf-8")).hexdigest()[:8]


def runtime_codec_repo(runtime) -> str:
    key = getattr(runtime, "key", None)
    return str(getattr(key, "codec_repo", None) or DEFAULT_CODEC_REPO)


def latent_path(voice_file: Path, codec_repo: str) -> Path:
    return voice_file.with_name(f"{voice_stem(voice_file)}.{_codec_tag(codec_repo)}.latent.pt")


def wav_path(voice_file: Path) -> Path:
    return voice_file.with_name(f"{voice_stem(voice_file)}.wav")


def encode_latent(runtime, wav_file: Path, target: Path) -> Path:
    """参照音声を runtime と同じ設定でエンコードし、ref_latent で読める形で保存する。"""
    import torch
    from irodori_tts.inference_runtime import SamplingRequest, _load_audio

    defaults = SamplingRequest.__dataclass_fields__
    wav, sample_rate = _load_audio(str(wav_file))
    with torch.inference_mode():
        latent = runtime.codec.encode_waveform(
            wav.unsqueeze(0), sample_rate=int(sample_rate),
            normalize_db=defaults["ref_normalize_db"].default,
            ensure_max=bool(defaults["ref_ensure_max"].default),
        )
    latent = latent.detach().to("cpu", dtype=torch.float32)[0].contiguous()
    if latent.ndim != 2 or latent.shape[0] == 0:
        raise ValueError("声のエンコードに失敗しました（音声が短すぎます）")
    fd, name = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(latent, name)
        os.replace(name, target)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return target


def ensure_latent(voice_file: Path, runtime) -> Path:
    """読み込み中のモデルの codec 用の latent を返す。無ければ wav から作る。"""
    if runtime is None or not hasattr(runtime, "codec"):
        raise ValueError("このバックエンドでは声づくりの声を使えません")
    target = latent_path(Path(voice_file), runtime_codec_repo(runtime))
    if target.is_file() and target.stat().st_size > 0:
        return target
    source = wav_path(Path(voice_file))
    if not source.is_file():
        raise FileNotFoundError(f"声の音声ファイルがありません: {source}")
    return encode_latent(runtime, source, target)


def take_key(query: dict, speaker_id) -> str:
    """セリフの合成条件（エディタが合成に送る query と話者）から、照合用のキーを作る。

    同じ条件で合成した音声は同じキーになる。シードが空欄（ランダム）のセリフは、
    最後に合成したものが残る。
    """
    payload = json.dumps({"speaker": speaker_id, "query": query},
                         sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class TakeStore:
    """このエンジンが合成した音声（テイク）を少しのあいだ覚えておく。声の保存はここからだけ。"""

    def __init__(self, limit: int = 24):
        self.limit = limit
        self._takes: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, wav: bytes, params: dict, key: str | None = None) -> str:
        take = key or secrets.token_hex(8)
        with self._lock:
            self._takes.pop(take, None)
            self._takes[take] = {"wav": wav, "params": dict(params)}
            while len(self._takes) > self.limit:
                self._takes.popitem(last=False)
        return take

    def get(self, take) -> dict:
        with self._lock:
            value = self._takes.get(take) if isinstance(take, str) else None
        if value is None:
            raise ValueError("このセリフで生成した音声が見つかりません。もう一度生成してください")
        return value


def _clean_name(name) -> str:
    if not isinstance(name, str):
        raise ValueError("名前を入力してください")
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip()
    if not 1 <= len(name) <= MAX_NAME_CHARS:
        raise ValueError(f"名前は1〜{MAX_NAME_CHARS}文字で入力してください")
    return name


def _decode_icon(data_url) -> bytes | None:
    """アイコンの data URL を PNG にする（話者一覧の画像と同じ扱い）。"""
    if data_url is None:
        return None
    if not isinstance(data_url, str) or not re.match(r"data:image/(png|jpeg|webp);base64,", data_url):
        raise ValueError("アイコンは PNG・JPEG・WebP の画像を選んでください")
    try:
        raw = base64.b64decode(data_url.split(",", 1)[1], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("アイコンの画像を読めませんでした") from exc
    if len(raw) > MAX_ICON_BYTES:
        raise ValueError("アイコンの画像は2MB以下にしてください")
    from PIL import Image
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGBA")
    except Exception as exc:  # noqa: BLE001 - 壊れた画像は入力エラーとして返す
        raise ValueError("アイコンの画像を読めませんでした") from exc
    image.thumbnail((512, 512))
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


class VoiceLibrary:
    """speakers/声づくり/ の声の一覧・保存・変更・削除。"""

    def __init__(self, speaker_dir: Path):
        self.root = Path(speaker_dir) / VOICE_FOLDER
        self._lock = threading.Lock()

    def _file(self, voice_id) -> Path:
        if not isinstance(voice_id, str) or not VOICE_ID.fullmatch(voice_id):
            raise ValueError("声の ID が正しくありません")
        path = self.root / voice_id / f"{voice_id}{VOICE_SUFFIX}"
        if not path.is_file():
            raise FileNotFoundError("声が見つかりません。一覧を更新してください")
        return path

    def list(self) -> list[dict]:
        voices = []
        if not self.root.is_dir():
            return voices
        for path in sorted(self.root.glob(f"*/*{VOICE_SUFFIX}")):
            if not VOICE_ID.fullmatch(voice_stem(path)):
                continue
            try:
                meta = read_meta(path)
            except (OSError, ValueError):
                continue
            voices.append({
                "id": voice_stem(path), "name": display_name(path),
                "caption": meta.get("caption", ""), "text": meta.get("text", ""),
                "seed": meta.get("seed"), "steps": meta.get("steps"), "model": meta.get("model"),
                "created": meta.get("created"), "updated": meta.get("updated"),
                "hasIcon": path.with_name(f"{voice_stem(path)}.png").is_file(),
            })
        voices.sort(key=lambda voice: voice.get("created") or 0)
        return voices

    def names(self) -> set[str]:
        return {voice["name"] for voice in self.list()}

    def audio(self, voice_id) -> bytes:
        return wav_path(self._file(voice_id)).read_bytes()

    def icon(self, voice_id) -> bytes | None:
        path = self._file(voice_id)
        icon = path.with_name(f"{voice_stem(path)}.png")
        return icon.read_bytes() if icon.is_file() else None

    @staticmethod
    def _take_meta(take: dict) -> dict:
        params = take["params"]
        return {key: params.get(key) for key in ("caption", "text", "seed", "steps", "model", "speaker")}

    def create(self, take: dict, name, icon=None, encode=None) -> str:
        """テイクを声として保存する。encode(voice_file) は latent を作る（GPU を使う）。"""
        name = _clean_name(name)
        icon_png = _decode_icon(icon)
        with self._lock:
            if name in self.names():
                raise ValueError("同じ名前の声があります。別の名前にしてください")
            voice_id = "voice-" + secrets.token_hex(5)
            folder = self.root / voice_id
            folder.mkdir(parents=True, exist_ok=False)
            try:
                path = folder / f"{voice_id}{VOICE_SUFFIX}"
                wav_path(path).write_bytes(take["wav"])
                if icon_png:
                    path.with_name(f"{voice_id}.png").write_bytes(icon_png)
                now = time.time()
                meta = {"version": 1, "name": name, **self._take_meta(take),
                        "created": now, "updated": now}
                if encode is not None:
                    encode(path)
                path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            except BaseException:
                shutil.rmtree(folder, ignore_errors=True)
                raise
        return voice_id

    def update(self, voice_id, name=None, take: dict | None = None, icon=None, encode=None) -> str:
        """名前・アイコンの変更、または声（テイク）の差し替え。ID は変えない。"""
        path = self._file(voice_id)
        icon_png = _decode_icon(icon)
        with self._lock:
            meta = read_meta(path)
            if name is not None:
                name = _clean_name(name)
                if name != meta.get("name") and name in self.names():
                    raise ValueError("同じ名前の声があります。別の名前にしてください")
                meta["name"] = name
            if take is not None:
                # 古い latent は別の音声のものなので消してから作り直す
                for old in path.parent.glob(f"{voice_id}.*.latent.pt"):
                    old.unlink()
                wav_path(path).write_bytes(take["wav"])
                meta.update(self._take_meta(take))
                if encode is not None:
                    encode(path)
            if icon_png:
                path.with_name(f"{voice_id}.png").write_bytes(icon_png)
            meta["updated"] = time.time()
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, path)
        return voice_id

    def delete(self, voice_id) -> None:
        path = self._file(voice_id)
        with self._lock:
            shutil.rmtree(path.parent)
