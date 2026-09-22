"""Single VOICEVOX fork endpoint with lazy, exclusive Irodori backends."""
import argparse
import base64
import gc
import hashlib
import json
import math
import os
import re
import struct
import threading
import time
import tempfile
import uuid
from collections import deque
from pathlib import Path
from http.server import ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from voicevox_engine import (Handler, VoicevoxAdapter, ROOT, ENGINE_UUID_NAMESPACE,
                              TINY_PNG, _query, SESSION_TOKEN, MAX_BODY_BYTES, MAX_TEXT_CHARS,
                              DEFAULT_SPEAKER_NAME)
from speaker_catalog import (
    _fallback_icon,
    credit_for,
    display_name_for,
    policy_for,
    blink_thumbnail_for,
    mouth_open_thumbnail_for,
    mouth_parts_for,
    portrait_for,
    speaker_catalog,
)
from asr_timeline import (AsrTimeline, decode_wav, ensure_asr_model, asr_package_error,
                          MODEL_DIR as ASR_MODEL_DIR)
from tts_cli import SpeakerCassette, resolve_embed_dirs
from speaker_mix import compose_speaker_mix

# 初回のASRモデル取得でリクエストを待たせる上限（秒）。待ち切れなくても
# ダウンロードは続くので、次の要求で揃っていれば使える。
ASR_DOWNLOAD_WAIT_SECONDS = float(os.environ.get("IRODORI_ASR_WAIT_SECONDS", "45"))

REPO_ROOT = ROOT.parent
MODEL_DIR = Path(os.environ.get("IRODORI_MODEL_DIR", str(REPO_ROOT / "models")))
SPEAKER_DIR = Path(os.environ.get("IRODORI_EMBED_DIR", str(REPO_ROOT / "speakers")))

# Hugging Face のリポジトリID。model.safetensors を含むリポジトリを指定する。
HF_MODEL_PATTERN = re.compile(
    r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*")
# 既定ステップ数: RF は 8、MeanFlow（蒸留モデル）は 4。
DEFAULT_STEPS_RF = 8
DEFAULT_STEPS_MEANFLOW = 4
# モデル追加ダウンロードも既存のランタイムと同じキャッシュへ置く。
os.environ.setdefault("IRODORI_HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HOME", os.environ["IRODORI_HF_HOME"])

def _wav_seconds(data):
    """WAVバイト列から音声長（秒）を求める。読めなければ None。"""
    try:
        if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            return None
        offset = 12
        byte_rate = None
        while offset + 8 <= len(data):
            chunk_id = data[offset:offset + 4]
            size = int.from_bytes(data[offset + 4:offset + 8], "little")
            body = data[offset + 8:offset + 8 + size]
            if chunk_id == b"fmt " and len(body) >= 16:
                channels = int.from_bytes(body[2:4], "little")
                sample_rate = int.from_bytes(body[4:8], "little")
                bits = int.from_bytes(body[14:16], "little")
                if channels and sample_rate and bits:
                    byte_rate = sample_rate * channels * (bits // 8)
            elif chunk_id == b"data" and byte_rate:
                return size / byte_rate
            offset += 8 + size + (size & 1)
        return None
    except Exception:
        return None


class EditorAdapter:
    DEFAULT_SETTINGS = dict(backend="cpu", model="Aratako/Irodori-TTS-v4.1-Small", seed=4763674,
                            sway_coeff=-1.0)

    def __init__(self):
        self.backend_name = "editor"
        self.operation_lock = threading.RLock()
        self.state_lock = threading.RLock()
        self.progress_lock = threading.Lock()
        self.synthesis_slots = threading.BoundedSemaphore(2)
        self.progress = dict(active=False, percent=0, stage="idle")
        self._progress_percent = 0
        self._prewarm_lock = threading.Lock()
        self._prewarm_scheduled = False
        self.logs = deque(maxlen=80)
        self._phase_marks = []
        # 口パクのタイムラインを ASR で作るための生成音声キャッシュ（直近数件）
        self._wav_cache = deque(maxlen=6)
        self.timeline_reader = AsrTimeline()
        self.delegate = None
        # モデルごとの flow_parameterization / 既定ステップ数の判定結果。
        self._model_info_cache = {}
        self.config_path = ROOT / "editor-settings.json"
        self.settings = dict(backend="cpu", model="Aratako/Irodori-TTS-v4.1-Small", seed=4763674,
                             sway_coeff=-1.0)
        try:
            saved = json.loads(self.config_path.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                saved.pop("caption", None)
                saved.pop("cfg", None)
                # 旧バージョンの共通設定。現在はセリフごとのクエリから受け取る。
                saved.pop("steps", None)
                saved.pop("t_schedule_mode", None)
                saved.pop("seconds", None)
                if saved.get("model") == "model.safetensors":
                    saved["model"] = self.DEFAULT_SETTINGS["model"]
            saved = {**self.DEFAULT_SETTINGS, **saved}
            self.validate(saved)
            self.settings = saved
        except (OSError, ValueError, TypeError, KeyError):
            pass
        self.refresh()

    def available_backends(self):
        result = {"cpu": True, "cuda": False, "trt": False, "radeon": False}
        try:
            import torch
            result["cuda"] = bool(torch.cuda.is_available())
        except ImportError:
            pass
        import importlib.util
        result["trt"] = (result["cuda"] and importlib.util.find_spec("tensorrt") is not None
                         and torch.cuda.is_bf16_supported()
                         and (ROOT / 'bf16-fallback/fallback_bf16.py').is_file())
        result["radeon"] = (REPO_ROOT / "work" / "amd-dml-venv").is_dir() and (REPO_ROOT / "work" / "codec_decoder.onnx").is_file()
        return result

    def models(self):
        MODEL_DIR.mkdir(exist_ok=True)
        return sorted(str(p.relative_to(MODEL_DIR)) for p in MODEL_DIR.rglob("*.safetensors")
                      if not p.name.endswith(".speaker.safetensors"))

    @staticmethod
    def _plan_path():
        return REPO_ROOT / "irodori-tts" / "bf16-fallback" / "fallback_bf16.plan"

    @staticmethod
    def _is_hf_model_source(source):
        """Hugging Face の repo_id（任意でサブフォルダ）だけを受け付ける。"""
        return bool(HF_MODEL_PATTERN.fullmatch(str(source).strip()))

    def _resolve_local_model(self, source):
        """models フォルダ内のローカル .safetensors を返す。無ければ None。"""
        model = str(source).strip()
        if not model:
            return None
        try:
            candidate = Path(model).expanduser().resolve() if Path(model).is_absolute() else (MODEL_DIR.resolve() / model).resolve()
        except OSError:
            return None
        if not candidate.is_file() or candidate.suffix != ".safetensors":
            return None
        if not Path(model).is_absolute() and MODEL_DIR.resolve() not in candidate.parents:
            return None
        return candidate

    def _validate_model_source(self, source):
        """ローカルの .safetensors か Hugging Face の repo_id であることを確かめる。"""
        model = str(source).strip()
        if not model:
            raise ValueError("モデルを指定してください")
        if self._resolve_local_model(model) is not None or self._is_hf_model_source(model):
            return model
        raise ValueError(
            "モデルは .safetensors のパスか Hugging Face の repo_id を指定してください")

    def _resolve_model(self, source):
        """読み込みに使うローカルパスを返す。HF はキャッシュ済みのときだけ解決できる。"""
        model = str(source).strip()
        local = self._resolve_local_model(model)
        if local is not None:
            return local
        if self._is_hf_model_source(model):
            cached = self._hf_checkpoint_from_cache(model)
            if cached is not None:
                return cached
            raise ValueError(
                "Hugging Face モデルはまだダウンロードされていません。設定を適用すると取得します")
        raise ValueError(
            "モデルは .safetensors のパスか Hugging Face の repo_id を指定してください")

    @staticmethod
    def _split_hf_source(source):
        parts = str(source).strip().strip("/").split("/")
        return "/".join(parts[:2]), ("/".join(parts[2:]) or None)

    @staticmethod
    def _hf_checkpoint_from_cache(source):
        """ダウンロード済みの HF 重みをキャッシュから探す（取得はしない）。"""
        try:
            from huggingface_hub import try_to_load_from_cache
        except ImportError:
            return None
        repo_id, subfolder = EditorAdapter._split_hf_source(source)
        filename = "model.safetensors" if subfolder is None else f"{subfolder}/model.safetensors"
        try:
            cached = try_to_load_from_cache(repo_id, filename)
        except Exception:
            return None
        if isinstance(cached, str) and Path(cached).is_file():
            return Path(cached)
        return None

    def _download_hf_model(self, source):
        """Hugging Face の重みをキャッシュへ取得してローカルパスを返す。"""
        self._set_progress("downloading model from Hugging Face", 15)
        self._runtime_log(f"downloading hugging face model: {source}")
        from tts_cli import _import_runtime
        _import_runtime()
        from irodori_tts.inference_runtime import download_hf_checkpoint
        path = Path(download_hf_checkpoint(source))
        self._runtime_log(f"downloaded hugging face model: {path}")
        return path

    @staticmethod
    def _read_safetensors_config(path):
        """safetensors ヘッダの config_json だけを読む（重みは読まない）。"""
        with open(path, "rb") as handle:
            raw = handle.read(8)
            if len(raw) != 8:
                return {}
            header_len = struct.unpack("<Q", raw)[0]
            if not 0 < header_len <= 64 * 1024 * 1024:
                return {}
            header = json.loads(handle.read(header_len).decode("utf-8", "replace"))
        metadata = header.get("__metadata__") or {}
        return json.loads(metadata.get("config_json") or "{}")

    @staticmethod
    def _read_hf_config(source):
        """HF 上のヘッダだけを Range で読む（数GBの重みは取得しない）。"""
        from huggingface_hub import hf_hub_url
        from urllib.request import Request, urlopen

        repo_id, subfolder = EditorAdapter._split_hf_source(source)
        filename = "model.safetensors" if subfolder is None else f"{subfolder}/model.safetensors"
        url = hf_hub_url(repo_id=repo_id, filename=filename)

        def fetch(start, end=None):
            header = {"Range": f"bytes={start}-{end}" if end is not None else f"bytes={start}-"}
            with urlopen(Request(url, headers=header), timeout=5) as response:
                return response.read()

        header_len = struct.unpack("<Q", fetch(0, 7))[0]
        if not 0 < header_len <= 64 * 1024 * 1024:
            return {}
        header = json.loads(fetch(8, 8 + header_len - 1).decode("utf-8", "replace"))
        metadata = header.get("__metadata__") or {}
        return json.loads(metadata.get("config_json") or "{}")

    def model_info(self, source=None, refresh=False):
        """選択中モデルの種類と既定ステップ数（MeanFlow は4ステップ）。"""
        with self.state_lock:
            model = str(self.settings.get("model") if source is None else source).strip()
            cached = self._model_info_cache.get(model)
        now = time.time()
        if cached is not None and not refresh:
            info, stamped = cached
            # 判定できなかった結果は短時間だけ使い回し、次回のポーリングで取り直す。
            if info.get("metadataAvailable") or now - stamped < 60:
                return info
        config = {}
        resolved = None
        try:
            local = self._resolve_local_model(model)
            if local is None and self._is_hf_model_source(model):
                local = self._hf_checkpoint_from_cache(model)
                if local is None:
                    config = self._read_hf_config(model)
            if local is not None:
                resolved = str(local)
                config = self._read_safetensors_config(local)
            elif not config and not self._is_hf_model_source(model):
                raise ValueError("unrecognized model source")
        except Exception as exc:
            self._runtime_log(f"model metadata unavailable: {exc}")
            config = {}
        flow = str(config.get("flow_parameterization") or "rf_velocity").strip().lower()
        info = dict(
            source=model,
            kind="hf" if self._is_hf_model_source(model) else "local",
            resolved=resolved,
            downloaded=bool(resolved),
            flowParameterization=flow,
            meanflow=flow == "meanflow",
            defaultSteps=DEFAULT_STEPS_MEANFLOW if flow == "meanflow" else DEFAULT_STEPS_RF,
            metadataAvailable=bool(config),
        )
        known = {
            "Aratako/Irodori-TTS-v4.1-Small": ("MIT", "https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small"),
            "Aratako/Irodori-TTS-v4.1-Small-MF": ("MIT", "https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-MF"),
            "phasefield-audio/Irodori-TTS-v4.1-Anime": ("MIT", "https://huggingface.co/phasefield-audio/Irodori-TTS-v4.1-Anime"),
        }
        license_name, license_url = known.get(model, (None, f"https://huggingface.co/{model}" if self._is_hf_model_source(model) else None))
        info["license"] = license_name
        info["licenseUrl"] = license_url
        with self.state_lock:
            self._model_info_cache[model] = (info, now)
        return info

    def _default_steps(self):
        try:
            return int(self.model_info().get("defaultSteps") or DEFAULT_STEPS_RF)
        except Exception:
            return DEFAULT_STEPS_RF

    def validate(self, value):
        if value["backend"] not in ("cpu", "cuda", "trt", "radeon"):
            raise ValueError("未対応のエンジンです")
        if not self.available_backends()[value["backend"]]:
            raise ValueError(f"{value['backend']} はこのPCでは利用できません")
        model = str(value.get("model", "")).strip()
        self._validate_model_source(model)
        if not isinstance(value["seed"], int) or not 0 <= value["seed"] < 2**31:
            raise ValueError("seedは0〜2147483647です")
        try:
            sway_coeff = float(value.get("sway_coeff", -1.0))
        except (TypeError, ValueError) as exc:
            raise ValueError("sway_coeff must be a finite number") from exc
        if not math.isfinite(sway_coeff):
            raise ValueError("sway_coeff must be a finite number")

    @staticmethod
    def _line_cfg(query, key, default):
        raw = query.get(key, default)
        try:
            value = float(default if raw is None else raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be a number") from exc
        if not math.isfinite(value) or not 0.0 <= value <= 20.0:
            raise ValueError(f"{key} must be between 0 and 20")
        return value

    def _line_steps(self, query):
        """セリフごとのステップ数。未指定ならモデルの既定（MeanFlowは4）。"""
        raw = query.get("irodori_steps", self._default_steps())
        if isinstance(raw, bool) or not isinstance(raw, int) or not 1 <= raw <= 80:
            raise ValueError("irodori_steps must be an integer between 1 and 80")
        return raw

    @staticmethod
    def _line_schedule(query):
        value = query.get("irodori_schedule", "sway")
        if value not in ("linear", "sway"):
            raise ValueError("irodori_schedule must be linear or sway")
        return value

    @staticmethod
    def _line_seconds(query):
        raw = query.get("irodori_seconds")
        if raw is None:
            return None
        if isinstance(raw, bool):
            raise ValueError("irodori_seconds must be a number or null")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("irodori_seconds must be a number or null") from exc
        if not math.isfinite(value) or not 0.1 <= value <= 60:
            raise ValueError("irodori_seconds must be between 0.1 and 60")
        return value

    def _set_progress(self, stage, percent, active=True):
        with self.progress_lock:
            percent = min(100, int(percent))
            if percent != 0:
                percent = max(self.progress["percent"], percent)
            self._progress_percent = percent
            self.progress = dict(active=active, percent=percent, stage=str(stage))

    def _progress_event(self, message, percent=None):
        if percent is not None:
            self._runtime_log(message)
            self._set_progress(message, percent)
            return
        self._runtime_log(message)

    def _runtime_log(self, message):
        message = str(message).strip()
        if not message:
            return
        if not hasattr(self, "logs"):
            self.logs = deque(maxlen=80)
        if not hasattr(self, "_phase_marks"):
            self._phase_marks = []
        phase_percent = {
            "start synthesize": 76,
            "prepare_lora": 77,
            "tokenize_text": 80,
            "prepare_reference": 82,
            "predict_duration": 85,
            "sample_rf": 90,
            "unpatchify_latent": 93,
            "decode_latent": 96,
            "silentcipher_watermark": 98,
            "done synthesize": 99,
        }
        stamped = time.perf_counter()
        matched = next((phase for phase in phase_percent if phase in message), None)
        if matched is None and "total_to_decode" in message:
            matched = "total_to_decode"
        with self.progress_lock:
            self.logs.append(message)
            current = self.progress["percent"]
            self._phase_marks.append((matched, stamped))
        percent = next((value for phase, value in phase_percent.items() if phase in message), current)
        print(f"[irodori] {time.strftime('%H:%M:%S')} stage: {message}", flush=True)
        self._set_progress(message, percent)

    def _report_synthesis(self, speaker_name, query, wav, started):
        """生成の計測値を1行にまとめる（デバッグ時はコンソールにも流れる）。"""
        elapsed = max(0.0, time.perf_counter() - started)
        audio_seconds = _wav_seconds(wav)
        with self.progress_lock:
            marks = list(self._phase_marks)
        phases = []
        previous = started
        for label, stamped in marks:
            if label is None:
                continue
            phases.append(f"{label}={max(0.0, stamped - previous):.1f}s")
            previous = stamped
        text = str(query.get("irodori_text") or query.get("kana") or "")
        if wav:
            self._cache_wav(text, speaker_name, wav)
        with self.state_lock:
            backend = self.settings.get("backend")
            model = self.settings.get("model")
        audio_text = f"{audio_seconds:.2f}s" if audio_seconds else "unknown"
        rtf = f"{elapsed / audio_seconds:.2f}x" if audio_seconds else "-"
        speaker_label = speaker_name or "話者なし"
        parts = [
            f"total={elapsed:.1f}s",
            f"audio={audio_text}",
            f"rtf={rtf}",
            f"wav={len(wav)}B",
            f"speaker={speaker_label}",
            f"text={len(text)}chars",
            f"steps={query.get('irodori_steps')}",
            f"schedule={query.get('irodori_schedule')}",
            f"seconds={query.get('irodori_seconds')}",
            f"seed={query.get('irodori_seed')}",
            f"cfg={query.get('irodori_cfg_text')}/{query.get('irodori_cfg_caption')}/{query.get('irodori_cfg_speaker')}",
            f"backend={backend}",
            f"model={model}",
        ]
        line = "synth done: " + " ".join(parts)
        if phases:
            line += " | phases: " + " ".join(phases)
        print(f"[irodori] {line}", flush=True)
        with self.progress_lock:
            self.logs.append(line)

    # ------------------------------------------------------------------
    # 口パク用タイムライン（ASR）
    # ------------------------------------------------------------------
    @staticmethod
    def _wav_key(text):
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def _cache_wav(self, text, speaker_name, wav):
        """直近に生成した音声を、セリフをキーにして保持する。"""
        if not text or not wav:
            return
        entry = (self._wav_key(text), text, speaker_name, bytes(wav))
        with self.state_lock:
            for existing in list(self._wav_cache):
                if existing[0] == entry[0]:
                    self._wav_cache.remove(existing)
            self._wav_cache.append(entry)

    def _cached_wav(self, text):
        key = self._wav_key(text)
        with self.state_lock:
            for entry in reversed(self._wav_cache):
                if entry[0] == key:
                    return entry
        return None

    def asr_timeline(self, value):
        """セリフの文字ごとの発話時刻を返す。ASRモデルが無いときは available=False。"""
        if not isinstance(value, dict):
            raise ValueError("body must be an object")
        text = str(value.get("text") or "").strip()
        if not text:
            raise ValueError("text is required")
        if MAX_TEXT_CHARS and len(text) > MAX_TEXT_CHARS:
            raise ValueError(f"text is too long: {len(text)} > {MAX_TEXT_CHARS}")
        if not self.timeline_reader.available:
            # 初回利用時にモデルを取得する。待ち切れなくても取得は続くので、
            # 次の要求（もう一度読み込む等）で揃っていれば使える。
            if os.environ.get("IRODORI_ASR_AUTO_DOWNLOAD", "1") != "0":
                state = ensure_asr_model(progress=self._set_progress, log=self._runtime_log,
                                         wait_seconds=ASR_DOWNLOAD_WAIT_SECONDS)
                if not state["ready"]:
                    if state["pending"]:
                        reason = "ASR model is downloading"
                        self._runtime_log("asr timeline: waiting for the ASR model download")
                    else:
                        reason = f"ASR model is not installed ({state['error'] or 'download failed'})"
                    return {"available": False, "reason": reason, "downloading": state["pending"],
                            "modelFolder": str(ASR_MODEL_DIR), "text": text}
            else:
                return {"available": False, "reason": "ASR model is not installed",
                        "modelFolder": str(ASR_MODEL_DIR), "text": text}
        payload = value.get("wav")
        speaker_name = value.get("speaker")
        if payload:
            raw = base64.b64decode(payload)
        else:
            cached = self._cached_wav(text)
            if cached is None:
                raise KeyError("no cached audio for this text; generate the line first")
            speaker_name = speaker_name or cached[2]
            raw = cached[3]
        samples = decode_wav(raw)
        try:
            result = self.timeline_reader.anchors(text, samples)
        except ImportError as exc:
            raise ValueError(asr_package_error()) from exc
        result["available"] = True
        result["speaker"] = speaker_name
        result["source"] = "cached" if not payload else "request"
        print(f"[irodori] asr timeline: {len(result['anchors'])} chars "
              f"audio={result['audioSeconds']}s asr={result['asrSeconds']}s "
              f"text={text[:24]!r}", flush=True)
        with self.progress_lock:
            self.logs.append(
                f"asr timeline: chars={len(result['anchors'])} "
                f"audio={result['audioSeconds']}s asr={result['asrSeconds']}s")
        return result

    def status(self):
        available = self.available_backends()
        with self.state_lock:
            if not available.get(self.settings["backend"], False):
                self.settings["backend"] = "cpu"
            settings = {**self.DEFAULT_SETTINGS, **self.settings}
            loaded = self.delegate is not None
        with self.progress_lock:
            progress = dict(self.progress)
            logs = list(self.logs)
        return dict(settings=settings, models=self.models(), loaded=loaded,
                    asr=self.timeline_reader.status(),
                    modelInfo=self.model_info(settings.get("model", "")),
                    availableBackends=available, progress=progress, logs=logs,
                    modelFolder=str(MODEL_DIR), speakerFolder=str(SPEAKER_DIR))

    def configure(self, value):
        value = {**self.DEFAULT_SETTINGS, **value}
        value.pop("caption", None)
        value.pop("cfg", None)
        # 旧クライアントから送られる共通値も保存しない。
        value.pop("steps", None)
        value.pop("t_schedule_mode", None)
        value.pop("seconds", None)
        self.validate(value)
        changed_backend = False
        retry_trt = False
        with self.operation_lock:
            tmp = self.config_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.config_path)
            with self.state_lock:
                changed_backend = any(value[k] != self.settings[k] for k in ("backend", "model"))
            if changed_backend:
                self._close_locked()
            with self.state_lock:
                self.settings = value
            retry_trt = value['backend'] == 'trt' and getattr(self, '_trt_failure', None) is not None
            self._trt_failure = None
        if changed_backend or retry_trt:
            self.schedule_prewarm()
        return self.status()

    def _build_delegate(self, prewarm=False):
        key = (self.settings['backend'], str(self.settings['model']).strip())
        failure = getattr(self, '_trt_failure', None)
        if key[0] == 'trt' and failure and failure[0] == key:
            raise RuntimeError(f'{failure[1]}\n自動再試行は停止しています。原因を解消後に「設定を適用」で再試行するか、CUDA を選んでください')
        try:
            return self._build_delegate_impl(prewarm)
        except Exception as exc:
            if key[0] == 'trt':
                self._trt_failure = (key, str(exc))
            raise

    def _build_delegate_impl(self, prewarm=False):
        self._set_progress("preloading model/backend" if prewarm else "initializing model/backend", 15)
        model_source = str(self.settings["model"]).strip()
        local_model = self._resolve_local_model(model_source)
        if local_model is None and self._is_hf_model_source(model_source):
            # HF の重みは初回だけ取得する（キャッシュ済みなら即返る）。
            local_model = self._hf_checkpoint_from_cache(model_source)
            if local_model is None:
                local_model = self._download_hf_model(model_source)
        elif local_model is None:
            raise ValueError(
            "モデルは .safetensors のパスか Hugging Face の repo_id を指定してください")
        os.environ["IRODORI_CHECKPOINT"] = str(local_model)
        # 判定結果をローカルパス付きで更新しておく（次回のポーリング用）。
        # メタデータの更新は付随処理なので、失敗しても読み込みは続ける。
        try:
            self.model_info(model_source, refresh=True)
        except Exception as exc:  # noqa: BLE001
            self._runtime_log(f"model metadata refresh skipped: {exc}")
        plan = self._plan_path()
        if self.settings['backend'] == 'trt':
            from trt_cache import ensure_plan
            plan = ensure_plan(local_model, self._set_progress, self._runtime_log)
        return VoicevoxAdapter(self.settings["backend"], 50125, resolve_embed_dirs(),
                               plan, self._progress_event)

    def schedule_prewarm(self):
        with self._prewarm_lock:
            if self._prewarm_scheduled:
                return False
            self._prewarm_scheduled = True
        # Reset completed synthesis progress before background work reports its phases.
        self._set_progress("preloading model/backend", 0)
        thread = threading.Thread(target=self._prewarm, name="irodori-model-prewarm", daemon=True)
        thread.start()
        return True

    def _prewarm(self):
        try:
            with self.operation_lock:
                with self.state_lock:
                    delegate = self.delegate
                if delegate is None:
                    delegate = self._build_delegate(prewarm=True)
                    with self.state_lock:
                        self.delegate = delegate
                self._set_progress("complete", 100, active=False)
        except Exception as exc:
            self._runtime_log(f"preload failed: {exc}")
            self._set_progress("preload failed", 100, active=False)
        finally:
            with self._prewarm_lock:
                self._prewarm_scheduled = False

    def refresh(self):
        with self.operation_lock:
            cassette = SpeakerCassette(resolve_embed_dirs())
            names = cassette.speakers
            names = [n for n in names if not (n.endswith(".speaker") and n[:-8] in names)]
            id_to_name = {0: ""}
            speakers_json = []
            catalog = dict(speaker_catalog(cassette.dirs))
            fallback = _fallback_icon()
            fallback_payload = fallback[1] if fallback else TINY_PNG
            for name in [""] + names:
                uid = uuid.uuid5(ENGINE_UUID_NAMESPACE, name or "no-speaker")
                sid = (uid.int % (2**31 - 1)) if name else 0
                if sid in id_to_name and name:
                    raise ValueError("話者IDが衝突しました。話者ファイルの名前を変更してください")
                id_to_name[sid] = name
                label = display_name_for(name) if name else "話者なし"
                image = catalog.get(name)
                encoded = image[1] if image else fallback_payload
                source = cassette.path_for(name) if name else None
                portrait = portrait_for(source) if source else None
                mouth = mouth_open_thumbnail_for(source) if source else None
                blink = blink_thumbnail_for(source) if source else None
                mouth_parts = mouth_parts_for(source) if source else None
                credit = credit_for(source) if source else None
                policy = policy_for(source) if source else None
                folder = cassette.folder_for(name) if name else None
                speakers_json.append(dict(name=label, speaker_uuid=str(uid),
                    styles=[dict(name="ノーマル", id=sid, type="talk")], version="0.2.0",
                    icon=encoded, portrait=portrait[1] if portrait else encoded,
                    mouth_open=mouth[1] if mouth else None,
                    blink=blink[1] if blink else None,
                    mouth_parts=mouth_parts, credit=credit, policy=policy,
                    irodori_folder=folder))
            # エディタはここで話者一覧を独自に組み立てる。話者IDは保ったまま
            # 表示順だけを入れ替え、初回の既定話者をつくよみちゃんにする。
            speakers_json.sort(
                key=lambda speaker: speaker["name"]
                != display_name_for(DEFAULT_SPEAKER_NAME)
            )
            with self.state_lock:
                self.id_to_name = id_to_name
                self.speakers_json = speakers_json
                delegate = self.delegate
            if delegate:
                delegate.refresh()

    def mix_speakers(self):
        cassette = SpeakerCassette(resolve_embed_dirs())
        return [dict(id=name, name=display_name_for(name)) for name in cassette.speakers
                if not (name.endswith(".speaker") and name[:-8] in cassette.speakers)]

    def mix_recipes(self):
        recipes = []
        if not SPEAKER_DIR.exists():
            return recipes
        for path in sorted(SPEAKER_DIR.rglob("*.mix.json")):
            if len(recipes) >= 100:
                break
            if path.stat().st_size > 128_000:
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if value.get("version") != 1 or not isinstance(value.get("tokens"), list) or len(value["tokens"]) != 16:
                    continue
                recipes.append({"name": path.name[:-len(".mix.json")], "tokens": value["tokens"]})
            except (OSError, UnicodeError, ValueError, TypeError, AttributeError):
                continue
        return recipes

    def _mix_tensor(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("mix request must be an object")
        return compose_speaker_mix(payload.get("tokens"), SpeakerCassette(resolve_embed_dirs()))

    def mix_preview(self, payload):
        text = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_CHARS:
            raise ValueError(f"text must contain 1 to {MAX_TEXT_CHARS} characters")
        tensor = self._mix_tensor(payload)
        with self.operation_lock:
            if self.delegate is None:
                self.delegate = self._build_delegate()
            with self.delegate.lock:
                with tempfile.TemporaryDirectory(prefix="irodori-mix-", dir=str(ROOT / "wrapper")) as temp:
                    output = Path(temp) / "preview.wav"
                    self.delegate.tts.synthesize(
                        text=text.strip(), speaker="", out_wav=output,
                        seed=int(self.settings["seed"]),
                        num_steps=self.delegate.default_steps,
                        cfg_scale_text=3.0, cfg_scale_speaker=5.0,
                        cfg_scale_caption=3.0,
                        speaker_tensor_override=tensor,
                    )
                    return output.read_bytes()

    def mix_save(self, payload):
        name = payload.get("name") if isinstance(payload, dict) else None
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", name):
            raise ValueError("name must be 1-48 ASCII letters, digits, _ or -")
        tensor = self._mix_tensor(payload)
        if not tensor.any():
            raise ValueError("at least one token must contain a speaker")
        from safetensors.torch import save_file
        folder = SPEAKER_DIR / name
        target = folder / f"{name}.speaker.safetensors"
        if target.exists():
            raise ValueError("speaker name already exists")
        folder.mkdir(parents=True, exist_ok=True)
        save_file({"speaker_embedding": tensor.contiguous()}, str(target))
        (folder / f"{name}.mix.json").write_text(
            json.dumps({"version": 1, "tokens": payload["tokens"]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.refresh()
        return {"name": name}

    def synthesize(self, query, speaker_id):
        try:
            with self.operation_lock:
                with self.progress_lock:
                    if not hasattr(self, "logs"):
                        self.logs = deque(maxlen=80)
                    self.logs.clear()
                    self._phase_marks = []
                self._set_progress("validating synthesis", 0)
                return self._synthesize(query, speaker_id)
        except Exception as exc:
            # DirectML\u30ef\u30fc\u30ab\u30fc\u304c\u843d\u3061\u305f\u3068\u304d\u306f\u30c7\u30ea\u30b2\u30fc\u30c8\u3092\u6368\u3066\u3066\u3001\u6b21\u306e\u751f\u6210\u3067
            # \u4f5c\u308a\u76f4\u3059\u3002\u6368\u3066\u306a\u3044\u3068\u518d\u8d77\u52d5\u3059\u308b\u307e\u3067\u5168\u8a71\u8005\u304c500\u306e\u307e\u307e\u306b\u306a\u308b\u3002
            if self._is_worker_failure(exc):
                self._runtime_log("radeon worker stopped; rebuilding on the next request")
                self._close_locked()
            self._set_progress("error", 100, active=False)
            raise

    @staticmethod
    def _is_worker_failure(exc):
        text = str(exc)
        markers = (
            "Radeon worker pipe failed",
            "Radeon worker failed during startup",
            "Radeon worker pipe closed",
            "Radeon worker checkpoint identity",
            "Radeon worker precision mismatch",
            "NoneType' object has no attribute 'stdin'",
            "UnicodeDecodeError",
        )
        return any(marker in text for marker in markers)

    def _synthesize(self, query, speaker_id):
        self.validate(self.settings)
        if True:
            if speaker_id not in self.id_to_name:
                raise ValueError("話者が見つかりません。話者一覧を更新してください")
            if self.delegate is None:
                try:
                    self.delegate = self._build_delegate()
                except Exception:
                    self._set_progress("error", 100, active=False)
                    raise
            self._set_progress("generating audio", 76)
            name = self.id_to_name[speaker_id] or "話者なし"
            sid = self.delegate.name_to_id.get(name)
            if sid is None:
                raise ValueError("話者一覧を更新してください")
            line_steps = self._line_steps(query)
            line_schedule = self._line_schedule(query)
            line_seconds = self._line_seconds(query)
            query.update(irodori_steps=line_steps,
                         irodori_seconds=line_seconds,
                         irodori_cfg_text=self._line_cfg(query, "irodori_cfg_text", 3.0),
                         irodori_cfg_speaker=self._line_cfg(query, "irodori_cfg_speaker", 5.0),
                         irodori_cfg_caption=self._line_cfg(query, "irodori_cfg_caption", 3.0),
                         irodori_schedule=line_schedule,
                         irodori_sway_coeff=float(self.settings["sway_coeff"]))
            if "irodori_seed" not in query:
                query["irodori_seed"] = self.settings["seed"]
            started = time.perf_counter()
            try:
                result = self.delegate.synthesize(query, sid)
            except Exception:
                self._set_progress("error", 100, active=False)
                raise
            self._report_synthesis(name, query, result, started)
        self._set_progress("complete", 100, active=False)
        return result

    def close(self):
        with self.operation_lock:
            self._close_locked()

    def _close_locked(self):
        with self.state_lock:
            delegate = self.delegate
            self.delegate = None
        if delegate:
            delegate.tts.close()
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class EditorHandler(Handler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/irodori/mix/speakers":
            if not self._authorized():
                return self._json(403, {"detail": "invalid local origin/session"})
            return self._json(200, self.adapter.mix_speakers())
        if path == "/irodori/mix/recipes":
            if not self._authorized():
                return self._json(403, {"detail": "invalid local origin/session"})
            return self._json(200, self.adapter.mix_recipes())
        if path == "/irodori/settings":
            if not self._authorized():
                return self._json(403, {"detail": "invalid local origin/session"})
            self._json(200, self.adapter.status())
        elif path == "/irodori/timeline" and self._authorized():
            # GET でも使えるように（デバッグ用に ?text= と ?wav= は無し）
            try:
                self._json(200, self.adapter.asr_timeline({"text": parse_qs(urlparse(self.path).query).get("text", [""])[0]}))
            except KeyError as exc:
                self._json(409, {"detail": str(exc)})
            except ValueError as exc:
                self._json(400, {"detail": str(exc)})
            except Exception as exc:
                self._json(500, {"detail": str(exc)})
        else:
            super().do_GET()

    def do_POST(self):
        if not self._authorized():
            return self._json(403, {"detail": "invalid local origin/session"})
        if urlparse(self.path).path in ("/irodori/mix/preview", "/irodori/mix/save"):
            try:
                payload = self._body_json()
                if self.path.endswith("/preview"):
                    return self._send(200, self.adapter.mix_preview(payload), "audio/wav")
                return self._json(200, self.adapter.mix_save(payload))
            except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
                return self._json(400, {"detail": str(exc)})
            except Exception as exc:
                return self._json(500, {"detail": str(exc)})
        if urlparse(self.path).path == "/irodori/timeline":
            try:
                return self._json(200, self.adapter.asr_timeline(self._body_json()))
            except KeyError as exc:
                return self._json(409, {"detail": str(exc)})
            except ValueError as exc:
                return self._json(400, {"detail": str(exc)})
            except Exception as exc:
                return self._json(500, {"detail": str(exc)})
        if urlparse(self.path).path == "/irodori/shutdown":
            # The server is bound to loopback, but keep this endpoint explicitly
            # local-only in case the bind configuration is changed later.
            if self.client_address[0] not in ("127.0.0.1", "::1", "localhost"):
                return self._json(403, {"detail": "shutdown is local-only"})
            self._json(200, {"shutdown": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        if urlparse(self.path).path in ("/irodori/open-models", "/irodori/open-speakers"):
            folder = MODEL_DIR if self.path.endswith("open-models") else SPEAKER_DIR
            try:
                folder.mkdir(exist_ok=True)
                os.startfile(str(folder))
                return self._json(200, {"opened": str(folder)})
            except OSError as exc:
                return self._json(500, {"detail": str(exc)})
        if urlparse(self.path).path == "/audio_query":
            text = parse_qs(urlparse(self.path).query).get("text", [""])[0]
            return self._json(200, _query(text))
        if urlparse(self.path).path != "/irodori/settings":
            return super().do_POST()
        try:
            self._json(200, self.adapter.configure(self._body_json()))
        except (ValueError, KeyError, TypeError) as exc:
            self._json(400, {"detail": str(exc)})
        except Exception as exc:
            self._json(500, {"detail": str(exc)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1", "localhost"])
    parser.add_argument("--port", default=50125, type=int)
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()
    for key, folder in dict(IRODORI_RUNTIME_DIR="runtime", IRODORI_HF_HOME=".cache/huggingface",
                            IRODORI_CACHE_DIR=".cache/irodori-tts-cache").items():
        os.environ.setdefault(key, str(REPO_ROOT / folder))
    os.environ.setdefault("IRODORI_EMBED_DIR", str(SPEAKER_DIR))
    os.environ.pop("IRODORI_BACKEND", None)
    EditorHandler.adapter = EditorAdapter()
    server = ThreadingHTTPServer((args.host, args.port), EditorHandler)
    if os.environ.get("IRODORI_NO_PREWARM") == "1":
        print("[irodori] prewarm skipped (IRODORI_NO_PREWARM=1)", flush=True)
    else:
        EditorHandler.adapter.schedule_prewarm()
    try:
        server.serve_forever()
    finally:
        server.server_close()
        EditorHandler.adapter.close()
