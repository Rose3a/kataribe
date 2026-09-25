"""VOICEVOX Engine compatible HTTP adapter for Irodori-TTS.

The official VOICEVOX editor talks to an engine through a small HTTP API.
This adapter exposes the endpoints the editor needs while keeping Irodori's
speaker cassette and backend selection in the resident process.

Run one process per backend/port, for example::

    python wrapper/voicevox_engine.py --backend cpu --port 50021
    python wrapper/voicevox_engine.py --backend cuda --port 50022
    python wrapper/voicevox_engine.py --backend trt --port 50023

Radeon uses the resident DirectML worker when the benchmark environment and
ONNX codec are available on this machine.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import hashlib
import math
import mimetypes
import os
import re
import sys
import tempfile
import threading
import uuid
import base64
import binascii
import secrets
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tts_cli import IrodoriTTS, resolve_embed_dirs  # noqa: E402
from reading_dictionary import ENGLISH_READINGS, KANA_STYLES, READING_DICTIONARY, make_word
from third_party_licenses import dependency_licenses
from speaker_catalog import blink_thumbnail_for, credit_for, display_name_for, policy_for, mouth_open_thumbnail_for, mouth_parts_for, portrait_for, speaker_catalog, _fallback_icon  # noqa: E402


ENGINE_VERSION = "0.1.1"
ENGINE_UUID_NAMESPACE = uuid.UUID("1d9f9d29-5a2a-4a4d-81e9-bb5f78a89d4b")
DEFAULT_SPEAKER_NAME = "tsukuyomi"
MAX_BODY_BYTES = 16 * 1024 * 1024
# 30秒前後の日本語音声（目安180〜240文字）を収めつつ、
# DirectMLワーカーのメモリ使用量が急増する長文を防ぐ。
MAX_TEXT_CHARS = 256
SESSION_TOKEN = secrets.token_urlsafe(32)
ALLOWED_ORIGINS = frozenset({
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "app://.",
    "file://",
})
# A valid 1x1 PNG keeps the official editor's character cards lightweight.
TINY_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAF"
    "gAI/7jS7WQAAAABJRU5ErkJggg=="
)


def _speaker_list_payload(speakers: list[dict]) -> list[dict]:
    """Build the lightweight ``/speakers`` response.

    Portraits and icons belong to ``/speaker_info``.  Sending them in this
    list made the initial response grow by one base64 image per speaker (and
    then sent the same data again from ``/speaker_info``), which is prohibitive
    for large local speaker collections.
    """
    fields = (
        "name",
        "speaker_uuid",
        "styles",
        "version",
        "supported_features",
        "irodori_folder",
    )
    return [
        {field: speaker[field] for field in fields if field in speaker}
        for speaker in speakers
    ]


def _speaker_resource_index(speakers: list[dict]) -> dict[str, str]:
    """Index unique base64 resources once when the speaker catalog is built."""
    resources: dict[str, str] = {}
    digests_by_value: dict[str, str] = {}
    for speaker in speakers:
        values = [
            speaker.get("icon"),
            speaker.get("portrait"),
            speaker.get("mouth_open"),
            speaker.get("blink"),
        ]
        mouth_parts = speaker.get("mouth_parts")
        if isinstance(mouth_parts, dict):
            values.extend(mouth_parts.values())
        for value in values:
            if not isinstance(value, str):
                continue
            digest = digests_by_value.get(value)
            if digest is None:
                digest = hashlib.sha256(value.encode("ascii")).hexdigest()
                digests_by_value[value] = digest
            resources.setdefault(digest, value)
    return resources


def _speaker_uuid_index(speakers: list[dict]) -> dict[str, dict]:
    """Index speaker records so /speaker_info stays fast for large libraries."""
    index = {}
    for speaker in speakers:
        speaker_uuid = speaker.get("speaker_uuid")
        if isinstance(speaker_uuid, str):
            index.setdefault(speaker_uuid, speaker)
    return index


class RequestBodyError(ValueError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class RequestValidationError(ValueError):
    """Invalid request parameters, reported like VOICEVOX (FastAPI) as HTTP 422.

    ``errors`` uses the FastAPI shape ``{"type", "loc", "msg", "input"}`` so that
    VOICEVOX clients that already understand 422 responses can read it as is.
    """

    def __init__(self, errors: list[dict]):
        super().__init__("; ".join(error["msg"] for error in errors))
        self.errors = errors


def _field_error(loc: list, msg: str, kind: str, value=None) -> dict:
    return {"type": kind, "loc": loc, "msg": msg, "input": value}


# 話速。0 などを受け付けると、何も言わずに極端に長い音声が返るため範囲を決める。
SPEED_SCALE_MIN = 0.25
SPEED_SCALE_MAX = 4.0

_STRENGTH = {"type": "number", "minimum": 0.0, "maximum": 1.0}

# 音声クエリで受け付ける Irodori 拡張フィールド。入力チェック（_validate_query）と
# /openapi.json の両方がこの表を参照するので、仕様と実装がずれない。
IRODORI_QUERY_FIELDS: dict[str, dict] = {
    "irodori_text": {
        "type": "string",
        "description": "元の文章。合成時に読み辞書を再適用する。省略時は kana を使う",
    },
    "irodori_seed": {
        "type": "integer", "nullable": True, "default": 4763674,
        "description": "シード。null でランダム",
    },
    "irodori_steps": {
        "type": "integer", "minimum": 1, "maximum": 80,
        "description": "ステップ数。省略時はモデル既定（RF は8、MeanFlow は4）",
    },
    "irodori_schedule": {
        "type": "string", "enum": ["sway", "linear"], "default": "sway",
        "description": "ステップの刻み方。MeanFlow モデルでは無視される",
    },
    "irodori_seconds": {
        "type": "number", "nullable": True, "minimum": 0.1, "maximum": 60.0,
        "description": "音声長（秒）。null で自動",
    },
    "irodori_caption": {
        "type": "string", "nullable": True, "maxLength": 2000,
        "description": "場面・話し方・感情などの指示",
    },
    "irodori_caption_strength": {**_STRENGTH, "default": 1.0},
    "irodori_reference_strength": {**_STRENGTH, "default": 1.0},
    "irodori_speaker_strength": {**_STRENGTH, "default": 1.0},
    "irodori_cfg_text": {
        "type": "number", "minimum": 0.0, "maximum": 20.0, "default": 3.0,
        "description": "MeanFlow モデルでは無視される",
    },
    "irodori_cfg_caption": {
        "type": "number", "minimum": 0.0, "maximum": 20.0, "default": 3.0,
        "description": "MeanFlow モデルでは無視される",
    },
    "irodori_cfg_speaker": {
        "type": "number", "minimum": 0.0, "maximum": 20.0, "default": 5.0,
        "description": "MeanFlow モデルでは無視される",
    },
    "irodori_additional_speakers": {
        "type": "array", "nullable": True, "maxItems": 3,
        "items": {
            "type": "object", "required": ["style_id"],
            "properties": {
                "style_id": {"type": "integer"},
                "strength": {**_STRENGTH, "default": 0.5},
            },
        },
        "description": "混ぜる話者（最大3人）。音声参照とは同時に使えない",
    },
    "irodori_reference_audio": {
        "type": "object", "nullable": True, "required": ["dataUrl"],
        "properties": {
            "dataUrl": {"type": "string", "description": "data:audio/...;base64,...（最大10MB）"},
            "mime": {"type": "string"},
            "name": {"type": "string"},
        },
        "description": "話者・声質の参考音声",
    },
    "irodori_sway_coeff": {
        "type": "number",
        "description": "エディタ経由では共通設定の値が優先される",
    },
    "irodori_english_reading": {
        "type": "string", "enum": list(ENGLISH_READINGS), "default": "katakana",
        "description": "英単語・英文の読み。off は変換しない、katakana / hiragana はその表記に変換してから合成する。エディタ経由では共通設定の値が優先される",
    },
    "irodori_kana_style": {
        "type": "string", "enum": list(KANA_STYLES), "default": "katakana",
        "description": "hiragana なら文中のカタカナをひらがなにして読ませる。エディタ経由では共通設定の値が優先される",
    },
    "irodori_secondary_speaker_style_id": {
        "type": "integer", "nullable": True, "deprecated": True,
        "description": "旧形式。irodori_additional_speakers を使う",
    },
    "irodori_secondary_speaker_strength": {**_STRENGTH, "deprecated": True},
}


def _snake_case(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _check_value(value, schema: dict, loc: list) -> list[dict]:
    """Check one value against a small subset of JSON Schema used above."""
    if value is None:
        return [] if schema.get("nullable") else [
            _field_error(loc, "null は指定できません", "none_forbidden", value)]
    kind = schema.get("type")
    if kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            return [_field_error(loc, "整数を指定してください", "int_type", value)]
    elif kind == "number":
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            return [_field_error(loc, "数値を指定してください", "float_type", value)]
    elif kind == "string":
        if not isinstance(value, str):
            return [_field_error(loc, "文字列を指定してください", "string_type", value)]
        if "enum" in schema and value not in schema["enum"]:
            return [_field_error(
                loc, f"{' / '.join(schema['enum'])} のいずれかを指定してください", "enum", value)]
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            return [_field_error(
                loc, f"{schema['maxLength']}文字以内にしてください", "string_too_long", None)]
    elif kind == "array":
        if not isinstance(value, list):
            return [_field_error(loc, "配列を指定してください", "list_type", value)]
        if len(value) > schema.get("maxItems", len(value)):
            return [_field_error(
                loc, f"最大{schema['maxItems']}件までです", "too_long", None)]
        errors = []
        for index, item in enumerate(value):
            errors += _check_value(item, schema["items"], [*loc, index])
        return errors
    elif kind == "object":
        if not isinstance(value, dict):
            return [_field_error(loc, "オブジェクトを指定してください", "dict_type", value)]
        errors = [
            _field_error([*loc, name], "必須です", "missing")
            for name in schema.get("required", []) if name not in value
        ]
        for name, child in schema.get("properties", {}).items():
            if name in value:
                errors += _check_value(value[name], child, [*loc, name])
        return errors
    if "minimum" in schema and not schema["minimum"] <= value <= schema["maximum"]:
        return [_field_error(
            loc, f"{schema['minimum']}〜{schema['maximum']}の範囲で指定してください",
            "range", value)]
    return []


def _validate_query(query: dict) -> None:
    """Reject invalid Irodori fields before synthesis instead of ignoring them.

    VOICEVOX clients never send ``irodori*`` keys, so unknown keys with that
    prefix are almost always a typo or a camelCase name, which would otherwise
    silently fall back to defaults.
    """
    errors = []
    for key, value in query.items():
        if not key.lower().startswith("irodori"):
            continue
        schema = IRODORI_QUERY_FIELDS.get(key)
        if schema is None:
            suggestion = _snake_case(key)
            hint = (f"。{suggestion} のことですか？"
                    if suggestion in IRODORI_QUERY_FIELDS else "")
            errors.append(_field_error(
                ["body", key], f"未知のフィールドです{hint}", "extra_forbidden", value))
            continue
        errors += _check_value(value, schema, ["body", key])
    errors += _check_value(
        query.get("speedScale", 1.0),
        {"type": "number", "minimum": SPEED_SCALE_MIN, "maximum": SPEED_SCALE_MAX},
        ["body", "speedScale"])
    if errors:
        raise RequestValidationError(errors)


def _int_query_param(params: dict, name: str) -> int:
    """Read a required integer query parameter (VOICEVOX's ``speaker`` etc.)."""
    values = params.get(name)
    if not values:
        raise RequestValidationError([_field_error(["query", name], "必須です", "missing")])
    try:
        return int(values[0])
    except ValueError:
        raise RequestValidationError([_field_error(
            ["query", name], "整数を指定してください", "int_parsing", values[0])]) from None


def _reference_audio_extension(reference: dict) -> str:
    """Return a safe extension for a materialized reference-audio file."""
    name_suffix = Path(str(reference.get("name", ""))).suffix.lower()
    if re.fullmatch(r"\.[a-z0-9]{1,10}", name_suffix):
        return name_suffix
    mime = str(reference.get("mime", "")).split(";", 1)[0].strip().lower()
    mime_suffix = mimetypes.guess_extension(mime) or ""
    if re.fullmatch(r"\.[a-z0-9]{1,10}", mime_suffix):
        return mime_suffix
    return ".wav"


def _cfg_value(query: dict, key: str, default: float) -> float:
    """Read one per-line CFG value while keeping old queries compatible."""
    raw = query.get(key, default)
    try:
        value = float(default if raw is None else raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a number") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 20.0:
        raise ValueError(f"{key} must be between 0 and 20")
    return value


def _strength_value(query: dict, key: str, default: float = 1.0) -> float:
    """Read one per-line condition strength in the inclusive 0..1 range."""
    raw = query.get(key, default)
    try:
        value = float(default if raw is None else raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a number") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{key} must be between 0 and 1")
    return value


# 既定ステップ数はモデル種別で決まる。RF はエンジン既定の8、MeanFlow（蒸留
# モデル）は蒸留時の4。エディタ側の /irodori/settings と同じ規則に揃える。
DEFAULT_STEPS_RF = 8
DEFAULT_STEPS_MEANFLOW = 4


def _read_safetensors_config(path: str | Path) -> dict:
    """safetensors ヘッダの ``config_json`` だけを読む（重みは読まない）。"""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        metadata = handle.metadata() or {}
    return json.loads(metadata.get("config_json") or "{}")


def _resolve_flow_parameterization(checkpoint: str | Path | None) -> tuple[str, str]:
    """(flow_parameterization, 判定元) を返す。

    1. 環境変数 ``IRODORI_FLOW_PARAMETERIZATION`` の明示指定
    2. チェックポイントの safetensors ヘッダ ``config_json``（重みは読まない）
    3. 判定できない場合は RF とみなす

    MeanFlow 蒸留モデルだけが ``config_json`` に ``meanflow`` を持ち、RF モデルは
    鍵そのものが無い（= ``rf_velocity``）。
    """
    override = os.environ.get("IRODORI_FLOW_PARAMETERIZATION", "").strip().lower()
    if override in {"meanflow", "rf_velocity"}:
        return override, "env"
    if checkpoint:
        path = Path(checkpoint)
        if path.is_file():
            try:
                config = _read_safetensors_config(path)
            except Exception:
                config = {}
            value = str(config.get("flow_parameterization") or "").strip().lower()
            if value in {"meanflow", "rf_velocity"}:
                return value, "metadata"
    return "rf_velocity", "default"


def _sampling_settings(flow_parameterization: str, default_steps: int, query: dict) -> dict:
    """1リクエスト分のサンプリング設定を決める。

    MeanFlow は蒸留時に教師の軌跡へ CFG を融合してあるので、推論は1ステップ
    あたり条件付き1回の評価で済む。RF 用の CFG スケールと Sway Sampling は
    効かないため、0 と linear を渡して「効いているように見える」リクエストを
    作らない。ステップ数だけは品質と速度を直接動かすので明示指定を受け付ける。
    """
    raw_steps = query.get("irodori_steps")
    if isinstance(raw_steps, bool) or raw_steps in ("", None):
        num_steps = int(default_steps)
    else:
        try:
            num_steps = int(raw_steps)
        except (TypeError, ValueError) as exc:
            raise ValueError("irodori_steps must be an integer") from exc
    if num_steps <= 0:
        raise ValueError("irodori_steps must be >= 1")
    if str(flow_parameterization).strip().lower() == "meanflow":
        return {
            "num_steps": num_steps,
            "cfg_scale_text": 0.0,
            "cfg_scale_speaker": 0.0,
            "cfg_scale_caption": 0.0,
            "t_schedule_mode": "linear",
            "sway_coeff": -1.0,
        }
    return {
        "num_steps": num_steps,
        "cfg_scale_text": _cfg_value(query, "irodori_cfg_text", 3.0),
        "cfg_scale_speaker": _cfg_value(query, "irodori_cfg_speaker", 5.0),
        "cfg_scale_caption": _cfg_value(query, "irodori_cfg_caption", 3.0),
        "t_schedule_mode": str(query.get("irodori_schedule", "sway")),
        "sway_coeff": float(query.get("irodori_sway_coeff", -1.0)),
    }


def _manifest_sampling_fields(adapter) -> dict:
    """Return manifest fields for both the direct and editor-backed adapters.

    EditorAdapter lazily creates its VoicevoxAdapter delegate, so it does not
    expose the delegate's sampling attributes before the first synthesis.
    Keep the public manifest available during editor startup.
    """
    flow = str(getattr(adapter, "flow_parameterization", "rf_velocity"))
    if flow not in {"rf_velocity", "meanflow"}:
        flow = "rf_velocity"
    return {
        "irodori_flow_parameterization": flow,
        "irodori_flow_parameterization_source": str(
            getattr(adapter, "flow_parameterization_source", "default")),
        "irodori_default_steps": int(getattr(
            adapter, "default_steps",
            DEFAULT_STEPS_MEANFLOW if flow == "meanflow" else DEFAULT_STEPS_RF)),
    }


def _openapi_document() -> dict:
    """Describe the VOICEVOX-compatible subset and the Irodori extensions.

    The Irodori fields come from IRODORI_QUERY_FIELDS, the same table the
    request validation uses.
    """
    speaker = {"name": "speaker", "in": "query", "required": True,
               "schema": {"type": "integer"}, "description": "/speakers の styles[].id"}
    json_ok = {"description": "OK", "content": {"application/json": {}}}
    invalid = {"description": "入力エラー（VOICEVOX と同じ FastAPI 形式）",
               "content": {"application/json": {
                   "schema": {"$ref": "#/components/schemas/HTTPValidationError"}}}}
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "Irodori-TTS VOICEVOX compatible engine",
            "version": ENGINE_VERSION,
            "description": (
                "VOICEVOX エンジン互換の API。ブラウザ以外のローカルクライアント（Origin ヘッダーなし）は"
                "トークン不要で使える。ブラウザからは許可された Origin と /irodori/session の"
                "トークン（X-Irodori-Session）が必要。"
            ),
        },
        "paths": {
            "/version": {"get": {"responses": {"200": json_ok}}},
            "/core_versions": {"get": {"responses": {"200": json_ok}}},
            "/speakers": {"get": {"responses": {"200": json_ok}}},
            "/speaker_info": {"get": {
                "parameters": [{"name": "speaker_uuid", "in": "query", "required": True,
                                "schema": {"type": "string"}}],
                "responses": {"200": json_ok, "404": json_ok}}},
            "/engine_manifest": {"get": {"responses": {"200": json_ok}}},
            "/user_dict": {"get": {"responses": {"200": json_ok}}},
            "/audio_query": {"post": {
                "parameters": [
                    {"name": "text", "in": "query", "required": True,
                     "schema": {"type": "string"}, "description": "最大256文字"},
                    speaker,
                ],
                "responses": {
                    "200": {"description": "OK", "content": {"application/json": {
                        "schema": {"$ref": "#/components/schemas/AudioQuery"}}}},
                    "422": invalid,
                }}},
            "/synthesis": {"post": {
                "parameters": [speaker],
                "requestBody": {"required": True, "content": {"application/json": {
                    "schema": {"$ref": "#/components/schemas/AudioQuery"}}}},
                "responses": {
                    "200": {"description": "WAV", "content": {"audio/wav": {}}},
                    "400": json_ok,
                    "422": invalid,
                    "429": {"description": "合成の同時実行数の上限。Retry-After 秒後に再試行"},
                }}},
        },
        "components": {"schemas": {
            "AudioQuery": {
                "type": "object",
                "description": "VOICEVOX の AudioQuery。accent_phrases などの韻律フィールドは互換のため"
                               "受け付けるが、Irodori は文章から直接合成するため使わない",
                "properties": {
                    "accent_phrases": {"type": "array", "items": {"type": "object"}},
                    "speedScale": {"type": "number", "minimum": SPEED_SCALE_MIN,
                                   "maximum": SPEED_SCALE_MAX, "default": 1.0},
                    "pitchScale": {"type": "number"},
                    "intonationScale": {"type": "number"},
                    "volumeScale": {"type": "number"},
                    "prePhonemeLength": {"type": "number"},
                    "postPhonemeLength": {"type": "number"},
                    "outputSamplingRate": {"type": "integer"},
                    "outputStereo": {"type": "boolean"},
                    "kana": {"type": "string"},
                    **IRODORI_QUERY_FIELDS,
                },
            },
            "HTTPValidationError": {"type": "object", "properties": {"detail": {
                "type": "array", "items": {"type": "object", "properties": {
                    "type": {"type": "string"},
                    "loc": {"type": "array", "items": {}},
                    "msg": {"type": "string"},
                    "input": {},
                }}}}},
        }},
    }


def _stderr_progress(message: str, percent: int | None = None) -> None:
    """進捗と実行時ログを同じ 1 行形式で stderr に出す。

    アダプタは進捗コールバック（``message``, ``percent``）としても、
    ``tts_cli`` の ``log_fn``（``message`` のみ）としても同じ関数を渡す。
    ランタイムの ``sample_meanflow`` / ``sample_rf`` 段がここに出るので、
    どのサンプラーとステップ数で動いたかを後から確認できる。
    """
    suffix = "" if percent is None else f" ({percent}%)"
    print(f"[voicevox] {message}{suffix}", file=sys.stderr, flush=True)


def _unique_speakers(tts: IrodoriTTS) -> list[str]:
    """Hide the convenience ``name`` alias when ``name.speaker`` exists."""
    names = set(tts.speakers())
    result: list[str] = []
    added: set[str] = set()
    for name in tts.speakers():
        if name.endswith(".speaker") and name[: -len(".speaker")] in names:
            continue
        if name not in added:
            result.append(name)
            added.add(name)
    return result


def _speaker_table(tts: IrodoriTTS, progress_callback=None) -> tuple[list[dict], dict[int, str]]:
    if progress_callback:
        progress_callback("preparing speaker table", 65)
    styles = ["話者なし"] + _unique_speakers(tts)
    catalog = dict(speaker_catalog(
        tts.embed_dirs,
        ((name, tts.cassette.path_for(name)) for name in styles[1:]),
    ))
    id_to_name: dict[int, str] = {}
    output: list[dict] = []
    fallback = _fallback_icon()
    fallback_payload = fallback[1] if fallback else TINY_PNG
    for index, name in enumerate(styles, start=1):
        style_id = index
        id_to_name[style_id] = name
        display_name = display_name_for(name)
        speaker_uuid = str(uuid.uuid5(ENGINE_UUID_NAMESPACE, name))
        folder = (
            tts.cassette.folder_for(name)
            if name and name != "話者なし"
            else None
        )
        # 「話者なし」も赤い空画像ではなく、話者一覧と同じ汎用SVGを表示する。
        icon = portrait = fallback_payload
        thumb = catalog.get(name)
        if thumb:
            mime, encoded = thumb
            # VOICEVOX expects the raw base64 payload in these fields.
            icon = portrait = encoded
        # "話者なし" is a UI choice, not an embedding filename.
        source = tts.cassette.path_for(name) if name != "話者なし" else None
        open_mouth = mouth_open_thumbnail_for(source) if source else None
        blink = blink_thumbnail_for(source) if source else None
        mouth_parts = mouth_parts_for(source) if source else None
        credit = credit_for(source) if source else None
        policy = policy_for(source) if source else None
        original = portrait_for(source) if source else None
        if original:
            portrait = original[1]
        output.append({
            "name": display_name,
            "speaker_uuid": speaker_uuid,
            "styles": [{"name": "ノーマル", "id": style_id, "type": "talk"}],
            "version": ENGINE_VERSION,
            "icon": icon,
            "portrait": portrait,
            "mouth_open": open_mouth[1] if open_mouth else None,
            "blink": blink[1] if blink else None,
            # 母音ごとの口パーツ（任意）
            "mouth_parts": mouth_parts,
            "credit": credit,
            "policy": policy,
            "irodori_folder": folder,
        })
    # 話者 ID は既存プロジェクトとの互換性のため生成順のまま維持し、
    # 表示順だけを変えて初回起動時の既定話者をつくよみちゃんにする。
    output.sort(
        key=lambda speaker: speaker["name"]
        != display_name_for(DEFAULT_SPEAKER_NAME)
    )
    if progress_callback:
        progress_callback("speaker table ready", 75)
    return output, id_to_name


def _reading_options(query: dict) -> dict:
    """英単語の読み変換とカナ表記の指定（エディタの共通設定から届く）。"""
    english = query.get("irodori_english_reading", "katakana")
    kana_style = query.get("irodori_kana_style", "katakana")
    if english not in ENGLISH_READINGS:
        raise ValueError(
            f"irodori_english_reading must be one of {', '.join(ENGLISH_READINGS)}")
    if kana_style not in KANA_STYLES:
        raise ValueError(f"irodori_kana_style must be one of {', '.join(KANA_STYLES)}")
    return dict(english=english, kana_style=kana_style)


def _query(text: str) -> dict:
    # Irodori accepts text directly and does not currently expose VOICEVOX's
    # accent-phrase/mora editor.  Keep a valid empty phrase list and preserve
    # the original text in an extension field for /synthesis.
    return {
        "accent_phrases": [],
        "speedScale": 1.0,
        "pitchScale": 0.0,
        "intonationScale": 1.0,
        "volumeScale": 1.0,
        "prePhonemeLength": 0.1,
        "postPhonemeLength": 0.1,
        "outputSamplingRate": 48000,
        "outputStereo": False,
        "kana": READING_DICTIONARY.convert(text),
        "irodori_text": text,
    }


class VoicevoxAdapter:
    def __init__(self, backend: str, port: int, embed_dirs: list[Path], plan: Path | None,
                 progress_callback=None):
        self.backend_name = backend
        self.port = port
        self.progress_callback = progress_callback
        if progress_callback:
            progress_callback("initializing model/backend", 15)
        self.tts = IrodoriTTS(backend=backend, embed_dirs=embed_dirs, plan_path=plan, allow_no_speaker=True)
        if progress_callback:
            progress_callback("preparing speaker table", 65)
        self.speakers_json, self.id_to_name = _speaker_table(self.tts, progress_callback)
        self.resource_index = _speaker_resource_index(self.speakers_json)
        self.resource_digests = {value: digest for digest, value in self.resource_index.items()}
        self.speaker_index = _speaker_uuid_index(self.speakers_json)
        self.name_to_id = {name: sid for sid, name in self.id_to_name.items()}
        self.lock = threading.Lock()
        self.synthesis_slots = threading.BoundedSemaphore(2)
        # サンプリング既定はモデル種別で決まる。MeanFlow 蒸留モデルなら4ステップ・
        # CFG/スケジュール無効、RF なら従来どおり8ステップ＋CFG＋sway。
        self.checkpoint = os.environ.get("IRODORI_CHECKPOINT", "")
        self.flow_parameterization, self.flow_parameterization_source = (
            _resolve_flow_parameterization(self.checkpoint))
        self.is_meanflow = self.flow_parameterization == "meanflow"
        self.default_steps = (
            DEFAULT_STEPS_MEANFLOW if self.is_meanflow else DEFAULT_STEPS_RF)

    def refresh(self) -> None:
        with self.lock:
            self.tts.refresh_cassette()
            self.speakers_json, self.id_to_name = _speaker_table(self.tts)
            self.resource_index = _speaker_resource_index(self.speakers_json)
            self.resource_digests = {value: digest for digest, value in self.resource_index.items()}
            self.speaker_index = _speaker_uuid_index(self.speakers_json)
            self.name_to_id = {name: sid for sid, name in self.id_to_name.items()}

    def synthesize(self, query: dict, speaker_id: int) -> bytes:
        text = str(query.get("irodori_text") or query.get("kana") or "").strip()
        text = READING_DICTIONARY.convert(text, **_reading_options(query))
        if not text:
            raise ValueError("audio query does not contain text (irodori_text/kana)")
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError(
                f"セリフが長すぎます（{len(text)}文字 / 上限{MAX_TEXT_CHARS}文字）。"
                "句読点の位置で分割してください"
            )
        if speaker_id not in self.id_to_name:
            raise ValueError(f"unknown speaker style id: {speaker_id}")
        speaker_name = (
            "" if self.id_to_name[speaker_id] == "話者なし"
            else self.id_to_name[speaker_id])
        # VOICEVOX's prosody fields are retained in the query for compatibility;
        # text-to-audio controls will be added when the runtime exposes them.
        # サンプリングはモデル種別で決まる（MeanFlow は4ステップ・CFG/スケジュール無効）。
        # 古い経路で組まれたアダプタはRF既定にフォールバックする。
        flow = getattr(self, "flow_parameterization", "rf_velocity")
        sampling = _sampling_settings(
            flow, getattr(self, "default_steps", DEFAULT_STEPS_RF), query)
        caption_strength = _strength_value(
            query, "irodori_caption_strength")
        reference_strength = _strength_value(
            query, "irodori_reference_strength")
        speaker_strength = _strength_value(
            query, "irodori_speaker_strength")
        additions = query.get("irodori_additional_speakers")
        if additions is None:
            legacy_id = query.get("irodori_secondary_speaker_style_id")
            additions = ([] if legacy_id is None else [{
                "style_id": legacy_id,
                "strength": _strength_value(query, "irodori_secondary_speaker_strength", 0.5),
            }])
        if not isinstance(additions, list) or len(additions) > 3:
            raise ValueError("追加話者は最大3人までです")
        additional_speakers = []
        seen_speakers = set()
        for entry in additions:
            if not isinstance(entry, dict):
                raise ValueError("追加話者の指定が正しくありません")
            style_id = entry.get("style_id")
            if (isinstance(style_id, bool) or not isinstance(style_id, int)
                    or style_id not in self.id_to_name):
                raise ValueError("追加話者が見つかりません")
            name = self.id_to_name[style_id]
            if name == "話者なし" or name == speaker_name or name in seen_speakers:
                raise ValueError("追加話者が重複しているか、話者なしが指定されています")
            seen_speakers.add(name)
            strength = _strength_value(entry, "strength", 0.5)
            additional_speakers.append((name, strength))
        # The slider controls the optional user-provided reference audio. If no
        # such file is attached, preserve the normal speaker condition.
        if not query.get("irodori_reference_audio"):
            reference_strength = 1.0
        print(
            f"[voicevox] synth: flow={flow} steps={sampling['num_steps']} "
            f"cfg={sampling['cfg_scale_text']}/{sampling['cfg_scale_caption']}"
            f"/{sampling['cfg_scale_speaker']} schedule={sampling['t_schedule_mode']} "
            f"speaker={speaker_name or '(none)'}",
            file=sys.stderr, flush=True,
        )
        reference = query.get("irodori_reference_audio")
        with self.lock:
            # The runtime reads reference audio from a file, so only that case
            # needs a temporary folder; the output WAV stays in memory.
            with (tempfile.TemporaryDirectory(prefix="irodori-vv-", dir=str(ROOT / "wrapper"))
                  if reference else contextlib.nullcontext()) as tmp:
                wav = io.BytesIO()
                ref_wav = None
                if reference:
                    data_url = str(reference.get("dataUrl", ""))
                    if not data_url.startswith("data:audio/") or ";base64," not in data_url:
                        raise ValueError("irodori reference audio must be a base64 audio data URL")
                    encoded = data_url.split(",", 1)[1]
                    if len(encoded) > 14_000_000:
                        raise ValueError("irodori reference audio is too large")
                    try:
                        raw = base64.b64decode(encoded, validate=True)
                    except (ValueError, binascii.Error) as exc:
                        raise ValueError("invalid irodori reference audio data URL") from exc
                    if len(raw) > 10 * 1024 * 1024:
                        raise ValueError("irodori reference audio is too large")
                    ref_wav = str(Path(tmp) / f"reference-audio{_reference_audio_extension(reference)}")
                    Path(ref_wav).write_bytes(raw)
                seed_value = query.get("irodori_seed", 4763674)
                self.tts.synthesize(
                    text=text,
                    speaker=speaker_name,
                    out_wav=wav,
                    seed=(None if seed_value is None else int(seed_value)),
                    num_steps=sampling["num_steps"],
                    seconds=query.get("irodori_seconds"),
                    caption=str(query.get("irodori_caption") or "").strip() or None,
                    caption_strength=caption_strength,
                    reference_strength=reference_strength,
                    speaker_strength=speaker_strength,
                    additional_speakers=additional_speakers,
                    duration_scale=1.0 / max(0.1, float(query.get("speedScale", 1.0))),
                    cfg_scale_text=sampling["cfg_scale_text"],
                    cfg_scale_speaker=sampling["cfg_scale_speaker"],
                    cfg_scale_caption=sampling["cfg_scale_caption"],
                    t_schedule_mode=sampling["t_schedule_mode"],
                    sway_coeff=sampling["sway_coeff"],
                    ref_wav=ref_wav,
                    log_fn=self.progress_callback,
                )
                return wav.getvalue()


class Handler(BaseHTTPRequestHandler):
    adapter: VoicevoxAdapter

    def log_message(self, fmt, *args):
        sys.stderr.write("[voicevox] " + (fmt % args) + "\n")

    def _send(self, status: int, body: bytes, content_type: str = "application/json; charset=utf-8",
              cache_control: str | None = None, retry_after: int | None = None):
        self.send_response(status)
        if retry_after is not None:
            self.send_header("Retry-After", str(retry_after))
        origin = self.headers.get("Origin")
        if origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Irodori-Session")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS")
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if cache_control:
            self.send_header("Cache-Control", cache_control)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, value: object, retry_after: int | None = None):
        self._send(status, json.dumps(value, ensure_ascii=False).encode("utf-8"),
                   retry_after=retry_after)

    def _body_json(self) -> dict:
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length) if raw_length is not None else 0
        except ValueError as exc:
            raise RequestBodyError(400, "Content-Length must be an integer") from exc
        if length <= 0:
            raise RequestBodyError(400, "request body is required")
        if length > MAX_BODY_BYTES:
            raise RequestBodyError(413, f"request body exceeds {MAX_BODY_BYTES} bytes")
        previous_timeout = self.connection.gettimeout()
        self.connection.settimeout(10.0)
        try:
            payload = self.rfile.read(length)
        except TimeoutError as exc:
            raise RequestBodyError(400, "request body read timed out") from exc
        finally:
            self.connection.settimeout(previous_timeout)
        if len(payload) != length:
            raise RequestBodyError(400, "request body is incomplete")
        try:
            value = json.loads(payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RequestBodyError(400, "request body must be valid JSON") from exc
        if not isinstance(value, dict):
            raise RequestBodyError(400, "request body must be a JSON object")
        return value

    def _bootstrap_allowed(self) -> bool:
        origin = self.headers.get("Origin")
        return self._loopback_host() and origin in ALLOWED_ORIGINS

    def _loopback_host(self) -> bool:
        # Host の確認は DNS rebinding 対策。
        host = self.headers.get("Host", "")
        hostname = host.rsplit(":", 1)[0] if ":" in host and not host.startswith("[") else host
        if host.startswith("["):
            hostname = host[1:].split("]", 1)[0]
        return hostname in ("127.0.0.1", "localhost", "::1")

    def _native_client(self) -> bool:
        """VOICEVOX と同じく、ブラウザ以外のローカルクライアントはトークンなしで通す。

        ブラウザは POST や CORS の要求に必ず Origin を付け、no-cors の GET
        （img タグなど）にも Sec-Fetch-Site を付ける。どちらも無い要求は同じ PC の
        プログラムからのもの。そうしたプログラムは Origin を偽装すればトークンも
        取れるので、トークンを求めても防御にはならない。
        """
        if "Origin" in self.headers:
            return False
        if self.headers.get("Sec-Fetch-Site", "none") not in ("none", "same-origin"):
            return False
        return self.client_address[0] in ("127.0.0.1", "::1") and self._loopback_host()

    def _authorized(self) -> bool:
        if self._native_client():
            return True
        return self._bootstrap_allowed() and self.headers.get("X-Irodori-Session") == SESSION_TOKEN

    def _resource_url(self, value: str) -> str:
        """Create a content-addressed, loopback-only URL for a speaker image."""
        digests = getattr(self.adapter, "resource_digests", None)
        digest = digests.get(value) if isinstance(digests, dict) else None
        if digest is None:
            digest = hashlib.sha256(value.encode("ascii")).hexdigest()
        host = self.headers.get("Host", "127.0.0.1")
        return f"http://{host}/irodori/resource/{digest}"

    def _resource_value(self, digest: str) -> str | None:
        """Find a cached catalog resource by its content digest."""
        resource_index = getattr(self.adapter, "resource_index", None)
        if isinstance(resource_index, dict):
            return resource_index.get(digest)

        # Compatibility fallback for adapters that have not built the index.
        for speaker in self.adapter.speakers_json:
            values = [
                speaker.get("icon"),
                speaker.get("portrait"),
                speaker.get("mouth_open"),
                speaker.get("blink"),
            ]
            mouth_parts = speaker.get("mouth_parts")
            if isinstance(mouth_parts, dict):
                values.extend(mouth_parts.values())
            for value in values:
                if isinstance(value, str) and hashlib.sha256(
                    value.encode("ascii")
                ).hexdigest() == digest:
                    return value
        return None

    @staticmethod
    def _image_mime(data: bytes) -> str:
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if data.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if data.startswith((b"GIF87a", b"GIF89a")):
            return "image/gif"
        if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
            return "image/webp"
        if data.lstrip().startswith((b"<svg", b"<?xml")):
            return "image/svg+xml"
        return "application/octet-stream"

    def _send_resource(self, digest: str) -> None:
        # Image requests cannot attach the session header.  The engine binds to
        # loopback, and resources were already available from /speaker_info.
        if self.client_address[0] not in ("127.0.0.1", "::1"):
            self._json(403, {"detail": "resource access is local-only"})
            return
        value = self._resource_value(digest)
        if value is None:
            self._json(404, {"detail": "speaker resource not found"})
            return
        try:
            data = base64.b64decode(value, validate=True)
        except (ValueError, binascii.Error):
            self._json(500, {"detail": "invalid speaker resource"})
            return
        self._send(200, data, self._image_mime(data),
                   "private, max-age=31536000, immutable")

    def _speaker_info_payload(self, speaker: dict, use_resource_url: bool) -> dict:
        def resource(value):
            if not isinstance(value, str):
                return value
            return self._resource_url(value) if use_resource_url else value

        mouth_parts = speaker.get("mouth_parts")
        return {
            "policy": (speaker.get("policy") or "").replace("\n", "  \n"),
            "credit": speaker.get("credit"),
            "portrait": resource(speaker.get("portrait", TINY_PNG)),
            "style_infos": [
                {
                    "id": style["id"],
                    "icon": resource(speaker.get("icon", TINY_PNG)),
                    "portrait": resource(speaker.get("portrait", TINY_PNG)),
                    "voice_samples": [],
                    "mouth_open": resource(speaker.get("mouth_open")),
                    "blink": resource(speaker.get("blink")),
                    "mouth_parts": (
                        {shape: resource(value) for shape, value in mouth_parts.items()}
                        if isinstance(mouth_parts, dict) else None
                    ),
                }
                for style in speaker.get("styles", [])
            ],
        }

    def do_OPTIONS(self):
        if not self._bootstrap_allowed():
            self._send(403, b"forbidden")
            return
        self._send(204, b"")

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/irodori/session":
            if not self._bootstrap_allowed():
                self._json(403, {"detail": "invalid local origin/host"})
                return
            self._json(200, {"token": SESSION_TOKEN})
        elif parsed.path == "/refresh" and not self._authorized():
            self._json(403, {"detail": "invalid local origin/session"})
            return
        elif parsed.path.startswith("/irodori/resource/"):
            self._send_resource(parsed.path.removeprefix("/irodori/resource/"))
        elif parsed.path == "/version":
            self._json(200, ENGINE_VERSION)
        elif parsed.path == "/core_versions":
            self._json(200, [ENGINE_VERSION])
        elif parsed.path == "/openapi.json":
            self._json(200, _openapi_document())
        elif parsed.path == "/speakers":
            self._json(200, _speaker_list_payload(self.adapter.speakers_json))
        elif parsed.path == "/irodori/speaker_bundle":
            if not self._authorized():
                self._json(403, {"detail": "invalid local origin/session"})
                return
            speakers = self.adapter.speakers_json
            self._json(200, {
                "speakers": _speaker_list_payload(speakers),
                "speaker_infos": {
                    speaker["speaker_uuid"]: self._speaker_info_payload(speaker, True)
                    for speaker in speakers
                },
            })
        elif parsed.path in ("/engine_manifest", "/engine_manifest.json"):
            manifest = {
                "manifest_version": "0.13.1",
                "name": f"Irodori ({self.adapter.backend_name})",
                "brand_name": "Irodori-TTS",
                "irodori_checkpoint": os.environ.get("IRODORI_CHECKPOINT", ""),
                "uuid": str(uuid.uuid5(ENGINE_UUID_NAMESPACE, self.adapter.backend_name)),
                "url": "https://github.com/Rose3a/kataribe",
                "icon": TINY_PNG,
                "default_sampling_rate": 48000,
                "frame_rate": 93.75,
                "terms_of_service": "",
                "update_infos": [],
                "dependency_licenses": dependency_licenses(),
                "supported_features": {
                    "adjust_mora_pitch": False,
                    "adjust_phoneme_length": False,
                    "adjust_speed_scale": True,
                    "adjust_pitch_scale": False,
                    "adjust_intonation_scale": False,
                    "adjust_volume_scale": False,
                    "adjust_pause_length": False,
                    "interrogative_upspeak": False,
                    "synthesis_morphing": False,
                    "sing": False,
                    "manage_library": False,
                    "return_resource_url": True,
                    "apply_katakana_english": False,
                },
            }
            # MeanFlow 判定とサンプリング既定をクライアントから確認できるようにする。
            manifest.update(_manifest_sampling_fields(self.adapter))
            self._json(200, manifest)
        elif parsed.path == "/supported_devices":
            backend = self.adapter.backend_name
            self._json(200, {
                "cpu": backend in ("cpu", "radeon"),
                "cuda": backend in ("cuda", "trt"),
                "dml": backend == "radeon",
            })
        elif parsed.path == "/speaker_info":
            speaker_uuid = (parse_qs(parsed.query).get("speaker_uuid") or [""])[0]
            speaker_index = getattr(self.adapter, "speaker_index", None)
            if isinstance(speaker_index, dict):
                speaker = speaker_index.get(speaker_uuid)
            else:
                # Compatibility for adapters created before the index existed.
                speaker = next(
                    (item for item in self.adapter.speakers_json
                     if item.get("speaker_uuid") == speaker_uuid),
                    None,
                )
            if speaker is None:
                self._json(404, {"detail": "speaker not found"})
                return
            use_resource_url = (
                (parse_qs(parsed.query).get("resource_format") or [""])[0]
                == "url"
            )
            self._json(200, self._speaker_info_payload(speaker, use_resource_url))
        elif parsed.path == "/is_initialized_speaker":
            self._json(200, True)
        elif parsed.path == "/user_dict":
            self._json(200, READING_DICTIONARY.snapshot())
        elif parsed.path == "/refresh":
            self.adapter.refresh()
            self._json(200, _speaker_list_payload(self.adapter.speakers_json))
        else:
            self._json(404, {"detail": "Not Found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        if not self._authorized():
            self._json(403, {"detail": "invalid local origin/session"})
            return
        try:
            if parsed.path == "/user_dict_word":
                params = parse_qs(parsed.query)
                word = make_word(params.get("surface", [""])[0],
                                 params.get("pronunciation", [""])[0],
                                 params.get("accent_type", ["0"])[0],
                                 params.get("priority", ["5"])[0])
                self._json(200, READING_DICTIONARY.put(word))
                return
            if parsed.path == "/import_user_dict":
                params = parse_qs(parsed.query)
                READING_DICTIONARY.import_words(self._body_json(),
                    params.get("override", ["false"])[0].lower() == "true")
                self._send(204, b"")
                return
            if parsed.path == "/audio_query":
                # VOICEVOX と同じく text と speaker は必須。空の text は許す。
                params = parse_qs(parsed.query, keep_blank_values=True)
                if "text" not in params:
                    raise RequestValidationError(
                        [_field_error(["query", "text"], "必須です", "missing")])
                _int_query_param(params, "speaker")
                self._json(200, _query(params["text"][0]))
                return
            if parsed.path == "/synthesis":
                params = parse_qs(parsed.query)
                speaker = _int_query_param(params, "speaker")
                query = self._body_json()
                _validate_query(query)
                if not self.adapter.synthesis_slots.acquire(timeout=0.1):
                    self._json(429, {"detail": "synthesis queue is full"}, retry_after=1)
                    return
                try:
                    data = self.adapter.synthesize(query, speaker)
                finally:
                    self.adapter.synthesis_slots.release()
                self._send(200, data, "audio/wav")
                return
            if parsed.path == "/initialize_speaker":
                self._send(204, b"")
                return
            self._json(404, {"detail": "Not Found"})
        except RequestBodyError as exc:
            self.log_message("request rejected (%s): %s", exc.status, exc)
            self._json(exc.status, {"detail": str(exc)})
        except RequestValidationError as exc:
            self.log_message("request rejected (422): %s", exc)
            self._json(422, {"detail": exc.errors})
        except (ValueError, KeyError) as exc:
            self.log_message("request rejected (400): %s", exc)
            self._json(400, {"detail": str(exc)})
        except Exception as exc:
            self.log_message("request failed (500): %s", exc)
            self._json(500, {"detail": str(exc)})

    def _change_dictionary(self, delete=False):
        if not self._authorized():
            return self._json(403, {"detail": "invalid local origin/session"})
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/user_dict_word/"):
            return self._json(404, {"detail": "Not Found"})
        try:
            key = parsed.path.removeprefix("/user_dict_word/")
            if delete:
                READING_DICTIONARY.delete(key)
            else:
                params = parse_qs(parsed.query)
                word = make_word(params.get("surface", [""])[0],
                                 params.get("pronunciation", [""])[0],
                                 params.get("accent_type", ["0"])[0],
                                 params.get("priority", ["5"])[0])
                READING_DICTIONARY.put(word, key)
            self._send(204, b"")
        except KeyError as exc:
            self._json(404, {"detail": str(exc)})
        except (ValueError, TypeError) as exc:
            self._json(400, {"detail": str(exc)})
        except Exception as exc:
            self._json(500, {"detail": str(exc)})

    def do_PUT(self):
        self._change_dictionary()

    def do_DELETE(self):
        self._change_dictionary(delete=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda", "trt", "radeon"), default="cpu")
    parser.add_argument("--port", type=int, default=50021)
    parser.add_argument("--embed-dir", action="append", type=Path, default=[])
    parser.add_argument("--plan", type=Path, default=None)
    args = parser.parse_args(argv)
    # Make the adapter self-contained when launched from the official editor
    # (which does not inherit the project's batch-file environment).
    os.environ.setdefault("IRODORI_RUNTIME_DIR", str(ROOT / "runtime"))
    os.environ.setdefault("IRODORI_CHECKPOINT", str(ROOT / "models" / "model.safetensors"))
    os.environ.setdefault("IRODORI_HF_HOME", str(ROOT / ".cache" / "huggingface"))
    os.environ.setdefault("IRODORI_CACHE_DIR", str(ROOT / ".cache" / "irodori-tts-cache"))
    adapter = VoicevoxAdapter(args.backend, args.port,
                              resolve_embed_dirs(args.embed_dir), args.plan,
                              progress_callback=_stderr_progress)
    Handler.adapter = adapter
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(
        f"Irodori VOICEVOX engine: http://127.0.0.1:{args.port} backend={args.backend} "
        f"checkpoint={adapter.checkpoint or '(unset)'} "
        f"flow={adapter.flow_parameterization}({adapter.flow_parameterization_source}) "
        f"default_steps={adapter.default_steps}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
        adapter.tts.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
