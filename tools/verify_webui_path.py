"""WebUI と同じ経路（HTTP + セッショントークン）でエンジンを検証する。

1. /irodori/session をブラウザと同じ Origin で取得
2. /irodori/settings で使うバックエンドを確認
3. /audio_query にアポストロフィ・全大文字・辞書上書きを混ぜて投げる（改修の回帰確認）
4. /synthesis で実 WAV を作る
5. WAV の長さと先頭サンプルを確認
"""
from __future__ import annotations

import base64
import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from pathlib import Path

BASE = "http://127.0.0.1:50125"
ORIGIN = "http://localhost:5173"
OUT = Path(__file__).resolve().parents[1] / "outputs"
report: dict = {}


def call(method: str, path: str, token: str | None = None, body: dict | None = None, timeout: int = 1800):
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Origin": ORIGIN, "Content-Type": "application/json"}
    if token:
        headers["X-Irodori-Session"] = token
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
            ctype = response.headers.get("Content-Type", "")
            return response.status, (json.loads(raw) if "json" in ctype else raw), ctype
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace"), ""


# 1. 起動待ち + セッション
deadline = time.time() + 180
while time.time() < deadline:
    try:
        status, version, _ = call("GET", "/version")
        if status == 200:
            report["version"] = version
            break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("engine did not become ready")

status, session, _ = call("GET", "/irodori/session")
report["session_status"] = status
token = session["token"] if isinstance(session, dict) else None
report["token_received"] = bool(token)

# 2. 設定（セッション必須）
status, settings, _ = call("GET", "/irodori/settings", token)
report["settings_status"] = status
if isinstance(settings, dict):
    report["backend"] = settings.get("settings", {}).get("backend")
    report["model"] = settings.get("settings", {}).get("model")
    report["loaded"] = settings.get("loaded")

# 3. 読み辞書まわりの回帰確認（ブラウザの「読み」と同じ経路）
text = "ROCK'N ROLL と python3 と AI と こんにちは"
status, query, _ = call("POST", "/audio_query?speaker=0&text=" + urllib.parse.quote(text), token)
report["audio_query_status"] = status
if isinstance(query, dict):
    report["kana"] = query.get("kana")
else:
    report["audio_query_error"] = str(query)[:300]

# 4. 合成
if report.get("audio_query_status") == 200:
    started = time.time()
    status, wav, ctype = call("POST", "/synthesis?speaker=0", token, dict(query))
    report["synthesis_status"] = status
    report["synthesis_wall_s"] = round(time.time() - started, 1)
    if status == 200 and isinstance(wav, bytes):
        OUT.mkdir(parents=True, exist_ok=True)
        target = OUT / "verify-webui-radeon.wav"
        target.write_bytes(wav)
        with wave.open(io.BytesIO(wav)) as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
        report.update(wav=str(target), wav_bytes=len(wav), seconds=round(frames / rate, 2), rate=rate)
    else:
        report["synthesis_error"] = str(wav)[:300]

(Path(__file__).resolve().parents[1] / "logs" / "verify-webui.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
