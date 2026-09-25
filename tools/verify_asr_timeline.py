"""口パク用ASR（/irodori/timeline）の検証。

初回はエンジンがモデル（約626MB）を自動取得するので、取得完了まで再送しながら
anchors が返ることを確かめる。結果は logs\\verify-asr.json に残す。
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:50125"
ORIGIN = "http://localhost:5173"
BOX = Path(__file__).resolve().parents[1]
TEXT = "こんにちは、口パクのテストです"
DEADLINE_SECONDS = 900


def call(method: str, path: str, token: str | None = None, body: dict | None = None,
         timeout: int = 600):
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Origin": ORIGIN, "Content-Type": "application/json"}
    if token:
        headers["X-Irodori-Session"] = token
    request = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            kind = response.headers.get("Content-Type", "")
            return response.status, (json.loads(raw) if "json" in kind else raw)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def main() -> int:
    report: dict = {}
    started = time.time()
    while time.time() - started < 120:
        try:
            status, version = call("GET", "/version", timeout=10)
            if status == 200:
                report["version"] = version
                break
        except Exception:
            time.sleep(1)
    else:
        print("engine did not become ready")
        return 1

    status, session = call("GET", "/irodori/session")
    token = session["token"] if isinstance(session, dict) else None
    report["session_status"] = status
    status, settings = call("GET", "/irodori/settings", token)
    report["backend"] = settings.get("settings", {}).get("backend") if isinstance(settings, dict) else None
    report["asr_before"] = settings.get("asr") if isinstance(settings, dict) else None

    status, query = call("POST", "/audio_query?speaker=0&text=" + urllib.parse.quote(TEXT), token)
    report["audio_query_status"] = status
    if status != 200:
        report["error"] = f"audio_query failed: {query}"
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1
    status, wav = call("POST", "/synthesis?speaker=0", token, dict(query))
    report["synthesis_status"] = status
    report["wav_bytes"] = len(wav) if isinstance(wav, bytes) else 0

    download_started = time.time()
    attempts = 0
    result = None
    while time.time() - download_started < DEADLINE_SECONDS:
        attempts += 1
        status, payload = call("POST", "/irodori/timeline", token, {"text": TEXT}, timeout=180)
        if status == 200 and isinstance(payload, dict) and payload.get("available"):
            result = payload
            break
        if status == 200 and isinstance(payload, dict):
            print(f"[{attempts}] waiting: reason={payload.get('reason')} "
                  f"downloading={payload.get('downloading')}", flush=True)
        else:
            print(f"[{attempts}] status={status} body={str(payload)[:200]}", flush=True)
        time.sleep(15)

    report["timeline_attempts"] = attempts
    report["asr_wait_seconds"] = round(time.time() - download_started, 1)
    if result is None:
        report["ok"] = False
        report["error"] = f"timeline did not become available within {DEADLINE_SECONDS}s"
    else:
        anchors = result.get("anchors") or []
        report.update(
            ok=bool(anchors),
            anchor_count=len(anchors),
            anchors_head=anchors[:5],
            audio_seconds=result.get("audioSeconds"),
            asr_seconds=result.get("asrSeconds"),
            resolution=result.get("resolution"),
        )
    asr_dir = BOX / "models" / "asr"
    report["asr_files"] = {p.name: p.stat().st_size for p in sorted(asr_dir.glob("*")) if p.is_file()}
    (BOX / "logs" / "verify-asr.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
