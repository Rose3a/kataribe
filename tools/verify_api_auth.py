"""ローカルAPIの入口（オリジンとセッショントークン）の確認。

エディタは起動ごとに変わるトークンを使うので、入口が揃っているかを
実物のエンジンに対して確かめる。結果は logs\\verify-api-auth.json に残す。
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:50125"
ORIGIN = "http://localhost:5173"
FOREIGN_ORIGIN = "http://evil.example"
BOX = Path(__file__).resolve().parents[1]


def call(method: str, path: str, token: str | None = None, origin: str | None = ORIGIN,
         body: dict | None = None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if origin:
        headers["Origin"] = origin
    if token:
        headers["X-Irodori-Session"] = token
    request = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except urllib.error.URLError as exc:
        return 0, str(exc).encode()


def main() -> int:
    report: dict[str, object] = {}
    checks: list[tuple[str, bool, str]] = []

    status, _ = call("GET", "/irodori/settings")
    checks.append(("トークン無しの設定取得は拒否", status == 403, f"status={status}"))

    status, _ = call("POST", "/irodori/settings", body={"backend": "cpu"})
    checks.append(("トークン無しの設定保存は拒否", status == 403, f"status={status}"))

    status, raw = call("GET", "/irodori/session", origin=FOREIGN_ORIGIN)
    checks.append(("許可外オリジンにはトークンを渡さない", status == 403, f"status={status}"))

    status, raw = call("GET", "/irodori/session")
    token = ""
    if status == 200:
        token = json.loads(raw)["token"]
    checks.append(("正しいオリジンにはトークンを渡す", status == 200 and bool(token),
                   f"status={status}"))

    status, raw = call("GET", "/irodori/settings", token=token)
    body = json.loads(raw) if status == 200 else {}
    checks.append(("有効なトークンで設定を取得できる",
                   status == 200 and "settings" in body, f"status={status}"))
    report["settings_keys"] = sorted(body.keys())
    report["progress"] = body.get("progress")

    status, _ = call("GET", "/irodori/settings", token="stale-token")
    checks.append(("古いトークンは拒否（エディタはここで取り直す）", status == 403,
                   f"status={status}"))

    status, _ = call("GET", "/version")
    checks.append(("VOICEVOX 互換の /version はトークン不要", status == 200, f"status={status}"))

    status, _ = call("POST", "/audio_query?text=a&speaker=0", origin=None)
    checks.append(("Origin 無しのローカルクライアントはトークン不要（VOICEVOX 互換）",
                   status == 200, f"status={status}"))

    status, _ = call("POST", "/audio_query?text=a&speaker=0", origin=FOREIGN_ORIGIN)
    checks.append(("許可外オリジンからの合成系は拒否", status == 403, f"status={status}"))

    report["checks"] = [{"name": name, "ok": ok, "detail": detail}
                        for name, ok, detail in checks]
    report["ok"] = all(ok for _, ok, _ in checks)
    (BOX / "logs").mkdir(exist_ok=True)
    (BOX / "logs" / "verify-api-auth.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, ok, detail in checks:
        print(("OK   " if ok else "NG   ") + f"{name} ({detail})")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
