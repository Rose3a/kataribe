"""Own the editor engine and Vite processes for the browser editor session."""
from __future__ import annotations

import argparse
import atexit
import ctypes
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ENGINE_HOST = "127.0.0.1"
ENGINE_PORT = 50125
BROWSER_URL = "http://127.0.0.1:5173"
STARTUP_TIMEOUT = 90.0
POLL_INTERVAL = 0.5
# open_browser.bat が IRODORI_DEBUG=1 を渡すとデバッグモードになり、
# エンジンとUIのログ(生成時間などの計測値を含む)をこのコンソールへ流す。
# exe化したときは同じ環境変数で切り替えられる。
DEBUG = os.environ.get("IRODORI_DEBUG", "").strip().lower() not in ("", "0", "false", "no")
LOG_TAIL_INTERVAL = 0.4
# 1.5秒ごとの設定ポーリング(200)でコンソールが埋まらないように落とす。
# 失敗したポーリングは残して異常に気づけるようにする。
LOG_SKIP_PATTERNS = ('/irodori/settings HTTP/1.1" 200 -', "OPTIONS /")

if os.name == "nt":
    from ctypes import wintypes

    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9

    class _JobObjectBasicLimitInformation(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64), ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t), ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD), ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

    class _IoCounters(ctypes.Structure):
        _fields_ = [("ReadOperationCount", ctypes.c_uint64), ("WriteOperationCount", ctypes.c_uint64), ("OtherOperationCount", ctypes.c_uint64), ("ReadTransferCount", ctypes.c_uint64), ("WriteTransferCount", ctypes.c_uint64), ("OtherTransferCount", ctypes.c_uint64)]

    class _JobObjectExtendedLimitInformation(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _JobObjectBasicLimitInformation), ("IoInfo", _IoCounters), ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]



def _load_console_ui():
    """見やすい表示（wrapper/console_ui.py）。rich が無い・端末でないときは None（従来の 1 行表示）。"""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "irodori-tts" / "wrapper"))
        import console_ui
    except Exception:  # noqa: BLE001
        return None
    return console_ui if console_ui.fancy() else None


_CONSOLE_UI = _load_console_ui()


def _show_log_line(label: str, line: str) -> None:
    """ログ 1 行をコンソールへ。端末なら色分け・整形し、そうでなければ従来どおり。"""
    if _CONSOLE_UI is not None:
        try:
            markup = _CONSOLE_UI.log_line_markup(label, line)
            if markup:  # 空は「細かすぎるので出さない」行（ログファイルには残る）
                _CONSOLE_UI.print_markup(markup)
            return
        except Exception:  # noqa: BLE001 - 整形に失敗しても素の行は必ず出す
            pass
    print(f"[{label}] {line.rstrip()}", flush=True)


class LogStreamer(threading.Thread):
    """デバッグ用: ログファイルの追記をコンソールへ流す(tail -f 相当)。"""

    def __init__(self, path: Path, label: str, skip_patterns=()):
        super().__init__(name=f"log-stream-{label}", daemon=True)
        self.path = path
        self.label = label
        self.skip_patterns = skip_patterns
        self._stop = threading.Event()
        self._position: int | None = None

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                size = self.path.stat().st_size
                if self._position is None:
                    # 既存の履歴は流さず、起動後の追記だけを見る
                    self._position = size
                elif size < self._position:
                    self._position = 0
                if size != self._position:
                    with self.path.open("r", encoding="utf-8", errors="replace") as handle:
                        handle.seek(self._position)
                        for line in handle:
                            if not any(pattern in line for pattern in self.skip_patterns):
                                _show_log_line(self.label, line)
                        self._position = handle.tell()
            except FileNotFoundError:
                pass
            except (OSError, ValueError):
                pass
            self._stop.wait(LOG_TAIL_INTERVAL)

def _create_job() -> int | None:
    if os.name != "nt":
        return None
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None
        limits = _JobObjectExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(job, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(limits), ctypes.sizeof(limits)):
            kernel32.CloseHandle(job)
            return None
        return int(job)
    except Exception:
        return None


def _assign_to_job(job: int | None, process: subprocess.Popen) -> bool:
    if job is None or os.name != "nt":
        return False
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        return bool(kernel32.AssignProcessToJobObject(job, process._handle))
    except Exception:
        return False


def _close_job(job: int | None) -> None:
    if job is None or os.name != "nt":
        return
    try:
        ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(job)
    except Exception:
        pass


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_python(box: Path) -> Path:
    candidates = (box / "irodori-tts" / ".venv" / "Scripts" / "python.exe", box / ".local" / "venv" / "Scripts" / "python.exe")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Python runtime was not found. Checked: " + ", ".join(map(str, candidates)))


def _request(path: str, method: str = "GET", timeout: float = 1.0, headers=None):
    request = urllib.request.Request(
        f"http://{ENGINE_HOST}:{ENGINE_PORT}{path}", method=method, headers=headers or {})
    return urllib.request.urlopen(request, timeout=timeout)

def _session_token() -> str:
    with _request("/irodori/session", headers={"Origin": BROWSER_URL}) as response:
        import json
        return json.loads(response.read(4096))["token"]


class EngineAuthError(RuntimeError):
    """エンジンは応答したが、Origin かセッショントークンが合わず 403 が返った。"""


def _authorized_request(path: str, method: str = "GET", timeout: float = 1.0):
    """Irodori のエンドポイントは Origin とセッショントークンの両方が要る。

    `/irodori/session` はトークンを配る口なので `_bootstrap_allowed()` しか見ないが、
    `/irodori/settings` などは `_authorized()` を通る。トークンを付けずに叩くと 403。
    """
    token = _session_token()
    return _request(path, method=method, timeout=timeout,
                    headers={"Origin": BROWSER_URL, "X-Irodori-Session": token})


def _engine_settings(timeout: float = 1.0):
    """設定を取る。繋がらなければ None、認証が通らなければ EngineAuthError。

    `HTTPError` は `URLError`/`OSError` のサブクラスなので、まとめて捕まえると
    「エンジンが落ちている」と「セッションが拒否された」の区別が消える。
    403 は設定のずれなので黙って False にせず投げる。
    """
    try:
        return _authorized_request("/irodori/settings", timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 403:
            raise EngineAuthError(
                f"engine rejected the editor session (403 {exc.reason}). "
                f"Check Origin ({BROWSER_URL}) against the engine's ALLOWED_ORIGINS "
                "and refresh the session token."
            ) from exc
        return None
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        return None


def engine_ready() -> bool:
    response = _engine_settings()
    if response is None:
        return False
    try:
        return response.status == 200
    finally:
        response.close()


def shutdown_engine(timeout: float = 3.0) -> bool:
    """Stop the project engine if its identifying endpoint is present."""
    response = _engine_settings(timeout=0.8)
    if response is None:
        return False
    response.close()
    try:
        request = urllib.request.Request(
            f"http://{ENGINE_HOST}:{ENGINE_PORT}/irodori/shutdown", method="POST",
            headers={"Origin": BROWSER_URL, "X-Irodori-Session": _session_token()})
        with urllib.request.urlopen(request, timeout=1.5) as response:
            if not 200 <= response.status < 300:
                return False
    except (OSError, urllib.error.URLError):
        return False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not engine_ready():
            return True
        time.sleep(0.1)
    return not engine_ready()


def _creation_flags() -> int:
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    if os.name == "nt":
        flags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return flags


def start_batch(batch: Path, box: Path) -> subprocess.Popen:
    return subprocess.Popen([os.environ.get("ComSpec", "cmd.exe"), "/d", "/c", str(batch)], cwd=str(box), creationflags=_creation_flags(), close_fds=False)


def stop_process(process: subprocess.Popen | None, wait: float = 3.0) -> None:
    if process is None or process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        try:
            process.terminate()
            process.wait(timeout=wait)
            return
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
    try:
        process.wait(timeout=wait)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _browser_ready() -> bool:
    try:
        with urllib.request.urlopen(BROWSER_URL, timeout=1.0) as response:
            if not 200 <= response.status < 500:
                return False
            body = response.read(256 * 1024).decode("utf-8", "ignore")
            return "VOICEVOX" in body and "Irodori" in body
    except (OSError, urllib.error.URLError):
        return False


class BrowserSession:
    def __init__(self, box: Path, launcher: Path):
        self.box = box
        self.launcher = launcher
        self.engine: subprocess.Popen | None = None
        self.browser: subprocess.Popen | None = None
        self._job = _create_job()
        self.cleaned = False
        self.streamers: list[LogStreamer] = []

    def _start_batch(self, batch: Path) -> subprocess.Popen:
        process = start_batch(batch, self.box)
        _assign_to_job(self._job, process)
        return process

    def cleanup(self) -> None:
        if self.cleaned:
            return
        self.cleaned = True
        for streamer in self.streamers:
            streamer.stop()
        self.streamers = []
        if self.engine is not None and self.engine.poll() is None:
            shutdown_engine()
        stop_process(self.engine)
        stop_process(self.browser)
        _close_job(self._job)
        self._job = None

    def start_debug_streams(self) -> None:
        """デバッグ時: エンジンとUIのログをコンソールへ流し始める。"""
        self.streamers = [
            LogStreamer(self.box / "logs" / "browser-engine.log", "engine", LOG_SKIP_PATTERNS),
            LogStreamer(self.box / "logs" / "browser-ui.log", "ui"),
        ]
        for streamer in self.streamers:
            streamer.start()

    def wait_ready(self) -> None:
        deadline = time.monotonic() + STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if not self.launcher.exists():
                raise RuntimeError("launcher file was deleted")
            if self.engine and self.engine.poll() is not None:
                raise RuntimeError(f"editor engine exited with code {self.engine.returncode}")
            if self.browser and self.browser.poll() is not None:
                raise RuntimeError(f"browser UI exited with code {self.browser.returncode}")
            if engine_ready() and _browser_ready():
                return
            time.sleep(POLL_INTERVAL)
        raise RuntimeError("services did not become ready before timeout")

    def run(self) -> int:
        if engine_ready() and not shutdown_engine():
            raise RuntimeError("existing editor engine did not shut down")
        if DEBUG:
            print("=== デバッグモード: 生成時間などの計測値をこのコンソールに流します ===", flush=True)
            self.start_debug_streams()
        self.engine = self._start_batch(self.box / "bat" / "serve_editor_engine.bat")
        self.browser = self._start_batch(self.box / "bat" / "serve_browser.bat")
        self.wait_ready()
        if DEBUG:
            if _CONSOLE_UI is not None:
                _CONSOLE_UI.print_markup(f"[bold #8fd9b6]✔ 起動しました[/]  [#7f948a]エンジン[/] :{ENGINE_PORT}"
                                         f"  [#7f948a]UI[/] [underline #8ecbff]{BROWSER_URL}[/]")
            else:
                print("[session] engine and browser UI are ready", flush=True)
        webbrowser.open(BROWSER_URL)
        while self.launcher.exists():
            if self.engine.poll() is not None or self.browser.poll() is not None:
                raise RuntimeError("a session service exited")
            time.sleep(POLL_INTERVAL)
        return 0


def check(box: Path) -> int:
    python = resolve_python(box)
    required = [box / "open_browser.bat", box / "bat" / "serve_editor_engine.bat", box / "bat" / "serve_browser.bat", box / "irodori-tts" / "wrapper" / "editor_engine.py"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))
    print(f"project root: {box}")
    print(f"python: {python}")
    print("session files: OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher", type=Path, default=project_root() / "open_browser.bat")
    parser.add_argument("--check", action="store_true", help="validate paths without starting services")
    parser.add_argument("--no-pause", action="store_true", help="accepted for launcher compatibility")
    parser.add_argument("--debug", action="store_true", help="stream engine/UI logs and timings to the console")
    args = parser.parse_args(argv)
    if args.debug:
        global DEBUG
        DEBUG = True
    box = project_root()
    if args.check:
        return check(box)
    session = BrowserSession(box, args.launcher.resolve())
    atexit.register(session.cleanup)
    for signal_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        signal_number = getattr(signal, signal_name, None)
        if signal_number is not None:
            signal.signal(signal_number, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        return session.run()
    except KeyboardInterrupt:
        return 130
    except (OSError, RuntimeError, FileNotFoundError) as exc:
        print(f"Browser session failed: {exc}", file=sys.stderr)
        print(f"Engine log: {box / 'logs' / 'browser-engine.log'}", file=sys.stderr)
        print(f"Browser log: {box / 'logs' / 'browser-ui.log'}", file=sys.stderr)
        return 1
    finally:
        session.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
