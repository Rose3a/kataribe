"""エンジン側の話者学習ジョブ: speaker_train.py を子プロセスで動かし、進捗をエディタに返す。

- 同時に動かすのは 1 つだけ。子プロセスはエンジンと同じ venv の python で起動する
  （process_job.tie_children_to_this_process で、エンジンが終われば一緒に止まる）。
- 子プロセスは標準出力に JSON 1 行ずつで進捗を出す（speaker_train.emit）。それ以外の行はログとして末尾だけ残す。
- 中止は子プロセスとその子（書き起こしの Haas）をまとめて止める。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG_LINES = 200

# フォルダ選びの画面（Windows では新しい形式のフォルダ選択）。Tk はエンジンのスレッドと混ぜないよう、別プロセスで開く
_PICK_FOLDER_SCRIPT = r"""
import sys
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()
try:
    root.attributes("-topmost", True)
except tk.TclError:
    pass
root.update()
path = filedialog.askdirectory(parent=root, title=sys.argv[1], initialdir=sys.argv[2] or None, mustexist=True)
root.destroy()
sys.stdout.write(path or "")
"""
_pick_lock = threading.Lock()


def pick_folder(initial: str = "", title: str = "音声のフォルダを選ぶ") -> dict:
    """エンジンを動かしている PC で、フォルダ選択の画面を開く。選んだフォルダ（やめたら None）を返す。"""
    if not _pick_lock.acquire(blocking=False):
        raise RuntimeError("フォルダを選ぶ画面がもう開いています")
    try:
        start = initial if initial and Path(initial).is_dir() else ""
        proc = subprocess.run([sys.executable, "-c", _PICK_FOLDER_SCRIPT, title, start], capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=1800,
                              env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        if proc.returncode != 0:
            raise RuntimeError("フォルダを選ぶ画面を開けませんでした: " + (proc.stderr or "").strip()[-300:])
        path = proc.stdout.strip()
        return {"path": str(Path(path)) if path else None}
    finally:
        _pick_lock.release()


class SpeakerTrainJob:
    def __init__(self, on_finished=None):
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.on_finished = on_finished
        self.state = self._idle()

    @staticmethod
    def _idle() -> dict:
        return {"state": "idle", "percent": 0, "stage": "", "message": "", "log": [], "result": None,
                "error": None, "params": None, "startedAt": None, "finishedAt": None, "detail": {}}

    # ------------------------------------------------------------ 状態
    def running(self) -> bool:
        with self.lock:
            return self.state["state"] == "running"

    def status(self) -> dict:
        with self.lock:
            value = dict(self.state)
            value["log"] = list(self.state["log"])[-60:]
            return value

    @staticmethod
    def options() -> dict:
        import speaker_train as st
        try:
            import torch
            cuda = bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001
            cuda = False
        return {
            "baseModels": st.list_base_models(),
            "haasAvailable": st.haas_python() is not None,
            "cudaAvailable": cuda,
            "defaults": {"steps": st.DEFAULT_STEPS, "speed": "learn", "device": "cuda" if cuda else "cpu"},
            "speakerFolder": str(st.speaker_root() / st.SPEAKER_SUBDIR),
            "existing": sorted(p.parent.name for p in (st.speaker_root() / st.SPEAKER_SUBDIR).glob("*/*.speaker.safetensors")),
        }

    # ------------------------------------------------------------ 開始・中止
    def start(self, payload: dict, before_start=None) -> dict:
        import speaker_train as st

        if not isinstance(payload, dict):
            raise ValueError("request must be an object")
        name = st.validate_name(str(payload.get("name") or ""))
        data = str(payload.get("dataDir") or "").strip().strip('"')
        if not data or not Path(data).is_dir():
            raise ValueError(f"音声のフォルダが見つかりません: {data or '（未指定）'}")
        base = str(payload.get("baseModel") or "").strip()
        speed = payload.get("speed", "learn")
        if speed not in ("learn", "keep"):
            raise ValueError("speed は learn か keep です")
        device = payload.get("device", "auto")
        if device not in ("auto", "cuda", "cpu"):
            raise ValueError("device は auto / cuda / cpu です")
        steps = payload.get("steps", st.DEFAULT_STEPS)
        if isinstance(steps, bool) or not isinstance(steps, int) or not 50 <= steps <= 5000:
            raise ValueError("ステップ数は 50〜5000 です")
        verify = bool(payload.get("verifyText", True))
        overwrite = bool(payload.get("overwrite", False))
        target, _ = st.output_paths(name)
        if target.exists() and not overwrite:
            raise ValueError(f"「{name}」という学習した話者はもうあります。別の名前にするか、上書きを選んでください")

        with self.lock:
            if self.state["state"] == "running":
                raise RuntimeError("話者学習はもう動いています。終わるか中止してから始めてください")
            params = {"name": name, "dataDir": data, "baseModel": base or None, "speed": speed, "device": device,
                      "steps": steps, "verifyText": verify, "overwrite": overwrite}
            self.state = self._idle()
            self.state.update(state="running", message="準備しています", params=params, startedAt=time.time())
        if before_start is not None:
            before_start(device)

        cmd = [sys.executable, str(HERE / "speaker_train.py"), "--name", name, "--data", data,
               "--speed", speed, "--steps", str(steps), "--device", device, "--verify-text", "1" if verify else "0"]
        if base:
            cmd += ["--base", base]
        if overwrite:
            cmd.append("--overwrite")
        env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                         encoding="utf-8", errors="replace", env=env, cwd=str(HERE),
                                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as exc:
            with self.lock:
                self.state.update(state="error", error=str(exc), finishedAt=time.time())
            raise
        threading.Thread(target=self._read, args=(self.proc,), daemon=True, name="speaker-train").start()
        return self.status()

    def cancel(self) -> dict:
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return self.status()
        with self.lock:
            self.state["cancelRequested"] = True
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            proc.kill()
        return self.status()

    # ------------------------------------------------------------ 子プロセスの出力
    def _read(self, proc: subprocess.Popen) -> None:
        log: deque[str] = deque(maxlen=LOG_LINES)
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\r\n")
            msg = None
            if line.startswith("{"):
                try:
                    msg = json.loads(line)
                except ValueError:
                    msg = None
            with self.lock:
                if msg is None:
                    if line.strip():
                        log.append(line)
                        self.state["log"] = list(log)
                    continue
                event = msg.get("event")
                if event == "progress":
                    self.state.update(percent=msg.get("percent", self.state["percent"]),
                                      stage=msg.get("stage", ""), message=msg.get("message", ""))
                    detail = {k: v for k, v in msg.items() if k not in ("event", "stage", "percent", "message")}
                    if detail:
                        self.state["detail"] = {**self.state["detail"], **detail}
                elif event == "done":
                    self.state["result"] = msg.get("result")
                elif event == "error":
                    self.state["error"] = msg.get("message")
                    if msg.get("traceback"):
                        log.append(msg["traceback"])
                        self.state["log"] = list(log)
        code = proc.wait()
        with self.lock:
            if self.state.get("cancelRequested"):
                self.state.update(state="cancelled", message="中止しました")
            elif code == 0 and self.state["result"]:
                self.state.update(state="done", percent=100, message="完了しました")
            else:
                self.state.update(state="error", error=self.state["error"] or f"学習が止まりました（終了コード {code}）")
            self.state["finishedAt"] = time.time()
            finished = dict(self.state)
        self.proc = None
        if self.on_finished is not None:
            try:
                self.on_finished(finished)
            except Exception:  # noqa: BLE001 - 一覧の更新に失敗しても学習の結果は残す
                pass
