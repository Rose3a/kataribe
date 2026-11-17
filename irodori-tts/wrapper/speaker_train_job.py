"""エンジン側の話者学習ジョブ: speaker_train.py を子プロセスで動かし、進捗をエディタに返す。

- 同時に動かすのは 1 つだけ。子プロセスはエンジンと同じ venv の python で起動する
  （process_job.tie_children_to_this_process で、エンジンが終われば一緒に止まる）。
- 子プロセスは標準出力に JSON 1 行ずつで進捗を出す（speaker_train.emit）。それ以外の行はログとして末尾だけ残す。
  学習中の loss などは event=metrics（kind=train / val）で届き、history にためてエディタのグラフに出す。
- 中止は子プロセスとその子（書き起こしの Haas）をまとめて止める。「保存して終える」は止めずに合図のファイルを置き、
  学習側がその時点の話者を保存して終わる。
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
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
LOG_LINES = 200
STOP_FILE = HERE.parents[1] / ".cache" / "speaker-train" / "stop.flag"

# エディタの要求のキー → speaker_train の引数名（数の設定）
NUMBER_KEYS = {"steps": "steps", "lr": "lr", "warmup": "warmup", "batch": "batch", "durWeight": "dur_weight",
               "valFrac": "val_frac", "valEvery": "val_every", "seed": "seed"}

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
                "error": None, "params": None, "startedAt": None, "finishedAt": None, "detail": {},
                "config": None, "history": {"train": [], "val": []}, "stopRequested": False}

    # ------------------------------------------------------------ 状態
    def running(self) -> bool:
        with self.lock:
            return self.state["state"] == "running"

    def status(self) -> dict:
        with self.lock:
            value = dict(self.state)
            value["log"] = list(self.state["log"])[-60:]
            value["history"] = {k: list(v) for k, v in self.state["history"].items()}
            value.pop("cancelRequested", None)
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
            "defaults": {"steps": st.DEFAULT_STEPS, "speed": "learn", "device": "cuda" if cuda else "cpu",
                         "lr": st.DEFAULT_LR, "lrSchedule": "constant", "warmup": 0, "batch": 0,
                         "durWeight": st.DEFAULT_DUR_WEIGHT, "valFrac": st.DEFAULT_VAL_FRAC, "valEvery": 0,
                         "keep": "last", "seed": 0},
            "limits": {key: [st.LIMITS[arg][0], st.LIMITS[arg][1]] for key, arg in NUMBER_KEYS.items()},
            "valMinTotal": st.VAL_MIN_TOTAL,
            "speakerFolder": str(st.speaker_root() / st.SPEAKER_SUBDIR),
            "existing": sorted(p.parent.name for p in (st.speaker_root() / st.SPEAKER_SUBDIR).glob("*/*.speaker.safetensors")),
            "runs": SpeakerTrainJob.runs(),
        }

    # ------------------------------------------------------------ 過去の学習（speakers/学習した話者/<名前>/<名前>.train.json）
    @staticmethod
    def runs() -> list[dict]:
        """過去の学習の一覧（新しい順）。グラフ用の履歴は record() で取る。"""
        import speaker_train as st

        found = []
        for path in (st.speaker_root() / st.SPEAKER_SUBDIR).glob(f"*/*{st.TRAIN_META_SUFFIX}"):
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(rec, dict):
                continue
            found.append({"name": path.parent.name, "created": rec.get("created"), "steps": rec.get("steps"),
                          "lr": rec.get("lr"), "hasHistory": bool((rec.get("history") or {}).get("train")),
                          "metrics": rec.get("metrics") or {}})
        return sorted(found, key=lambda r: str(r.get("created") or ""), reverse=True)

    @staticmethod
    def record(name: str) -> dict:
        import speaker_train as st

        try:
            name = st.validate_name(name)
        except st.TrainError as exc:
            raise ValueError(str(exc)) from exc
        _, path = st.output_paths(name)
        if not path.is_file():
            raise FileNotFoundError(f"「{name}」の学習の記録がありません")
        return json.loads(path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------ 開始・中止
    @staticmethod
    def _settings(payload: dict) -> SimpleNamespace:
        """学習の設定（数・選択）を読み、speaker_train.check_settings で範囲を確かめる。"""
        import speaker_train as st

        args = SimpleNamespace(lr_schedule=payload.get("lrSchedule", "constant"), keep=payload.get("keep", "last"))
        defaults = {"steps": st.DEFAULT_STEPS, "lr": st.DEFAULT_LR, "warmup": 0, "batch": 0,
                    "dur_weight": st.DEFAULT_DUR_WEIGHT, "val_frac": st.DEFAULT_VAL_FRAC, "val_every": 0, "seed": 0}
        for key, arg in NUMBER_KEYS.items():
            value = payload.get(key, defaults[arg])
            # 整数の設定に 500.0 のような値が来たら整数にする（JSON の数は区別しない）
            if isinstance(st.LIMITS[arg][0], int) and isinstance(value, float) and value.is_integer():
                value = int(value)
            elif isinstance(st.LIMITS[arg][0], float) and isinstance(value, int) and not isinstance(value, bool):
                value = float(value)
            setattr(args, arg, value)
        try:
            st.check_settings(args)
        except st.TrainError as exc:
            raise ValueError(str(exc)) from exc
        return args

    def start(self, payload: dict, before_start=None) -> dict:
        import speaker_train as st

        if not isinstance(payload, dict):
            raise ValueError("request must be an object")
        try:
            name = st.validate_name(str(payload.get("name") or ""))
        except st.TrainError as exc:
            raise ValueError(str(exc)) from exc
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
        args = self._settings(payload)
        verify = bool(payload.get("verifyText", True))
        overwrite = bool(payload.get("overwrite", False))
        target, _ = st.output_paths(name)
        if target.exists() and not overwrite:
            raise ValueError(f"「{name}」という学習した話者はもうあります。別の名前にするか、上書きを選んでください")

        with self.lock:
            if self.state["state"] == "running":
                raise RuntimeError("話者学習はもう動いています。終わるか中止してから始めてください")
            params = {"name": name, "dataDir": data, "baseModel": base or None, "speed": speed, "device": device,
                      "steps": args.steps, "lr": args.lr, "lrSchedule": args.lr_schedule, "warmup": args.warmup,
                      "batch": args.batch, "durWeight": args.dur_weight, "valFrac": args.val_frac,
                      "valEvery": args.val_every, "keep": args.keep, "seed": args.seed,
                      "verifyText": verify, "overwrite": overwrite}
            self.state = self._idle()
            self.state.update(state="running", message="準備しています", params=params, startedAt=time.time())
        if before_start is not None:
            before_start(device)

        STOP_FILE.parent.mkdir(parents=True, exist_ok=True)
        STOP_FILE.unlink(missing_ok=True)
        cmd = [sys.executable, str(HERE / "speaker_train.py"), "--name", name, "--data", data,
               "--speed", speed, "--steps", str(args.steps), "--device", device, "--verify-text", "1" if verify else "0",
               "--lr", repr(float(args.lr)), "--lr-schedule", args.lr_schedule, "--warmup", str(args.warmup),
               "--batch", str(args.batch), "--dur-weight", repr(float(args.dur_weight)),
               "--val-frac", repr(float(args.val_frac)), "--val-every", str(args.val_every), "--keep", args.keep,
               "--seed", str(args.seed), "--stop-file", str(STOP_FILE)]
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

    def stop(self) -> dict:
        """学習をここで終え、その時点の話者を保存させる（中止と違って結果が残る）。学習のステップに入ってから使える。"""
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return self.status()
        with self.lock:
            if self.state["stage"] != "train":
                raise RuntimeError("「保存して終える」は学習のステップに入ってから使えます（今は準備中です）")
            self.state.update(stopRequested=True, message="今の話者を保存して終えます")
        STOP_FILE.parent.mkdir(parents=True, exist_ok=True)
        STOP_FILE.write_text("stop", encoding="utf-8")
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
                    message = msg.get("message", "")
                    if self.state.get("stopRequested") and msg.get("stage") == "train":
                        message = "今の話者を保存して終えます"
                    self.state.update(percent=msg.get("percent", self.state["percent"]),
                                      stage=msg.get("stage", ""), message=message)
                    detail = {k: v for k, v in msg.items() if k not in ("event", "stage", "percent", "message")}
                    if detail:
                        self.state["detail"] = {**self.state["detail"], **detail}
                elif event == "metrics":
                    kind = msg.get("kind")
                    if kind in ("train", "val"):
                        self.state["history"][kind].append(
                            {k: v for k, v in msg.items() if k not in ("event", "kind")})
                elif event == "config":
                    self.state["config"] = {k: v for k, v in msg.items() if k != "event"}
                elif event == "done":
                    self.state["result"] = msg.get("result")
                elif event == "error":
                    self.state["error"] = msg.get("message")
                    if msg.get("traceback"):
                        log.append(msg["traceback"])
                        self.state["log"] = list(log)
        code = proc.wait()
        try:
            STOP_FILE.unlink(missing_ok=True)
        except OSError:
            pass
        with self.lock:
            if self.state.get("cancelRequested"):
                self.state.update(state="cancelled", message="中止しました")
            elif code == 0 and self.state["result"]:
                stopped = ((self.state["result"] or {}).get("metrics") or {}).get("stopped_at")
                self.state.update(state="done", percent=100,
                                  message="完了しました" if stopped is None else f"ステップ {stopped} で終えて保存しました")
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
