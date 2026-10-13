"""エンジンが終わると、エンジンが起動した子プロセスも一緒に終わるようにする。

TensorRT の plan 構築（数分・GPU と VRAM を使う）や Radeon の作業プロセスが、エンジンを
閉じたあと（強制終了も含む）に残らないようにする。Windows では、エンジン自身を
「閉じたら中のプロセスを全部止める」ジョブオブジェクトに入れる。以後に起動した子・孫は
自動で同じジョブに入り、エンジンが終わってジョブのハンドルが閉じると止まる。

trt_cache.py などの子プロセスを起動する側は、編集すると TensorRT plan のキャッシュが
無効になる（ソースのハッシュが plan の識別に入っている）ので、こちらで包む。
"""

from __future__ import annotations

import os
import subprocess
import sys

JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x0800
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
CREATE_BREAKAWAY_FROM_JOB = 0x01000000

# ジョブのハンドル。プロセスが終わるまで閉じない（閉じると中のプロセスが止まる）。
_job = None


def tie_children_to_this_process() -> bool:
    """以後に起動する子プロセスを、このプロセスの終了と一緒に止める。Windows 以外では何もしない。"""
    global _job
    if os.name != "nt":
        return False
    if _job is not None:
        return True
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.SetInformationJobObject.argtypes = (
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)

        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class BasicLimits(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                        ("PerJobUserTimeLimit", ctypes.c_longlong),
                        ("LimitFlags", wintypes.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t),
                        ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BasicLimits),
                        ("IoInfo", IoCounters),
                        ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise OSError(ctypes.get_last_error(), "CreateJobObject failed")
        info = ExtendedLimits()
        # フォルダを開く Explorer などは、明示すればジョブの外へ出せるようにしておく。
        info.BasicLimitInformation.LimitFlags = (
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_BREAKAWAY_OK)
        if not kernel32.SetInformationJobObject(
                job, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info)):
            error = ctypes.get_last_error()
            kernel32.CloseHandle(job)
            raise OSError(error, "SetInformationJobObject failed")
        if not kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
            error = ctypes.get_last_error()
            kernel32.CloseHandle(job)
            raise OSError(error, "AssignProcessToJobObject failed")
        _job = job
        return True
    except Exception as exc:
        print(f"[irodori] warning: child processes may outlive the engine ({exc})",
              file=sys.stderr, flush=True)
        return False


def open_folder(path) -> None:
    """フォルダを Explorer で開く。エンジンを閉じてもウィンドウが一緒に閉じないよう、ジョブの外で起動する。"""
    if os.name != "nt":
        raise OSError("opening folders is supported on Windows only")
    flags = CREATE_BREAKAWAY_FROM_JOB if _job is not None else 0
    try:
        subprocess.Popen(["explorer", str(path)], creationflags=flags, close_fds=True)
    except OSError:
        os.startfile(str(path))
