import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

WRAPPER = Path(__file__).resolve().parents[1] / "wrapper"

# エンジン役: ジョブに入ってから、長く眠る子（plan 構築役）を起動し、その pid を出して待つ。
ENGINE = f"""
import subprocess, sys, time
sys.path.insert(0, {str(WRAPPER)!r})
from process_job import tie_children_to_this_process
assert tie_children_to_this_process()
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
print(child.pid, flush=True)
time.sleep(60)
"""


def alive(pid):
    import ctypes
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(0x1000 | 0x00100000, False, pid)  # QUERY_LIMITED | SYNCHRONIZE
    if not handle:
        return False
    try:
        return kernel32.WaitForSingleObject(handle, 0) == 0x102  # WAIT_TIMEOUT
    finally:
        kernel32.CloseHandle(handle)


@unittest.skipUnless(os.name == "nt", "Windows job objects only")
class ProcessJobTest(unittest.TestCase):
    def test_child_dies_when_engine_is_killed(self):
        engine = subprocess.Popen([sys.executable, "-c", ENGINE], stdout=subprocess.PIPE, text=True)
        try:
            child = int(engine.stdout.readline())
            self.assertTrue(alive(child))
            engine.kill()  # 強制終了（タスクマネージャーで止めたのと同じ）
            engine.wait(10)
            deadline = time.monotonic() + 10
            while alive(child) and time.monotonic() < deadline:
                time.sleep(0.1)
            self.assertFalse(alive(child))
        finally:
            if engine.poll() is None:
                engine.kill()


if __name__ == "__main__":
    unittest.main()
