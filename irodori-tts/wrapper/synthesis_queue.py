"""合成の順番待ち。

GPU では1件ずつしか合成しないので、同時に届いた要求は届いた順に待たせる。
VOICEVOX と同じく待たせるのが基本で、待ちが上限を超えたときだけ断る（429）。
待っている間に接続が切れた要求は、順番が来ても合成せずに飛ばす。
"""

from __future__ import annotations

import select
import socket
import threading
from collections import deque
from contextlib import contextmanager

# 待ち行列の上限。一括書き出しや外部ツールの並列要求を受け止めつつ、際限なく溜めない。
QUEUE_LIMIT = 32
# 順番待ち中に、接続が切れていないかを確かめる間隔（秒）
POLL_SECONDS = 0.5


class QueueFull(Exception):
    pass


class ClientGone(Exception):
    pass


def connection_closed(sock) -> bool:
    """相手が接続を閉じたか。読める状態で 0 バイトなら閉じている。判定できないときは False。"""
    if sock is None:
        return False
    try:
        readable, _, _ = select.select([sock], [], [], 0)
        if not readable:
            return False
        return sock.recv(1, socket.MSG_PEEK) == b""
    except (ConnectionError, OSError):
        return True
    except Exception:
        return False


class SynthesisQueue:
    def __init__(self, limit: int = QUEUE_LIMIT):
        self.limit = limit
        self._cond = threading.Condition()
        self._waiting: deque[object] = deque()
        self._running = False

    @property
    def waiting(self) -> int:
        with self._cond:
            return len(self._waiting)

    @contextmanager
    def turn(self, gone=lambda: False):
        """順番が来たら合成してよい。gone() が真になったら ClientGone で抜ける。"""
        me = object()
        with self._cond:
            if len(self._waiting) >= self.limit:
                raise QueueFull()
            self._waiting.append(me)
            try:
                while self._running or self._waiting[0] is not me:
                    self._cond.wait(POLL_SECONDS)
                    if gone():
                        raise ClientGone()
                if gone():
                    raise ClientGone()
            except BaseException:
                self._waiting.remove(me)
                self._cond.notify_all()
                raise
            self._waiting.popleft()
            self._running = True
        try:
            yield
        finally:
            with self._cond:
                self._running = False
                self._cond.notify_all()
