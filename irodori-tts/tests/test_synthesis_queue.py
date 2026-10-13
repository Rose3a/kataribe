import socket
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wrapper"))

from synthesis_queue import ClientGone, QueueFull, SynthesisQueue, connection_closed  # noqa: E402


class SynthesisQueueTest(unittest.TestCase):
    def test_runs_one_at_a_time_in_arrival_order(self):
        queue = SynthesisQueue()
        order = []
        active = []
        first = threading.Event()

        def job(i):
            with queue.turn():
                active.append(i)
                self.assertEqual(len(active), 1)
                if i == 0:
                    first.wait(5)
                order.append(i)
                active.remove(i)

        threads = [threading.Thread(target=job, args=(0,))]
        threads[0].start()
        time.sleep(0.05)
        for i in range(1, 5):
            thread = threading.Thread(target=job, args=(i,))
            thread.start()
            threads.append(thread)
            time.sleep(0.05)
        first.set()
        for thread in threads:
            thread.join(10)
        self.assertEqual(order, [0, 1, 2, 3, 4])

    def test_rejects_when_waiting_line_is_full(self):
        queue = SynthesisQueue(limit=1)
        with queue.turn():
            waiter = threading.Thread(target=lambda: queue.turn().__enter__())
            waiter.start()
            time.sleep(0.1)
            with self.assertRaises(QueueFull):
                with queue.turn():
                    pass
        waiter.join(5)

    def test_waiter_leaves_when_client_is_gone(self):
        queue = SynthesisQueue()
        gone = threading.Event()
        result = []

        def waiter():
            try:
                with queue.turn(gone.is_set):
                    result.append("ran")
            except ClientGone:
                result.append("gone")

        with queue.turn():
            thread = threading.Thread(target=waiter)
            thread.start()
            gone.set()
            thread.join(5)
        self.assertEqual(result, ["gone"])
        self.assertEqual(queue.waiting, 0)
        with queue.turn():
            pass

    def test_connection_closed_detects_peer_close(self):
        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        client = socket.create_connection(server.getsockname())
        conn, _ = server.accept()
        try:
            self.assertFalse(connection_closed(conn))
            client.close()
            time.sleep(0.05)
            self.assertTrue(connection_closed(conn))
        finally:
            conn.close()
            server.close()


if __name__ == "__main__":
    unittest.main()
