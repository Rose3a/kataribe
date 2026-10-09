"""窓デコード（ストリーミング再生）の契約テスト。実モデルは読まない。

コーデックは畳み込みなので、前後に余白をつけた窓で分けて復号しても、一度に復号した
結果と同じになる。ここでは「窓の割り方」と「sink を渡さなければ素通し」「渡すと
最後の窓だけ手元に残して他を送る」を、受容野が有限な偽のコーデックで確かめる。
"""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wrapper"))
from stream_decode import (FIRST_FRAMES, MARGIN, StreamingDecode, current_sink,  # noqa: E402
                           stream_to, windows)

HOP = 4
RECEPTIVE = 3  # 偽コーデックが片側に見るフレーム数（MARGIN 以下）


def fake_decode(latent: torch.Tensor) -> torch.Tensor:
    """各フレームを、前後 RECEPTIVE フレームの和から HOP サンプルへ展開する。"""
    frames = latent[..., 0]
    padded = torch.nn.functional.pad(frames, (RECEPTIVE, RECEPTIVE))
    mixed = sum(padded[:, i:i + frames.shape[1]] for i in range(2 * RECEPTIVE + 1))
    return mixed.repeat_interleave(HOP, dim=1).unsqueeze(1)


class WindowTests(unittest.TestCase):
    def test_windows_tile_the_latent(self):
        for total in (1, FIRST_FRAMES, FIRST_FRAMES + 1, 100, 393):
            spans = list(windows(total))
            self.assertEqual(spans[0][0], 0)
            self.assertEqual(spans[-1][1], total)
            for (_, end), (start, _) in zip(spans, spans[1:]):
                self.assertEqual(end, start)

    def test_first_window_is_small_then_grows(self):
        sizes = [end - start for start, end in windows(300)]
        self.assertEqual(sizes[:4], [FIRST_FRAMES, 24, 48, 48])


class StreamingDecodeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.latent = torch.randn(1, 200, 1)
        self.wrapper = StreamingDecode(fake_decode, HOP, 48000)

    def test_without_sink_is_a_pass_through(self):
        self.assertTrue(torch.equal(self.wrapper(self.latent), fake_decode(self.latent)))

    def test_windowed_result_equals_one_shot(self):
        self.assertLessEqual(RECEPTIVE, MARGIN)
        sent = []
        self.wrapper.begin(lambda data, rate: sent.append((data, rate)))
        windowed = self.wrapper(self.latent)
        self.wrapper.end()
        self.assertTrue(torch.allclose(windowed, fake_decode(self.latent)))
        self.assertGreater(len(sent), 1)
        self.assertEqual(sent[0][1], 48000)
        # 最初の窓は FIRST_FRAMES ぶん（int16 で 2 バイト/サンプル）。
        self.assertEqual(len(sent[0][0]), FIRST_FRAMES * HOP * 2)

    def test_last_window_is_held_back(self):
        sent = []
        self.wrapper.begin(lambda data, rate: sent.append(data))
        result = self.wrapper(self.latent)
        self.wrapper.end()
        sent_samples = sum(len(data) // 2 for data in sent)
        self.assertEqual(sent_samples, self.wrapper.sent_samples)
        self.assertLess(sent_samples, result.shape[-1])

    def test_short_latent_is_not_windowed(self):
        sent = []
        self.wrapper.begin(lambda data, rate: sent.append(data))
        short = torch.randn(1, FIRST_FRAMES + MARGIN, 1)
        self.assertTrue(torch.equal(self.wrapper(short), fake_decode(short)))
        self.assertEqual(sent, [])

    def test_batches_are_not_streamed(self):
        sent = []
        self.wrapper.begin(lambda data, rate: sent.append(data))
        batch = torch.randn(2, 100, 1)
        self.wrapper(batch)
        self.assertEqual(sent, [])


class SinkContextTests(unittest.TestCase):
    def test_sink_is_scoped(self):
        self.assertIsNone(current_sink())
        marker = object()
        with stream_to(marker):
            self.assertIs(current_sink(), marker)
        self.assertIsNone(current_sink())


if __name__ == "__main__":
    unittest.main()
