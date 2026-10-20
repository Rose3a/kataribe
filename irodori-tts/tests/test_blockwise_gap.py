"""前のセリフから続けて生成（blockwise）の行間そろえ。

- 行間（前の行の末尾の無音＋続きの行の先頭の無音）は既定ではモデルの判断のまま、
- prefix_gap_scale 倍に縮める・prefix_gap_max 秒までに切る、ができる（伸ばしはしない）、
- 末尾の無音は tail_silence_max 秒までに切る、
- 前の行の WAV から末尾の無音の長さを測る。

聞き比べ（work/blockwise_listen/gen_gap.py）では、行間 0.3〜1.7 秒が上限 0.6 秒で 0.3〜0.63 秒になった。
"""
from pathlib import Path
import io
import sys
import types
import unittest

import numpy as np
import torch

BOX = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BOX / "irodori-tts" / "wrapper"))
sys.path.insert(0, str(BOX / "runtime" / "trt-lab" / "repo"))

from irodori_tts.inference_runtime import (  # noqa: E402
    InferenceRuntime, SamplingRequest, leading_silence_seconds, lookahead_cut_frame)
from context_latents import trailing_silence_seconds  # noqa: E402

HOP, RATE = 100, 2500   # 25 フレーム/秒（本物の codec と同じ比）


class _Codec:
    """latent の 1 列目が 0 のフレームを無音、それ以外を音ありとしてデコードする。"""
    sample_rate = RATE
    model = types.SimpleNamespace(hop_length=HOP)

    def __init__(self):
        self.calls = 0

    def decode_latent(self, latent):
        self.calls += 1
        loud = (latent[..., 0] != 0).float()                       # (B, T)
        return loud.repeat_interleave(HOP, dim=1)[:, None, :] * 0.5  # (B, 1, T*HOP)


def _runtime():
    runtime = InferenceRuntime.__new__(InferenceRuntime)
    runtime.codec = _Codec()
    return runtime


def _latent(head, speech, tail):
    z = torch.zeros(head + speech + tail, 4)
    z[head:head + speech, 0] = 1.0
    return z


class BlockwiseGapTest(unittest.TestCase):
    def test_leading_silence(self):
        audio = torch.cat([torch.zeros(RATE // 2), torch.full((RATE,), 0.5)])
        self.assertAlmostEqual(leading_silence_seconds(audio, RATE), 0.5, places=2)

    def test_long_gap_is_cut_to_the_limit(self):
        runtime = _runtime()
        req = SamplingRequest(text="x", prefix_gap_max=0.6, prefix_tail_seconds=0.2)
        # 先頭 0.9 秒（23 フレーム弱）の無音 → 0.6 - 0.2 = 0.4 秒を残して切る
        head, tail = runtime._edge_trim_frames(_latent(22, 50, 3), req, prefixed=True)
        self.assertEqual(head, int((22 / 25 - 0.4) * 25))
        self.assertEqual(tail, 0)

    def test_model_gap_is_kept_by_default(self):
        runtime = _runtime()
        req = SamplingRequest(text="x", prefix_tail_seconds=0.2)
        self.assertEqual(runtime._edge_trim_frames(_latent(22, 50, 3), req, prefixed=True), (0, 0))
        self.assertEqual(runtime.codec.calls, 0)

    def test_gap_scale_keeps_the_ratio(self):
        runtime = _runtime()
        req = SamplingRequest(text="x", prefix_gap_scale=0.5, prefix_tail_seconds=0.2)
        # 行間 0.2 + 1.0 = 1.2 秒 → 0.6 秒。先頭は 0.4 秒を残す
        head, _ = runtime._edge_trim_frames(_latent(25, 50, 0), req, prefixed=True)
        self.assertEqual(head, int((1.0 - 0.4) * 25))

    def test_short_gap_is_kept(self):
        runtime = _runtime()
        req = SamplingRequest(text="x", prefix_gap_max=0.6, prefix_tail_seconds=0.2)
        self.assertEqual(runtime._edge_trim_frames(_latent(5, 50, 3), req, prefixed=True), (0, 0))

    def test_onset_margin_when_previous_tail_is_long(self):
        runtime = _runtime()
        req = SamplingRequest(text="x", prefix_gap_max=0.6, prefix_tail_seconds=1.0)
        head, _ = runtime._edge_trim_frames(_latent(20, 50, 0), req, prefixed=True)
        self.assertEqual(head, int((20 / 25 - 0.05) * 25))

    def test_first_line_head_is_untouched_but_tail_is_capped(self):
        runtime = _runtime()
        req = SamplingRequest(text="x", tail_silence_max=0.3)
        head, tail = runtime._edge_trim_frames(_latent(20, 50, 25), req, prefixed=False)
        self.assertEqual(head, 0)
        self.assertEqual(tail, int((25 / 25 - 0.3) * 25))

    def test_trial_decode_bypasses_the_streaming_wrapper(self):
        runtime = _runtime()
        inner = runtime.codec.decode_latent
        wrapper = types.SimpleNamespace(inner=inner)
        runtime.codec.decode_latent = wrapper   # 包み自体を呼ぶと TypeError になる
        req = SamplingRequest(text="x", prefix_gap_max=0.6)
        self.assertGreater(runtime._edge_trim_frames(_latent(20, 50, 0), req, prefixed=True)[0], 0)

    def _speech(self, *parts):
        """(秒, 音ありか) の並びから波形を作る。"""
        return torch.cat([torch.full((int(sec * RATE),), 0.5 if loud else 0.0) for sec, loud in parts])

    def test_lookahead_cuts_at_the_end_of_this_line(self):
        # この行 2 秒（途中に 0.2 秒の間）→ 行の間 0.6 秒 → 次の行 1 秒
        audio = self._speech((1.0, True), (0.2, False), (0.8, True), (0.6, False), (1.0, True))
        frames = audio.numel() // HOP
        cut = lookahead_cut_frame(audio, RATE, HOP, ratio=0.62, total_frames=frames)
        # 行の間の頭（2.0 秒）＋減衰ぶん 0.04 秒
        self.assertEqual(cut, int(np.ceil((2.0 + 0.04) * 25)))

    def test_lookahead_gives_up_without_a_clear_gap(self):
        # 「お代官様！」の直後に笑い声（ハとハの間に 0.1 秒の無音）。笑いの中で切らない
        audio = self._speech((1.0, True), (0.2, True), (0.1, False), (0.2, True), (0.1, False), (0.2, True))
        frames = audio.numel() // HOP
        self.assertIsNone(lookahead_cut_frame(audio, RATE, HOP, ratio=0.6, total_frames=frames))

    def test_trailing_silence_of_previous_wav(self):
        import soundfile as sf
        audio = np.concatenate([np.full(24000, 0.3), np.zeros(12000)]).astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, audio, 24000, format="WAV", subtype="PCM_16")
        self.assertAlmostEqual(trailing_silence_seconds(buf.getvalue()), 0.5, places=2)


if __name__ == "__main__":
    unittest.main()
