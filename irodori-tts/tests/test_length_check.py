"""長さの測り直し（2 パス合成、length_check.py）の契約テスト。実モデルは読まない。"""
import dataclasses
from pathlib import Path
import sys
import types
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wrapper"))
import length_check as lc  # noqa: E402

SMALL = types.SimpleNamespace(flow_parameterization="rf_velocity", num_layers=12, num_heads=20,
                              model_dim=1280, latent_patch_size=1)


@dataclasses.dataclass
class Request:
    text: str = "テスト"
    seconds: float | None = None
    seed: int | None = None


@dataclasses.dataclass
class Result:
    used_seed: int
    total_to_decode: float = 0.1


class FakeAttention(torch.nn.Module):
    """層 0 の注意の代わり。呼ばれたことだけ分かればよい（ポインタは recorder を差し替えて与える）。"""

    def forward(self, **kwargs):
        return kwargs["x"]


class FakeRuntime:
    def __init__(self, cfg=SMALL):
        self.model_cfg = cfg
        self.codec = types.SimpleNamespace(sample_rate=48000, model=types.SimpleNamespace(hop_length=1920))
        self.model = types.SimpleNamespace(blocks=[types.SimpleNamespace(attention=FakeAttention())])
        self.requests = []

    def synthesize(self, request, log_fn=None):
        self.requests.append(request)
        return Result(used_seed=request.seed if request.seed is not None else 99)


class EstimateTest(unittest.TestCase):
    def test_end_is_first_frame_on_second_to_last_token_plus_tail(self):
        # トークン 6 個（BOS 込み）。4 に届くのは 30 ステップ目（0 始まり）
        pointer = torch.tensor([0] * 10 + [2] * 10 + [3] * 10 + [4] * 5 + [5] * 45)
        frames = lc.estimated_frames(pointer, text_len=6, frames_per_step=1, frames_per_second=25.0)
        self.assertEqual(frames, int(torch.tensor((31 + 0.5 * 25) * lc.MARGIN).ceil()))

    def test_not_reached_returns_none(self):
        pointer = torch.tensor([0, 1, 2, 3])
        self.assertIsNone(lc.estimated_frames(pointer, text_len=8, frames_per_step=1, frames_per_second=25.0))

    def test_only_small_rf_architecture(self):
        self.assertTrue(lc.supported(SMALL))
        self.assertFalse(lc.supported(types.SimpleNamespace(**{**vars(SMALL), "flow_parameterization": "meanflow"})))
        self.assertFalse(lc.supported(types.SimpleNamespace(**{**vars(SMALL), "num_layers": 24})))


class TwoPassTest(unittest.TestCase):
    def run_with_pointer(self, pointer, text_len, runtime=None, request=None):
        runtime = runtime or FakeRuntime()

        class Recorder:
            def __init__(self, model):
                self.pointer, self.text_len = pointer, text_len

            def close(self):
                pass

        original = lc.PointerRecorder
        lc.PointerRecorder = Recorder
        try:
            result, info = lc.synthesize(runtime, request or Request())
        finally:
            lc.PointerRecorder = original
        return runtime, result, info

    def test_overlong_is_resynthesized_shorter_with_the_same_seed(self):
        # 200 ステップのうち 40 で読み終えている
        pointer = torch.tensor([0] * 20 + [3] * 20 + [4] * 160)
        runtime, _, info = self.run_with_pointer(pointer, text_len=6)
        self.assertEqual(info["length_check"], "resynthesized")
        self.assertEqual(len(runtime.requests), 2)
        second = runtime.requests[1]
        self.assertEqual(second.seed, 99)  # 1 回目のランダムなシードを引き継ぐ
        frames = round(second.seconds * 48000 / 1920)
        self.assertEqual(frames, info["estimated_frames"])
        self.assertLess(frames, 200)

    def test_fitting_length_is_kept(self):
        pointer = torch.tensor([0] * 40 + [3] * 40 + [4] * 20)
        runtime, _, info = self.run_with_pointer(pointer, text_len=6)
        self.assertEqual(info["length_check"], "kept")
        self.assertEqual(len(runtime.requests), 1)

    def test_too_short_is_never_lengthened(self):
        pointer = torch.tensor([0] * 20 + [2] * 20)  # 最後まで届かない（話速を上げすぎ）
        runtime, _, info = self.run_with_pointer(pointer, text_len=6)
        self.assertEqual(info["length_check"], "kept")
        self.assertEqual(len(runtime.requests), 1)

    def test_manual_seconds_and_other_models_are_skipped(self):
        pointer = torch.tensor([0] * 20 + [4] * 180)
        runtime, _, info = self.run_with_pointer(pointer, 6, request=Request(seconds=3.0))
        self.assertEqual((info["length_check"], len(runtime.requests)), ("skipped", 1))
        mf = FakeRuntime(types.SimpleNamespace(**{**vars(SMALL), "flow_parameterization": "meanflow"}))
        runtime, _, info = self.run_with_pointer(pointer, 6, runtime=mf)
        self.assertEqual((info["length_check"], len(runtime.requests)), ("skipped", 1))


class StreamingAndContextTest(unittest.TestCase):
    def test_first_pass_is_not_streamed_and_sink_is_restored(self):
        runtime = FakeRuntime()
        decoder = types.SimpleNamespace(sink="client")
        seen = []
        decoder.end = lambda: setattr(decoder, "sink", None)
        decoder.begin = lambda sink: setattr(decoder, "sink", sink)
        original_synth = runtime.synthesize

        def synth(request, log_fn=None):
            seen.append(decoder.sink)
            return original_synth(request, log_fn)

        runtime.synthesize = synth

        class Recorder:
            def __init__(self, model):
                self.pointer, self.text_len = torch.tensor([0] * 20 + [3] * 20 + [4] * 160), 6

            def close(self):
                pass

        original = lc.PointerRecorder
        lc.PointerRecorder = Recorder
        try:
            lc.synthesize(runtime, Request(), stream_decoder=decoder)
        finally:
            lc.PointerRecorder = original
        self.assertEqual(seen, [None, "client"])  # 1 回目は流さず、作り直しは流す
        self.assertEqual(decoder.sink, "client")

    def test_prefix_and_lookahead_are_skipped(self):
        @dataclasses.dataclass
        class ContextRequest(Request):
            prefix_latent: object = None
            lookahead_text: str | None = None

        for request in (ContextRequest(prefix_latent=torch.zeros(1)), ContextRequest(lookahead_text="次の行")):
            runtime = FakeRuntime()
            _, info = lc.synthesize(runtime, request)
            self.assertEqual((info["length_check"], len(runtime.requests)), ("skipped", 1))


class QueryFieldTest(unittest.TestCase):
    def test_engine_accepts_on_off_and_rejects_others(self):
        from voicevox_engine import RequestValidationError, _validate_query
        _validate_query({"irodori_length_check": "on"})
        _validate_query({"irodori_length_check": "off"})
        with self.assertRaises(RequestValidationError):
            _validate_query({"irodori_length_check": "auto"})


if __name__ == "__main__":
    unittest.main()
