"""長さの測り直し（2 パス合成、length_check.py）の契約テスト。実モデルは読まない。"""
import dataclasses
import math
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
    num_steps: int | None = 4


@dataclasses.dataclass
class Result:
    used_seed: int
    total_to_decode: float = 0.1


class FakeAttention(torch.nn.Module):
    """層 0 の注意の代わり。呼ばれたことだけ分かればよい（ポインタは recorder を差し替えて与える）。"""

    def forward(self, **kwargs):
        return kwargs["x"]


class FakeRecorder:
    """PointerRecorder の代わり。FakeRuntime.synthesize が呼び出しごとにポインタを入れて on_call を呼ぶ。"""
    active = None

    def __init__(self, model, on_call=None):
        self.pointer, self.text_len, self.calls, self.on_call = None, 0, 0, on_call
        FakeRecorder.active = self

    def close(self):
        FakeRecorder.active = None


class FakeRuntime:
    """passes[i] が i 回目の合成の、呼び出しごとのポインタ（最後の 1 つが最後のステップ）。足りなければ最後を繰り返す。"""

    def __init__(self, cfg=SMALL, passes=None, text_len=6):
        self.model_cfg = cfg
        self.codec = types.SimpleNamespace(sample_rate=48000, model=types.SimpleNamespace(hop_length=1920))
        self.model = types.SimpleNamespace(blocks=[types.SimpleNamespace(attention=FakeAttention())])
        self.requests = []
        self.passes, self.text_len = passes or [], text_len

    def synthesize(self, request, log_fn=None):
        self.requests.append(request)
        calls = self.passes[min(len(self.requests), len(self.passes)) - 1] if self.passes else []
        for pointer in calls:
            rec = FakeRecorder.active
            if rec is None:
                break
            rec.pointer, rec.text_len = torch.as_tensor(pointer), self.text_len
            rec.calls += 1
            if rec.on_call is not None:
                rec.on_call(rec)
        return Result(used_seed=request.seed if request.seed is not None else 99)


def steps(pointer, n=4):
    """どのステップでも同じポインタ（n 回の呼び出し）。"""
    return [pointer] * n


class EstimateTest(unittest.TestCase):
    def test_end_is_first_frame_on_second_to_last_token_plus_tail(self):
        # トークン 6 個（BOS 込み）。4 に届くのは 30 ステップ目（0 始まり）
        pointer = torch.tensor([0] * 10 + [2] * 10 + [3] * 10 + [4] * 5 + [5] * 45)
        frames = lc.estimated_frames(pointer, text_len=6, frames_per_step=1, frames_per_second=25.0)
        self.assertEqual(frames, int(torch.tensor((31 + 0.5 * 25) * lc.MARGIN).ceil()))

    def test_not_reached_returns_none(self):
        pointer = torch.tensor([0, 1, 2, 3])
        self.assertIsNone(lc.estimated_frames(pointer, text_len=8, frames_per_step=1, frames_per_second=25.0))

    def test_momentary_jump_to_the_last_token_is_ignored(self):
        # 文の途中（トークン 2 を読んでいる間）に 1 ステップだけ「。」（5）に飛ぶ。旧来の「最初に届いた位置」は 15
        pointer = torch.tensor([0] * 5 + [1] * 5 + [2] * 5 + [5] + [2] * 5 + [3] * 10 + [4] * 5 + [5] * 40)
        self.assertEqual(lc.reading_end(pointer, text_len=6), 31)

    def test_jump_during_a_pause_is_ignored(self):
        # 「、」（3）の間に 2 回「。」に飛ぶ（BASIC5000_0477 の形）
        pointer = [0] * 4 + [1] * 6 + [3] * 8 + [6, 3, 3, 6] + [3] * 6 + [4] * 8 + [5] * 6 + [6] * 30
        self.assertEqual(lc.reading_end(pointer, text_len=7), 36)

    def test_looping_back_after_the_end_keeps_the_first_end(self):
        # 読み終えた後に前へ戻って読み直す
        pointer = [0] * 5 + [1] * 5 + [2] * 5 + [3] * 5 + [4] * 5 + [1] * 5 + [2] * 5 + [3] * 5 + [4] * 5
        self.assertEqual(lc.reading_end(pointer, text_len=6), 20)

    def test_frequent_final_punct_during_reading_is_ignored_when_the_text_ends_with_it(self):
        # 「、」の間などで何度も「。」（6）を指す（BASIC5000_3942 の形）。最後の語（5）に届くのは 40
        reading = [0] * 4 + [1] * 6 + [2] * 4 + [6] * 3 + [2] * 3 + [3] * 5 + [6, 6, 3, 6, 3, 3] + [4] * 9
        pointer = reading + [5] * 6 + [6] * 30
        self.assertEqual(len(reading), 40)
        self.assertEqual(lc.reading_end(pointer, text_len=7, final_punct=True), 40)
        frames = lc.estimated_frames(pointer, 7, 1, 25.0, text="テストの文です。")
        self.assertEqual(frames, math.ceil((41 + 0.5 * 25) * lc.MARGIN))

    def test_long_stretches_on_the_final_punct_mid_text_are_not_the_end(self):
        # 複数の文: 文中の「。」と取り違えて、読んでいる途中で長く文末（20）を指す（cv3_uttid_464 の形）
        pointer = [0] * 5 + [v for v in range(1, 10) for _ in range(3)] + [20] * 25 + [v for v in range(10, 20) for _ in range(3)] + [20] * 40
        reach = 5 + 27 + 25 + 27  # 最後の語（19）に初めて届くところ
        self.assertEqual(lc.reading_end(pointer, text_len=21, final_punct=True), reach)
        # 距離に上限が無く飛ばし放題だと、記号の扱いが無ければ途中で終わったとみなしていた
        self.assertEqual(lc.reading_end(pointer, text_len=21), reach)

    def test_a_long_spike_to_the_last_word_mid_text_is_not_the_end(self):
        # トークン 12 個、6 フレームずつ読む（最後の語 10 に届くのは 60）。12 フレーム目から 15 フレーム（0.6 秒）
        # 最後の語を指す。距離に上限が無く飛ばし放題だと、そこで読み終えたとみなして 12 に縮めていた
        pointer = [v for v in range(11) for _ in range(6)] + [11] * 30
        pointer[12:27] = [10] * 15
        self.assertEqual(lc.reading_end(pointer, text_len=12, final_punct=True), 60)
        original = lc.DIST_CAP, lc.SKIP_COST
        lc.DIST_CAP, lc.SKIP_COST = float("inf"), 0.0
        try:
            self.assertEqual(lc.reading_end(pointer, text_len=12, final_punct=True), 12)
        finally:
            lc.DIST_CAP, lc.SKIP_COST = original

    def test_last_word_time_comes_from_the_duration_predictor(self):
        # BASIC5000_0071「さよならを言わなければなりません。」: 最後の語「ければなりません」（予測 27.6 フレーム）に
        # 21〜22 フレーム目で届き、71 フレーム目まで読む。届いた時点 + 0.5 秒だと約 40 フレーム（1.6 秒・10 カナ/秒）
        pointer = [0] * 3 + [1] * 11 + [2] * 6 + [3] + [5] * 30 + [6] + [5] * 10 + [6] * 6 + [5] * 3 + [6] * 44
        text = "さよならを言わなければなりません。"
        tokens = {"frames": [20.0, 10.0, 3.0, 5.0, 3.5, 27.6, 41.7], "content_frames": 51.6}
        end = lc.reading_end(pointer, 7, final_punct=True)
        self.assertIn(end, (21, 22))
        self.assertEqual(lc.estimated_frames(pointer, 7, 1, 25.0, text), math.ceil((end + 1 + 12.5) * lc.MARGIN))
        self.assertEqual(lc.estimated_frames(pointer, 7, 1, 25.0, text, tokens),
                         math.ceil((end + 1 + 27.6 + lc.LAST_WORD_TAIL_SECONDS * 25) * lc.MARGIN))  # 64（2.56 秒）
        # 記号で終わらない文は、最後の 2 つ（L-2 と L-1）の予測を足す
        self.assertEqual(lc.estimated_frames(pointer, 7, 1, 25.0, "さよならを言わなければなりません", tokens),
                         max(math.ceil((lc.reading_end(pointer, 7) + 1 + 27.6 + 41.7 + 7.5) * lc.MARGIN),
                             math.ceil(lc.FLOOR * 51.6)))

    def test_the_estimate_never_goes_below_the_floor(self):
        # 読み始めの 40 フレームのうち 20 フレームが最後の語へのスパイク: 経路はそこで届いたとみなす
        pointer = [v for v in range(11) for _ in range(4)] + [11] * 30
        pointer[12:32] = [10] * 20
        self.assertLess(lc.reading_end(pointer, text_len=12, final_punct=True), 20)
        tokens = {"frames": [20.0] + [4.0] * 10 + [0.5], "content_frames": 40.0}
        frames = lc.estimated_frames(pointer, 12, 1, 25.0, "テストの文です。", tokens)
        self.assertEqual(frames, math.ceil(lc.FLOOR * 40.0))
        # 続きの行では先頭の余白（lead）も足す
        self.assertEqual(lc.estimated_frames(pointer, 12, 1, 25.0, "テストの文です。", tokens, lead=20),
                         20 + math.ceil(lc.FLOOR * 40.0))

    def test_tokens_of_another_text_are_not_used(self):
        runtime = types.SimpleNamespace(last_duration_tokens={"frames": [1.0, 2.0, 3.0], "content_frames": 2.0})
        self.assertIsNone(lc.duration_tokens(runtime, 4))
        self.assertEqual(lc.duration_tokens(runtime, 3)["content_frames"], 2.0)
        self.assertIsNone(lc.duration_tokens(types.SimpleNamespace(), 3))

    def test_text_without_final_punct_uses_the_last_token(self):
        self.assertTrue(lc.ends_with_punct("聞いてる？ "))
        self.assertFalse(lc.ends_with_punct("弁当を選ぶようになった"))
        pointer = [0] * 10 + [3] * 10 + [5] * 20  # 最後のトークン（5）が中身の語
        self.assertEqual(lc.estimated_frames(pointer, 6, 1, 25.0, text="弁当を選ぶようになった"),
                         math.ceil((21 + 0.5 * 25) * lc.MARGIN))

    def test_engine_pointer_reads_the_plan_output(self):
        engine = types.SimpleNamespace(pointer_name="pointer", last_pointer=torch.tensor([1, 2]),
                                       last_text_mask=torch.tensor([[True] * 5 + [False] * 3]))
        recorder = lc.EnginePointer(engine)
        self.assertIsNone(recorder.pointer)  # 1 回目の前に前の合成の値を捨てる
        engine.last_pointer = torch.tensor([0, 1, 2])
        self.assertEqual(recorder.pointer.tolist(), [0, 1, 2])
        self.assertEqual(recorder.text_len, 5)
        self.assertTrue(lc.EnginePointer.available(engine))
        self.assertFalse(lc.EnginePointer.available(types.SimpleNamespace(pointer_name=None)))

    def test_only_small_rf_architecture(self):
        self.assertTrue(lc.supported(SMALL))
        self.assertFalse(lc.supported(types.SimpleNamespace(**{**vars(SMALL), "flow_parameterization": "meanflow"})))
        self.assertFalse(lc.supported(types.SimpleNamespace(**{**vars(SMALL), "num_layers": 24})))


class TwoPassTest(unittest.TestCase):
    def setUp(self):
        self.original = lc.PointerRecorder, lc.EARLY_CALL
        lc.PointerRecorder = FakeRecorder

    def tearDown(self):
        lc.PointerRecorder, lc.EARLY_CALL = self.original

    def run_with_pointer(self, pointer, text_len, runtime=None, request=None, second=None):
        if runtime is None:
            second = second if second is not None else [0] * 10 + [4] * 30
            runtime = FakeRuntime(passes=[steps(pointer), steps(second)], text_len=text_len)
        result, info = lc.synthesize(runtime, request or Request())
        return runtime, result, info

    def test_overlong_is_resynthesized_shorter_with_the_same_seed(self):
        # 200 ステップのうち 40 で読み終えている
        pointer = torch.tensor([0] * 20 + [3] * 20 + [4] * 160)
        runtime, _, info = self.run_with_pointer(pointer, text_len=6)
        self.assertEqual(info["length_check"], "resynthesized")
        self.assertEqual(len(runtime.requests), 2)
        first, second = runtime.requests
        self.assertIsNotNone(first.seed)  # 無指定でも先に決めて、1 回目と同じ初期ノイズで作り直す
        self.assertEqual(second.seed, first.seed)
        frames = round(second.seconds * 48000 / 1920)
        self.assertEqual(frames, info["estimated_frames"])
        self.assertLess(frames, 200)

    def test_fitting_length_is_kept(self):
        pointer = torch.tensor([0] * 40 + [3] * 40 + [4] * 20)
        runtime, _, info = self.run_with_pointer(pointer, text_len=6)
        self.assertEqual(info["length_check"], "kept")
        self.assertEqual(len(runtime.requests), 1)

    def test_a_small_shrink_is_kept(self):
        # 見積もりは 1 回目の約 93%（5% 以上・10% 未満短い）: 作り直しても読みが変わるだけなので 1 回目のまま
        pointer = torch.tensor([0] * 40 + [3] * 30 + [4] * 30)
        frames = lc.estimated_frames(pointer, 6, 1, 25.0, "テスト")
        self.assertTrue(100 * (1 - lc.RESYNTH_SHRINK) <= frames < 100 * (1 - lc.MIN_SHRINK))
        runtime, _, info = self.run_with_pointer(pointer, text_len=6)
        self.assertEqual((info["length_check"], len(runtime.requests)), ("kept", 1))

    def test_resynthesis_uses_the_runtime_duration_tokens(self):
        # 最後の語（4）の予測 30 フレーム: 届いた時点（40 フレーム目＝41 フレーム分）+ 30 + 0.3 秒で作り直す
        pointer = torch.tensor([0] * 20 + [3] * 20 + [4] * 160)
        runtime = FakeRuntime(passes=[steps(pointer), steps([0] * 10 + [4] * 30)], text_len=6)
        runtime.last_duration_tokens = {"frames": [20.0, 5.0, 5.0, 5.0, 30.0, 0.5], "content_frames": 45.0}
        _, info = lc.synthesize(runtime, Request(text="テストです。"))
        self.assertEqual(info["length_check"], "resynthesized")
        self.assertEqual(info["estimated_frames"], math.ceil((41 + 30.0 + 7.5) * lc.MARGIN))
        self.assertEqual(round(runtime.requests[1].seconds * 25), info["estimated_frames"])

    def test_too_short_is_never_lengthened(self):
        pointer = torch.tensor([0] * 20 + [2] * 20)  # 最後まで届かない（話速を上げすぎ）
        runtime, _, info = self.run_with_pointer(pointer, text_len=6)
        self.assertEqual(info["length_check"], "kept")
        self.assertEqual(len(runtime.requests), 1)

    def test_second_pass_that_does_not_reach_the_end_falls_back_to_the_first(self):
        pointer = torch.tensor([0] * 20 + [3] * 20 + [4] * 160)
        runtime, _, info = self.run_with_pointer(pointer, 6, second=[0] * 20 + [3] * 20)
        self.assertEqual(info["length_check"], "fallback")
        # 少し長くした作り直しも届かない。1 回目は最後のステップで止めた（デコードしていない）ので、
        # 1 回目と同じ長さ・シードでもう一度作る
        first, second, retry, last = runtime.requests
        self.assertAlmostEqual(retry.seconds / second.seconds, lc.RETRY_GROWTH, delta=0.02)
        self.assertEqual((last.seconds, last.seed), (None, first.seed))

    def test_preview_decides_the_length_for_slow_speech(self):
        @dataclasses.dataclass
        class SlowRequest(Request):
            duration_scale: float = 2.0
            num_steps: int | None = 4

        lc.PREVIEW_STEPS = 2
        try:
            overlong = [0] * 20 + [3] * 20 + [4] * 160
            runtime = FakeRuntime(passes=[steps(overlong, 2), steps([0] * 10 + [4] * 30)])
            _, info = lc.synthesize(runtime, SlowRequest())
        finally:
            lc.PREVIEW_STEPS = None
        preview, second = runtime.requests
        self.assertEqual((preview.num_steps, second.num_steps), (2, 4))  # 1 回目（4 ステップ）は作らない
        self.assertEqual((info["length_check"], info["early_stop_call"]), ("resynthesized", "preview2"))
        self.assertEqual(info["estimated_frames"],
                         math.ceil(lc.estimated_frames(overlong, 6, 1, 25.0, "テスト") * lc.PREVIEW_GROWTH))

    def test_truncated_second_is_retried_longer(self):
        overlong = [0] * 20 + [3] * 20 + [4] * 160
        runtime = FakeRuntime(passes=[steps(overlong), steps([0] * 30), steps([0] * 10 + [4] * 30)])
        _, info = lc.synthesize(runtime, Request())
        self.assertEqual((info["length_check"], info["retries"]), ("resynthesized", 1))
        self.assertEqual(len(runtime.requests), 3)

    def test_early_call_stops_the_first_pass(self):
        lc.EARLY_CALL = 2
        overlong = [0] * 20 + [3] * 20 + [4] * 160
        runtime = FakeRuntime(passes=[steps(overlong), steps([0] * 10 + [4] * 30)])
        _, info = lc.synthesize(runtime, Request())
        self.assertEqual((info["length_check"], info["early_stop_call"]), ("resynthesized", 2))
        self.assertEqual(len(runtime.requests), 2)

    def test_early_stop_then_truncated_second_reruns_the_first(self):
        lc.EARLY_CALL = 2
        overlong = [0] * 20 + [3] * 20 + [4] * 160
        runtime = FakeRuntime(passes=[steps(overlong), steps([0] * 30), steps([0] * 30), []])
        _, info = lc.synthesize(runtime, Request())
        self.assertEqual(info["length_check"], "fallback")
        first, _, _, third = runtime.requests
        self.assertEqual((third.seconds, third.seed), (None, first.seed))

    def test_manual_seconds_and_other_models_are_skipped(self):
        pointer = torch.tensor([0] * 20 + [4] * 180)
        runtime, _, info = self.run_with_pointer(pointer, 6, request=Request(seconds=3.0))
        self.assertEqual((info["length_check"], len(runtime.requests)), ("skipped", 1))
        mf = FakeRuntime(types.SimpleNamespace(**{**vars(SMALL), "flow_parameterization": "meanflow"}),
                         passes=[steps(pointer)])
        runtime, _, info = self.run_with_pointer(pointer, 6, runtime=mf)
        self.assertEqual((info["length_check"], len(runtime.requests)), ("skipped", 1))


class StreamingAndContextTest(unittest.TestCase):
    def test_first_pass_is_not_streamed_and_sink_is_restored(self):
        runtime = FakeRuntime(passes=[steps([0] * 20 + [3] * 20 + [4] * 160), steps([0] * 10 + [4] * 30)])
        decoder = types.SimpleNamespace(sink="client")
        seen = []
        decoder.end = lambda: setattr(decoder, "sink", None)
        decoder.begin = lambda sink: setattr(decoder, "sink", sink)
        original_synth = runtime.synthesize

        def synth(request, log_fn=None):
            seen.append(decoder.sink)
            return original_synth(request, log_fn)

        runtime.synthesize = synth

        original = lc.PointerRecorder
        lc.PointerRecorder = FakeRecorder
        try:
            lc.synthesize(runtime, Request(), stream_decoder=decoder)
        finally:
            lc.PointerRecorder = original
        self.assertEqual(seen, [None, "client"])  # 1 回目は流さず、作り直しは流す
        self.assertEqual(decoder.sink, "client")

    def test_lookahead_is_skipped(self):
        @dataclasses.dataclass
        class ContextRequest(Request):
            lookahead_text: str | None = None

        runtime = FakeRuntime()
        _, info = lc.synthesize(runtime, ContextRequest(lookahead_text="次の行"))
        self.assertEqual((info["length_check"], len(runtime.requests)), ("skipped", 1))

    def test_prefix_resynthesizes_without_the_lead_frames(self):
        # 続きの行: ポインタは先頭の余白（20 フレーム）を含む。ランタイムが seconds にも余白を足すので引いて渡す
        @dataclasses.dataclass
        class PrefixRequest(Request):
            prefix_latent: object = None
            prefix_lead_frames: int = 20

        original = lc.PointerRecorder
        lc.PointerRecorder = FakeRecorder
        try:
            overlong = [0] * 30 + [3] * 20 + [4] * 150
            runtime = FakeRuntime(passes=[steps(overlong), steps([0] * 10 + [4] * 30)])
            runtime.supports_prefix = True
            _, info = lc.synthesize(runtime, PrefixRequest(prefix_latent=torch.zeros(1)))
        finally:
            lc.PointerRecorder = original
        self.assertEqual(info["length_check"], "resynthesized")
        second = runtime.requests[1]
        self.assertEqual(round(second.seconds * 48000 / 1920), info["estimated_frames"] - 20)
        self.assertIsNotNone(second.prefix_latent)  # 作り直しも続きのまま


class QueryFieldTest(unittest.TestCase):
    def test_engine_accepts_on_off_and_rejects_others(self):
        from voicevox_engine import RequestValidationError, _validate_query
        _validate_query({"irodori_length_check": "on"})
        _validate_query({"irodori_length_check": "off"})
        with self.assertRaises(RequestValidationError):
            _validate_query({"irodori_length_check": "auto"})


if __name__ == "__main__":
    unittest.main()
