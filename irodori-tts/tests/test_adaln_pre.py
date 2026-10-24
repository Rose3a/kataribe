"""adaLN の事前計算（adaln_pre.py / trt_adaln.py）の契約テスト。GPU も TensorRT も使わない。

小さな DiT もどきで、事前計算した mod が各層の adaLN の計算と同じ値になること、合成の中での
呼び出し順の扱い（ステップを数える・表に無い呼び出しは直接計算する）を確かめる。
"""
from pathlib import Path
import json
import struct
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wrapper"))
sys.path.insert(0, str(ROOT.parent / "runtime" / "trt-lab" / "repo"))
try:
    import torch
    import adaln_pre
    from irodori_tts.model import LowRankAdaLN
except ModuleNotFoundError as exc:
    if exc.name not in ("torch", "irodori_tts"):
        raise
    torch = None

DIM, RANK, LAYERS, TIME_DIM = 16, 4, 3, 8


class Block(torch.nn.Module if torch else object):
    def __init__(self):
        super().__init__()
        self.attention_adaln = LowRankAdaLN(DIM, RANK, 1e-5)
        self.mlp_adaln = LowRankAdaLN(DIM, RANK, 1e-5)


class TinyDiT(torch.nn.Module if torch else object):
    def __init__(self):
        super().__init__()
        self.cfg = types.SimpleNamespace(timestep_embed_dim=TIME_DIM)
        self.cond_module = torch.nn.Sequential(
            torch.nn.Linear(TIME_DIM, DIM, bias=False), torch.nn.SiLU(),
            torch.nn.Linear(DIM, DIM, bias=False), torch.nn.SiLU(),
            torch.nn.Linear(DIM, DIM * 3, bias=False))
        self.blocks = torch.nn.ModuleList(Block() for _ in range(LAYERS))
        self.delta_cond_module = None
        # 本物は up が 0 初期化なので、値の違いが出るよう乱数にする
        for p in self.parameters():
            torch.nn.init.normal_(p, std=0.3)


def tiny_model():
    torch.manual_seed(0)
    return TinyDiT().eval()


@unittest.skipIf(torch is None, "torch が無い環境")
class TableMatchesLayers(unittest.TestCase):
    def test_precomputed_mod_gives_the_same_adaln_output_as_the_layer(self):
        from irodori_tts import model as model_module
        model = tiny_model()
        t = torch.tensor([1.0, 0.7, 0.2, 0.05])
        pre = adaln_pre.AdalnPrecompute(model)
        table = pre.table(t)
        self.assertEqual(tuple(table.shape), (4, LAYERS, 2, 3, DIM))
        x = torch.randn(4, 5, DIM)
        with torch.no_grad():
            cond = model.cond_module(model_module.get_timestep_embedding(t, TIME_DIM))[:, None, :]
            expected = [[block.attention_adaln(x, cond), block.mlp_adaln(x, cond)] for block in model.blocks]
        patched = tiny_model()
        adaln_pre.patch_blocks(patched.blocks)
        with torch.no_grad():
            for i, block in enumerate(patched.blocks):
                for j, adaln in enumerate((block.attention_adaln, block.mlp_adaln)):
                    got_x, got_gate = adaln(x, table[:, i])
                    want_x, want_gate = expected[i][j]
                    torch.testing.assert_close(got_x, want_x, atol=1e-5, rtol=1e-5)
                    torch.testing.assert_close(got_gate, want_gate, atol=1e-5, rtol=1e-5)

    def test_unsupported_models_say_why(self):
        model = tiny_model()
        model.delta_cond_module = torch.nn.Linear(2, 2)
        with self.assertRaises(RuntimeError) as caught:
            adaln_pre.AdalnPrecompute(model)
        self.assertIn("MeanFlow", str(caught.exception))
        self.assertIsNotNone(adaln_pre.supports(types.SimpleNamespace(delta_cond_module=None, blocks=[])))


@unittest.skipIf(torch is None, "torch が無い環境")
class Schedule(unittest.TestCase):
    def test_matches_rf_formula(self):
        schedule = adaln_pre.t_schedule(16, "sway", -1.0, "cpu")
        self.assertEqual(schedule.numel(), 17)
        self.assertAlmostEqual(float(schedule[0]), 0.999, places=6)
        self.assertAlmostEqual(float(schedule[-1]), 0.0, places=6)
        self.assertTrue(bool((schedule[:-1] > schedule[1:]).all()))
        u = torch.linspace(0.0, 1.0, 17)
        u = (u + (-1.0) * (torch.cos(0.5 * torch.pi * u) + u - 1.0)).clamp(0.0, 1.0)
        torch.testing.assert_close(schedule, (1.0 - u) * 0.999)
        linear = adaln_pre.t_schedule(4, "linear", -1.0, "cpu")
        torch.testing.assert_close(linear, (1.0 - torch.linspace(0.0, 1.0, 5)) * 0.999)


@unittest.skipIf(torch is None, "torch が無い環境")
class PerSynthesis(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()
        self.pre = adaln_pre.AdalnPrecompute(self.model)
        self.steps = adaln_pre.t_schedule(4, "sway", -1.0, "cpu")[:-1]

    def test_calls_take_the_steps_in_order_and_batches_share_the_row(self):
        self.pre.begin(4, "sway", -1.0, True)
        table = self.pre.table(self.steps)
        for step in range(4):
            batch = 3 if step < 2 else 1
            mod = self.pre.mod_for(self.steps[step].repeat(batch))
            self.assertEqual(tuple(mod.shape), (batch, LAYERS, 2, 3, DIM))
            self.assertTrue(mod.is_contiguous())
            for row in range(batch):
                torch.testing.assert_close(mod[row], table[step])

    def test_calls_beyond_the_table_and_other_cfg_modes_compute_from_t(self):
        self.pre.begin(2, "linear", -1.0, True)
        for _ in range(2):
            self.pre.mod_for(torch.tensor([0.9]))
        extra = self.pre.mod_for(torch.tensor([0.5, 0.25]))
        torch.testing.assert_close(extra, self.pre.table(torch.tensor([0.5, 0.25])))
        self.pre.begin(4, "sway", -1.0, False)  # joint / alternating: 数えずに t から計算する
        got = self.pre.mod_for(torch.tensor([0.3]))
        torch.testing.assert_close(got, self.pre.table(torch.tensor([0.3])))
        self.assertEqual(self.pre.state["n"], 0)
        self.pre.end()
        outside = self.pre.mod_for(torch.tensor([0.8]))  # 合成の外（state なし）
        torch.testing.assert_close(outside, self.pre.table(torch.tensor([0.8])))

    def test_check_mode_rejects_a_t_that_does_not_match_the_table(self):
        self.pre.check = True
        self.pre.begin(4, "sway", -1.0, True)
        self.pre.mod_for(self.steps[0].repeat(2))
        with self.assertRaises(RuntimeError):
            self.pre.mod_for(torch.tensor([0.123, 0.123]))

    def test_adapter_replaces_t_with_mod_and_delegates_the_rest(self):
        seen = {}

        class Inner:
            compact = True

            def __call__(self, **kw):
                seen.update(kw)
                return "out"

        adapter = adaln_pre.AdalnAdapter(Inner(), self.pre)
        self.pre.begin(4, "sway", -1.0, True)
        self.assertEqual(adapter(x_t=1, t=self.steps[0].repeat(2)), "out")
        self.assertEqual(tuple(seen["t"].shape), (2, LAYERS, 2, 3, DIM))
        self.assertEqual(seen["x_t"], 1)
        self.assertTrue(adapter.compact)

    def test_sample_hook_builds_the_table_around_each_synthesis_and_restores_state(self):
        calls = []

        def original(*args, **kwargs):
            calls.append((self.pre.state is not None and self.pre.state["count"], self.pre.state["independent"]))
            return "latent"

        module = types.SimpleNamespace(sample_euler_rf_cfg=original)
        adaln_pre.install_sample_hook(self.pre, module)
        self.assertEqual(module.sample_euler_rf_cfg(model=1, num_steps=8, t_schedule_mode="sway"), "latent")
        self.assertEqual(module.sample_euler_rf_cfg(model=1, num_steps=5, cfg_guidance_mode="Joint"), "latent")
        self.assertEqual(calls, [(8, True), (5, False)])
        self.assertIsNone(self.pre.state)
        # 入れ直しても元の関数を二重に包まない
        adaln_pre.install_sample_hook(self.pre, module)
        self.assertIs(module.sample_euler_rf_cfg._adaln_original, original)

    def test_low_vram_keeps_the_weights_on_the_host_and_gives_the_same_table(self):
        low = adaln_pre.AdalnPrecompute(self.model, low_vram=True)
        torch.testing.assert_close(low.table(self.steps), self.pre.table(self.steps))
        self.assertGreater(low.bytes, 0)


def write_header_only(path, metadata):
    header = json.dumps({"__metadata__": metadata}).encode()
    Path(path).write_bytes(struct.pack("<Q", len(header)) + header)


class FakeEngine:
    def __init__(self, rank):
        self.engine = types.SimpleNamespace(get_tensor_shape=lambda name: (-1,) * rank)


@unittest.skipIf(torch is None, "torch が無い環境")
class PlanSelection(unittest.TestCase):
    def test_a_plan_whose_t_has_five_dims_takes_mod(self):
        self.assertTrue(adaln_pre.plan_takes_mod(FakeEngine(5)))
        self.assertFalse(adaln_pre.plan_takes_mod(FakeEngine(1)))
        self.assertFalse(adaln_pre.plan_takes_mod(object()))

    def test_only_bf16_and_trt_ready_w8a8_checkpoints_get_an_adaln_plan(self):
        try:
            import trt_adaln
        except ModuleNotFoundError as exc:  # tensorrt などが無い環境
            self.skipTest(str(exc))
        quant = lambda payload: {"irodori_quantization_json": json.dumps(payload)}  # noqa: E731
        with tempfile.TemporaryDirectory() as tmp:
            plain, int4, int8, int8_trt = (Path(tmp) / n for n in ("plain", "int4", "int8", "int8trt"))
            write_header_only(plain, {})
            write_header_only(int4, quant({"quantization_type": "int4_weight_only"}))
            write_header_only(int8, quant({"quantization_type": "int8_dynamic_activation_int8_weight"}))
            write_header_only(int8_trt, quant({"quantization_type": "int8_dynamic_activation_int8_weight",
                                               "trt_activation_amax": {"blocks.0.mlp.w2": 1.0}}))
            self.assertEqual(trt_adaln.base_kind(plain), "bf16")
            self.assertIsNone(trt_adaln.base_kind(int4))
            self.assertIsNone(trt_adaln.base_kind(int8))
            self.assertEqual(trt_adaln.base_kind(int8_trt), "int8")
            trt_adaln._failed.clear()
            self.assertTrue(trt_adaln.usable(plain))
            self.assertFalse(trt_adaln.usable(int4))
            with unittest.mock.patch.dict("os.environ", {"IRODORI_ADALN": "0"}):
                self.assertFalse(trt_adaln.usable(plain))
            trt_adaln._failed.add(str(plain))
            self.assertFalse(trt_adaln.usable(plain))
            trt_adaln._failed.clear()

    def test_a_failed_build_falls_back_to_the_regular_plan_and_is_not_retried(self):
        try:
            import trt_adaln
        except ModuleNotFoundError as exc:
            self.skipTest(str(exc))
        messages = []
        def boom(*args):
            raise RuntimeError("変換に失敗")
        with unittest.mock.patch.object(trt_adaln, "ensure_plan", boom):
            trt_adaln._failed.clear()
            plan, used = trt_adaln.ensure_plan_or_fallback(
                "model.safetensors", lambda *a: None, messages.append, lambda *a: "regular.plan")
        self.assertEqual((plan, used), ("regular.plan", False))
        self.assertIn("model.safetensors", trt_adaln._failed)
        self.assertTrue(any("従来の plan" in m for m in messages))
        trt_adaln._failed.clear()


import unittest.mock  # noqa: E402

if __name__ == "__main__":
    unittest.main()
