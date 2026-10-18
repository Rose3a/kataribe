"""前のセリフから続けて生成（blockwise）を TensorRT で動かす配線。

- prefix の K/V とマスクは caption 枠の後ろにつなぐ（caption は有効な長さまで詰める）、
- 同じ cache には同じつないだ cache を返す（Adapter の入力の使い回しが効く）、
- caption 枠の上限を超える分は prefix の古い側から捨てる、
- prefix 付きの K/V 計算は CUDA Graph を撮らない、
- vram_trim は prefix の射影を GPU に残す、
- PyTorch の attention は cuDNN の SDPA を使わない（形ごとの計画作りで約0.8秒かかる）。

実モデルでの一致（TRT と PyTorch の DiT 出力の差が prefix なしと同程度）は、
ベンチ用のスクリプトで確かめた（相対 RMSE 平均 0.9%）。
"""
from pathlib import Path
import os
import sys
import types
import unittest
from unittest.mock import patch

import torch

BOX = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BOX / "irodori-tts" / "wrapper"))
sys.path.insert(0, str(BOX / "runtime" / "trt-lab" / "repo"))

import cuda_graphs  # noqa: E402
import tts_cli  # noqa: E402
import vram_trim  # noqa: E402

CUDA = torch.cuda.is_available()


def fake_model(static_kv=6):
    attention = types.SimpleNamespace(num_static_context_kv=static_kv)
    return types.SimpleNamespace(blocks=[types.SimpleNamespace(attention=attention)])


def layer(batch, text, speaker, caption, prefix, seed):
    gen = torch.Generator().manual_seed(seed)
    shape = lambda n: (batch, n, 2, 4)  # noqa: E731
    return tuple(torch.randn(shape(n), generator=gen)
                 for n in (text, text, speaker, speaker, caption, caption, prefix, prefix))


class PrefixToCaptionTest(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.adapter = lambda **kw: self.calls.append(kw) or torch.zeros(1)
        self.cache = [layer(3, 5, 4, 512, 6, seed) for seed in range(2)]
        self.caption_mask = torch.zeros((3, 512), dtype=torch.bool)
        self.caption_mask[0, :2] = True  # 条件ありの行だけ caption がある
        self.prefix_mask = torch.ones((3, 6), dtype=torch.bool)

    def call(self, wrapper, **extra):
        return wrapper(x_t=torch.zeros(3, 7, 32), t=torch.ones(3), context_kv_cache=self.cache,
                       caption_mask=self.caption_mask, prefix_mask=self.prefix_mask.clone(),
                       prefix_freqs=torch.zeros(6, 2), **extra)

    def test_prefix_follows_the_trimmed_caption(self):
        wrapper = tts_cli._PrefixToCaption(self.adapter, fake_model(), max_caption=512)
        self.call(wrapper)
        kw = self.calls[0]
        self.assertNotIn("prefix_mask", kw)
        self.assertNotIn("prefix_freqs", kw)
        mask = kw["caption_mask"]
        self.assertEqual(tuple(mask.shape), (3, 2 + 6))
        self.assertTrue(torch.equal(mask[:, :2], self.caption_mask[:, :2]))
        self.assertTrue(bool(mask[:, 2:].all()))
        for merged, original in zip(kw["context_kv_cache"], self.cache):
            self.assertEqual(len(merged), 6)
            for i in range(4):
                self.assertIs(merged[i], original[i])
            self.assertTrue(torch.equal(merged[4], torch.cat([original[4][:, :2], original[6]], 1)))
            self.assertTrue(torch.equal(merged[5], torch.cat([original[5][:, :2], original[7]], 1)))

    def test_empty_caption_keeps_one_masked_slot(self):
        self.caption_mask[:] = False
        wrapper = tts_cli._PrefixToCaption(self.adapter, fake_model(), max_caption=512)
        self.call(wrapper)
        mask = self.calls[0]["caption_mask"]
        self.assertEqual(mask.shape[1], 1 + 6)
        self.assertFalse(bool(mask[:, 0].any()))

    def test_same_cache_reuses_the_merged_cache(self):
        wrapper = tts_cli._PrefixToCaption(self.adapter, fake_model(), max_caption=512)
        self.call(wrapper)
        self.call(wrapper)  # rf.py はマスクを毎ステップ作り直す
        self.assertIs(self.calls[0]["context_kv_cache"], self.calls[1]["context_kv_cache"])
        self.cache = [tuple(t.clone() for t in l) for l in self.cache]
        self.call(wrapper)
        self.assertIsNot(self.calls[1]["context_kv_cache"], self.calls[2]["context_kv_cache"])

    def test_cap_drops_the_oldest_prefix_tokens(self):
        wrapper = tts_cli._PrefixToCaption(self.adapter, fake_model(), max_caption=6)
        self.call(wrapper)
        kw = self.calls[0]
        self.assertEqual(kw["caption_mask"].shape[1], 6)
        merged, original = kw["context_kv_cache"][0], self.cache[0]
        self.assertTrue(torch.equal(merged[4][:, 2:], original[6][:, -4:]))

    def test_without_prefix_passes_through(self):
        wrapper = tts_cli._PrefixToCaption(self.adapter, fake_model(), max_caption=512)
        wrapper(x_t=torch.zeros(1), context_kv_cache=self.cache, caption_mask=self.caption_mask)
        self.assertIs(self.calls[0]["context_kv_cache"], self.cache)
        self.assertIs(self.calls[0]["caption_mask"], self.caption_mask)

    def test_model_without_caption_is_refused(self):
        wrapper = tts_cli._PrefixToCaption(self.adapter, fake_model(static_kv=4), max_caption=512)
        with self.assertRaises(ValueError):
            self.call(wrapper)


class DropStepInputsTest(unittest.TestCase):
    """合成が終わったら、Adapter と _PrefixToCaption が持つ plan の入力（文脈 K/V）を捨てる。

    残すと長文・話者ありで 0.4〜0.5GB が待機中も GPU に残り、その塊ごと VRAM を返せなかった。
    """

    def test_clears_the_adapter_chain(self):
        inner = types.SimpleNamespace(prepared={("cache", 100): ("kv",)})
        outer = tts_cli._PrefixToCaption(inner, fake_model(), 512)
        outer._merged = ("cache", (1, 4), ["merged"], "mask", (1, 512))
        backend = object.__new__(tts_cli.TrtBackend)
        backend.adapter = outer
        backend._drop_step_inputs()
        self.assertEqual(inner.prepared, {})
        self.assertIsNone(outer._merged)
        # 包んでいない Adapter だけでも動く
        backend.adapter = types.SimpleNamespace(prepared={1: 2})
        backend._drop_step_inputs()
        self.assertEqual(backend.adapter.prepared, {})


class SupportsPrefixTest(unittest.TestCase):
    def tts(self, backend):
        tts = tts_cli.IrodoriTTS.__new__(tts_cli.IrodoriTTS)
        tts.backend = backend
        return tts

    def test_trt_needs_a_prefix_capable_plan(self):
        runtime = types.SimpleNamespace(supports_prefix=True)
        trt = types.SimpleNamespace(name="trt", runtime=runtime, prefix_capable=True)
        self.assertTrue(self.tts(trt).supports_prefix)
        trt.prefix_capable = False
        self.assertFalse(self.tts(trt).supports_prefix)

    def test_torch_and_radeon(self):
        runtime = types.SimpleNamespace(supports_prefix=True)
        self.assertTrue(self.tts(types.SimpleNamespace(name="torch", runtime=runtime)).supports_prefix)
        self.assertFalse(self.tts(types.SimpleNamespace(name="radeon", runtime=runtime)).supports_prefix)


class VramTrimTest(unittest.TestCase):
    def test_prefix_projection_stays_on_gpu(self):
        self.assertIn("wk_prefix", vram_trim.KV_PROJECTION)
        self.assertIn("wv_prefix", vram_trim.KV_PROJECTION)


class CudnnAttentionTest(unittest.TestCase):
    def setUp(self):
        from irodori_tts import attention
        self.attention = attention
        self.saved = list(attention._SDPA_PRIORITY)

    def tearDown(self):
        self.attention._SDPA_PRIORITY[:] = self.saved

    def test_cudnn_is_left_out(self):
        from torch.nn.attention import SDPBackend
        with patch.dict(os.environ, {"IRODORI_CUDNN_ATTENTION": ""}):
            tts_cli._avoid_cudnn_attention()
        self.assertNotIn(SDPBackend.CUDNN_ATTENTION, self.attention._SDPA_PRIORITY)
        self.assertIn(SDPBackend.EFFICIENT_ATTENTION, self.attention._SDPA_PRIORITY)

    def test_can_be_kept(self):
        with patch.dict(os.environ, {"IRODORI_CUDNN_ATTENTION": "1"}):
            tts_cli._avoid_cudnn_attention()
        self.assertEqual(self.attention._SDPA_PRIORITY, self.saved)


@unittest.skipUnless(CUDA, "CUDA required")
class GraphedPrefixKvTest(unittest.TestCase):
    def test_prefix_kv_runs_eagerly(self):
        def build(text_state, speaker_state, caption_state, prefix_state=None, prefix_freqs=None):
            out = text_state * 2
            return [(out,) if prefix_state is None else (out, prefix_state + 1)]
        model = types.SimpleNamespace(text_encoder=None, caption_encoder=None,
                                      build_context_kv_cache=build)
        wrappers = cuda_graphs.install(model, log=self.fail)
        with torch.inference_mode():
            x = torch.randn(2, 4, device="cuda")
            p = torch.randn(2, 3, device="cuda")
            kv = model.build_context_kv_cache(text_state=x, speaker_state=None, caption_state=None,
                                              prefix_state=p, prefix_freqs=p)
            self.assertTrue(torch.equal(kv[0][1], p + 1))
            model.build_context_kv_cache(text_state=x, speaker_state=None, caption_state=None)
        self.assertEqual(wrappers["context_kv"].captures, 1)


if __name__ == "__main__":
    unittest.main()
