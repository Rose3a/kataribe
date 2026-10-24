"""Radeon（DirectML）で読むモデルの違い（条件の有無・bf16）を、実 GPU なしで確かめる。"""
import sys
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "irodori-tts" / "wrapper"
sys.path.insert(0, str(WRAPPER))

import radeon_bridge  # noqa: E402


def fake_bridge(has_speaker, has_caption):
    bridge = radeon_bridge.Bridge.__new__(radeon_bridge.Bridge)
    attention = types.SimpleNamespace(has_speaker_condition=has_speaker,
                                      has_caption_condition=has_caption)
    model = types.SimpleNamespace(blocks=[types.SimpleNamespace(attention=attention)])
    bridge.runtime = types.SimpleNamespace(model=model)
    bridge.compact = True
    bridge.contexts = {}
    bridge.stats = {"calls": 0, "roundtrip_s": 0.0, "worker_compute_s": 0.0}
    bridge.sent = []

    def command(obj):
        bridge.sent.append(obj)
        if obj["op"] == "forward":
            return {"output": radeon_bridge.pack(torch.zeros(1, 4, 8)), "compute_s": 0.0}
        return {"ok": True}

    bridge.command = command
    return bridge


def mask(length, used):
    value = torch.zeros(1, length, dtype=torch.bool)
    value[:, :used] = True
    return value


def kv(length):
    return torch.zeros(1, length, 2, 4)


class RadeonBridgeContextTests(unittest.TestCase):
    def forward(self, bridge, masks, cache):
        bridge.forward(x_t=torch.zeros(1, 4, 8), t=torch.zeros(1), latent_mask=None,
                       context_kv_cache=cache, **masks)
        context = bridge.sent[0]["context"]
        return [[shape for _, shape, _ in layer] for layer in context["kv"]], context

    def test_text_speaker_caption_kv_are_trimmed_to_their_own_masks(self):
        bridge = fake_bridge(True, True)
        cache = [(kv(10), kv(10), kv(6), kv(6), kv(9), kv(9))]
        shapes, _ = self.forward(bridge, dict(text_mask=mask(10, 3), speaker_mask=mask(6, 5),
                                              caption_mask=mask(9, 2)), cache)
        self.assertEqual([s[1] for s in shapes[0]], [3, 3, 5, 5, 2, 2])

    def test_model_without_caption_has_four_kv(self):
        # 500M v3: speaker あり・caption なし
        bridge = fake_bridge(True, False)
        cache = [(kv(10), kv(10), kv(6), kv(6))]
        shapes, _ = self.forward(bridge, dict(text_mask=mask(10, 3), speaker_mask=mask(6, 5),
                                              caption_mask=None), cache)
        self.assertEqual([s[1] for s in shapes[0]], [3, 3, 5, 5])

    def test_caption_only_model_uses_the_caption_mask_length(self):
        # speaker の K/V が無いと caption が2番目の組になる。speaker のマスクで詰めない。
        bridge = fake_bridge(False, True)
        cache = [(kv(10), kv(10), kv(9), kv(9))]
        shapes, context = self.forward(bridge, dict(text_mask=mask(10, 3), speaker_mask=mask(6, 5),
                                                    caption_mask=mask(9, 2)), cache)
        self.assertEqual([s[1] for s in shapes[0]], [3, 3, 2, 2])
        self.assertEqual(context["caption_mask"][1], (1, 2))


class RadeonWorkerSourceTests(unittest.TestCase):
    source = (WRAPPER / "radeon_worker.py").read_text(encoding="utf-8")

    def test_worker_does_not_index_kv_by_fixed_position(self):
        # K/V の数はモデルの条件で変わる（4 か 6）。kv[4] のような固定位置は使わない。
        self.assertNotIn("kv[4]", self.source)
        self.assertNotIn("kv[2]", self.source)
        self.assertIn('context_kv=context["kv"][i]', self.source)

    def test_fp16_adaln_applies_silu_before_down_like_the_model(self):
        # model.LowRankAdaLN: up(down(silu(x))) + x。逆順だと FP16 の出力が音声にならなかった。
        for name in ("shift", "scale", "gate"):
            self.assertIn(f"self.{name}_up(self.{name}_down(torch.nn.functional.silu({name})))",
                          self.source)

    def test_fp16_numerically_sensitive_parts_use_fp32(self):
        # 1000*t の cos/sin と、あふれやすい二乗平均は FP32 で計算する。
        self.assertIn("timestep[:, None].float()", self.source)
        self.assertIn("x.float().square().mean(", self.source)
        self.assertNotIn("(x * x).mean(", self.source)

    def test_bf16_weights_are_loaded_as_fp32(self):
        # DirectML は BFloat16 を扱えない（Blockwise の bf16 版で落ちていた）。
        self.assertIn("torch.bfloat16", self.source)
        self.assertIn("t.float()", self.source)


if __name__ == "__main__":
    unittest.main()
