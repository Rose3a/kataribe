"""省VRAM のテキストエンコーダ CPU 化の契約テスト（GPU なしの小さな偽モデル）。

- 有効な長さまで切って計算し、元の長さにゼロで戻す（全長で計算した値と一致）
- 本体を CPU の fp32 にしても model.dtype（最初のパラメータ）は変わらない
- IRODORI_TEXT_ON_CPU で上書きできる
"""
from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wrapper"))
import vram_trim  # noqa: E402


class Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)

    def forward(self, input_ids, mask):
        state = self.backbone(input_ids.float().unsqueeze(-1).expand(-1, -1, 4))
        return state * mask.unsqueeze(-1).to(state.dtype)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained_text_backbone = Wrapper()  # 先に登録される
        self.text_encoder = nn.Linear(4, 4).to(torch.bfloat16)


class TextOnCpuTests(unittest.TestCase):
    def test_trim_and_pad_match_full_length(self):
        model = Model()
        ids = torch.tensor([[3, 5, 7, 0, 0, 0]])
        mask = torch.tensor([[True, True, True, False, False, False]])
        expected = model.pretrained_text_backbone(ids, mask)
        moved = vram_trim.park_text_backbone(model, "cpu")
        self.assertGreater(moved, 0)
        got = model.pretrained_text_backbone(ids, mask)
        self.assertEqual(got.shape, expected.shape)
        torch.testing.assert_close(got, expected)
        self.assertTrue(torch.equal(got[:, 3:], torch.zeros_like(got[:, 3:])))

    def test_model_dtype_stays_with_the_remaining_parameters(self):
        model = Model()
        vram_trim.park_text_backbone(model, "cpu")
        self.assertEqual(next(model.parameters()).dtype, torch.bfloat16)

    def test_env_override(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("IRODORI_TEXT_ON_CPU", None)
            self.assertFalse(vram_trim.text_on_cpu_enabled(False))
            self.assertTrue(vram_trim.text_on_cpu_enabled(True))
            os.environ["IRODORI_TEXT_ON_CPU"] = "0"
            self.assertFalse(vram_trim.text_on_cpu_enabled(True))


class LowVramDefaultsTests(unittest.TestCase):
    """省VRAM で既定オンになる部品と、解放までの秒数（環境変数で上書きできる）。"""

    def test_defaults_follow_low_vram(self):
        with patch.dict(os.environ, {}, clear=False):
            for name in ("IRODORI_PREFIX_ON_DEMAND", "IRODORI_KV_ON_DEMAND", "IRODORI_UNLOAD_AFTER"):
                os.environ.pop(name, None)
            self.assertTrue(vram_trim.low_vram_flag("IRODORI_PREFIX_ON_DEMAND", True))
            self.assertTrue(vram_trim.low_vram_flag("IRODORI_KV_ON_DEMAND", True))
            self.assertFalse(vram_trim.low_vram_flag("IRODORI_KV_ON_DEMAND", False))
            self.assertEqual(vram_trim.unload_after_seconds(True), 30.0)
            self.assertEqual(vram_trim.unload_after_seconds(False), 0.0)

    def test_env_overrides(self):
        with patch.dict(os.environ, {"IRODORI_KV_ON_DEMAND": "0", "IRODORI_UNLOAD_AFTER": "0"}):
            self.assertFalse(vram_trim.low_vram_flag("IRODORI_KV_ON_DEMAND", True))
            self.assertEqual(vram_trim.unload_after_seconds(True), 0.0)


class SwappedTests(unittest.TestCase):
    def test_weights_stay_on_host_outside_the_call_and_values_are_kept(self):
        module = nn.Linear(3, 3)
        before = module.weight.detach().clone()
        swap = vram_trim._Swapped([module], "cpu")
        with swap:
            torch.testing.assert_close(module.weight.detach(), before)
        torch.testing.assert_close(module.weight.detach(), before)
        self.assertGreater(swap.bytes, 0)


if __name__ == "__main__":
    unittest.main()
