"""長さの予測で、文頭 <s> と文末の記号が足す長さに上限を付ける（duration_edge_cap_frames）。

- 文頭・文末のトークンだけを同じ割合で縮め、中身のトークンは変えない。上限より短ければ何もしない、
- 上限を掛ける位置は BOS と、文末に続く記号だけのトークン（文中の読点は入れない）、
- CUDA Graph で記録したあとに上限の値を変えても、再生に反映される（ホストと同期する分岐を使わない）。

実機の確認（TensorRT、話者なし）: 「胸が痛みます。」は <s> 21.2 + 中身 21.5 + 「。」37.8 フレームが、
上限 12 で 33.6 フレーム（1.36 秒）になる。
"""
from pathlib import Path
import sys
import types
import unittest

import torch

BOX = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BOX / "irodori-tts" / "wrapper"))
sys.path.insert(0, str(BOX / "runtime" / "trt-lab" / "repo"))

from irodori_tts.inference_runtime import InferenceRuntime  # noqa: E402
from irodori_tts.model import cap_edge_frames  # noqa: E402

CUDA = torch.cuda.is_available()


class CapEdgeFramesTest(unittest.TestCase):
    def test_only_the_edges_shrink(self):
        frames = torch.tensor([[21.2, 10.1, 7.5, 3.9, 37.8, 5.0]])
        mask = torch.tensor([[True, True, True, True, True, False]])  # 最後は詰め物
        edge = torch.tensor([[True, False, False, False, True, False]])
        out = cap_edge_frames(frames, mask, edge, torch.tensor([12.0]))
        self.assertAlmostEqual(float(out[0, 0] + out[0, 4]), 12.0, places=4)
        self.assertAlmostEqual(float(out[0, 0] / out[0, 4]), 21.2 / 37.8, places=4)
        torch.testing.assert_close(out[0, 1:4], frames[0, 1:4])
        self.assertEqual(float(out[0, 5]), 5.0)  # 詰め物には触らない

    def test_short_edges_are_kept(self):
        frames = torch.tensor([[4.0, 10.0, 3.0]])
        edge = torch.tensor([[True, False, True]])
        out = cap_edge_frames(frames, torch.ones_like(edge), edge, torch.tensor([12.0]))
        torch.testing.assert_close(out, frames)

    def test_each_row_has_its_own_cap(self):
        frames = torch.tensor([[20.0, 5.0, 20.0], [20.0, 5.0, 20.0]])
        edge = torch.tensor([[True, False, True]] * 2)
        out = cap_edge_frames(frames, torch.ones_like(edge), edge, torch.tensor([10.0, 30.0]))
        self.assertAlmostEqual(float(out[0, 0] + out[0, 2]), 10.0, places=4)
        self.assertAlmostEqual(float(out[1, 0] + out[1, 2]), 30.0, places=4)


class EdgeMaskTest(unittest.TestCase):
    def runtime(self, add_bos=True):
        vocab = {"<s>": 1, "胸が": 10, "痛み": 11, "ます": 12, "。": 3, "、": 4, "!?": 5, "▁": 6, "ね": 13}
        tok = types.SimpleNamespace(tokenizer=types.SimpleNamespace(get_vocab=lambda: vocab), add_bos=add_bos)
        return types.SimpleNamespace(tokenizer=tok)

    def edge(self, ids, n, add_bos=True):
        ids = torch.tensor([ids])
        mask = torch.zeros_like(ids, dtype=torch.bool)
        mask[0, :n] = True
        return InferenceRuntime._duration_edge_mask(self.runtime(add_bos), ids, mask)[0].tolist()

    def test_bos_and_trailing_punctuation(self):
        # <s> 胸が 痛み ます 。 [詰め物]
        self.assertEqual(self.edge([1, 10, 11, 12, 3, 0], 5), [True, False, False, False, True, False])

    def test_a_run_of_trailing_marks(self):
        self.assertEqual(self.edge([1, 13, 5, 3], 4), [True, False, True, True])

    def test_commas_inside_are_left_alone(self):
        self.assertEqual(self.edge([1, 10, 4, 11, 12], 5), [True, False, False, False, False])

    def test_without_bos(self):
        self.assertEqual(self.edge([10, 11, 3], 3, add_bos=False), [False, False, True])

    def test_a_space_only_token_is_not_punctuation(self):
        self.assertEqual(self.edge([1, 13, 6], 3), [True, False, False])


@unittest.skipUnless(CUDA, "needs CUDA")
class CudaGraphTest(unittest.TestCase):
    def test_a_new_cap_takes_effect_on_replay(self):
        import cuda_graphs
        call = cuda_graphs.GraphedCall(cap_edge_frames, "edge_cap", log=lambda m: None)
        frames = torch.tensor([[21.2, 10.1, 7.5, 3.9, 37.8]], device="cuda")
        mask = torch.ones_like(frames, dtype=torch.bool)
        edge = torch.tensor([[True, False, False, False, True]], device="cuda")
        with torch.no_grad():  # 推論と同じ（勾配が要る呼び出しは記録しない）
            for cap in (12.0, 12.0, 30.0, 8.0):  # 2 回目から記録した graph の再生
                out = call(frames, mask, edge, torch.tensor([cap], device="cuda"))
                self.assertAlmostEqual(float(out[0, 0] + out[0, 4]), cap, places=3)
        self.assertGreaterEqual(call.replays, 2)


if __name__ == "__main__":
    unittest.main()
