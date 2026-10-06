"""vram_trim が torchao の量子化テンソル（v4-Large INT4）を壊さないことの契約テスト。

量子化テンソルは data_ptr() が常に 0 で、repack がそれらを1つにまとめて最初の重みで上書きしていた。
実際の torchao は読まず、同じ性質の代役で確かめる。
"""
from pathlib import Path
import sys
import unittest

IRODORI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IRODORI_ROOT / "wrapper"))
try:
    import torch
    import vram_trim
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = None


class QuantizedData:
    """torchao のテンソルに似せたもの: 素の Tensor ではなく、実体は qdata と scale_and_zero。"""

    def __init__(self, qdata, scale_and_zero):
        self.qdata, self.scale_and_zero = qdata, scale_and_zero

    def __tensor_flatten__(self):
        return ["qdata", "scale_and_zero"], None


class QuantizedParam:
    is_cuda = True

    def __init__(self, logical_numel, qdata, scale_and_zero):
        self.data = QuantizedData(qdata, scale_and_zero)
        self.logical_numel = logical_numel

    def numel(self):
        return self.logical_numel

    def element_size(self):
        return 2  # 見かけは BF16。実際の大きさとは違う

    def data_ptr(self):
        return 0


@unittest.skipIf(torch is None, "torch is not installed in this interpreter")
class QuantizedWeightTests(unittest.TestCase):
    def test_plain_tensors_are_plain_and_quantized_ones_are_not(self):
        plain = torch.nn.Parameter(torch.zeros(4, 4))
        self.assertTrue(vram_trim._is_plain(plain))
        self.assertFalse(vram_trim._is_plain(QuantizedParam(1, None, None)))

    def test_size_counts_the_stored_bytes_not_the_logical_elements(self):
        plain = torch.nn.Parameter(torch.zeros(4, 4, dtype=torch.bfloat16))
        self.assertEqual(vram_trim._param_bytes(plain), 32)
        qdata = torch.zeros(8, 4, dtype=torch.int32)          # 128 B
        scale_and_zero = torch.zeros(2, 8, 2, dtype=torch.bfloat16)   # 64 B
        # 論理サイズ 2048 要素 × 2 B = 4096 B ではなく、持っている 192 B
        self.assertEqual(vram_trim._param_bytes(QuantizedParam(2048, qdata, scale_and_zero)), 192)

    def test_repack_leaves_quantized_weights_alone(self):
        class Holder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.first = QuantizedParam(16, torch.zeros(2, 2), torch.zeros(1, 2, 2))
                self.second = QuantizedParam(32, torch.zeros(3, 3), torch.zeros(1, 3, 2))

            def parameters(self, recurse=True):
                return iter([self.first, self.second])

            def buffers(self, recurse=True):
                return iter([])

        holder = Holder()
        before = (holder.first.data, holder.second.data)
        vram_trim.repack(holder)       # 以前は2つの data_ptr（どちらも 0）を同じ組と見て、片方で上書きした
        self.assertIs(holder.first.data, before[0])
        self.assertIs(holder.second.data, before[1])
        self.assertIsNot(holder.first.data, holder.second.data)


if __name__ == "__main__":
    unittest.main()
