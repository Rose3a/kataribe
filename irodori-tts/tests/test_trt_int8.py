"""W8A8 モデルの TensorRT 変換（int8_builder.py / trt_int8.py）の契約テスト。

GPU も TensorRT も使わない。ONNX の組み替えは、小さな numpy のインタプリタで数値まで確かめる。
"""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wrapper"))
sys.path.insert(0, str(ROOT / "bf16-fallback"))
import int8_builder  # noqa: E402
import trt_int8  # noqa: E402


def write_header_only(path, metadata):
    header = json.dumps({"__metadata__": metadata}).encode()
    Path(path).write_bytes(struct.pack("<Q", len(header)) + header)


def bf16_to_float(raw):
    return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def bf16_round(values):
    return bf16_to_float(int8_builder.bf16_bytes(values)).reshape(values.shape)


def evaluate(model, feeds):
    """rewrite_graph が出す演算だけを解く numpy のインタプリタ。"""
    values = dict(feeds)
    for init in model.graph.initializer:
        dims = tuple(init.dims)
        if init.data_type == TensorProto.BFLOAT16:
            values[init.name] = bf16_to_float(init.raw_data).reshape(dims)
        else:
            values[init.name] = onnx.numpy_helper.to_array(init).astype(np.float32)
    for node in model.graph.node:
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        ins = [values[name] for name in node.input]
        if node.op_type == "DequantizeLinear":
            scale = ins[1] if np.ndim(ins[1]) == 0 or attrs.get("axis", 1) != 1 else ins[1][None, :]
            out = ins[0] * scale
        elif node.op_type == "QuantizeLinear":
            out = np.clip(np.rint(ins[0] / ins[1]), -128, 127)
        elif node.op_type == "MatMul":
            out = ins[0] @ ins[1]
        elif node.op_type == "Split":
            bounds = np.cumsum(ins[1].astype(np.int64))[:-1]
            for name, part in zip(node.output, np.split(ins[0], bounds, axis=attrs["axis"])):
                values[name] = part
            continue
        else:
            raise AssertionError(node.op_type)
        values[node.output[0]] = out
    return values[model.graph.output[0].name]


def make_graph(weights):
    """x → MatMul(W0) → MatMul(W1) …（BF16 の定数の重み）。"""
    nodes, inits, x = [], [], "x"
    for i, w in enumerate(weights):
        tensor = TensorProto()
        tensor.name, tensor.data_type = f"w{i}", TensorProto.BFLOAT16
        tensor.dims.extend(w.shape)
        tensor.raw_data = int8_builder.bf16_bytes(w)
        inits.append(tensor)
        nodes.append(helper.make_node("MatMul", [x, f"w{i}"], [f"y{i}"]))
        x = f"y{i}"
    graph = helper.make_graph(
        nodes, "g", [helper.make_tensor_value_info("x", TensorProto.BFLOAT16, [2, weights[0].shape[0]])],
        [helper.make_tensor_value_info(x, TensorProto.BFLOAT16, None)], inits)
    return helper.make_model(graph)


def quantized(rng, k, n):
    q = rng.integers(-127, 128, size=(n, k)).astype(np.int8)       # torchao の並び (N, K)
    scale = bf16_round(rng.uniform(0.001, 0.01, size=n).astype(np.float32))
    dense = bf16_round(q.astype(np.float32) * scale[:, None])        # dequantize_blocks と同じ丸め
    return q, scale, dense


class RewriteTest(unittest.TestCase):
    def test_weights_and_activation_quantization(self):
        rng = np.random.default_rng(0)
        q0, s0, d0 = quantized(rng, 8, 6)
        q1, s1, d1 = quantized(rng, 6, 4)
        model = make_graph([d0.T.copy(), d1.T.copy()])
        layers = {
            int8_builder.weight_key(int8_builder.bf16_bytes(d0.T.copy())): ("blocks.0.mlp.w1", q0.T.copy(), s0, 2.0),
            int8_builder.weight_key(int8_builder.bf16_bytes(d1.T.copy())): ("blocks.0.mlp.w2", q1.T.copy(), s1, None),
        }
        done, missing = int8_builder.rewrite_graph(model, layers)
        self.assertEqual(sorted(done), ["blocks.0.mlp.w1", "blocks.0.mlp.w2"])
        self.assertEqual(missing, [])
        ops = [n.op_type for n in model.graph.node]
        # w1: 重み DQ ＋ 活性 Q/DQ ＋ MatMul、w2: 重み DQ ＋ MatMul（活性は BF16 のまま）
        self.assertEqual(ops, ["DequantizeLinear", "QuantizeLinear", "DequantizeLinear", "MatMul",
                               "DequantizeLinear", "MatMul"])
        self.assertFalse({"w0", "w1"} & {i.name for i in model.graph.initializer})

        x = rng.uniform(-2, 2, size=(2, 8)).astype(np.float32)
        a_scale = bf16_round(np.array(2.0 / 127, dtype=np.float32))
        xq = np.clip(np.rint(x / a_scale), -128, 127) * a_scale
        expected = (xq @ (q0.T * s0[None, :])) @ (q1.T * s1[None, :])
        np.testing.assert_allclose(evaluate(model, {"x": x}), expected, rtol=1e-5, atol=1e-5)

    def test_shared_input_is_merged(self):
        """同じ入力の Linear（wq/wk など）は1つの Q/DQ・MatMul にまとめ、Split で分ける。"""
        rng = np.random.default_rng(2)
        (qa, sa, da), (qb, sb, db) = quantized(rng, 8, 6), quantized(rng, 8, 4)
        inits = []
        for name, d in (("wa", da), ("wb", db)):
            t = TensorProto(); t.name, t.data_type = name, TensorProto.BFLOAT16
            t.dims.extend(d.T.shape); t.raw_data = int8_builder.bf16_bytes(d.T.copy()); inits.append(t)
        graph = helper.make_graph(
            [helper.make_node("MatMul", ["x", "wa"], ["ya"]), helper.make_node("MatMul", ["x", "wb"], ["yb"]),
             helper.make_node("MatMul", ["ya", "ya_t"], ["za"])],
            "g", [helper.make_tensor_value_info("x", TensorProto.BFLOAT16, [2, 8])],
            [helper.make_tensor_value_info("za", TensorProto.BFLOAT16, None),
             helper.make_tensor_value_info("yb", TensorProto.BFLOAT16, None)],
            inits + [onnx.numpy_helper.from_array(np.eye(6, dtype=np.float32), "ya_t")])
        model = helper.make_model(graph)
        layers = {
            int8_builder.weight_key(int8_builder.bf16_bytes(da.T.copy())): ("blocks.0.attention.wq", qa.T.copy(), sa, 2.0),
            int8_builder.weight_key(int8_builder.bf16_bytes(db.T.copy())): ("blocks.0.attention.wk", qb.T.copy(), sb, 2.0),
        }
        done, missing = int8_builder.rewrite_graph(model, layers)
        self.assertEqual(missing, [])
        self.assertEqual(sorted(done), ["blocks.0.attention.wk", "blocks.0.attention.wq"])
        ops = [n.op_type for n in model.graph.node]
        self.assertEqual(ops, ["DequantizeLinear", "QuantizeLinear", "DequantizeLinear", "MatMul", "Split", "MatMul"])
        x = rng.uniform(-2, 2, size=(2, 8)).astype(np.float32)
        a_scale = bf16_round(np.array(2.0 / 127, dtype=np.float32))
        xq = np.clip(np.rint(x / a_scale), -128, 127) * a_scale
        np.testing.assert_allclose(evaluate(model, {"x": x}), xq @ (qa.T * sa[None, :]), rtol=1e-5, atol=1e-5)

    def test_missing_layer_is_reported(self):
        rng = np.random.default_rng(1)
        q, s, d = quantized(rng, 4, 4)
        model = make_graph([d.T.copy()])
        layers = {"0" * 40: ("blocks.3.attention.wq", q.T.copy(), s, 1.0)}
        done, missing = int8_builder.rewrite_graph(model, layers)
        self.assertEqual(done, [])
        self.assertEqual(missing, ["blocks.3.attention.wq"])


class SettingsTest(unittest.TestCase):
    def test_trt_settings(self):
        payload = {"quantization_type": "int8_dynamic_activation_int8_weight",
                   "trt_activation_amax": {"blocks.0.mlp.w1": 3}, "trt_bf16_activation": ["blocks.0.mlp.w2"]}
        amax, bf16 = int8_builder.trt_settings(payload)
        self.assertEqual(amax, {"blocks.0.mlp.w1": 3.0})
        self.assertEqual(bf16, {"blocks.0.mlp.w2"})
        with self.assertRaises(RuntimeError):
            int8_builder.trt_settings({"quantization_type": "int8_dynamic_activation_int8_weight"})
        with self.assertRaises(RuntimeError):
            int8_builder.trt_settings(dict(payload, quantization_type="int8_weight_only"))

    def test_is_w8a8_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.safetensors"
            base = {"quantization_type": "int8_dynamic_activation_int8_weight"}
            write_header_only(path, {"irodori_quantization_json": json.dumps(dict(base, trt_activation_amax={"a": 1}))})
            self.assertTrue(trt_int8.is_w8a8_checkpoint(path))
            # 本家の量子化コマンドで作った INT8（活性スケールなし）は TensorRT では使わない
            write_header_only(path, {"irodori_quantization_json": json.dumps(base)})
            self.assertFalse(trt_int8.is_w8a8_checkpoint(path))
            write_header_only(path, {"irodori_quantization_json": json.dumps({"quantization_type": "int4_weight_only"})})
            self.assertFalse(trt_int8.is_w8a8_checkpoint(path))
            write_header_only(path, {})
            self.assertFalse(trt_int8.is_w8a8_checkpoint(path))


if __name__ == "__main__":
    unittest.main()
