"""torchao INT4 モデルの TensorRT 変換（int4_builder.py / trt_int4.py）の契約テスト。

GPU も TensorRT も使わない。ONNX の組み替えは、小さな numpy のインタプリタで数値まで確かめる。
"""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import TensorProto, helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wrapper"))
sys.path.insert(0, str(ROOT / "bf16-fallback"))
import int4_builder  # noqa: E402
import trt_int4  # noqa: E402

G = int4_builder.GROUP


def write_header_only(path, metadata):
    header = json.dumps({"__metadata__": metadata}).encode()
    Path(path).write_bytes(struct.pack("<Q", len(header)) + header)


def unpack_int4(raw, count):
    data = np.frombuffer(raw, dtype=np.uint8)
    nibbles = np.stack([data & 0xF, data >> 4], axis=1).reshape(-1)[:count].astype(np.int16)
    return np.where(nibbles > 7, nibbles - 16, nibbles).astype(np.float32)


def bf16_to_float(raw):
    return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def evaluate(model, feeds):
    """rewrite_graph が出す演算だけを解く numpy のインタプリタ。"""
    values = dict(feeds)
    for init in model.graph.initializer:
        dims = tuple(init.dims)
        if init.data_type == TensorProto.INT4:
            values[init.name] = unpack_int4(init.raw_data, int(np.prod(dims))).reshape(dims)
        elif init.data_type == TensorProto.BFLOAT16:
            values[init.name] = bf16_to_float(init.raw_data).reshape(dims)
        else:
            values[init.name] = onnx.numpy_helper.to_array(init)
    for node in model.graph.node:
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        ins = [values[name] for name in node.input]
        if node.op_type == "DequantizeLinear":  # axis=0, block_size: (K,N) 整数 × (K/block,N) スケール
            weight, scale = ins
            out = weight * np.repeat(scale, attrs["block_size"], axis=0)
        elif node.op_type == "MatMul":
            out = ins[0] @ ins[1]
        elif node.op_type == "Reshape":
            shape = [d if d != 0 else ins[0].shape[i] for i, d in enumerate(ins[1])]
            out = ins[0].reshape(shape)
        elif node.op_type == "ReduceSum":
            out = ins[0].sum(axis=tuple(int(a) for a in ins[1]), keepdims=bool(attrs.get("keepdims", 1)))
        elif node.op_type == "Add":
            out = ins[0] + ins[1]
        elif node.op_type == "Split":
            edges = np.cumsum(ins[1])[:-1]
            parts = np.split(ins[0], edges, axis=attrs["axis"])
            for name, part in zip(node.output, parts):
                values[name] = part
            continue
        elif node.op_type in ("Cast", "Identity"):
            out = ins[0]
        else:
            raise AssertionError(f"unexpected op {node.op_type}")
        values[node.output[0]] = out
    return values


class PackingTests(unittest.TestCase):
    def test_int4_nibble_order_matches_onnx(self):
        # 先の要素が下位ニブル。-8..7 を 2 の補数の 4 ビットで詰める
        packed = int4_builder.pack_int4(np.array([-8, 7, 0, 1, -1], dtype=np.int8))
        self.assertEqual(packed, bytes([0x78, 0x10, 0x0F]))
        values = np.arange(-8, 8, dtype=np.int8)
        np.testing.assert_array_equal(unpack_int4(int4_builder.pack_int4(values), 16), values)

    def test_bf16_rounds_to_nearest_even(self):
        self.assertEqual(int4_builder.bf16_bytes(np.array([1.0], dtype=np.float32)), bytes([0x80, 0x3F]))
        tie_to_even = np.array([0x3F808000], dtype=np.uint32).view(np.float32)   # ちょうど中間
        tie_to_odd = np.array([0x3F818000], dtype=np.uint32).view(np.float32)
        self.assertEqual(int4_builder.bf16_bytes(tie_to_even), bytes([0x80, 0x3F]))
        self.assertEqual(int4_builder.bf16_bytes(tie_to_odd), bytes([0x82, 0x3F]))

    def test_context_projections_are_not_traced(self):
        for name in ("blocks.0.attention.wk_text", "blocks.3.attention.wv_speaker", "blocks.1.attention.wk_caption"):
            self.assertTrue(int4_builder.is_context_projection(name), name)
        for name in ("blocks.0.attention.wq", "blocks.0.attention.wk", "blocks.5.mlp.w2"):
            self.assertFalse(int4_builder.is_context_projection(name), name)


class RewriteGraphTests(unittest.TestCase):
    def build(self):
        """x(bf16) を共有する QMM が2つ、別入力 x2(float32) の QMM が1つ。K は 256 と 128。"""
        def qmm(lid, n_out, in_dt, source, out):
            return helper.make_node("QMM", [source], [out], domain="irodori", name=f"qmm{lid}",
                                    lid=lid, n_out=n_out, in_dt=in_dt)
        nodes = [qmm(0, 8, int4_builder.DT_BFLOAT16, "x", "a"), qmm(1, 4, int4_builder.DT_BFLOAT16, "x", "b"),
                 qmm(2, 6, int4_builder.DT_FLOAT, "x2", "c")]
        graph = helper.make_graph(nodes, "g", [
            helper.make_tensor_value_info("x", TensorProto.BFLOAT16, ["B", "L", 256]),
            helper.make_tensor_value_info("x2", TensorProto.FLOAT, ["B", "L", 128])], [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, None) for name in "abc"])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("irodori", 1)])
        rng = np.random.default_rng(0)

        def layer(n, k):
            q = rng.integers(-8, 8, size=(n, k)).astype(np.int8)
            scale = rng.uniform(0.01, 0.05, size=(n, k // G)).astype(np.float32)
            zero = rng.uniform(-0.1, 0.1, size=(n, k // G)).astype(np.float32)
            return q, scale, zero
        data = {0: layer(8, 256), 1: layer(4, 256), 2: layer(6, 128)}
        return model, data

    @staticmethod
    def reference(layer, x):
        q, scale, zero = layer
        n, k = q.shape
        weight = q.astype(np.float32) * np.repeat(scale, G, axis=1) + np.repeat(zero, G, axis=1)
        return x @ weight.T

    def test_graph_structure(self):
        model, data = self.build()
        rewritten, groups = int4_builder.rewrite_graph(model, data, ["l0", "l1", "l2"])
        ops = [n.op_type for n in rewritten.graph.node]
        self.assertNotIn("QMM", ops)
        self.assertEqual(groups, 2)                       # 入力を共有する2つは1つにまとめる
        self.assertEqual(ops.count("DequantizeLinear"), 2)
        self.assertEqual(ops.count("Split"), 1)
        self.assertEqual(ops.count("Identity"), 1)
        # float32 の入力は BF16 にして計算し、結果を元の型に戻す（torchao の Linear と同じ）
        casts = [(n.name, helper.get_attribute_value(n.attribute[0])) for n in rewritten.graph.node
                 if n.op_type == "Cast"]
        self.assertEqual(sorted(to for _, to in casts), [int4_builder.DT_FLOAT, int4_builder.DT_BFLOAT16])
        self.assertEqual([o.domain for o in rewritten.opset_import], [""])
        shapes = {i.name: (i.data_type, tuple(i.dims)) for i in rewritten.graph.initializer}
        self.assertEqual(shapes["i4_0.w"], (TensorProto.INT4, (256, 12)))      # (K, N0+N1)
        self.assertEqual(shapes["i4_0.s"], (TensorProto.BFLOAT16, (2, 12)))    # (K/128, N)
        self.assertEqual(shapes["i4_2.w"], (TensorProto.INT4, (128, 6)))

    def test_numerics_match_the_torchao_weights(self):
        model, data = self.build()
        rewritten, _ = int4_builder.rewrite_graph(model, data, ["l0", "l1", "l2"])
        rng = np.random.default_rng(1)
        x = rng.normal(size=(2, 3, 256)).astype(np.float32)
        x2 = rng.normal(size=(2, 3, 128)).astype(np.float32)
        values = evaluate(rewritten, {"x": x, "x2": x2})
        # スケール・ゼロ点は BF16 に丸めて入るので、その分の誤差（約 0.4%）だけ許す
        for name, layer, source in (("a", data[0], x), ("b", data[1], x), ("c", data[2], x2)):
            expected = self.reference(layer, source)
            error = np.abs(values[name] - expected).max() / np.abs(expected).max()
            self.assertLess(error, 0.02, name)

    def test_group_by_input_keeps_order(self):
        model, _ = self.build()
        groups = int4_builder.group_by_input(model.graph.node)
        self.assertEqual([n.name for n in groups["x"]], ["qmm0", "qmm1"])
        self.assertEqual([n.name for n in groups["x2"]], ["qmm2"])


class QuantizationDetectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "model.safetensors"

    def test_detects_int4(self):
        write_header_only(self.path, {"irodori_quantization_json": json.dumps(
            {"quantization_type": "int4_weight_only"})})
        self.assertEqual(trt_int4.checkpoint_quantization(self.path), "int4_weight_only")
        self.assertTrue(trt_int4.is_int4_checkpoint(self.path))

    def test_other_quantizations_and_plain_models_are_not_int4(self):
        write_header_only(self.path, {"irodori_quantization_json": json.dumps(
            {"quantization_type": "int8_weight_only"})})
        self.assertEqual(trt_int4.checkpoint_quantization(self.path), "int8_weight_only")
        self.assertFalse(trt_int4.is_int4_checkpoint(self.path))
        write_header_only(self.path, {"config_json": "{}"})
        self.assertIsNone(trt_int4.checkpoint_quantization(self.path))

    def test_unreadable_files_are_not_int4(self):
        self.path.write_bytes(b"not a safetensors file")
        self.assertIsNone(trt_int4.checkpoint_quantization(self.path))
        self.assertFalse(trt_int4.is_int4_checkpoint(Path(self.temp.name) / "missing.safetensors"))


class EnsurePlanTests(unittest.TestCase):
    def patches(self, box, run):
        return (patch.object(trt_int4, "BOX", box),
                patch.object(trt_int4, "cache_identity", return_value={"model": "a", "variant": "int4"}),
                patch.object(trt_int4, "model_digest", return_value="a"),
                patch.object(trt_int4.subprocess, "run", side_effect=run))

    def test_failed_build_never_publishes_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            box = Path(temp)

            def fail(args, **kwargs):
                kwargs["stdout"].write("Traceback:\nRuntimeError: 変換エラー\n")
                return type("Result", (), {"returncode": 1})()
            first, second, third, fourth = self.patches(box, fail)
            with first, second, third, fourth:
                with self.assertRaisesRegex(RuntimeError, "RuntimeError: 変換エラー[\s\S]*CUDA"):
                    trt_int4.ensure_plan(box / "model", lambda *a: None, lambda *a: None)
            self.assertEqual(list(box.rglob("ready.json")), [])

    def test_builds_in_order_and_publishes_an_int4_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            box = Path(temp)
            calls = []

            def ok(args, **kwargs):
                calls.append(args[1:])
                work = Path(kwargs["env"]["IRODORI_TRT_WORK_DIR"])
                self.assertEqual(kwargs["env"]["IRODORI_TRT_OUTPUT_DIR"], str(work))
                if "build" in args:
                    (work / "fallback_int4.plan").write_bytes(b"plan")
                return type("Result", (), {"returncode": 0})()
            first, second, third, fourth = self.patches(box, ok)
            with first, second, third, fourth:
                plan = trt_int4.ensure_plan(box / "model", lambda *a: None, lambda *a: None)
                self.assertEqual(plan.name, "fallback_int4.plan")
                self.assertEqual(plan.read_bytes(), b"plan")
                self.assertEqual([Path(c[0]).name for c in calls],
                                 ["int4_builder.py", "int4_builder.py", "fallback_bf16.py", "trt_int4.py"])
                self.assertEqual(calls[0][1:], ["capture"])
                self.assertEqual(calls[1][1:], ["export", "--label", "fallback_int4"])
                self.assertEqual(calls[2][1:], ["build", "--label", "fallback_int4"])
                self.assertEqual(calls[3][1], "--verify")
                # 2回目は変換せずに再利用する
                calls.clear()
                self.assertEqual(trt_int4.ensure_plan(box / "model", lambda *a: None, lambda *a: None), plan)
                self.assertEqual(calls, [])
            record = json.loads((plan.parent / "ready.json").read_text(encoding="utf-8"))
            self.assertEqual(record["identity"]["variant"], "int4")
            self.assertFalse(list((box / ".cache/trt").glob("build-*")))   # 作業フォルダは消える

    def test_int4_sources_cover_only_the_int4_scripts(self):
        # INT4 のスクリプトだけをハッシュに入れる。BF16 モデルの plan の識別（trt_cache.py）には混ぜない
        # ので、INT4 のスクリプトを直しても Small の plan を作り直させない。
        self.assertEqual(sorted(Path(k).as_posix() for k in trt_int4.int4_sources()),
                         ["irodori-tts/bf16-fallback/int4_builder.py", "irodori-tts/wrapper/trt_int4.py"])


if __name__ == "__main__":
    unittest.main()
