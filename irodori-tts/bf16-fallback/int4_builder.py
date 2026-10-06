"""torchao INT4 (W4A16, group 128) の DiT を TensorRT 用の ONNX にする。

torchao の INT4 は PyTorch 専用のカーネル（tinygemm）の形式なので、plan にはそのまま渡せない。
公開されている重み（Aratako/Irodori-TTS-v4-Large-Quantized/int4-weight-only）から整数値・スケール・
ゼロ点を取り出し、ONNX の INT4 ``DequantizeLinear``（ブロック量子化）＋ ``MatMul`` に組み直す。

torchao のゼロ点は浮動小数の非対称（w = s*(q-8) + z）だが、TensorRT の INT4 はゼロ点なしの対称
量子化が前提。対称に量子化し直すと誤差が大きくなる（Large で相対 RMSE 5%）ので、z はグループごとの
入力の合計に掛ける補正項（``ReduceSum`` → ``MatMul`` → ``Add``）として別に足し、元の重みを正確に
再現する。入力を共有する Linear（wq/wk/wv/gate、w1/w3）は1つの MatMul にまとめる。

    int4_builder.py capture                      # DiT に渡る入力を保存する（trt_experiment.py capture 相当）
    int4_builder.py export --label fallback_int4 # QMM 置換 → ONNX → INT4 DQ への置き換え

plan の構築は fallback_bf16.py の ``build --label fallback_int4`` を使う（入出力の約束は同じ）。
環境変数は fallback_bf16.py と同じ（IRODORI_CHECKPOINT / IRODORI_TRT_LAB_DIR / IRODORI_TRT_WORK_DIR /
IRODORI_TRT_OUTPUT_DIR）。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import fallback_bf16 as fb  # noqa: E402  (同じフォルダ。LAB / NAMES / artifact_path / primitive_sdpa)

GROUP = 128
OPSET = 21  # INT4 と BFLOAT16 のブロック量子化 DequantizeLinear は opset 21 から
# ONNX の TensorProto.DataType（onnx を import しなくても使えるように数値で持つ）
DT_FLOAT, DT_FLOAT16, DT_BFLOAT16 = 1, 10, 16
SCALAR_TYPE_TO_DT = {"Float": DT_FLOAT, "Half": DT_FLOAT16, "BFloat16": DT_BFLOAT16}
# DiT の中で実際に走る INT4 Linear は blocks.N.attention.{wq,wk,wv,gate,wo} と blocks.N.mlp.{w1,w2,w3}。
# wk_text などの文脈側の射影は K/V キャッシュを作る PyTorch 側が使うので、plan には入らない。
CONTEXT_PROJECTIONS = ("_text", "_speaker", "_caption")
log = logging.getLogger("int4_builder")


# ------------------------------------------------------------------ 純粋な関数（テストしやすいもの）
def pack_int4(values: np.ndarray) -> bytes:
    """符号付き INT4（-8..7）を ONNX の並びで詰める。先の要素が下位ニブル。"""
    flat = (values.reshape(-1).astype(np.int16) & 0xF).astype(np.uint8)
    if flat.size % 2:
        flat = np.append(flat, np.uint8(0))
    return (flat[0::2] | (flat[1::2] << 4)).astype(np.uint8).tobytes()


def bf16_bytes(values: np.ndarray) -> bytes:
    """float32 → bfloat16（最近接偶数丸め）の生バイト列。"""
    bits = np.ascontiguousarray(values.astype(np.float32)).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16).tobytes()


def is_context_projection(name: str) -> bool:
    return any(word in name.rsplit(".", 1)[-1] for word in CONTEXT_PROJECTIONS)


def group_by_input(nodes) -> dict:
    """QMM ノードを入力テンソル名ごとにまとめる（出現順）。"""
    groups: dict = {}
    for node in nodes:
        if node.op_type == "QMM":
            groups.setdefault(node.input[0], []).append(node)
    return groups


# ------------------------------------------------------------------ 取り出し（GPU）
def extract_linear(weight, probe_cls):
    """torchao INT4 の重みから (q-8 の int8 (N,K), scale (N,K/G), zero (N,K/G)) を取り出す。"""
    import torch
    n_out, n_in = weight.shape
    scale_and_zero = weight.scale_and_zero  # (K_pad/G, N_pad, 2)
    probe = scale_and_zero.clone()
    probe[..., 0] = 1
    probe[..., 1] = 0
    # scale=1, zero=0 の tinygemm に単位行列を通すと、整数値 q-8 がそのまま出てくる
    pw = probe_cls(qdata=weight.qdata, scale_and_zero=probe, block_size=weight.block_size, shape=weight.shape)
    eye = torch.eye(n_in, dtype=torch.bfloat16, device=weight.device)
    out = torch.nn.functional.linear(eye, pw)  # (K, N)
    q = out.t().to(torch.int8).contiguous()
    if not torch.equal(out.float(), q.t().float()) or int(q.min()) < -8 or int(q.max()) > 7:
        raise RuntimeError("INT4 の整数値を取り出せませんでした（torchao の形式が想定と違います）")
    scale = scale_and_zero[: n_in // GROUP, :n_out, 0].t().contiguous()
    zero = scale_and_zero[: n_in // GROUP, :n_out, 1].t().contiguous()
    return q.cpu().numpy(), scale.float().cpu().numpy(), zero.float().cpu().numpy()


# ------------------------------------------------------------------ ONNX の組み替え
def rewrite_graph(model, data: dict, layers: list[str]):
    """QMM ノードを INT4 の DequantizeLinear + MatMul（+ 補正項）に置き換える。"""
    from onnx import TensorProto, helper, numpy_helper

    def int4_tensor(name, values):
        tensor = TensorProto()
        tensor.name, tensor.data_type = name, TensorProto.INT4
        tensor.dims.extend(values.shape)
        tensor.raw_data = pack_int4(values)
        return tensor

    def bf16_tensor(name, values):
        tensor = TensorProto()
        tensor.name, tensor.data_type = name, TensorProto.BFLOAT16
        tensor.dims.extend(values.shape)
        tensor.raw_data = bf16_bytes(values)
        return tensor

    groups = group_by_input(model.graph.node)
    first_of = {id(members[0]): members for members in groups.values()}
    skipped = {id(node) for members in groups.values() for node in members[1:]}
    nodes, initializers = [], []
    for node in model.graph.node:
        if node.op_type != "QMM":
            nodes.append(node)
            continue
        if id(node) in skipped:
            continue
        members = first_of[id(node)]
        lids = [next(a.i for a in m.attribute if a.name == "lid") for m in members]
        in_dt = next(a.i for a in members[0].attribute if a.name == "in_dt")
        tag = f"i4_{lids[0]}"
        q = np.concatenate([data[lid][0] for lid in lids], axis=0)      # (N_total, K)
        scale = np.concatenate([data[lid][1] for lid in lids], axis=0)  # (N_total, K/G)
        zero = np.concatenate([data[lid][2] for lid in lids], axis=0)
        sizes = [data[lid][0].shape[0] for lid in lids]
        groups_per_row = q.shape[1] // GROUP
        initializers += [
            int4_tensor(f"{tag}.w", np.ascontiguousarray(q.T)),                 # (K, N_total)
            bf16_tensor(f"{tag}.s", np.ascontiguousarray(scale.T)),             # (K/G, N_total)
            bf16_tensor(f"{tag}.z", np.ascontiguousarray(zero.T)),
            numpy_helper.from_array(np.array([0, 0, groups_per_row, GROUP], dtype=np.int64), f"{tag}.shape"),
            numpy_helper.from_array(np.array([3], dtype=np.int64), f"{tag}.axes"),
            numpy_helper.from_array(np.array(sizes, dtype=np.int64), f"{tag}.split"),
        ]
        x = node.input[0]
        if in_dt != DT_BFLOAT16:  # torchao の Linear は入力を BF16 にして、結果を元の型に戻す
            nodes.append(helper.make_node("Cast", [x], [f"{tag}.xb"], name=f"{tag}.cast_in", to=DT_BFLOAT16))
            x = f"{tag}.xb"
        outputs = [m.output[0] for m in members]
        split_outputs = outputs if in_dt == DT_BFLOAT16 else [f"{tag}.yb{i}" for i in range(len(outputs))]
        nodes += [
            helper.make_node("DequantizeLinear", [f"{tag}.w", f"{tag}.s"], [f"{tag}.dq"], name=f"{tag}.dq",
                             axis=0, block_size=GROUP),
            helper.make_node("MatMul", [x, f"{tag}.dq"], [f"{tag}.mm"], name=f"{tag}.mm"),
            helper.make_node("Reshape", [x, f"{tag}.shape"], [f"{tag}.xg"], name=f"{tag}.group"),
            helper.make_node("ReduceSum", [f"{tag}.xg", f"{tag}.axes"], [f"{tag}.xs"], name=f"{tag}.sum", keepdims=0),
            helper.make_node("MatMul", [f"{tag}.xs", f"{tag}.z"], [f"{tag}.corr"], name=f"{tag}.corr"),
            helper.make_node("Add", [f"{tag}.mm", f"{tag}.corr"], [f"{tag}.y"], name=f"{tag}.add"),
        ]
        if len(outputs) == 1:
            nodes.append(helper.make_node("Identity", [f"{tag}.y"], [split_outputs[0]], name=f"{tag}.id"))
        else:
            nodes.append(helper.make_node("Split", [f"{tag}.y", f"{tag}.split"], split_outputs,
                                          name=f"{tag}.split", axis=-1))
        if in_dt != DT_BFLOAT16:
            for produced, wanted in zip(split_outputs, outputs):
                nodes.append(helper.make_node("Cast", [produced], [wanted], name=f"{produced}.cast_out", to=in_dt))
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.graph.initializer.extend(initializers)
    kept = [o for o in model.opset_import if o.domain == ""]
    del model.opset_import[:]
    model.opset_import.extend(kept)
    return model, len(groups)


# ------------------------------------------------------------------ サブコマンド
def _load_runtime():
    """INT4 のモデルは inference_mode の外で読む（中だと torchao の .to() が失敗する）。"""
    import torch
    torch.zeros(1, device="cuda")  # 先に CUDA を初期化しておく
    sys.path.insert(0, str(fb.LAB))
    import trt_experiment as te
    return te, te.runtime()


def capture() -> None:
    te, rt = _load_runtime()
    te.runtime = lambda: rt
    te.export(capture_only=True)


def export(label: str) -> Path:
    import torch
    import torch.nn.functional as F
    import onnx
    from torch.onnx import symbolic_helper
    from torchao.quantization import Int4TilePackedTo4dTensor

    te, rt = _load_runtime()
    out = fb.artifact_path(label, ".onnx")
    wrapper = te.CachedDiT(rt.model).eval()
    inputs = sorted(fb.WORK.glob("inputs_b*.pt"), key=lambda p: int(p.stem.split("_b")[-1]))
    if not inputs:
        raise FileNotFoundError("先に int4_builder.py capture で入力を保存してください")
    xs_cpu = torch.load(inputs[-1], weights_only=True, map_location="cpu")
    old_rope, old_sdpa = te.model_module.apply_rotary_emb, F.scaled_dot_product_attention
    te.model_module.apply_rotary_emb = te.real_rope
    try:
        # primitive attention（FP32 softmax）に差し替えても eager の結果が変わらないことを確かめる
        with torch.no_grad():
            xs = [x.cuda() for x in xs_cpu]
            expected = wrapper(*xs)
            fb._CAPTURED_SDPA = old_sdpa
            F.scaled_dot_product_attention = fb.primitive_sdpa
            if not torch.equal(expected, wrapper(*xs)):
                raise RuntimeError("eager primitive wrapper changed output")
            del xs, expected
        # INT4 の Linear を取り出して、独自演算子 QMM に置き換える
        # 形状と型だけが要る（重みの値は後で ONNX に差し込む）ので、中身は 0 を返すだけでよい
        @torch.library.custom_op("irodori::qmm", mutates_args=(), schema="(Tensor x, int lid, int n_out) -> Tensor")
        def qmm(x, lid, n_out):
            return x.new_zeros(*x.shape[:-1], n_out)

        @qmm.register_fake
        def _(x, lid, n_out):
            return x.new_empty(*x.shape[:-1], n_out)

        def symbolic(g, x, lid, n_out):
            n = symbolic_helper._maybe_get_const(n_out, "i")
            scalar = x.type().scalarType()
            node = g.op("irodori::QMM", x, lid_i=symbolic_helper._maybe_get_const(lid, "i"), n_out_i=n,
                        in_dt_i=SCALAR_TYPE_TO_DT.get(scalar, DT_BFLOAT16))
            node.setType(x.type().with_sizes([None, None, n]))  # dtype を伝えて、混在する型の Cast を出させる
            return node

        torch.onnx.register_custom_op_symbolic("irodori::qmm", symbolic, OPSET)

        class QLinear(torch.nn.Module):
            def __init__(self, lid, n_out):
                super().__init__()
                self.lid, self.n_out = lid, n_out

            def forward(self, x):
                return torch.ops.irodori.qmm(x, self.lid, self.n_out)

        layers, data = [], {}
        with torch.no_grad():
            for name, module in list(wrapper.named_modules()):
                if not isinstance(module, torch.nn.Linear):
                    continue
                if not type(module.weight).__module__.startswith("torchao."):
                    continue
                parent_name, _, child = name.rpartition(".")
                parent = wrapper.get_submodule(parent_name)
                if is_context_projection(name):
                    setattr(parent, child, torch.nn.Linear(1, 1, bias=False))  # K/V キャッシュ側で使い、トレースされない
                    continue
                if module.bias is not None:
                    raise RuntimeError(f"bias のある INT4 Linear は未対応です: {name}")
                data[len(layers)] = extract_linear(module.weight, Int4TilePackedTo4dTensor)
                setattr(parent, child, QLinear(len(layers), module.weight.shape[0]))
                layers.append(name)
        if not layers:
            raise RuntimeError("INT4 の Linear が見つかりません。torchao INT4 のチェックポイントではありません")
        log.info("INT4 linears extracted: %d", len(layers))
        del rt
        torch.cuda.empty_cache()
        wrapper = wrapper.cpu()
        qmm_path = out.with_name(out.stem + ".qmm.onnx")
        log.info("exporting %s", qmm_path)
        with torch.no_grad():
            fb._CAPTURED_SDPA = old_sdpa
            F.scaled_dot_product_attention = fb.primitive_sdpa
            torch.onnx.export(wrapper, tuple(xs_cpu), str(qmm_path), input_names=fb.NAMES, output_names=["v"],
                              opset_version=OPSET, dynamo=False, dynamic_axes=fb.dynamic_axes(),
                              do_constant_folding=True, custom_opsets={"irodori": 1})
    finally:
        F.scaled_dot_product_attention = old_sdpa
        te.model_module.apply_rotary_emb = old_rope
        fb._CAPTURED_SDPA = None
    model = onnx.load(str(qmm_path))
    model, group_count = rewrite_graph(model, data, layers)
    onnx.save(model, str(out), save_as_external_data=True, all_tensors_to_one_file=True,
              location=out.name + ".data", size_threshold=1024)
    qmm_path.unlink()
    record = fb.inspect_export(out)
    record.update({"onnx": str(out), "sha256": fb.sha256(out), "opset": OPSET, "int4_linears": len(layers),
                   "int4_groups": group_count})
    fb.artifact_path(label, ".export.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    log.info("export complete: %s", json.dumps(record, sort_keys=True))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("capture", "export"))
    parser.add_argument("--label", default="fallback_int4")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    capture() if args.action == "capture" else export(args.label)


if __name__ == "__main__":
    main()
