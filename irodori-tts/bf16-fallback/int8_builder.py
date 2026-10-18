"""W8A8 モデル（torchao の int8_dynamic_activation_int8_weight）の DiT を TensorRT 用の ONNX にする。

torchao の INT8 Linear は活性をトークンごとに動的に量子化するが、TensorRT の INT8 は静的なスケールが
前提。そこで、重みは torchao の整数値（出力チャネルごと・対称）とスケールをそのまま
``DequantizeLinear`` に渡し、活性はチェックポイントのメタデータ（irodori_quantization_json の
``trt_activation_amax``、調整で集めた入力の大きさ）から作った静的な Q/DQ を前に入れる。
``trt_bf16_activation`` に挙がった層（外れ値の大きい mlp.w2）は活性を BF16 のままにする。

手順は、読み込んだモデルの INT8 Linear を BF16 に戻して（q × scale）ふつうの BF16 ONNX を書き出し、
重みの中身（転置した BF16 の並び）で MatMul を突き合わせて Q/DQ に置き換える。

    int8_builder.py capture                      # DiT に渡る入力を保存する
    int8_builder.py export --label fallback_int8 # BF16 ONNX → INT8 Q/DQ への置き換え

plan の構築は fallback_bf16.py の ``build --label fallback_int8``。環境変数は fallback_bf16.py と同じ。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import fallback_bf16 as fb  # noqa: E402

QUANTIZATION_TYPE = "int8_dynamic_activation_int8_weight"
log = logging.getLogger("int8_builder")


# ------------------------------------------------------------------ 純粋な関数（テストしやすいもの）
def bf16_bytes(values: np.ndarray) -> bytes:
    """float32 → bfloat16（最近接偶数丸め）の生バイト列。"""
    bits = np.ascontiguousarray(values.astype(np.float32)).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16).tobytes()


def weight_key(raw: bytes) -> str:
    return hashlib.sha1(raw).hexdigest()


def trt_settings(payload: dict) -> tuple[dict, set]:
    """メタデータから（層 → 活性の amax, 活性を BF16 のままにする層）を返す。"""
    amax = payload.get("trt_activation_amax")
    if payload.get("quantization_type") != QUANTIZATION_TYPE or not isinstance(amax, dict) or not amax:
        raise RuntimeError("TensorRT 用の活性スケール（trt_activation_amax）が無い INT8 モデルです")
    return {str(k): float(v) for k, v in amax.items()}, set(payload.get("trt_bf16_activation") or ())


def rewrite_graph(model, layers: dict) -> tuple[list, list]:
    """BF16 の MatMul（定数の重み）を INT8 の Q/DQ に置き換える。

    layers: {重みの BF16 生バイトの sha1: (層名, q (K,N) int8, scale (N,) float32, amax か None)}
    返り値: (置き換えた層, 見つからなかった層)
    """
    from onnx import TensorProto, helper, numpy_helper

    def bf16_tensor(name, values):
        tensor = TensorProto()
        tensor.name, tensor.data_type = name, TensorProto.BFLOAT16
        tensor.dims.extend(values.shape)
        tensor.raw_data = bf16_bytes(values)
        return tensor

    inits = {i.name: i for i in model.graph.initializer}
    consts = {n.output[0]: n for n in model.graph.node if n.op_type == "Constant"}

    def weight_raw(name):
        tensor = inits.get(name)
        if tensor is None and name in consts:
            tensor = next(a.t for a in consts[name].attribute if a.name == "value")
        if tensor is None or tensor.data_type != TensorProto.BFLOAT16 or len(tensor.dims) != 2:
            return None
        return tensor.raw_data or numpy_helper.to_array(tensor).view(np.uint16).tobytes()

    # 入力を共有する Linear（wq/wk/wv/gate、w1/w3）は1つの MatMul にまとめ、活性の Q/DQ も1回にする。
    # 短いセリフ（行列が小さい）では、別々の INT8 MatMul と Q/DQ の手間が BF16 より重くなるため。
    matched, groups = {}, {}
    for node in model.graph.node:
        raw = weight_raw(node.input[1]) if node.op_type == "MatMul" else None
        entry = layers.get(weight_key(raw)) if raw is not None else None
        if entry is not None:
            matched[id(node)] = entry
            groups.setdefault((node.input[0], entry[3] is None), []).append(node)
    first = {id(members[0]): members for members in groups.values()}

    nodes, new_inits, done = [], [], []
    for node in model.graph.node:
        if id(node) not in matched:
            nodes.append(node)
            continue
        members = first.get(id(node))
        if members is None:  # まとめた先頭の MatMul で出力済み
            continue
        entries = [matched[id(m)] for m in members]
        tag = "q8." + entries[0][0]
        q = np.concatenate([e[1] for e in entries], axis=1)      # (K, N の合計)
        scale = np.concatenate([e[2] for e in entries])
        new_inits += [numpy_helper.from_array(np.ascontiguousarray(q), tag + ".wq"),
                      bf16_tensor(tag + ".ws", scale)]
        nodes.append(helper.make_node("DequantizeLinear", [tag + ".wq", tag + ".ws"], [tag + ".w"],
                                      name=tag + ".wdq", axis=1))
        x = node.input[0]
        if entries[0][3] is not None:
            amax = max(e[3] for e in entries)  # 同じ入力なので調整値も同じ（念のため大きい方）
            new_inits += [bf16_tensor(tag + ".as", np.array(max(amax, 1e-6) / 127.0, dtype=np.float32)),
                          numpy_helper.from_array(np.array(0, dtype=np.int8), tag + ".az")]
            nodes += [helper.make_node("QuantizeLinear", [x, tag + ".as", tag + ".az"], [tag + ".aq"],
                                       name=tag + ".aq"),
                      helper.make_node("DequantizeLinear", [tag + ".aq", tag + ".as", tag + ".az"],
                                       [tag + ".a"], name=tag + ".adq")]
            x = tag + ".a"
        if len(members) == 1:
            nodes.append(helper.make_node("MatMul", [x, tag + ".w"], list(node.output),
                                          name=node.name or tag + ".mm"))
        else:
            sizes = np.array([e[1].shape[1] for e in entries], dtype=np.int64)
            new_inits.append(numpy_helper.from_array(sizes, tag + ".split"))
            nodes += [helper.make_node("MatMul", [x, tag + ".w"], [tag + ".y"], name=tag + ".mm"),
                      helper.make_node("Split", [tag + ".y", tag + ".split"],
                                       [m.output[0] for m in members], name=tag + ".split", axis=-1)]
        done += [e[0] for e in entries]
    used = {i for n in nodes for i in n.input}
    keep = [i for i in model.graph.initializer if i.name in used]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep + new_inits)
    missing = sorted({entry[0] for entry in layers.values()} - set(done))
    return done, missing


# ------------------------------------------------------------------ モデル側
def _checkpoint_payload() -> dict:
    import os
    from safetensors import safe_open
    with safe_open(os.environ["IRODORI_CHECKPOINT"], "pt") as handle:
        meta = handle.metadata() or {}
    return json.loads(meta.get("irodori_quantization_json") or "{}")


def dequantize_blocks(model, amax: dict, bf16_activation: set) -> dict:
    """DiT の INT8 Linear を BF16（q × scale）に戻し、ONNX の突き合わせ用の表を返す。"""
    import torch
    layers = {}
    for name, module in model.blocks.named_modules():
        weight = getattr(module, "weight", None)
        if not isinstance(module, torch.nn.Linear) or not hasattr(weight, "qdata"):
            continue
        key = f"blocks.{name}"
        q = weight.qdata.detach()
        scale = weight.scale.detach().reshape(-1)
        dense = (q.float() * scale.float()[:, None]).to(torch.bfloat16)
        module.weight = torch.nn.Parameter(dense, requires_grad=False)
        raw = dense.T.contiguous().view(torch.uint16).cpu().numpy().tobytes()
        if key not in amax:
            raise RuntimeError(f"{key} の活性スケールがありません")
        layers[weight_key(raw)] = (key, q.T.contiguous().cpu().numpy(),
                                   scale.float().cpu().numpy(),
                                   None if key in bf16_activation else amax[key])
    leftover = [n for n, m in model.named_modules()
                if isinstance(m, torch.nn.Linear) and hasattr(m.weight, "qdata")]
    if leftover:
        # DiT の外（テキスト・話者のエンコーダなど）の INT8 は plan に入らないが、想定外なので止める
        raise RuntimeError(f"DiT の外に INT8 の Linear があります: {leftover[:4]}")
    if not layers:
        raise RuntimeError("INT8 の Linear が見つかりません")
    return layers


def _load_runtime():
    import torch
    torch.zeros(1, device="cuda")
    sys.path.insert(0, str(fb.LAB))
    import trt_experiment as te
    with torch.inference_mode(False):
        rt = te.runtime()
    amax, bf16_activation = trt_settings(_checkpoint_payload())
    layers = dequantize_blocks(rt.model, amax, bf16_activation)
    te.runtime = lambda: rt  # fallback_bf16.load_wrapper もこれを使う
    return te, rt, layers


def capture() -> None:
    te, _, _ = _load_runtime()
    te.export(capture_only=True)


def export(label: str) -> Path:
    import onnx
    _, _, layers = _load_runtime()
    source = fb.export(label + "_bf16")
    model = onnx.load(str(source))
    done, missing = rewrite_graph(model, layers)
    if missing:
        raise RuntimeError(f"ONNX で見つからなかった INT8 層 {len(missing)} 個: {missing[:4]}")
    out = fb.artifact_path(label, ".onnx")
    onnx.save(model, str(out))
    record = {"onnx": str(out), "int8_layers": len(done),
              "bf16_activation": sum(1 for v in layers.values() if v[3] is None)}
    fb.artifact_path(label, ".export.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    source.unlink(missing_ok=True)
    log.info("INT8 export complete: %s", json.dumps(record))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("capture", "export"))
    parser.add_argument("--label", default="fallback_int8")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    capture() if args.action == "capture" else export(args.label)


if __name__ == "__main__":
    main()
