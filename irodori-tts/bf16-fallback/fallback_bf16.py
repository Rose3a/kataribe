"""Isolated BF16 primitive-attention export/build candidate.

The export monkeypatch is deliberately local to this process.  Eager forward
uses the original SDPA, while ONNX symbolic export emits the legacy primitive
graph with an FP32 softmax and a deliberate BF16 probability boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
LAB = Path(os.environ.get("IRODORI_TRT_LAB_DIR", SCRIPT_DIR.parents[1] / "runtime" / "trt-lab"))
WORK = Path(os.environ.get("IRODORI_TRT_WORK_DIR", LAB))
DEFAULT_LABEL = "fallback_bf16"
NAMES = [
    "x", "t", "text_mask", "speaker_mask", "caption_mask",
    "rope_cos", "rope_sin", *[f"kv_{i}" for i in range(6)],
]
CONTEXT_AXES = ("text", "text", "speaker", "speaker", "caption", "caption")
# Captured reference to the original F.scaled_dot_product_attention used by
# PrimitiveSDPA.forward. export() rebinds this right before swapping in the
# primitive_sdpa wrapper.
_CAPTURED_SDPA = None
log = logging.getLogger("fallback_bf16")


def artifact_path(label: str, suffix: str) -> Path:
    return Path(os.environ.get("IRODORI_TRT_OUTPUT_DIR", SCRIPT_DIR)) / f"{label}{suffix}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _constant(g, value, dtype):
    import torch
    return g.op("Constant", value_t=torch.tensor(value, dtype=dtype))


class PrimitiveSDPA(__import__("torch").autograd.Function):
    """Eager identity wrapper; symbolic is the isolated ONNX experiment."""

    @staticmethod
    def forward(ctx, q, k, v, mask):
        # IMPORTANT: use the function that was captured by export() before
        # F.scaled_dot_product_attention was monkeypatched. Importing F inside
        # forward would recurse through the patched attribute.
        return _CAPTURED_SDPA(q, k, v, attn_mask=mask, is_causal=False)

    @staticmethod
    def symbolic(g, q, k, v, mask):
        import torch

        # q/k/v are BF16 in the source model.  Keep QK BF16, then make the
        # precision boundary explicit before the numerically sensitive path.
        q_bf16 = g.op("Cast", q, to_i=16)
        k_bf16 = g.op("Cast", k, to_i=16)
        v_bf16 = g.op("Cast", v, to_i=16)
        kt = g.op("Transpose", k_bf16, perm_i=[0, 1, 3, 2])
        qk_bf16 = g.op("MatMul", q_bf16, kt)
        qk_f32 = g.op("Cast", qk_bf16, to_i=1)
        scale = _constant(g, 1.0 / (64.0 ** 0.5), torch.float32)
        scores = g.op("Mul", qk_f32, scale)

        # The model supplies a boolean validity mask.  Where preserves the
        # valid-token semantics without baking request lengths into the graph.
        neg_inf = _constant(g, float("-inf"), torch.float32)
        masked = g.op("Where", mask, scores, neg_inf)
        probs_f32 = g.op("Softmax", masked, axis_i=-1)
        probs_bf16 = g.op("Cast", probs_f32, to_i=16)
        pv_bf16 = g.op("MatMul", probs_bf16, v_bf16)
        return pv_bf16.setType(q.type())


def primitive_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
                   scale=None, enable_gqa=False):
    if attn_mask is None or is_causal or dropout_p != 0 or scale is not None or enable_gqa:
        raise AssertionError("unexpected SDPA form")
    return PrimitiveSDPA.apply(q, k, v, attn_mask)


def load_wrapper():
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, str(LAB))
    from trt_experiment import CachedDiT, NAMES as LAB_NAMES, model_module, real_rope, runtime
    if list(LAB_NAMES) != NAMES:
        raise RuntimeError(f"unexpected input contract: {LAB_NAMES}")
    rt = runtime()
    if rt.model.delta_cond_module is not None:
        NAMES.append('delta_t')
    wrapper = CachedDiT(rt.model).eval()
    inputs = sorted(WORK.glob("inputs_b*.pt"), key=lambda p: int(p.stem.split("_b")[-1]))
    if not inputs:
        raise FileNotFoundError("Run trt_experiment.py export first to capture inputs")
    xs = torch.load(inputs[-1], weights_only=True)
    if len(xs) != len(NAMES):
        raise RuntimeError(f"inputs_b3.pt has {len(xs)} tensors, expected {len(NAMES)}")
    return torch, F, model_module, real_rope, wrapper, xs


def dynamic_axes():
    axes = {
        "x": {0: "batch", 1: "latent"}, "t": {0: "batch"},
        "text_mask": {0: "batch", 1: "text"},
        "speaker_mask": {0: "batch", 1: "speaker"},
        "caption_mask": {0: "batch", 1: "caption"},
        "rope_cos": {0: "latent"}, "rope_sin": {0: "latent"},
        "v": {0: "batch", 1: "latent"},
    }
    for i, axis_name in enumerate(CONTEXT_AXES):
        axes[f"kv_{i}"] = {1: "batch", 2: axis_name}
    if 'delta_t' in NAMES:
        axes['delta_t'] = {0: 'batch'}
    return axes


def inspect_export(path: Path) -> dict:
    import onnx
    graph = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(graph)
    nodes = list(graph.graph.node)
    counts = {}
    for node in nodes:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    attention = counts.get("Attention", 0)
    softmax = counts.get("Softmax", 0)
    casts = counts.get("Cast", 0)
    matmul = counts.get("MatMul", 0)
    if attention != 0 or softmax != 12:
        raise RuntimeError({"Attention": attention, "Softmax": softmax, "counts": counts})
    if casts < 36 or matmul < 24:
        raise RuntimeError({"Cast": casts, "MatMul": matmul, "counts": counts})
    return {"bytes": path.stat().st_size, "nodes": len(nodes), "counts": counts,
            "attention_nodes": attention, "softmax_nodes": softmax,
            "comment": "QK BF16; scores/softmax FP32; probabilities, PV, and wo remain BF16."}


def export(label: str) -> Path:
    import torch

    out = artifact_path(label, ".onnx")
    manifest_path = artifact_path(label, ".export.json")
    if out.exists():
        if not manifest_path.exists():
            raise FileExistsError(f"refusing unverified existing output: {out}")
        inspect_export(out)
        log.info("verified existing export: %s", out)
        return out

    torch, F, model_module, real_rope, wrapper, xs = load_wrapper()
    old_rope = model_module.apply_rotary_emb
    old_sdpa = F.scaled_dot_product_attention
    model_module.apply_rotary_emb = real_rope
    try:
        expected = wrapper(*xs)
        global _CAPTURED_SDPA
        _CAPTURED_SDPA = old_sdpa
        F.scaled_dot_product_attention = primitive_sdpa
        candidate = wrapper(*xs)
        if not torch.equal(expected, candidate):
            raise RuntimeError("eager primitive wrapper changed output")
        log.info("exporting %s", out)
        torch.onnx.export(wrapper, xs, str(out), input_names=NAMES, output_names=["v"],
                          opset_version=18, dynamo=False, dynamic_axes=dynamic_axes(),
                          do_constant_folding=True)
    finally:
        F.scaled_dot_product_attention = old_sdpa
        model_module.apply_rotary_emb = old_rope
        _CAPTURED_SDPA = None

    record = inspect_export(out)
    record.update({"onnx": str(out), "sha256": sha256(out), "opset": 18})
    manifest_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    log.info("export complete: %s", json.dumps(record, sort_keys=True))
    return out


def build(label: str) -> Path:
    import tensorrt as trt

    onnx_path = artifact_path(label, ".onnx")
    out = artifact_path(label, ".plan")
    manifest_path = artifact_path(label, ".build.json")
    if not onnx_path.exists():
        raise FileNotFoundError(f"run export first: {onnx_path}")
    if out.exists():
        if not manifest_path.exists() or out.stat().st_size == 0:
            raise FileExistsError(f"refusing unverified existing output: {out}")
        log.info("verified existing plan: %s", out)
        return out

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        artifact_path(label, ".parse_failure.txt").write_text(errors, encoding="utf-8")
        raise RuntimeError(errors)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    config.builder_optimization_level = 3
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    # Read-only copy: never mutate the lab's shapes.json.
    shapes = json.loads((WORK / "shapes.json").read_text(encoding="utf-8"))
    profile = builder.create_optimization_profile()
    # Tune kernels for typical requests: 8 s of audio (200 frames) and a
    # compacted sentence (48 tokens).  MeanFlow runs batch 1; RF CFG batch 3.
    # Measured on an RTX 3060: ~5% faster per call at 36-183 frames, ~1%
    # slower at 700 frames, compared with opt=(batch 3, 625 frames, 128 tokens).
    opt_batch = 1 if "delta_t" in shapes else 3
    opt_frames, opt_text = 200, 48
    ranges = {
        "text_mask": (1, opt_text, 256), "speaker_mask": (1, 16, 64),
        "caption_mask": (1, 1, 512), "kv_0": (1, opt_text, 256),
        "kv_1": (1, opt_text, 256), "kv_2": (1, 16, 64), "kv_3": (1, 16, 64),
        "kv_4": (1, 1, 512), "kv_5": (1, 1, 512),
    }
    for i in range(network.num_inputs):
        name = network.get_input(i).name
        opt = list(shapes[name])
        low, high = opt.copy(), opt.copy()
        if name in {"x", "t", "delta_t", "text_mask", "speaker_mask", "caption_mask"}:
            low[0], opt[0], high[0] = 1, opt_batch, 3
        if name.startswith("kv_"):
            low[1], opt[1], high[1] = 1, opt_batch, 3
        if name == "x":
            low[1], opt[1], high[1] = 1, opt_frames, 750
        elif name.startswith("rope_"):
            low[0], opt[0], high[0] = 1, opt_frames, 750
        if name in ranges:
            axis = 1 if name.endswith("mask") else 2
            low[axis], opt[axis], high[axis] = ranges[name]
        profile.set_shape(name, low, opt, high)
        log.info("profile %s low=%s opt=%s high=%s", name, low, opt, high)
    config.add_optimization_profile(profile)
    started = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT returned no engine")
    out.write_bytes(bytes(plan))
    record = {"onnx": str(onnx_path), "plan": str(out), "bytes": out.stat().st_size,
              "sha256": sha256(out), "build_s": time.perf_counter() - started,
              "workspace_bytes": 4 << 30, "optimization_level": 3,
              "profiling_verbosity": "DETAILED",
              "opt_shape": {"batch": opt_batch, "frames": opt_frames, "text": opt_text}}
    manifest_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    log.info("build complete: %s", json.dumps(record, sort_keys=True))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("export", "build"))
    parser.add_argument("--label", default=DEFAULT_LABEL,
                        help="output label; use a new label instead of overwriting")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    (export if args.action == "export" else build)(args.label)


if __name__ == "__main__":
    main()
