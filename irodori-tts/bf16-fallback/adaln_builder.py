"""adaLN を事前計算にした DiT（plan の入力 t が mod: (B, 層, 2, 3, D)）の ONNX を作る。

BF16 モデルと W8A8 モデル（int8_builder.py の INT8 Q/DQ）に対応する。入力名は ``t`` のままにして
（中身が mod）、fallback_bf16.py の build・run_engine.Engine をそのまま使う。

    adaln_builder.py capture                       # 入力を保存し、t を mod に置き換える
    adaln_builder.py export --label fallback_adaln # CachedDiT を差し替えて ONNX にする（W8A8 は Q/DQ も）

plan の構築は fallback_bf16.py の ``build --label fallback_adaln``。環境変数は fallback_bf16.py と同じ。
mod の中身と実行時の扱いは wrapper/adaln_pre.py。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parents[0] / "wrapper"))
import fallback_bf16 as fb  # noqa: E402
import int8_builder as ib  # noqa: E402
import adaln_pre  # noqa: E402

log = logging.getLogger("adaln_builder")


def is_int8() -> bool:
    return ib._checkpoint_payload().get("quantization_type") == ib.QUANTIZATION_TYPE


def _load_runtime():
    """(te, rt, INT8 の突き合わせ表 or None)。ONNX 変換が同じ rt を使うよう te.runtime を差し替える。"""
    if is_int8():
        te, rt, layers = ib._load_runtime()
    else:
        import torch
        torch.zeros(1, device="cuda")
        sys.path.insert(0, str(fb.LAB))
        import trt_experiment as te
        rt = te.runtime()
        te.runtime = lambda: rt
        layers = None
    reason = adaln_pre.supports(rt.model)
    if reason:
        raise RuntimeError(reason)
    return te, rt, layers


def capture() -> None:
    import torch
    te, rt, _ = _load_runtime()
    te.export(capture_only=True)
    work = fb.WORK
    pre = adaln_pre.AdalnPrecompute(rt.model)
    mod = None
    for path in sorted(work.glob("inputs_b*.pt")):
        xs = list(torch.load(path, weights_only=True, map_location="cuda"))
        mod = pre.table(xs[1])  # (B, 層, 2, 3, D)。t は実行時と同じ dtype のまま渡す
        xs[1] = mod
        torch.save(tuple(xs), path)
    shapes = json.loads((work / "shapes.json").read_text())
    shapes["t"] = list(mod.shape)
    (work / "shapes.json").write_text(json.dumps(shapes, indent=2))
    print("mod shape", list(mod.shape), flush=True)


def export(label: str) -> Path:
    import onnx
    import torch
    te, rt, layers = _load_runtime()
    base = te.CachedDiT

    class PreDiT(base):
        def forward(self, x, mod, text_mask, speaker_mask, caption_mask, rope_cos, rope_sin, *kv):
            freqs = torch.stack((rope_cos, rope_sin), dim=-1)
            h = self.in_proj(x)
            for i, block in enumerate(self.blocks):
                cache = tuple(k[i] for k in kv)
                h = block(h, mod[:, i], cache[0], text_mask, cache[2], speaker_mask,
                          cache[4], caption_mask, freqs, context_kv=cache)
            return self.out_proj(self.out_norm(h)).to(x.dtype)

    adaln_pre.patch_blocks(rt.model.blocks)
    te.CachedDiT = PreDiT
    out = fb.artifact_path(label, ".onnx")
    record = {}
    if layers is None:
        source = fb.export(label)
        record["int8_layers"] = 0
    else:
        source = fb.export(label + "_bf16")
        model = onnx.load(str(source))
        done, missing = ib.rewrite_graph(model, layers)
        if missing:
            raise RuntimeError(f"ONNX で見つからなかった INT8 層 {len(missing)} 個: {missing[:4]}")
        onnx.save(model, str(out))
        source.unlink(missing_ok=True)
        record["int8_layers"] = len(done)
        record["bf16_activation"] = sum(1 for v in layers.values() if v[3] is None)
        fb.artifact_path(label, ".export.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    log.info("adaLN export complete: %s", json.dumps(record))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("capture", "export"))
    parser.add_argument("--label", default="fallback_adaln")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    capture() if args.action == "capture" else export(args.label)


if __name__ == "__main__":
    main()
