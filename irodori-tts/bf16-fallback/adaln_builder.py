"""adaLN を事前計算にした DiT（plan の入力 t が mod: (B, 層, 2, 3, D)）の ONNX を作る。

BF16 モデルと W8A8 モデル（int8_builder.py の INT8 Q/DQ）に対応する。入力名は ``t`` のままにして
（中身が mod）、fallback_bf16.py の build・run_engine.Engine をそのまま使う。

    adaln_builder.py capture                       # 入力を保存し、t を mod に置き換える
    adaln_builder.py export --label fallback_adaln # CachedDiT を差し替えて ONNX にする（W8A8 は Q/DQ も）

plan の構築は fallback_bf16.py の ``build --label fallback_adaln``。環境変数は fallback_bf16.py と同じ。
mod の中身と実行時の扱いは wrapper/adaln_pre.py。

長さの測り直し（wrapper/length_check.py）に対応するモデル（v4.1 Small と同じ構造の RF）では、2 つ目の出力
``pointer``（(S,) int32: cond の 1 行目で、各音声フレームが層 0 のヘッド 2・4 でいちばん強く見ている
テキスト位置）も出す。層 0 の q はその層の注意の計算と同じもの（_apply_rotary_half の出力）を使うので、
行列積は増えない。IRODORI_TRT_POINTER=0 で出さない。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parents[0] / "wrapper"))
import fallback_bf16 as fb  # noqa: E402
import int8_builder as ib  # noqa: E402
import adaln_pre  # noqa: E402
import length_check  # noqa: E402

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


def with_pointer(model_cfg) -> bool:
    return os.environ.get("IRODORI_TRT_POINTER", "1") != "0" and length_check.supported(model_cfg)


def pointer_from(q, k_text, text_mask):
    """length_check.PointerRecorder._hook と同じ計算。q は RoPE 後の層 0 の q、k_text は層 0 のテキストの K。"""
    import torch
    heads = list(length_check.POINTER_HEADS)
    logits = torch.einsum("bshd,bthd->bhst", q[:1, :, heads].float(), k_text[:1, :, heads].float())
    logits = logits.masked_fill(~text_mask[:1].bool()[:, None, None, :], float("-inf"))
    # argmax の結果（int64）の最大を TensorRT で取るため、いったん float にする
    return logits.argmax(-1).float().amax(1)[0].to(torch.int32)


def export(label: str) -> Path:
    import onnx
    import torch
    te, rt, layers = _load_runtime()
    base = te.CachedDiT
    pointer = with_pointer(rt.model_cfg)

    class PreDiT(base):
        def forward(self, x, mod, text_mask, speaker_mask, caption_mask, rope_cos, rope_sin, *kv):
            freqs = torch.stack((rope_cos, rope_sin), dim=-1)
            held = []
            if pointer:
                # 層 0 の注意の中で最初に _apply_rotary_half を通るのが q（次が self の K）
                attention = self.blocks[length_check.POINTER_LAYER].attention
                rotary = type(attention)._apply_rotary_half

                def capture(x_, freqs_):
                    y = rotary(attention, x_, freqs_)
                    held.append(y)
                    return y
                attention._apply_rotary_half = capture
            try:
                h = self.in_proj(x)
                for i, block in enumerate(self.blocks):
                    cache = tuple(k[i] for k in kv)
                    h = block(h, mod[:, i], cache[0], text_mask, cache[2], speaker_mask,
                              cache[4], caption_mask, freqs, context_kv=cache)
            finally:
                if pointer:
                    del attention._apply_rotary_half
            v = self.out_proj(self.out_norm(h)).to(x.dtype)
            if not pointer:
                return v
            return v, pointer_from(held[0], kv[0][length_check.POINTER_LAYER], text_mask)

    adaln_pre.patch_blocks(rt.model.blocks)
    te.CachedDiT = PreDiT
    out = fb.artifact_path(label, ".onnx")
    record = {"pointer": pointer}
    extra = {"pointer": {0: "latent"}} if pointer else None
    if layers is None:
        source = fb.export(label, extra)
        record["int8_layers"] = 0
    else:
        source = fb.export(label + "_bf16", extra)
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
