"""Export the Irodori DACVAE decoder for the DirectML worker."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch

BOX = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOX / "runtime" / "trt-lab" / "repo"))
from dacvae import DACVAE  # noqa: E402


def patch_runtime_watermark(model: torch.nn.Module) -> None:
    """Match DACVAECodec.load: omit watermark and retain audio projection."""
    decoder = getattr(model, "decoder", None)
    if decoder is None or not hasattr(decoder, "wm_model"):
        raise RuntimeError("DACVAE decoder does not contain the expected watermark model")
    decoder.alpha = 0.0

    def watermark_passthrough(x, message=None, _decoder=decoder):
        del message
        return _decoder.wm_model.encoder_block.forward_no_conv(x)

    decoder.watermark = watermark_passthrough


class Decoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, latent):
        audio = self.model.decode(latent.transpose(1, 2).contiguous())
        if audio.ndim != 3 or audio.shape[1] != 1:
            raise RuntimeError(f"DACVAE decoder returned unexpected shape {tuple(audio.shape)}")
        return audio


def _export(wrapper, destination, precision="fp32"):
    dtype = torch.float16 if precision == "fp16" else torch.float32
    dummy = torch.zeros(1, 8, 32, dtype=dtype)
    kwargs = dict(input_names=["latent"], output_names=["audio"], dynamic_axes={
        "latent": {0: "batch", 1: "frames"}, "audio": {0: "batch", 2: "samples"}},
        opset_version=18, do_constant_folding=True)
    try:
        torch.onnx.export(wrapper, dummy, str(destination), dynamo=False, **kwargs)
    except TypeError as exc:
        if "dynamo" not in str(exc):
            raise
        torch.onnx.export(wrapper, dummy, str(destination), **kwargs)


def _validate(model, onnx_path, precision="fp32"):
    import onnx
    import onnxruntime as ort

    graph = onnx.load(str(onnx_path), load_external_data=True)
    onnx.checker.check_model(graph)
    metadata = {p.key: p.value for p in graph.metadata_props}
    expected = {"irodori_codec_version": "2", "irodori_codec_hop_length": str(model.hop_length),
                "irodori_codec_sample_rate": str(model.sample_rate),
                "irodori_codec_precision": precision}
    if any(metadata.get(k) != v for k, v in expected.items()):
        raise RuntimeError(f"ONNX metadata contract mismatch: {metadata}")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    expected_type = "tensor(float16)" if precision == "fp16" else "tensor(float)"
    if (len(session.get_inputs()) != 1 or session.get_inputs()[0].type != expected_type
            or len(session.get_outputs()) != 1 or session.get_outputs()[0].type != expected_type):
        raise RuntimeError(
            f"ONNX dtype contract mismatch for {precision}: "
            f"input={session.get_inputs()[0].type!r}, output={session.get_outputs()[0].type!r}")
    torch.manual_seed(1234)
    with torch.inference_mode():
        for batch, frames in ((1, 8), (1, 16), (1, 27), (1, 64), (2, 16)):
            dtype = torch.float16 if precision == "fp16" else torch.float32
            latent = torch.randn(batch, frames, 32, dtype=dtype)
            reference = Decoder(model)(latent).numpy()
            actual = session.run(["audio"], {"latent": latent.numpy()})[0]
            expected_shape = (batch, 1, frames * int(model.hop_length))
            if actual.shape != expected_shape or reference.shape != expected_shape:
                raise RuntimeError(f"shape mismatch at batch={batch}, frames={frames}: {actual.shape}, {reference.shape} != {expected_shape}")
            if not torch.isfinite(torch.from_numpy(actual)).all() or not torch.isfinite(torch.from_numpy(reference)).all():
                raise RuntimeError(f"non-finite decoder output at batch={batch}, frames={frames}")
            error = float(abs(actual - reference).max())
            if error > (5e-3 if precision == "fp16" else 2e-4):
                raise RuntimeError(f"PyTorch/ORT mismatch at batch={batch}, frames={frames}: max_error={error:.3g}")
            print(f"validated batch={batch} frames={frames} shape={actual.shape} max_error={error:.3g}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=BOX / "work" / "codec_decoder.onnx")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download
    # Pinned revision so an existing ONNX codec can be reproduced later.
    # prepare.py and the runtime use HF_HOME=.cache\huggingface, whose cache is
    # the hub\ subfolder; pointing at the parent would download the codec twice.
    weights = hf_hub_download("Aratako/Semantic-DACVAE-Japanese-32dim", "weights.pth",
                              revision=os.environ.get("IRODORI_CODEC_REVISION",
                                                      "47376ee24834d7a05a48ebabfe3cde29b3c5e214"),
                              cache_dir=str(BOX / ".cache" / "huggingface" / "hub"))
    model = DACVAE.load(weights).eval().to(
        dtype=torch.float16 if args.precision == "fp16" else torch.float32)
    patch_runtime_watermark(model)
    wrapper = Decoder(model).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with torch.inference_mode():
            _export(wrapper, temporary, args.precision)
        import onnx
        graph = onnx.load(str(temporary), load_external_data=True)
        for key, value in {"irodori_codec_version": "2", "irodori_codec_hop_length": str(model.hop_length),
                           "irodori_codec_sample_rate": str(model.sample_rate),
                           "irodori_codec_precision": args.precision}.items():
            entry = graph.metadata_props.add()
            entry.key, entry.value = key, value
        onnx.save(graph, str(temporary))
        _validate(model, temporary, args.precision)
        if args.output.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup = args.output.with_name(f"{args.output.name}.bak-{stamp}")
            shutil.copy2(args.output, backup)
            print(f"backup={backup}")
        os.replace(temporary, args.output)
        print(f"exported={args.output}")
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
