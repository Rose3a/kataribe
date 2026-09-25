"""Benchmark the resident synthesis path (VoicevoxAdapter.synthesize, no HTTP).

Measures the same path the editor uses: reading dictionary, speaker cassette,
backend, WAV bytes.  Prints per-stage medians and writes the WAVs plus a JSON
summary so two runs (for example before/after an optimisation) can be compared
with ``--compare``.

    .local\\venv\\Scripts\\python.exe tools\\bench_trt_pipeline.py --out work\\bench\\after
    .local\\venv\\Scripts\\python.exe tools\\bench_trt_pipeline.py --compare work\\bench\\before work\\bench\\after
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

BOX = Path(__file__).resolve().parents[1]
WRAPPER = BOX / "irodori-tts" / "wrapper"
TEXTS = {
    "short": "おはようございます。",
    "medium": "こんにちは、私はAIです。これは音声合成のテストです。聞こえますか？",
    "long": ("今日はとても良い天気ですね。朝から散歩に出かけて、近所の公園でコーヒーを"
             "飲みました。帰り道に本屋へ寄って、前から気になっていた小説を買いました。"
             "夜はゆっくり読書をして過ごそうと思います。"),
}
STAGE = re.compile(r"\[runtime\] ([a-z_]+)(?: \([a-z]+\))?: ([0-9.]+) (ms|s)$")


def default_checkpoint() -> Path:
    snapshots = (BOX / ".cache" / "huggingface" / "hub"
                 / "models--Aratako--Irodori-TTS-v4.1-Small-MF" / "snapshots")
    found = sorted(snapshots.glob("*/model.safetensors"))
    if not found:
        raise SystemExit("MF checkpoint not found; pass --checkpoint")
    return found[-1]


def read_wav(data: bytes):
    import soundfile as sf
    audio, rate = sf.read(io.BytesIO(data), dtype="float32")
    return audio, rate


def run(args) -> int:
    checkpoint = Path(args.checkpoint) if args.checkpoint else default_checkpoint()
    os.environ["IRODORI_CHECKPOINT"] = str(checkpoint)
    os.environ.setdefault("HF_HOME", str(BOX / ".cache" / "huggingface"))
    sys.path.insert(0, str(WRAPPER))
    from voicevox_engine import VoicevoxAdapter

    plan = None
    if args.backend == "trt":
        from trt_cache import ensure_plan
        plan = (Path(args.plan) if args.plan else
                ensure_plan(checkpoint, lambda *a: None, lambda m: print(m, flush=True)))
        # Same as the editor (EditorEngine._prepare_trt_codec); IRODORI_TRT_CODEC=0 skips it.
        os.environ.pop("IRODORI_TRT_CODEC_PLAN", None)
        if os.environ.get("IRODORI_TRT_CODEC", "1") != "0":
            from trt_codec import ensure_codec_plan
            os.environ["IRODORI_TRT_CODEC_PLAN"] = str(
                ensure_codec_plan(lambda *a: None, lambda m: print(m, flush=True)))
    embed_dirs = [Path(p) for p in args.embed_dir] or [BOX / "speakers" / "tsukuyomi"]
    started = time.perf_counter()
    adapter = VoicevoxAdapter(args.backend, 0, embed_dirs, plan)
    load_s = time.perf_counter() - started
    speaker_id = adapter.name_to_id[args.speaker]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    stages: dict[str, dict[str, list[float]]] = {}
    walls: dict[str, list[float]] = {}

    def collect(bucket):
        def log(message: str, *_):
            match = STAGE.search(str(message))
            if match and bucket is not None:
                value = float(match.group(2)) / (1000.0 if match.group(3) == "ms" else 1.0)
                bucket.setdefault(match.group(1), []).append(value)
        return log

    for label, text in TEXTS.items():
        query = {"irodori_text": text, "irodori_seed": args.seed}
        adapter.progress_callback = collect(None)
        for _ in range(args.warmup):
            adapter.synthesize(query, speaker_id)
        bucket = stages.setdefault(label, {})
        adapter.progress_callback = collect(bucket)
        for i in range(args.repeat):
            t0 = time.perf_counter()
            wav = adapter.synthesize(query, speaker_id)
            walls.setdefault(label, []).append(time.perf_counter() - t0)
            if i == 0:
                (out / f"{label}.wav").write_bytes(wav)
    backend = adapter.tts.backend
    summary = {"checkpoint": str(checkpoint), "backend": args.backend, "load_s": load_s,
               "repeat": args.repeat, "texts": {},
               "trt_codec": getattr(backend, "codec", None) is not None,
               "cuda_graphs": {name: {"captures": g.captures, "replays": g.replays, "eager": g.eager}
                               for name, g in getattr(backend, "graphs", {}).items()}}
    for label in TEXTS:
        audio, rate = read_wav((out / f"{label}.wav").read_bytes())
        summary["texts"][label] = {
            "audio_s": len(audio) / rate,
            "wall_median_ms": 1000 * statistics.median(walls[label]),
            "wall_min_ms": 1000 * min(walls[label]),
            "stages_median_ms": {k: round(1000 * statistics.median(v), 2)
                                 for k, v in stages[label].items()},
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    close = getattr(adapter.tts, "close", None)
    if close:
        close()
    return 0


def compare(before: Path, after: Path) -> int:
    """Timing deltas plus waveform agreement for the WAVs of two runs."""
    import numpy as np
    a = json.loads((before / "summary.json").read_text(encoding="utf-8"))
    b = json.loads((after / "summary.json").read_text(encoding="utf-8"))
    for label in TEXTS:
        ta, tb = a["texts"][label], b["texts"][label]
        wa, _ = read_wav((before / f"{label}.wav").read_bytes())
        wb, _ = read_wav((after / f"{label}.wav").read_bytes())
        n = min(len(wa), len(wb))
        err = wa[:n] - wb[:n]
        snr = 10 * np.log10(np.sum(wa[:n] ** 2) / max(np.sum(err ** 2), 1e-30))
        identical = (before / f"{label}.wav").read_bytes() == (after / f"{label}.wav").read_bytes()
        print(f"{label:6s} wall {ta['wall_median_ms']:7.1f} -> {tb['wall_median_ms']:7.1f} ms"
              f"  len {len(wa)} / {len(wb)}  snr={snr:.1f} dB  max_abs={np.abs(err).max():.4g}"
              f"  {'IDENTICAL' if identical else ''}")
        for stage in sorted(set(ta["stages_median_ms"]) | set(tb["stages_median_ms"])):
            print(f"         {stage:22s} {ta['stages_median_ms'].get(stage, float('nan')):8.2f}"
                  f" -> {tb['stages_median_ms'].get(stage, float('nan')):8.2f} ms")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--backend", default="trt", choices=("trt", "cuda", "cpu"))
    ap.add_argument("--plan", help="DiT plan to use instead of the cached build (A/B checks)")
    ap.add_argument("--speaker", default="tsukuyomi")
    ap.add_argument("--embed-dir", action="append", default=[])
    ap.add_argument("--seed", type=int, default=4763674)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--out", default=str(BOX / "work" / "bench" / "latest"))
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    args = ap.parse_args()
    if args.compare:
        return compare(Path(args.compare[0]), Path(args.compare[1]))
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
