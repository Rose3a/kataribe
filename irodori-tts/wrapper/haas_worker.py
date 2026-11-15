"""話者学習のデータ作り: Haas-ja-v1.0 で音声を書き起こす（Haas 用の venv で動かす子プロセス）。

アプリの venv には NeMo を入れないので、speaker_train.py がこのファイルを Haas 用の venv
（既定 .local/venv-haas、tools/setup_haas.ps1 で作る）の python で起動する。
flash-linear-attention（Triton が要る）の代わりに haas_kda.py の PyTorch 版を使うので、Windows・CPU でも動く。

    <haas python> haas_worker.py --list in.jsonl --out out.jsonl [--device cuda|cpu]

in.jsonl は 1 行 1 本の {"wav": パス}。out.jsonl に {"wav", "hyp"} を書く。
進捗は標準出力に JSON 1 行ずつ: {"event": "progress", "done": n, "total": m} / {"event": "done"}。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))  # 同梱の haas_asr（Apache-2.0）

import haas_kda  # noqa: E402

LONG_SECONDS = 30.0
SAMPLE_RATE = 16000


def emit(**payload) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default="third-intelligence/Haas-ja-v1.0")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    haas_kda.install()
    import haas_asr

    wavs = [json.loads(l)["wav"] for l in Path(args.list).read_text(encoding="utf-8").splitlines() if l.strip()]
    emit(event="loading", device=args.device)
    model = haas_asr.load_model(args.model, device=args.device)
    out_rows = []
    short = []
    done = 0
    for w in wavs:
        try:
            audio = haas_asr.load_audio(w)
        except Exception as exc:  # noqa: BLE001 - 読めない音声は空の書き起こしにして続ける
            out_rows.append({"wav": w, "hyp": "", "error": str(exc)})
            done += 1
            continue
        if len(audio) / SAMPLE_RATE > LONG_SECONDS:
            out_rows.append({"wav": w, "hyp": haas_asr.transcribe_longform(model, audio, batch_size=args.batch)})
            done += 1
            emit(event="progress", done=done, total=len(wavs))
        else:
            short.append((w, audio))
    for i in range(0, len(short), args.batch):
        part = short[i:i + args.batch]
        outs = model.transcribe([a for _, a in part], verbose=False)
        for (w, _), o in zip(part, outs):
            out_rows.append({"wav": w, "hyp": o.text})
        done += len(part)
        emit(event="progress", done=done, total=len(wavs))
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows), encoding="utf-8")
    emit(event="done", total=len(wavs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
