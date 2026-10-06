"""torchao INT4 モデル（v4-Large の int4-weight-only など）の TensorRT plan を作る。

BF16 モデルの plan 構築は trt_cache.py。INT4 は重みの形式が違うので、bf16-fallback/int4_builder.py で
INT4 の DequantizeLinear を使う ONNX にしてから、同じ build（fallback_bf16.py build）で plan にする。
キャッシュは trt_cache.py と同じ `.cache/trt/<key>/` に置き、ready.json の identity に
``variant: int4`` を足して区別する（trt_cache.py を変えると既存の BF16 plan のキャッシュが無効になる
ので、こちらは別ファイルにしてある）。
"""
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile

import trt_cache
from trt_cache import BOX, cache_key, digest, env_identity, model_digest

PLAN_NAME = 'fallback_int4.plan'
LABEL = 'fallback_int4'
VARIANT = 'int4'
# INT4 plan の出力は torchao の DiT と相対 RMSE 1% 前後で一致する（Large で 0.9%）。BF16 plan の検証
# （trt_cache.verify の 0.1）より厳しくして、変換の取り違いを確実に弾く。
MAX_RELATIVE_RMSE = 0.05
INT4_QUANTIZATION = 'int4_weight_only'


def checkpoint_quantization(checkpoint):
    """safetensors ヘッダの irodori_quantization_json から、量子化の種類を返す（無ければ None）。"""
    try:
        with open(checkpoint, 'rb') as handle:
            raw = handle.read(8)
            if len(raw) != 8:
                return None
            header_len = struct.unpack('<Q', raw)[0]
            if not 0 < header_len <= 64 * 1024 * 1024:
                return None
            header = json.loads(handle.read(header_len).decode('utf-8', 'replace'))
        meta = (header.get('__metadata__') or {}).get('irodori_quantization_json')
        return (json.loads(meta).get('quantization_type') or 'unknown') if meta else None
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def is_int4_checkpoint(checkpoint):
    return checkpoint_quantization(checkpoint) == INT4_QUANTIZATION


def int4_sources():
    """INT4 の変換に使うスクリプトの内容ハッシュ（変わったら plan を作り直す）。"""
    sources = [BOX / 'irodori-tts/bf16-fallback/int4_builder.py', Path(__file__)]
    return {str(p.relative_to(BOX)): digest(p) for p in sources}


def cache_identity(checkpoint):
    return dict(model=model_digest(checkpoint), variant=VARIANT, int4_sources=int4_sources(),
                **env_identity())


def valid_cache(folder, identity):
    try:
        record = json.loads((folder / 'ready.json').read_text(encoding='utf-8'))
        plan = folder / PLAN_NAME
        return (record['identity'] == identity and plan.stat().st_size > 0
                and record['plan_sha256'] == digest(plan))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def ensure_plan(checkpoint, progress, log):
    """INT4 モデルの plan を返す（無ければ変換する）。"""
    progress('TensorRT: モデルと GPU を確認中', 18)
    identity = cache_identity(checkpoint)
    cache = BOX / '.cache/trt' / cache_key(identity)
    if valid_cache(cache, identity):
        log('TensorRT: 対応する INT4 plan を再利用')
        return cache / PLAN_NAME
    cache.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='build-', dir=cache.parent))
    lab = BOX / 'runtime/trt-lab'
    folder = BOX / 'irodori-tts/bf16-fallback'
    env = os.environ.copy()
    env.update(IRODORI_CHECKPOINT=str(checkpoint), IRODORI_TRT_LAB_DIR=str(lab),
               IRODORI_TRT_WORK_DIR=str(work), IRODORI_TRT_OUTPUT_DIR=str(work),
               PYTHONIOENCODING='utf-8', PYTHONUTF8='1', PYTHONUNBUFFERED='1')
    steps = [('変換用の入力を準備', 22, [str(folder / 'int4_builder.py'), 'capture']),
             ('INT4 ONNX 変換', 35, [str(folder / 'int4_builder.py'), 'export', '--label', LABEL]),
             ('plan 構築（初回は時間がかかります）', 48,
              [str(folder / 'fallback_bf16.py'), 'build', '--label', LABEL]),
             ('推論検証', 75, [str(Path(__file__)), '--verify', str(work)])]
    logfile = work / 'build.log'
    log(f'TensorRT build log: {logfile}')
    with logfile.open('w', encoding='utf-8') as output:
        for stage, percent, args in steps:
            progress('TensorRT: ' + stage, percent)
            output.write(stage + '\n')
            output.flush()
            result = subprocess.run([sys.executable, *args], env=env, cwd=BOX,
                                    stdout=output, stderr=subprocess.STDOUT,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            if result.returncode:
                output.flush()
                with logfile.open('rb') as detail:
                    detail.seek(0, os.SEEK_END)
                    detail.seek(max(0, detail.tell() - 12000))
                    tail = detail.read().decode('utf-8', errors='replace')
                log('TensorRT child process failed:\n' + tail)
                lines = [line.strip() for line in tail.splitlines() if line.strip()]
                reason = lines[-1][:1000] if lines else f'exit code {result.returncode}'
                raise RuntimeError(f'TensorRT {stage}に失敗しました: {reason}\nログ: {logfile}。CUDA に切り替えて利用できます')
    if model_digest(checkpoint) != identity['model']:
        raise RuntimeError('変換中にモデルが変更されました。設定を再適用してください')
    cache.mkdir(exist_ok=True)
    plan = work / f'{LABEL}.plan'
    record = dict(identity=identity, plan_sha256=digest(plan))
    plan.replace(cache / PLAN_NAME)
    marker = work / 'ready.json'
    marker.write_text(json.dumps(record, ensure_ascii=False), encoding='utf-8')
    marker.replace(cache / marker.name)
    # ONNX と取り出した INT4 は作業フォルダに残るだけなので消す
    shutil.rmtree(work, ignore_errors=True)
    return cache / PLAN_NAME


def verify(work):
    import torch
    sys.path.insert(0, str(BOX / 'runtime/trt-lab'))
    from run_engine import Engine
    inputs = sorted(work.glob('inputs_b*.pt'), key=lambda p: int(p.stem.split('_b')[-1]))
    xs = torch.load(inputs[-1], weights_only=True)
    expected = torch.load(work / 'export_expected.pt', weights_only=True).float()
    engine = Engine(work / f'{LABEL}.plan')
    actual = engine(xs).float()
    torch.cuda.synchronize()
    if (actual.shape != expected.shape or not torch.isfinite(actual).all()
            or not torch.isfinite(expected).all()):
        raise RuntimeError('TensorRT output shape or finiteness check failed')
    relative = (actual - expected).square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-6)
    print('TRT INT4 relative RMSE:', relative.item(), flush=True)
    if relative.item() > MAX_RELATIVE_RMSE:
        raise RuntimeError(f'TensorRT INT4 output differs from PyTorch (relative RMSE > {MAX_RELATIVE_RMSE})')


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] != '--verify':
        raise SystemExit('usage: trt_int4.py --verify WORK_DIR')
    verify(Path(sys.argv[2]))
