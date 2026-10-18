"""W8A8 モデル（torchao int8_dynamic_activation_int8_weight）の TensorRT plan を作る。

bf16-fallback/int8_builder.py で INT8 の Q/DQ を入れた ONNX にしてから、同じ build（fallback_bf16.py
build）で plan にする。活性のスケールはチェックポイントの irodori_quantization_json の
``trt_activation_amax`` を使うので、それが無い INT8 モデル（本家の量子化コマンドで作ったもの）は
TensorRT では使えない（is_w8a8_checkpoint が False）。
キャッシュの置き場所・作り方は trt_int4.py と同じで、identity に ``variant: int8`` を足して区別する。
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from trt_cache import BOX, cache_key, digest, env_identity, model_digest
from trt_int4 import checkpoint_quantization

PLAN_NAME = 'fallback_int8.plan'
LABEL = 'fallback_int8'
VARIANT = 'int8'
INT8_QUANTIZATION = 'int8_dynamic_activation_int8_weight'
# W8A8 plan と、INT8 の重みを BF16 に戻した DiT との差（活性の量子化の分）。Blockwise Small で 0.030（BF16 plan は 0.009 前後）。
MAX_RELATIVE_RMSE = 0.1


def _quantization_payload(checkpoint):
    import struct
    try:
        with open(checkpoint, 'rb') as handle:
            header_len = struct.unpack('<Q', handle.read(8))[0]
            if not 0 < header_len <= 64 * 1024 * 1024:
                return {}
            header = json.loads(handle.read(header_len).decode('utf-8', 'replace'))
        return json.loads((header.get('__metadata__') or {}).get('irodori_quantization_json') or '{}')
    except (OSError, ValueError, TypeError, AttributeError, struct.error):
        return {}


def is_w8a8_checkpoint(checkpoint):
    """TensorRT で使える W8A8 か（INT8 で、活性スケールがメタデータにある）。"""
    if checkpoint_quantization(checkpoint) != INT8_QUANTIZATION:
        return False
    return bool(_quantization_payload(checkpoint).get('trt_activation_amax'))


def int8_sources():
    sources = [BOX / 'irodori-tts/bf16-fallback/int8_builder.py', Path(__file__)]
    return {str(p.relative_to(BOX)): digest(p) for p in sources}


def cache_identity(checkpoint):
    return dict(model=model_digest(checkpoint), variant=VARIANT, int8_sources=int8_sources(),
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
    """W8A8 モデルの plan を返す（無ければ変換する）。"""
    progress('TensorRT: モデルと GPU を確認中', 18)
    identity = cache_identity(checkpoint)
    cache = BOX / '.cache/trt' / cache_key(identity)
    if valid_cache(cache, identity):
        log('TensorRT: 対応する INT8 plan を再利用')
        return cache / PLAN_NAME
    cache.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='build-', dir=cache.parent))
    lab = BOX / 'runtime/trt-lab'
    folder = BOX / 'irodori-tts/bf16-fallback'
    env = os.environ.copy()
    env.update(IRODORI_CHECKPOINT=str(checkpoint), IRODORI_TRT_LAB_DIR=str(lab),
               IRODORI_TRT_WORK_DIR=str(work), IRODORI_TRT_OUTPUT_DIR=str(work),
               PYTHONIOENCODING='utf-8', PYTHONUTF8='1', PYTHONUNBUFFERED='1')
    steps = [('変換用の入力を準備', 22, [str(folder / 'int8_builder.py'), 'capture']),
             ('INT8 ONNX 変換', 35, [str(folder / 'int8_builder.py'), 'export', '--label', LABEL]),
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
    record = dict(identity=identity, plan_sha256=digest(plan),
                  verify=json.loads((work / 'verify.json').read_text(encoding='utf-8')))
    log(f"TensorRT: INT8 plan の検証 relative RMSE {record['verify']['relative_rmse']:.4f}")
    plan.replace(cache / PLAN_NAME)
    marker = work / 'ready.json'
    marker.write_text(json.dumps(record, ensure_ascii=False), encoding='utf-8')
    marker.replace(cache / marker.name)
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
    print('TRT INT8 relative RMSE:', relative.item(), flush=True)
    (work / 'verify.json').write_text(json.dumps({'relative_rmse': relative.item()}), encoding='utf-8')
    if relative.item() > MAX_RELATIVE_RMSE:
        raise RuntimeError(f'TensorRT INT8 output differs from PyTorch (relative RMSE > {MAX_RELATIVE_RMSE})')


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] != '--verify':
        raise SystemExit('usage: trt_int8.py --verify WORK_DIR')
    verify(Path(sys.argv[2]))
