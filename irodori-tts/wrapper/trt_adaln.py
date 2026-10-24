"""adaLN を事前計算にした TensorRT plan（入力 t が mod）を作る。BF16 モデルと W8A8 モデルが対象。

bf16-fallback/adaln_builder.py で ONNX にしてから、同じ build（fallback_bf16.py build）で plan にする。
実行時の組み立ては adaln_pre.py（TrtBackend が plan の t の次元から自動で判定する）。
キャッシュの置き場所・作り方は trt_int8.py と同じで、identity に ``variant: adaln`` と元の種類（``base``）を足して
従来の plan と区別する。IRODORI_ADALN=0 で従来の plan に戻す。作れなかったときも従来の plan に戻る。
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
from trt_int8 import is_w8a8_checkpoint

PLAN_NAME = 'fallback_adaln.plan'
LABEL = 'fallback_adaln'
VARIANT = 'adaln'
# BF16 の plan と同じ基準（trt_cache.verify）。W8A8 は活性の量子化の分が乗る（trt_int8.MAX_RELATIVE_RMSE）。
MAX_RELATIVE_RMSE = 0.1
_failed = set()  # 作れなかったモデル（このプロセスの間は再挑戦しない）


def enabled():
    return os.environ.get('IRODORI_ADALN', '1') != '0'


def base_kind(checkpoint):
    """'bf16' / 'int8' / None（adaLN 版を作れない種類。INT4 などの量子化）。"""
    quantization = checkpoint_quantization(checkpoint)
    if quantization is None:
        return 'bf16'
    return 'int8' if is_w8a8_checkpoint(checkpoint) else None


def usable(checkpoint):
    return enabled() and str(checkpoint) not in _failed and base_kind(checkpoint) is not None


def adaln_sources():
    names = ['irodori-tts/bf16-fallback/adaln_builder.py', 'irodori-tts/bf16-fallback/int8_builder.py',
             'irodori-tts/wrapper/adaln_pre.py']
    sources = [BOX / name for name in names] + [Path(__file__)]
    return {str(p.relative_to(BOX)): digest(p) for p in sources}


def cache_identity(checkpoint):
    return dict(model=model_digest(checkpoint), variant=VARIANT, base=base_kind(checkpoint),
                adaln_sources=adaln_sources(), **env_identity())


def valid_cache(folder, identity):
    try:
        record = json.loads((folder / 'ready.json').read_text(encoding='utf-8'))
        plan = folder / PLAN_NAME
        return (record['identity'] == identity and plan.stat().st_size > 0
                and record['plan_sha256'] == digest(plan))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def ensure_plan(checkpoint, progress, log):
    """adaLN 版の plan を返す（無ければ変換する）。失敗は RuntimeError（呼び出し側が従来の plan に戻す）。"""
    progress('TensorRT: モデルと GPU を確認中', 18)
    identity = cache_identity(checkpoint)
    cache = BOX / '.cache/trt' / cache_key(identity)
    if valid_cache(cache, identity):
        log('TensorRT: 対応する adaLN 事前計算の plan を再利用')
        return cache / PLAN_NAME
    cache.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='build-', dir=cache.parent))
    lab = BOX / 'runtime/trt-lab'
    folder = BOX / 'irodori-tts/bf16-fallback'
    env = os.environ.copy()
    env.update(IRODORI_CHECKPOINT=str(checkpoint), IRODORI_TRT_LAB_DIR=str(lab),
               IRODORI_TRT_WORK_DIR=str(work), IRODORI_TRT_OUTPUT_DIR=str(work),
               PYTHONIOENCODING='utf-8', PYTHONUTF8='1', PYTHONUNBUFFERED='1')
    steps = [('変換用の入力を準備', 22, [str(folder / 'adaln_builder.py'), 'capture']),
             ('adaLN 事前計算の ONNX 変換', 35, [str(folder / 'adaln_builder.py'), 'export', '--label', LABEL]),
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
                raise RuntimeError(f'TensorRT {stage}に失敗しました: {reason}\nログ: {logfile}')
    if model_digest(checkpoint) != identity['model']:
        raise RuntimeError('変換中にモデルが変更されました。設定を再適用してください')
    cache.mkdir(exist_ok=True)
    plan = work / f'{LABEL}.plan'
    record = dict(identity=identity, plan_sha256=digest(plan),
                  verify=json.loads((work / 'verify.json').read_text(encoding='utf-8')))
    log(f"TensorRT: adaLN 事前計算の plan の検証 relative RMSE {record['verify']['relative_rmse']:.4f}")
    plan.replace(cache / PLAN_NAME)
    marker = work / 'ready.json'
    marker.write_text(json.dumps(record, ensure_ascii=False), encoding='utf-8')
    marker.replace(cache / marker.name)
    shutil.rmtree(work, ignore_errors=True)
    return cache / PLAN_NAME


def ensure_plan_or_fallback(checkpoint, progress, log, fallback):
    """adaLN 版の plan。作れなければ log を残して fallback(checkpoint, progress, log) の plan にする。"""
    try:
        return ensure_plan(checkpoint, progress, log), True
    except Exception as exc:  # noqa: BLE001 - 従来の plan で動かせる
        _failed.add(str(checkpoint))
        log(f'adaLN 事前計算の plan を作れなかったので従来の plan を使います: {exc}')
        return fallback(checkpoint, progress, log), False


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
    print('TRT adaLN relative RMSE:', relative.item(), flush=True)
    (work / 'verify.json').write_text(json.dumps({'relative_rmse': relative.item()}), encoding='utf-8')
    if relative.item() > MAX_RELATIVE_RMSE:
        raise RuntimeError(f'TensorRT adaLN output differs from PyTorch (relative RMSE > {MAX_RELATIVE_RMSE})')


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] != '--verify':
        raise SystemExit('usage: trt_adaln.py --verify WORK_DIR')
    verify(Path(sys.argv[2]))
