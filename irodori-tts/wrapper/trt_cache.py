"""Model/device-specific TensorRT builds in isolated child processes."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

BOX = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _digest_memo_path():
    return BOX / '.cache/trt/model-digests.json'


def _read_digest_memo():
    try:
        memo = json.loads(_digest_memo_path().read_text(encoding='utf-8'))
        return memo if isinstance(memo, dict) else {}
    except (OSError, ValueError):
        return {}


def model_digest(checkpoint):
    """sha256 of a checkpoint, remembered by path/size/mtime between starts.

    Hashing v4-Large takes tens of seconds, and the memo also lets the storage
    view tell which model a cached plan was built from.
    """
    path = Path(checkpoint).resolve()
    stat = path.stat()
    memo = _read_digest_memo()
    record = memo.get(str(path))
    if (isinstance(record, dict) and record.get('size') == stat.st_size
            and record.get('mtime_ns') == stat.st_mtime_ns and record.get('sha256')):
        return record['sha256']
    value = digest(path)
    memo = _read_digest_memo()
    memo[str(path)] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=value)
    target = _digest_memo_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix('.tmp')
    tmp.write_text(json.dumps(memo, ensure_ascii=False, indent=1), encoding='utf-8')
    tmp.replace(target)
    return value


def known_model_digests():
    """{sha256: checkpoint path} for every checkpoint hashed so far.

    The same weights can sit in both models/ and the HF cache; prefer the HF
    path because it names the repo.
    """
    result = {}
    for path, record in _read_digest_memo().items():
        if not isinstance(record, dict) or not record.get('sha256'):
            continue
        if record['sha256'] not in result or 'models--' in path:
            result[record['sha256']] = path
    return result


def env_identity():
    """Everything in the cache identity except the checkpoint itself."""
    import torch
    import tensorrt as trt
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('この TensorRT 変換には BF16 対応 CUDA GPU が必要です')
    gpu = torch.cuda.get_device_properties(torch.cuda.current_device())
    sources = [BOX / 'irodori-tts/bf16-fallback/fallback_bf16.py',
               BOX / 'runtime/trt-lab/trt_experiment.py',
               BOX / 'runtime/trt-lab/numerical_checks.py',
               BOX / 'runtime/trt-lab/run_engine.py', Path(__file__)]
    sources += sorted((BOX / 'runtime/trt-lab/repo/irodori_tts').rglob('*.py'))
    return dict(gpu=gpu.name,
                capability=[gpu.major, gpu.minor], device=str(getattr(gpu, 'uuid', '')),
                trt=trt.__version__, torch=torch.__version__, cuda=torch.version.cuda,
                sources={str(p.relative_to(BOX)): digest(p) for p in sources})


def cache_identity(checkpoint):
    return dict(model=model_digest(checkpoint), **env_identity())


def cache_key(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def valid_cache(folder, identity):
    try:
        record = json.loads((folder / 'ready.json').read_text(encoding='utf-8'))
        plan = folder / 'fallback_bf16.plan'
        return (record['identity'] == identity and plan.stat().st_size > 0
                and record['plan_sha256'] == digest(plan))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def ensure_plan(checkpoint, progress, log):
    progress('TensorRT: モデルと GPU を確認中', 18)
    identity = cache_identity(checkpoint)
    cache = BOX / '.cache/trt' / cache_key(identity)
    if valid_cache(cache, identity):
        log('TensorRT: 対応する plan を再利用')
        return cache / 'fallback_bf16.plan'
    cache.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='build-', dir=cache.parent))
    lab = BOX / 'runtime/trt-lab'
    builder = BOX / 'irodori-tts/bf16-fallback/fallback_bf16.py'
    env = os.environ.copy()
    env.update(IRODORI_CHECKPOINT=str(checkpoint), IRODORI_TRT_LAB_DIR=str(lab),
               IRODORI_TRT_WORK_DIR=str(work), IRODORI_TRT_OUTPUT_DIR=str(work),
               PYTHONIOENCODING='utf-8', PYTHONUTF8='1', PYTHONUNBUFFERED='1')
    steps = [('変換用の入力を準備', 22, [str(lab / 'trt_experiment.py'), 'capture']),
             ('BF16 ONNX 変換', 35, [str(builder), 'export']),
             ('plan 構築（初回は時間がかかります）', 48, [str(builder), 'build']),
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
    # Failed builds never gain a ready marker. Atomic plan/marker replacement
    # also lets a later attempt repair an incomplete or damaged cache entry.
    cache.mkdir(exist_ok=True)
    plan = work / 'fallback_bf16.plan'
    record = dict(identity=identity, plan_sha256=digest(plan))
    plan.replace(cache / plan.name)
    marker = work / 'ready.json'
    marker.write_text(json.dumps(record, ensure_ascii=False), encoding='utf-8')
    marker.replace(cache / marker.name)
    # The ONNX export stays behind otherwise (about 3 GB for v4-Large).
    shutil.rmtree(work, ignore_errors=True)
    return cache / plan.name


def verify(work):
    import torch
    sys.path.insert(0, str(BOX / 'runtime/trt-lab'))
    from run_engine import Engine
    inputs = sorted(work.glob('inputs_b*.pt'), key=lambda p: int(p.stem.split('_b')[-1]))
    xs = torch.load(inputs[-1], weights_only=True)
    expected = torch.load(work / 'export_expected.pt', weights_only=True).float()
    engine = Engine(work / 'fallback_bf16.plan')
    actual = engine(xs).float()
    torch.cuda.synchronize()
    if (actual.shape != expected.shape or not torch.isfinite(actual).all()
            or not torch.isfinite(expected).all()):
        raise RuntimeError('TensorRT output shape or finiteness check failed')
    relative = (actual - expected).square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-6)
    print('TRT relative RMSE:', relative.item(), flush=True)
    if relative.item() > 0.1:
        raise RuntimeError('TensorRT output differs from PyTorch (relative RMSE > 0.1)')


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] != '--verify':
        raise SystemExit('usage: trt_cache.py --verify WORK_DIR')
    verify(Path(sys.argv[2]))
