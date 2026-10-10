#!/usr/bin/env bash
# Linux / Google Colab 向けのセットアップ（setup.ps1 の CUDA 経路に相当）。
# 使い方:  bash tools/setup_colab.sh        （リポジトリのルートで）
# 環境は .local/venv に作る。Windows 版と同じ固定バージョンを入れる。
set -euo pipefail

BOX="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$BOX/.local/venv"
PY="$VENV/bin/python"
TOOLS="$BOX/tools"

command -v uv >/dev/null 2>&1 || pip install -q uv

if [ ! -x "$PY" ]; then
  uv python install 3.11
  uv venv --python 3.11 "$VENV"
fi

uv pip install --python "$PY" torch==2.10.0 torchaudio==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$PY" -r "$TOOLS/requirements-app.txt"
# descript-audiotools の protobuf 上限と ONNX の要求がぶつかるので、ONNX 系は --no-deps で入れる。
uv pip install --python "$PY" --upgrade 'protobuf>=4.25.1,<6' 'ml_dtypes>=0.5.4' flatbuffers coloredlogs packaging sympy
uv pip install --python "$PY" --no-deps 'onnx>=1.16,<1.23' 'onnxruntime>=1.24,<2' 'onnxscript>=0.2' 'onnx_ir>=0.1'

# setup.ps1 と同じ固定コミット（git アーカイブなので --no-deps）。
uv pip install --python "$PY" --no-deps \
  'dacvae @ https://github.com/facebookresearch/dacvae/archive/414c20785fc3a28373073ea8ef7a1316eeeaca6e.zip' \
  'silentcipher @ https://github.com/SesameAILabs/silentcipher/archive/d46d7d0893a583d8968ab3a6626e2289faec9152.zip'

if [ "${1:-}" = "--trt" ] || [ "${WITH_TRT:-0}" = "1" ]; then
  uv pip install --python "$PY" -r "$BOX/irodori-tts/requirements-trt.txt"
fi

"$PY" - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name() if torch.cuda.is_available() else "")
EOF
echo "setup done: $PY"
