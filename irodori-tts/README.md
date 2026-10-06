# Irodori-TTS backend

このフォルダが box のバックエンド側です。wrapper、`.venv`、`bf16-fallback`、`runtime` をここにまとめ、フロントエンドからは相対パスで参照します（配布レイアウトは `DEPLOYMENT_LAYOUT.md` を参照）。

## 推奨運用

通常は `model.safetensors`（Aratako/Irodori-TTS-v4.1-Small）を使います。エディタのモデル欄には、`models` フォルダ内のローカル `.safetensors` ファイル名、または同じアーキテクチャのHugging FaceリポジトリIDを指定できます。

例:

```text
model.safetensors
phasefield-audio/Irodori-TTS-v4.1-Anime
Aratako/Irodori-TTS-v4-Large
```

Hugging Faceのモデルは初回生成時に取得されます。互換性のないアーキテクチャや設定のモデルは使用できません。TensorRTはBF16のモデルと、INT4量子化版（`Aratako/Irodori-TTS-v4-Large-Quantized/int4-weight-only`）に対応し、モデルごとに専用のplanを初回に作ります。INT4はINT4の重みのままplanにします（`trt_int4.py`、`bf16-fallback/int4_builder.py`）。INT8・FP8の量子化版はTensorRTでは使えず、CUDAで使います。


このboxは高速化のため、Duration PredictorとRFサンプリングで条件エンコード結果を共有します。これにより推論時間とGPU/CPU負荷を抑えられます。話者の声質はキャプションではなく、事前学習済み話者埋め込みで指定してください。