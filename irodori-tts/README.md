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

Hugging Faceのモデルは初回生成時に取得されます。互換性のないアーキテクチャや設定のモデルは使用できません。TensorRTは `model.safetensors` と対応するplanの組み合わせのみ対応します。


このboxは高速化のため、Duration PredictorとRFサンプリングで条件エンコード結果を共有します。これにより推論時間とGPU/CPU負荷を抑えられますが、キャプション条件はサンプリングへ反映されません。話者の声質はキャプションではなく、事前学習済み話者埋め込みで指定してください。