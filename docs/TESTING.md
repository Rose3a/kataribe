# テストと検証

セットアップ済みの環境（`bat\first_setup.bat` 実行後の `.local\venv`）で動かす。

## ユニットテスト

テストは標準の `unittest` で書いてあるので pytest は不要。

```bat
.local\venv\Scripts\python.exe -m unittest discover -s irodori-tts\tests -t irodori-tts\tests -p "test_*.py"
```

pytest を使う場合:

```bat
.local\venv\Scripts\python.exe -m pytest irodori-tts\tests -q
```

`test_voicevox_meanflow.py`、`test_meanflow_condition_reuse.py`、`test_trt_meanflow.py`、
`test_trt_numerical_checks.py`、`test_inference_speedups.py` は torch（TensorRT 系は tensorrt も）を
import するため、依存が入っていない環境ではコレクション時に失敗する。それ以外は素の Python でも動く。
`test_inference_speedups.py` の CUDA Graph と CUDA 上の末尾トリムのテストは、GPU が無いとスキップされる。

## TensorRT 経路の速度と一致の確認（実モデル）

```bat
.local\venv\Scripts\python.exe tools\bench_trt_pipeline.py --out work\bench\after
.local\venv\Scripts\python.exe tools\bench_trt_pipeline.py --compare work\bench\before work\bench\after
```

エディタと同じ `VoicevoxAdapter.synthesize` を短文・中文・長文で繰り返し、段ごとの中央値と WAV を
`--out` に残す。`--compare` は2回分の時間差と WAV の一致（`IDENTICAL` はバイト一致）を出す。
`--plan` で DiT の plan を固定できるので、コード変更だけの効果を同じ plan で比べられる。
`IRODORI_TRT_CODEC=0` / `IRODORI_CUDA_GRAPHS=0` を付けると、それぞれを外した状態で測れる。
codec の TensorRT plan は構築時に FP32 の PyTorch codec と比べ、BF16 の PyTorch codec より
FP32 から離れていたら採用しない（`.cache\trt-codec\<key>\ready.json` に SNR を記録）。

## エンジンの検証

```bat
bat\verify.bat
```

`tools\verify.py` が一時ポートでエンジンを起動し、`/irodori/settings` と話者一覧を確認して
1 行だけ合成し、`outputs\verification.wav` と `logs\verification.json` を残す。

## WebUI と同じ経路の検証（ヘッドレス）

エンジンを別途起動したうえで:

```bat
.local\venv\Scripts\python.exe tools\verify_webui_path.py
```

`/irodori/session` でセッショントークンを取得し、`/audio_query`（読み辞書を通る）と
`/synthesis` をブラウザと同じヘッダで叩いて、WAV と `logs\verify-webui.json` を残す。

Vite 開発サーバー（`bat\serve_browser.bat`）とエンジンを起動して、実ブラウザで描画と
エンジン接続まで見る場合:

```bat
.local\venv\Scripts\python.exe tools\verify_webui_browser.py
```

ヘッドレス Chrome を CDP で操作し、UI の描画・コンソールエラー・失敗リクエスト・
ページ内からの `/version`、`/irodori/session`、`/irodori/settings` を確認して
`logs\webui-verification.json` と `logs\webui-verification.png` を残す。Chrome の場所は
スクリプト内で指定しているので、環境に合わせて変更する。

## 手で確認するとき

```text
1. bat\serve_editor_engine.bat      … エンジン（既定 127.0.0.1:50125）
2. bat\serve_browser.bat            … エディタ（既定 127.0.0.1:5173）
3. ブラウザで http://127.0.0.1:5173/ を開く
```

## セッション（トークン）まわりの検証

エンジンを別途起動したうえで:

```bat
.local\venv\Scripts\python.exe tools\verify_api_auth.py
```

ローカルAPIの入口を確かめて `logs\verify-api-auth.json` を残す。トークン無しの設定取得・保存が
403 になること、許可外オリジンにはトークンを渡さないこと、古いトークンが拒否されることを見る。
Origin を付けないローカルクライアント（一般の VOICEVOX クライアント）がトークン無しで
`/audio_query` を使えること、許可外オリジンからは拒否されることも確かめる。

```bat
.local\venv\Scripts\python.exe tools\verify_token_recovery.py
```

ヘッドレス Chrome でエディタを開き、ページ内の実クライアント（`helpers/irodoriEngine.ts` と
`infrastructures/EngineConnector.ts`）にトークンを取らせてからエンジンを再起動し、
「古いトークンは 403、クライアントは取り直して成功する」ことを確かめて
`logs\verify-token-recovery.json` を残す。確認に使う口はトークンが要るものに限ること。
GET `/speakers` と `/version` は `_authorized()` を通らないので、ここで使うと
古いトークンでも 200 が返り、成功しても何の証拠にもならない。

どのツールも、エンジン側と同じく `Origin` と `X-Irodori-Session` を付けて叩く。
`bat\verify.bat`（`tools\verify.py`）も同じ理由でトークンを取ってから
`/irodori/settings` と `/synthesis` を呼ぶ。

## 口パク用ASR（/irodori/timeline）

初回の呼び出しでエンジンがモデル（約626MB）を `models\asr` へ自動取得する。
手で確認する手順は次のとおり。

```text
1. エンジンを起動して 1 行合成する（WAV がエンジンのキャッシュに入る）
2. POST /irodori/timeline に {"text": "..."} を送る
   - 取得が終わっていれば anchors が返る
   - 取得中なら available=false と downloading=true が返るので、
     少し待って再送する（取得はエンジン側で続いている）
```

`logs\browser-engine.log` に `ASR model download:` の行が出る。モデルを先に置いて
おけば取得は走らず、`IRODORI_ASR_AUTO_DOWNLOAD=0` で自動取得を止められる。

エンジンとエディタのポート、`voicevox-editor\.env` の `VITE_DEFAULT_ENGINE_INFOS` の
`host`、エンジン側の許可オリジン（`irodori-tts\wrapper\voicevox_engine.py` の
`ALLOWED_ORIGINS`）を揃えること。ずれているとエンジンに接続できない。
