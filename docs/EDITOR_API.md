# Editor 経由の Irodori API

この文書は、起動中の kataribe / Irodori Editor エンジンを外部スクリプトから利用し、
Editor と同じ辞書・モデル・バックエンド設定で音声を生成する手順をまとめたものです。

エンジンは **VOICEVOX エンジン互換** です。VOICEVOX に対応したクライアントやライブラリは、
接続先を `http://127.0.0.1:50125` に変えるだけで使えます（VOICEVOX の既定ポートは 50021）。

## 使うエンドポイント

Editor 経由では、`http://127.0.0.1:50125` の API を使います。

| 用途 | メソッド | パス | 備考 |
|---|---:|---|---|
| バージョン | GET | `/version`, `/core_versions` | |
| API 仕様（OpenAPI 3） | GET | `/openapi.json` | Irodori 拡張フィールドの型・範囲・既定値を含む |
| 話者一覧 | GET | `/speakers` | |
| 話者情報 | GET | `/speaker_info?speaker_uuid=...` | |
| 音声クエリ作成 | POST | `/audio_query?text=...&speaker=...` | `text` と `speaker` は必須（VOICEVOX と同じ） |
| 音声合成 | POST | `/synthesis?speaker=...` | `speaker` は必須 |
| ユーザー辞書 | GET/POST/PUT/DELETE | `/user_dict`, `/user_dict_word...` | VOICEVOX 互換 |
| 辞書一括取込 | POST | `/import_user_dict?override=...` | |
| 共通設定・状態 | GET/POST | `/irodori/settings` | Irodori 独自 |

エンジンはループバック (`127.0.0.1`) にだけ bind されます。

### 認証

VOICEVOX エンジンと同じ考え方です。

- **ブラウザ以外のローカルクライアント**（PowerShell、Python、各種 VOICEVOX クライアント）は、
  `Origin` ヘッダーを付けなければトークン不要でそのまま使えます。
- **ブラウザ**から呼ぶ場合は、許可された `Origin`（Editor の画面）と、`GET /irodori/session`
  で得たトークンを `X-Irodori-Session` ヘッダーに付ける必要があります。許可外のサイトからの
  要求は 403 になります。

`tools/verify_api_auth.py` で、この入口の動作を実物のエンジンに対して確かめられます。

## 設定の引き継ぎ

Editor 経由の `/synthesis` では、次のように設定が適用されます。

| 設定 | 保存場所・入力 | 適用範囲 |
|---|---|---|
| `backend` | Editor の「エンジン」設定、`/irodori/settings` | 全リクエスト |
| `model` | Editor の「モデル」設定、`/irodori/settings` | 全リクエスト |
| `seed` | Editor の共通設定 | `irodori_seed` が省略された場合の既定値 |
| `sway_coeff` | Editor の共通設定 | 行クエリへ適用。入力側の同名値より優先 |
| CFG / step / Schedule / 音声長 / caption / 話速など | 音声クエリの `irodori_*` / `speedScale` | その1行だけ |
| ユーザー辞書 | プロジェクトルートの `user_dictionary.json` | `/audio_query` と合成時のテキスト変換 |

現在の backend・model・seed は `GET /irodori/settings` で確認できます。Editor のセリフごとの
設定まで同じにしたい場合は、`/audio_query` の応答へ同じ `irodori_*` 値を追加してから
`/synthesis` に渡します。

## PowerShell の最小例

次の例は、起動中の Editor から話者IDを取得し、ユーザー辞書を通した音声クエリを作り、
Editor の現在設定を使って `outputs\editor-api.wav` を生成します。

```powershell
$Base = "http://127.0.0.1:50125"

# 表示名から話者の style ID を解決する。IDを固定値で保存しないこと。
$speakers = Invoke-RestMethod "$Base/speakers"
$speakerInfo = $speakers | Where-Object { $_.name -eq "つくよみちゃん" } | Select-Object -First 1
$speaker = $speakerInfo.styles[0].id

# audio_query が user_dictionary.json を反映した kana を返す。
$text = "辞書を引き継いだテストです。"
$query = Invoke-RestMethod "$Base/audio_query?text=$([uri]::EscapeDataString($text))&speaker=$speaker" `
  -Method Post

# 省略した項目は Editor 側の既定値・共通設定が使われる。
$query | Add-Member -NotePropertyName irodori_steps -NotePropertyValue 8 -Force
$query | Add-Member -NotePropertyName irodori_schedule -NotePropertyValue "sway" -Force
$query | Add-Member -NotePropertyName irodori_caption -NotePropertyValue "落ち着いた読み方" -Force

New-Item -ItemType Directory -Force outputs | Out-Null
Invoke-WebRequest "$Base/synthesis?speaker=$speaker" `
  -Method Post -ContentType "application/json; charset=utf-8" `
  -Body ([Text.Encoding]::UTF8.GetBytes(($query | ConvertTo-Json -Depth 20))) `
  -OutFile "outputs\editor-api.wav"
```

`speaker` は `/speakers` の `styles[].id` を使います。話者ファイルの追加・更新でIDの
一覧が変わることがあるため、アプリケーション側で固定値を埋め込まず毎回解決してください。

## 音声クエリで指定できる Irodori 拡張値

JSON のキーは snake_case で指定します。`irodoriSteps` のような camelCase や綴りの間違いは
**422 で拒否**され、`msg` に正しい名前の候補が入ります（黙って既定値で合成はしません）。
型・範囲・既定値の一覧は `GET /openapi.json` の `components.schemas.AudioQuery` にもあります。

| キー | 型 | 範囲・既定 | 説明 |
|---|---|---|---|
| `irodori_text` | string | | 元の文章。合成時に辞書を再適用する。省略時は `kana` |
| `irodori_seed` | integer / null | 既定 4763674 | null でランダム |
| `irodori_steps` | integer | 1〜80、既定はモデル依存（RF 8 / MeanFlow 4） | |
| `irodori_schedule` | string | `sway`（既定） / `linear` | MeanFlow では無視 |
| `irodori_seconds` | number / null | 0.1〜60、null で自動 | 音声長 |
| `irodori_caption` | string / null | 2000文字まで | 場面・話し方・感情 |
| `irodori_caption_strength` | number | 0〜1、既定 1 | |
| `irodori_speaker_strength` | number | 0〜1、既定 1 | 基本話者の強さ |
| `irodori_cfg_text` / `irodori_cfg_caption` / `irodori_cfg_speaker` | number | 0〜20、既定 3 / 3 / 5 | MeanFlow では無視 |
| `irodori_additional_speakers` | array | 最大3件 | `[{"style_id": 123, "strength": 0.5}]`。音声参照と同時には使えない |
| `irodori_reference_audio` | object / null | 最大10MB | `{"dataUrl": "data:audio/...;base64,...", "mime": "...", "name": "..."}` |
| `irodori_reference_strength` | number | 0〜1、既定 1 | 参照音声の強さ |
| `irodori_english_reading` | string | `off` / `katakana`（既定） / `hiragana` | 英単語・英文の読み。`off` は英字のまま、ほかはその表記に変換してから合成する |
| `irodori_english_spacing` | string | `keep`（既定） / `join` | 変換した英語の前後の空白。`join` なら詰めてつなげて読む（`アイ ラブ ユー.` → `アイラブユー.`）。`off` のときは使わない |
| `irodori_kana_style` | string | `katakana`（既定） / `hiragana` | `hiragana` なら文中のカタカナをひらがなにして読ませる |
| `speedScale` | number | 0.25〜4.0、既定 1 | 話速（VOICEVOX と同じ名前） |

- `irodori_text` は元の文章として保持してください。`kana` だけを置き換えると、合成時
  に辞書が再適用されず、Editor と同じ結果にならない場合があります。
- `irodori_english_reading`・`irodori_english_spacing`・`irodori_kana_style` はセリフごとの
  設定です。Editor の「セリフの設定」→「読み方」で選んだ値がそのまま送られます。
- `irodori_english_spacing=join` は、変換した語とユーザー辞書の読みの前後の空白を詰めます。
  空白があると Irodori は語ごとに区切って読むためです。改行は詰めません。
- `pitchScale` や `accent_phrases` などの VOICEVOX の韻律フィールドは、互換のため受け付けますが
  使いません（Irodori は文章から直接合成します）。

## ユーザー辞書

Editor の辞書ダイアログで追加した単語は、同じエンジンの API に即時反映されます。
外部APIから登録する場合は、VOICEVOX互換の辞書エンドポイントを使います。

```powershell
$params = @{
  surface = "Irodori"
  pronunciation = "イロドリ"
  accent_type = 0
  priority = 10
}
$encoded = ($params.GetEnumerator() | ForEach-Object {
  "$([uri]::EscapeDataString($_.Key))=$([uri]::EscapeDataString([string]$_.Value))"
}) -join "&"

$wordId = Invoke-RestMethod "$Base/user_dict_word?$encoded" -Method Post
Invoke-RestMethod "$Base/user_dict"
```

辞書の一括取込は、`GET /user_dict` で得たオブジェクトと同じ形式を JSON body にして
`POST /import_user_dict?override=true` へ送ります。`override=false` なら既存項目を優先します。
辞書変更後は音声キャッシュを使い回さず、次の `/audio_query` と `/synthesis` を新しく
実行してください。

## エラー

| ステータス | 意味 | 本文 |
|---:|---|---|
| 400 | 実行できない要求（未知の話者、文章が長すぎる、空の文章など） | `{"detail": "説明"}` |
| 403 | 許可外の `Origin`、またはブラウザからの要求でトークンが無い・古い | `{"detail": "..."}` |
| 422 | 入力エラー（必須パラメータなし、型・範囲違い、未知の `irodori_*` キー） | VOICEVOX（FastAPI）と同じ `{"detail": [{"type", "loc", "msg", "input"}]}` |
| 429 | 同時合成数の上限 | `Retry-After` ヘッダーの秒数だけ待って再試行 |

- 文章は現在最大256文字です。長文は句読点などで分割してください。
- `/irodori/settings` の POST は Editor 共通設定を変更し、`irodori-tts/editor-settings.json`
  に保存します。通常はEditor画面の「設定を適用」を使い、外部APIから変更する場合は
  他の生成処理と競合しないようにしてください。変更は Editor の画面にすぐには反映されません。
- API はローカル利用を前提にしています。ポート50125を外部ネットワークへ公開しないでください。

## コマンドラインから使う場合

`irodori-tts/wrapper/tts_cli.py` はモデルを自分で読み込んで合成する独立した CLI です。
手軽ですが、Editor の辞書・共通設定・読み込み済みのモデルは使いません。Editor と同じ結果が
必要な場合は、この文書の API を使ってください。

## 関連ファイル

- `irodori-tts/wrapper/editor_engine.py` — Editor 経由のサーバー実装
- `irodori-tts/wrapper/voicevox_engine.py` — VOICEVOX互換エンドポイント、認証、入力チェック（`IRODORI_QUERY_FIELDS`）
- `irodori-tts/wrapper/reading_dictionary.py` — `user_dictionary.json` の読み変換
- `irodori-tts/wrapper/english_reading.py` — 英単語・英文のカタカナ読み（発音データは `tools/build_english_dictionary.py` で再生成）
- `irodori-tts/tests/test_voicevox_compat.py` — VOICEVOX 互換 API の契約テスト
- `voicevox-editor/IRODORI_EDITOR.md` — Editor画面とエンジンの設定説明
