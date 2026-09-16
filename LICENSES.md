# ライセンス一覧

このファイルは、リポジトリ内のコードと、標準設定で取得・利用するモデルについて、
どのライセンスが適用されるかを示す一覧です。個別のライセンステキストまたはモデルカードと
内容が異なる場合は、そちらを優先します。

## ソースコード

| 場所 | 対象 | ライセンス | 主な条件 | ライセンステキスト |
| --- | --- | --- | --- | --- |
| ルートの補助スクリプト、`bat/` | セットアップ・起動用スクリプト | MIT | コピーまたは重要な部分を配布するときは、著作権表示と許諾表示を残します。 | `LICENSE` |
| `irodori-tts/` | Irodori API ラッパー、エンジン連携コード、テスト | MIT | コピーまたは重要な部分を配布するときは、著作権表示と許諾表示を残します。 | `LICENSE` |
| `tools/` | セットアップ・検証用スクリプト | MIT | コピーまたは重要な部分を配布するときは、著作権表示と許諾表示を残します。 | `LICENSE` |
| `voicevox-editor/` | 改変を含む VOICEVOX Editor | LGPL-3.0、または VOICEVOX 上流から別途取得するライセンス | LGPL-3.0 を選ぶ場合は、改変部分を含む対応ソースコードの提供など、同ライセンスの条件に従います。別途取得したライセンスを使う場合は、その条件に従います。 | `voicevox-editor/LICENSE`、`voicevox-editor/LGPL_LICENSE`、`voicevox-editor/GPL_LICENSE` |
| `runtime/trt-lab/repo/` | 改変を含む Irodori-TTS ランタイム | MIT | コピーまたは重要な部分を配布するときは、著作権表示と許諾表示を残します。 | `runtime/trt-lab/repo/LICENSE` |

`voicevox-editor/` に含まれる Irodori 向けの追加・変更部分は、MIT には切り替わらず
LGPL-3.0 の対象です。
Editor の JavaScript 依存パッケージと実行環境の依存パッケージは、それぞれのパッケージの
ライセンスが適用されます。詳細はアプリの「ヘルプ → ライセンス情報」と
`voicevox-editor/public/licenses.json`、`dependency-licenses.json`、
`runtime-licenses.json` を確認してください。

## 標準モデル・データ

| 名前 | 用途 | ライセンス | 主な条件・確認先 |
| --- | --- | --- | --- |
| Irodori-TTS-v4.1-Small | 音声生成モデル | MIT License + モデルカードの Ethical Restrictions | 著作権表示と許諾表示を残して再配布します。なりすましや、誤情報を目的とする音声生成は禁止されています。配布ページ: <https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small> |
| Semantic-DACVAE-Japanese-32dim | 音声コーデックモデル | 配布ページの MIT 表記 | 再配布時は著作権表示と許諾表示を残します。配布ページ: <https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim> |
| ModernBERT-ja-310m | 日本語テキストエンコーダー・トークナイザー | MIT License | 再配布時は著作権表示と許諾表示を残します。配布ページ: <https://huggingface.co/sbintuitions/modernbert-ja-310m> |
| Parakeet TDT-CTC 0.6B Japanese | リップシンク用 ASR モデル | CC BY 4.0 | NVIDIA の CC BY 4.0 モデルを csukuangfj が ONNX / int8 に変換した版です。共有時はクレジット、出典、ライセンスへのリンク、変更内容を示します。元モデル: <https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja>、変換版: <https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8> |
| 話者埋め込み・参照音声・肖像画像 | 話者の音声・画像データ | 話者ごとに異なる | 話者ごとの `credit.txt` または配布元が示す利用条件に従います。本人または権利者の許可も必要です。 |

モデル重み、トークナイザー、話者データ、参照音声、画像、生成音声は、上のコードライセンスとは
別の配布物です。モデルや話者を追加・差し替えた場合は、その配布元が示すライセンスと利用条件を
確認してください。

## Irodori-TTS 音声生成モデルの用途上の制約

以下は LGPL-3.0 や MIT License の条件ではなく、標準の Irodori-TTS モデルカードで
示されている利用上の制約です。

- 本人の明示的な同意なしに、実在する個人の声を模倣・なりすましする目的で使用しないでください。
- 人を欺いたり、誤情報を広めたりする目的のディープフェイク音声を生成しないでください。
- 参照音声なしの生成でも、実在の人物の声に偶然似ることがあります。特定の人物の声だと断定して扱わないでください。
- 利用する地域・用途に適用される法令とサービス規約を確認してください。

## 上流プロジェクト

- Irodori-TTS: <https://github.com/Aratako/Irodori-TTS>
- VOICEVOX Editor: <https://github.com/VOICEVOX/voicevox>
- VOICEVOX Engine: <https://github.com/VOICEVOX/voicevox_engine>
