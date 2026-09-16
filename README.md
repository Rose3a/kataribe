
# kataribe

Windows 向けの Irodori-TTS 実行環境と、Irodori 対応 VOICEVOX Editor のリポジトリです。
環境の GPU を自動判定し、CUDA / DirectML (Radeon) / CPU の各バックエンドを構成します。

※モデルの重みファイル、話者データ、生成音声、仮想環境などはリポジトリに含まれません（セットアップ時に自動取得、または手動配置）。

## 動作要件

- Windows 10 / 11
- NVIDIA GPU（CUDA）または AMD Radeon GPU（DirectML）※CPUのみでも動作可能

Python 3.11、uv、Node.js、pnpm などのツール類は、初回セットアップ時に `.local` ディレクトリへ自動的にダウンロード・配置されます。

## セットアップと起動

初めて使う場合は、Git を使うなら次のように clone します。Git を使わない場合は、GitHub の「Code」から ZIP をダウンロードして展開してください。

```powershell
git clone https://github.com/Rose3a/kataribe.git
cd kataribe
```

展開または clone したリポジトリのルートで `setup.bat` を一度実行します。GPU の検出と、`.local` 配下への環境構築が行われます。完了後は `open_browser.bat` でブラウザ版を起動します。

```bat
setup.bat
open_browser.bat
```

### TensorRTによる高速化（CUDA環境）

`setup.bat` の完了後、対応する CUDA 環境では `bat\trt_setup.bat` を実行することで、TensorRT による推論の高速化を利用できます。

```bat
bat\trt_setup.bat
```

> **注意:** すべての CUDA 対応 GPU での動作を保証するものではありません。未確認の環境もあります。
>
> **動作確認済み環境:** RTX 3060（MFモデル / step4）では、約20秒の音声を1秒未満で推論できることを確認しています。

通常は `open_browser.bat` を使用してください。ブラウザ版のエディタとエンジンを起動します。

Electron Editor をソースからビルドして起動する場合は `bat\rebuild_and_open_editor.bat` を使用します。

- バックエンドを手動指定する場合: `bat\first_setup.bat -Backend cuda`
- 指定可能なオプションの確認: `bat\first_setup.bat -Help`

※モデルは初回実行時に Hugging Face から自動ダウンロードされます。各モデルの利用規約を確認の上で使用してください。
※話者データを利用する場合は、権利関係に問題のないデータのみを `speakers/` に配置してください。

## ディレクトリ構成

```text
bat/                セットアップ・起動用バッチファイル
irodori-tts/        Irodori API ラッパーおよびテスト
runtime/trt-lab/    Irodori-TTS ランタイム本体
tools/              セットアップ・検証用スクリプト
voicevox-editor/    Irodori 対応 VOICEVOX Editor ソース
models/             モデル配置ディレクトリ（git管理外）
speakers/           話者データ配置ディレクトリ（git管理外）
```

## 倫理的制約・免責事項

本ツールが利用する「Irodori-TTS」モデルには、ライセンス（MIT）に加えて以下の倫理的制約が定められています。本ツールの利用者もこれに従う必要があります。

- **なりすましの禁止**: 本人の明示的な同意を得ていない実在の個人（声優、著名人、公人等）の声をクローンしたり、なりすましを行ったりする目的で使用しないでください。
- **ディープフェイク・誤情報の生成禁止**: 他者を欺く意図を持った音声や、誤情報・偽情報を拡散する目的の音声生成は行わないでください。
- **生成音声の類似性に関する免責**: 参照音声を用いずテキスト/キャプションから生成した場合でも、偶然実在の人物に声が似る可能性があります。これは潜在空間における確率的な結果であり、特定の個人を再現することを意図した学習は行われていません。
- **免責事項**: 本モデルおよび本ツールの利用によって生じたトラブルや損害について、開発者は一切の責任を負いません。各国の法規制を確認の上、利用者自身の責任で利用してください。


## ライセンス・クレジット

コードと標準モデルに適用されるライセンスは [LICENSES.md](LICENSES.md) に一覧で記載しています。
ライセンステキストは [LICENSE](LICENSE) と各同梱ディレクトリの `LICENSE` を確認してください。
