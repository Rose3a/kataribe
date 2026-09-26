# kataribe

既存のVOICEVOXフォーク本体です。独自Tkinter GUIとは別です。

`release/win-unpacked/kataribe.exe` を起動します。右側のIrodori-TTSパネルでエンジン・モデルを選んで「設定を適用」します。音声長・ステップ数・Schedule・Seed・CFG・キャプション・音声リファレンスはセリフごとに設定でき、vvprojに保存されます。エンジンなどの共通設定はエンジンフォルダのeditor-settings.jsonに保存します。

セリフの設定の「読み方」→「英単語・英文」は、合成前に英語をカナ読みへ置き換えます。「カタカナで読む」（既定）「ひらがなで読む」「変換しない（英字のまま）」から選べます（API → エーピーアイ、server → サーバー、I love you → アイ ラブ ユー）。「英単語の区切り」を「つなげて読む」にすると、変換した語の前後の空白を詰めて区切らずに読ませます（アイラブユー）。発音辞書（CMUdict）と規則で変換するのでモデルは使わず、速度にはほぼ影響しません。ユーザー辞書の登録が優先されるので、読みが気になる語は辞書に登録してください。「文中のカタカナ語」を「ひらがなにする」にすると、英語の読みも含めて文中のカタカナをすべてひらがなにして読ませます。レンタルサーバーのようなカタカナ語で言いよどむときに試してください。いずれもセリフごとに保存され、新しく追加した行は直前の行の設定を引き継ぎます。

選択変更は設定適用時に確定します。CPU/CUDA/TensorRT/Radeonは単一のAPI接続先を共有し、話者と台本を維持します。モデルは初回生成時に読み込み、同じ構成なら再利用します。別の構成を適用すると前のモデルを解放します。非対応GPUや不足ファイルは生成時にエラー表示します。

## フォルダ配置

現在のビルドは、用意済みのエンジンフォルダに接続して使います。exe横の `irodori-engine-path.txt` にエンジンフォルダへの相対パスまたは絶対パスを指定できます。設定ファイルがなければexe横の `irodori-engine` フォルダを使います。

エンジンフォルダの `models` に対応するIrodoriチェックポイント(.safetensors)、`embeddings` に *.speaker.safetensors を置いて、画面の「一覧を更新」を押します。話者なしでも生成できます。話者のサムネイルは同じ名前の `.png` / `.jpg` / `.webp` を隣に置くとVOICEVOXの話者一覧へ表示されます。

Torchで追加した話者埋め込みはTensorRTでもそのまま共用します。話者追加のための変換やplan再構築は不要です。WebエディターのTensorRTは選択モデルを初回にONNX化して専用planを構築し、モデル内容・GPU・ランタイムに対応するキャッシュを再利用します。MeanFlowにも対応し、進捗表示が完了するとモデルが読み込まれます。Radeonは既存のDirectML PythonとONNX codecの追加準備が必要です。

配布フォルダを分ける場合は、フロントエンドの `irodori-engine-path.txt` にバックエンドフォルダを指定します。推奨レイアウトはエンジン側の `DEPLOYMENT_LAYOUT.md` を参照してください。

別PCではエンジンフォルダでsetup_venv.batを実行し、そのPC向けの依存関係とモデルを準備します。Pythonのvenvはコピー移植を保証しません。このビルドは全依存を内包するインストーラーではなく、準備済み環境に接続するElectronアプリです。Electronのwin-unpackedフォルダはexe以外のファイルも一緒に保持してください。

## 開発

`pnpm run electron:build:compile` の後に `node node_modules/electron-builder/cli.js --config build/irodori-builder.json --win dir --publish never` で梱包します。ブートストラップのC#ソースはbuild/IrodoriEngineBootstrap.csです。
