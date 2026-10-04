import { ref } from "vue";

export const IRODORI_DEFAULT_SEED = 4763674;
export const IRODORI_DEFAULT_STEPS = 8;
/** MeanFlow（蒸留）モデルの既定ステップ数。 */
export const IRODORI_MEANFLOW_DEFAULT_STEPS = 4;
export const IRODORI_DEFAULT_SCHEDULE = "sway" as const;
/** 英単語・英文の読み。off なら変換せず、katakana / hiragana はその表記にする。 */
export type IrodoriEnglishReading = "off" | "katakana" | "hiragana";
/** 合成前のカナ表記。hiragana ならカタカナ語もひらがなにして読ませる。 */
export type IrodoriKanaStyle = "katakana" | "hiragana";
/** 変換した英語の前後の空白。keep は語ごとに区切り、join は詰めてつなげて読む。 */
export type IrodoriEnglishSpacing = "keep" | "join";
/** 語彙分割辞書。on なら学習の少ないまとまりトークン（浦和レッズ など）を分けて読ませる。 */
export type IrodoriTokenSplit = "on" | "off";
export const IRODORI_DEFAULT_ENGLISH_READING: IrodoriEnglishReading =
  "katakana";
export const IRODORI_DEFAULT_KANA_STYLE: IrodoriKanaStyle = "katakana";
export const IRODORI_DEFAULT_ENGLISH_SPACING: IrodoriEnglishSpacing = "keep";
export const IRODORI_DEFAULT_TOKEN_SPLIT: IrodoriTokenSplit = "on";
export const IRODORI_DEFAULT_CFG_TEXT = 3;
export const IRODORI_DEFAULT_CFG_CAPTION = 3;
export const IRODORI_DEFAULT_CFG_SPEAKER = 5;
export const IRODORI_DEFAULT_CAPTION_STRENGTH = 1;
export const IRODORI_DEFAULT_REFERENCE_STRENGTH = 1;
export const IRODORI_DEFAULT_SPEAKER_STRENGTH = 1;

/**
 * 選択中モデルに応じた既定ステップ数。
 *
 * エンジンが返す modelInfo.defaultSteps（MeanFlow なら4、RF なら8）を
 * IrodoriSettings.vue が反映し、生成側もここを参照する。
 */
export const irodoriDefaultSteps = ref(IRODORI_DEFAULT_STEPS);

/**
 * 選択中モデルが MeanFlow か。MeanFlow では Schedule と CFG が使われないため、
 * 行設定の画面でそれらの入力を無効にする。
 */
export const irodoriMeanflow = ref(false);

/**
 * ここから下はエンジン（ローカルの HTTP サーバ）とのやり取りの型。
 *
 * `/irodori/*` の契約はこのファイルだけに書き、呼び出し側
 * （helpers/irodoriEngine.ts と各コンポーネント）はこれを共有する。
 */

/** GET /irodori/session の応答。 */
export type IrodoriSession = {
  token: string;
};

/** 口パク用ASRの、1文字ぶんの発話時刻（秒）。 */
export type AsrAnchor = {
  char: string;
  start: number;
  end: number;
};

/** POST /irodori/timeline の応答。 */
export type AsrTimelineResponse = {
  available: boolean;
  text?: string;
  asrText?: string;
  anchors?: AsrAnchor[];
  resolution?: number;
  audioSeconds?: number;
  asrSeconds?: number;
  /** available が false のときの理由。 */
  reason?: string;
  /** モデル取得中なら true。エンジン側では取得が続いている。 */
  downloading?: boolean;
  modelFolder?: string;
};

/** /irodori/settings に送る設定。 */
export type IrodoriSettings = {
  backend: string;
  model: string;
  seed: number;
  sway_coeff: number;
};

/** 選択中モデルの情報。 */
export type IrodoriModelInfo = {
  source: string;
  kind: "hf" | "local";
  resolved: string | null;
  downloaded: boolean;
  flowParameterization: string;
  meanflow: boolean;
  defaultSteps: number;
  metadataAvailable: boolean;
  license?: string;
  licenseUrl?: string;
};

/** エンジンが持つ進捗（生成だけでなくモデル取得でも動く）。 */
export type IrodoriProgress = {
  active: boolean;
  percent: number;
  stage: string;
};

/** GET /irodori/settings の応答。 */
export type IrodoriStatus = {
  settings: IrodoriSettings;
  loaded: boolean;
  modelFolder: string;
  speakerFolder: string;
  availableBackends: Record<string, boolean>;
  progress: IrodoriProgress;
  modelInfo?: IrodoriModelInfo;
};

/** ストレージ一覧の1項目（モデル・TensorRT キャッシュなど）。 */
export type IrodoriStorageEntry = {
  id: string;
  kind:
    | "hf"
    | "local"
    | "asr"
    | "trt"
    | "trt-build"
    | "codec"
    | "codec-build"
    | "legacy";
  path: string;
  label: string;
  detail: string;
  /** in_use: 使用中 / unused: 未使用だが有効 / stale: もう使われない / required: 必須 */
  status: "in_use" | "unused" | "stale" | "required";
  deletable: boolean;
  note: string;
  bytes: number;
  modified: number | null;
};

/** POST /irodori/tokenize の1トークン。split はトークンではなく見えない区切りの位置。 */
export type IrodoriToken =
  | {
      text: string;
      id: number;
      /** 出現度（Unigram の logp）。低いほど学習で見ていない。 */
      score: number;
      /** 出現度の低い側の複数文字トークン（読めないことが多い）。 */
      rare: boolean;
      /** 語彙分割辞書に登録がある（合成時に自動で分ける）。 */
      dictionary: boolean;
    }
  | { split: true };

/** POST /irodori/tokenize の応答。 */
export type IrodoriTokenView = {
  available: boolean;
  /** model: 読み込み中のモデル / fallback: 未読み込みのため既定のトークナイザ */
  source: "model" | "fallback";
  results: { text: string; tokens: IrodoriToken[] }[];
};

/**
 * 語彙分割辞書（読めない語句の対策）の1件。読み方＆アクセント辞書とは別に持つ。
 * text の | は見えない区切り、[ZW] はゼロ幅スペース。
 */
export type TokenSplitEntry = {
  surface: string;
  text: string;
  note?: string;
  /** 自動の登録だけ: 選ばれた書き換え方と、書き換え前後の OK率 */
  method?: string;
  ok_before?: number;
  ok_after?: number;
};

/** GET /irodori/storage と POST /irodori/storage/delete の応答。 */
export type IrodoriStorageStatus = {
  entries: IrodoriStorageEntry[];
  totalBytes: number;
  /** TensorRT plan とモデルの照合（ハッシュ計算）を裏で実行中。 */
  identifying: boolean;
  scannedAt: number;
  freedBytes?: number;
};
