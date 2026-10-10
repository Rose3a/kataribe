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
/** 語彙分割。on なら学習の少ないまとまりトークン（浦和レッズ など）を分けて読ませる。 */
export type IrodoriTokenSplit = "on" | "off";
/**
 * 語彙分割で避けるトークンの出現度の境目（全体の設定）。4文字以上で出現度がこれ以下の
 * トークンを使わない。0 に近い（-10）ほど分ける語が増える。none ならどのセリフにも当てない。
 * 出現度の尺度は modernbert-ja のものなので、v4 Small 系のモデルにだけ当たる。
 */
export type IrodoriTokenSplitThreshold = "-10" | "-11" | "-12" | "-13" | "none";
export const IRODORI_DEFAULT_TOKEN_SPLIT_THRESHOLD: IrodoriTokenSplitThreshold =
  "-13";
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
export const IRODORI_MIN_STEPS = 1;
export const IRODORI_MAX_STEPS = 80;

/**
 * 選択中モデルが MeanFlow か。MeanFlow では Schedule と CFG が使われないため、
 * 行設定の画面でそれらの入力を無効にする。
 */
export const irodoriMeanflow = ref(false);

/**
 * ストリーミング再生が有効か（設定の stream_playback）。
 * IrodoriGlobalSettings.vue が反映し、再生側（PLAY_AUDIO）が参照する。
 */
export const irodoriStreamPlayback = ref(false);

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
  /** 語彙分割の境目。古いエンジンは返さない（-13 として扱う）。 */
  token_split_threshold?: IrodoriTokenSplitThreshold;
  /** 生成の完了を待たずに鳴らし始める（ストリーミング再生）。古いエンジンは返さない。 */
  stream_playback?: boolean;
  /** 全行共通の既定ステップ数（1〜80）。MeanFlow も含めて既定は8。古いエンジンは返さない。 */
  default_steps?: number;
};

/** 選択中モデルの情報。 */
export type IrodoriModelInfo = {
  source: string;
  kind: "hf" | "local";
  resolved: string | null;
  downloaded: boolean;
  flowParameterization: string;
  meanflow: boolean;
  /** 全行共通の既定ステップ数（設定の default_steps）。 */
  defaultSteps: number;
  /** モデル自身の既定（MeanFlow なら4、RF なら8）。古いエンジンは返さない。 */
  modelDefaultSteps?: number;
  metadataAvailable: boolean;
  /** 量子化モデルの種類（int4_weight_only など）。量子化でなければ null。 */
  quantization?: string | null;
  /** モデルが使うテキストのトークナイザ。語彙分割を当てるかの判定に使う。 */
  textTokenizerRepo?: string | null;
  license?: string;
  licenseUrl?: string;
};

/** 語彙分割がいま選択中のモデルに効くか（GET /irodori/settings の tokenSplit）。 */
export type IrodoriTokenSplitState = {
  threshold: IrodoriTokenSplitThreshold;
  /** true なら、v4 Small 系（modernbert-ja）のモデルで、境目が none でなければ当てる。 */
  applies: boolean;
  modelTokenizer: string | null;
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
  tokenSplit?: IrodoriTokenSplitState;
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

/** GET /irodori/storage と POST /irodori/storage/delete の応答。 */
export type IrodoriStorageStatus = {
  entries: IrodoriStorageEntry[];
  totalBytes: number;
  /** TensorRT plan とモデルの照合（ハッシュ計算）を裏で実行中。 */
  identifying: boolean;
  scannedAt: number;
  freedBytes?: number;
};
