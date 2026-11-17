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
 * 長さの測り直し。on なら、時間予測が長すぎたとき（末尾の余計な発話・ループの原因）に、
 * 読み終えた位置を測って短く作り直す（遅い話速で効く。v4.1 Small 系を PyTorch か TensorRT の adaLN 版の plan で動かすときのみ）。
 */
export type IrodoriLengthCheck = "on" | "off";
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
export const IRODORI_DEFAULT_LENGTH_CHECK: IrodoriLengthCheck = "on";
export const IRODORI_DEFAULT_CFG_TEXT = 3;
export const IRODORI_DEFAULT_CFG_CAPTION = 3;
export const IRODORI_DEFAULT_CFG_SPEAKER = 3;
// CFG をかけるステップの範囲。1 で従来どおり、0 で CFG なし（エンジンが cfg_min_t に換算する）。
// 0.75（cfg_min_t 0.625）でも聞いて問題なく、1 より速いので既定にする。
export const IRODORI_DEFAULT_CFG_RANGE = 0.75;
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
 * 前のセリフから続けて生成するか（設定の blockwise が有効で、対応モデルを PyTorch か TensorRT で動かしているとき）。
 * IrodoriGlobalSettings.vue が反映し、生成側（FETCH_AUDIO・ストリーミング再生）が参照する。
 */
export const irodoriBlockwise = ref(false);

/**
 * 次のセリフも読んで生成するか（設定の blockwise_lookahead。前のセリフから続けて生成が有効なときだけ）。
 * IrodoriGlobalSettings.vue が反映し、生成側（FETCH_AUDIO・ストリーミング再生）が参照する。
 */
export const irodoriBlockwiseLookahead = ref(false);

/** 前のセリフから続けて生成できるエンジン。TensorRT は prefix を caption 枠に足して渡す。DirectML の DiT は prefix の入力を持たない。 */
export const IRODORI_BLOCKWISE_BACKENDS = ["cpu", "cuda", "trt"];

/** エンジンの状態から、前のセリフから続けて生成するか（irodoriBlockwise）を決める。 */
export function irodoriBlockwiseFromStatus(status: IrodoriStatus): boolean {
  return (
    status.settings.blockwise === true &&
    status.modelInfo?.blockwise === true &&
    IRODORI_BLOCKWISE_BACKENDS.includes(status.settings.backend)
  );
}

/** 生成済み音声をメモリに持っておく量の既定（MB）。エンジンの DEFAULT_AUDIO_CACHE_MB と同じ。 */
export const IRODORI_DEFAULT_AUDIO_CACHE_MB = 25;

/** 声づくりで保存した声のフォルダ（エンジンの voice_design.VOICE_FOLDER と同じ）。 */
export const IRODORI_VOICE_DESIGN_FOLDER = "声づくり";

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
  /** 文節の頭（助詞・助動詞のあとに自立語が来る位置）でも分ける。古いエンジンは返さない。 */
  token_split_bunsetsu?: boolean;
  /** 生成の完了を待たずに鳴らし始める（ストリーミング再生）。古いエンジンは返さない。 */
  stream_playback?: boolean;
  /**
   * 前のセリフから続けて生成する（blockwise 対応モデルだけ）。直前のセリフが同じ話者で生成済みなら、
   * その音声の末尾を聞かせてから作る。古いエンジンは返さない。
   */
  blockwise?: boolean;
  /**
   * 次のセリフも読んで生成する（blockwise が有効なときだけ）。次のセリフが同じ話者なら、その文も一緒に
   * 読ませて作り、このセリフの話し終わりで切る。古いエンジンは返さない。
   */
  blockwise_lookahead?: boolean;
  /** 全行共通の既定ステップ数（1〜80）。MeanFlow も含めて既定は8。古いエンジンは返さない。 */
  default_steps?: number;
  /** 生成済み音声をメモリに持っておく量（0〜1024MB、0 で持たない）。古いエンジンは返さない。 */
  audio_cache_mb?: number;
  /** 省VRAM（TensorRT・CUDA。codec の窓デコード・テキストエンコーダや参照音声用の部品を CPU に置く・待機中の VRAM をすぐ返す。TensorRT では CUDA Graph も使わず、30 秒の待機でエンジンも解放する）。古いエンジンは返さない。 */
  low_vram?: boolean;
  /** 他の TTS の演技タグ（[laughing]・<sigh>・(laughs) など）を絵文字にする。既定オン。古いエンジンは返さない。 */
  acting_tags?: boolean;
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
  /** 前のセリフから続けて生成できるモデルか（use_latent_prefix）。古いエンジンは返さない。 */
  blockwise?: boolean;
  license?: string;
  licenseUrl?: string;
};

/** 語彙分割がいま選択中のモデルに効くか（GET /irodori/settings の tokenSplit）。 */
export type IrodoriTokenSplitState = {
  threshold: IrodoriTokenSplitThreshold;
  /** true なら、v4 Small 系（modernbert-ja）のモデルで、境目が none でなければ当てる。 */
  applies: boolean;
  modelTokenizer: string | null;
  /** 文節の頭でも分ける設定か。古いエンジンは返さない。 */
  bunsetsu?: boolean;
  /** 文節の区切りに使う MeCab（fugashi・unidic-lite）が入っているか。古いエンジンは返さない。 */
  bunsetsuAvailable?: boolean;
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
  /** models フォルダ内のローカルモデル（フォルダからの相対パス） */
  models?: string[];
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

/** 話者学習でベースに選べるモデル（v4.1 Small と同じ構造で、手元にあるもの）。 */
export type IrodoriSpeakerTrainBaseModel = {
  path: string;
  label: string;
  source: "local" | "huggingface";
};

/** 学習の設定（数・選択）。エンジンの既定値と同じ形。 */
export type IrodoriSpeakerTrainSettings = {
  steps: number;
  lr: number;
  /** constant = 一定 / cosine = 最後に向けて 0 近くまで下げる。 */
  lrSchedule: "constant" | "cosine";
  warmup: number;
  /** 0 = 自動（GPU 16 / CPU 4）。 */
  batch: number;
  durWeight: number;
  /** 検証に回す音声の割合（0 で検証しない）。 */
  valFrac: number;
  /** 検証の間隔（ステップ）。0 = 自動。 */
  valEvery: number;
  /** 保存する話者: last = 最後 / best = 検証 loss が一番低かったとき。 */
  keep: "last" | "best";
  seed: number;
};

export type IrodoriSpeakerTrainNumberKey = Exclude<
  keyof IrodoriSpeakerTrainSettings,
  "lrSchedule" | "keep"
>;

/** 過去の学習（speakers/学習した話者/<名前>/<名前>.train.json）の一覧の 1 件。 */
export type IrodoriSpeakerTrainRun = {
  name: string;
  created: string | null;
  steps: number | null;
  lr: number | null;
  /** loss の履歴があるか（グラフで比べられるか）。 */
  hasHistory: boolean;
  metrics: Partial<IrodoriSpeakerTrainMetrics>;
};

/** GET /irodori/speaker-train/options の応答。 */
export type IrodoriSpeakerTrainOptions = {
  baseModels: IrodoriSpeakerTrainBaseModel[];
  /** テキストの無い音声を書き起こす Haas-ja が入っているか（bat\haas_setup.bat）。 */
  haasAvailable: boolean;
  cudaAvailable: boolean;
  defaults: IrodoriSpeakerTrainSettings & {
    speed: "learn" | "keep";
    device: "cuda" | "cpu";
  };
  /** 数の設定で選べる範囲 [下限, 上限]。 */
  limits: Record<IrodoriSpeakerTrainNumberKey, [number, number]>;
  /** これより少ない本数では検証に取り分けない。 */
  valMinTotal: number;
  speakerFolder: string;
  /** すでにある学習した話者の名前。 */
  existing: string[];
  runs: IrodoriSpeakerTrainRun[];
};

/** POST /irodori/speaker-train/start の要求。 */
export type IrodoriSpeakerTrainRequest = IrodoriSpeakerTrainSettings & {
  name: string;
  dataDir: string;
  baseModel: string;
  speed: "learn" | "keep";
  device: "auto" | "cuda" | "cpu";
  verifyText: boolean;
  overwrite: boolean;
};

/** 学習中の記録（学習: report_every ステップごとの平均）。 */
export type IrodoriSpeakerTrainPoint = {
  step: number;
  /** RF（速度場）の loss。 */
  loss: number;
  /** 時間予測の loss（Huber、log フレーム）。 */
  dur: number;
  /** 時間予測の誤差（秒）。 */
  dur_mae_sec: number;
  lr: number;
  /** 話者埋め込みの勾配ノルム。 */
  grad: number;
  /** 初期値（ゼロショット）からの変化（‖今 − 初期値‖ ÷ ‖初期値‖）。 */
  drift: number;
  steps_per_sec: number;
  elapsed_sec: number;
};

/** 検証の記録（決まった t・雑音で測った loss）。step 0 は学習前（ゼロショット）。 */
export type IrodoriSpeakerTrainValPoint = {
  step: number;
  loss: number;
  dur_mae_sec: number;
  /** その時点で一番低かった。 */
  best?: boolean;
};

export type IrodoriSpeakerTrainHistory = {
  train: IrodoriSpeakerTrainPoint[];
  val: IrodoriSpeakerTrainValPoint[];
};

/** 学習で実際に使った設定（自動の値を決めた後）。 */
export type IrodoriSpeakerTrainConfig = {
  batch: number;
  grad_ckpt: number;
  dtype: string;
  train_clips: number;
  val_clips: number;
  val_points: number;
  val_every: number;
  report_every: number;
  lr: number;
  lr_schedule: string;
  warmup: number;
  steps: number;
  dur_weight: number;
  params: number;
  keep: string;
  seed: number;
};

export type IrodoriSpeakerTrainMetrics = {
  actual_over_predicted: number;
  pace_vs_base: number;
  train_seconds: number;
  peak_vram_gb?: number;
  steps_done?: number;
  saved_step?: number;
  stopped_at?: number;
  final_train_loss?: number | null;
  drift?: number;
  val_loss_init?: number;
  val_loss_final?: number;
  val_loss_best?: number;
  best_step?: number;
  val_loss_saved?: number;
  val_dur_mae_sec_init?: number;
  val_dur_mae_sec_saved?: number;
};

/** 学習した結果（speaker_train.py の run の戻り値）。 */
export type IrodoriSpeakerTrainResult = {
  name: string;
  speaker_file: string;
  record: string;
  data: {
    files: number;
    unreadable: number;
    too_short: number;
    transcribed: number;
    text_replaced_by_asr: number;
    no_text: number;
    too_long: number;
    used: number;
    used_minutes: number;
    val?: number;
  };
  metrics: IrodoriSpeakerTrainMetrics;
};

/** GET /irodori/speaker-train/record?name= の応答（.train.json の中身）。古い記録には history・settings が無い。 */
export type IrodoriSpeakerTrainRecord = {
  version: number;
  name: string;
  created: string;
  steps: number;
  lr: number;
  speed: "learn" | "keep";
  dur_weight: number;
  batch?: number;
  settings?: {
    steps: number;
    lr: number;
    lr_schedule: "constant" | "cosine";
    warmup: number;
    batch: number;
    dur_weight: number;
    speed: "learn" | "keep";
    val_frac: number;
    val_every: number;
    keep: "last" | "best";
    seed: number;
  };
  metrics: Partial<IrodoriSpeakerTrainMetrics>;
  history?: IrodoriSpeakerTrainHistory;
};

/** GET /irodori/speaker-train の応答。 */
export type IrodoriSpeakerTrainStatus = {
  state: "idle" | "running" | "done" | "error" | "cancelled";
  percent: number;
  stage: string;
  message: string;
  log: string[];
  result: IrodoriSpeakerTrainResult | null;
  error: string | null;
  params: IrodoriSpeakerTrainRequest | null;
  startedAt: number | null;
  finishedAt: number | null;
  detail: {
    step?: number;
    steps?: number;
    eta_sec?: number;
    vram_gb?: number;
  };
  config: IrodoriSpeakerTrainConfig | null;
  history: IrodoriSpeakerTrainHistory;
  /** 「保存して終える」を押した。 */
  stopRequested: boolean;
};
