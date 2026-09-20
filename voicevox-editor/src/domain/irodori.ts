import { ref } from "vue";

export const IRODORI_DEFAULT_SEED = 4763674;
export const IRODORI_DEFAULT_STEPS = 8;
/** MeanFlow（蒸留）モデルの既定ステップ数。 */
export const IRODORI_MEANFLOW_DEFAULT_STEPS = 4;
export const IRODORI_DEFAULT_SCHEDULE = "sway" as const;
export const IRODORI_DEFAULT_CFG_TEXT = 3;
export const IRODORI_DEFAULT_CFG_CAPTION = 3;
export const IRODORI_DEFAULT_CFG_SPEAKER = 5;
export const IRODORI_DEFAULT_CAPTION_STRENGTH = 1;
export const IRODORI_DEFAULT_REFERENCE_STRENGTH = 1;

/**
 * 選択中モデルに応じた既定ステップ数。
 *
 * エンジンが返す modelInfo.defaultSteps（MeanFlow なら4、RF なら8）を
 * IrodoriSettings.vue が反映し、生成側もここを参照する。
 */
export const irodoriDefaultSteps = ref(IRODORI_DEFAULT_STEPS);

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
