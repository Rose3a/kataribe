import { toBase64 } from "fast-base64";
import type {
  AudioItem,
  AudioStoreState,
  EditorAudioQuery,
  FetchAudioResult,
  IEngineConnectorFactoryActionsMapper,
  IrodoriLineContext,
  SettingStoreState,
} from "./type";
import { convertAudioQueryFromEditorToEngine } from "./proxy";
import { generateTempUniqueId } from "./utility";
import {
  IRODORI_DEFAULT_CFG_CAPTION,
  IRODORI_DEFAULT_CFG_SPEAKER,
  IRODORI_DEFAULT_CFG_RANGE,
  IRODORI_DEFAULT_CFG_TEXT,
  IRODORI_DEFAULT_CAPTION_STRENGTH,
  IRODORI_DEFAULT_REFERENCE_STRENGTH,
  IRODORI_DEFAULT_SPEAKER_STRENGTH,
  IRODORI_DEFAULT_SEED,
  IRODORI_DEFAULT_SCHEDULE,
  IRODORI_DEFAULT_ENGLISH_READING,
  IRODORI_DEFAULT_KANA_STYLE,
  IRODORI_DEFAULT_TOKEN_SPLIT,
  IRODORI_DEFAULT_ENGLISH_SPACING,
  IRODORI_DEFAULT_AUDIO_CACHE_MB,
  irodoriBlockwise,
  irodoriDefaultSteps,
} from "@/domain/irodori";
import type { AudioKey } from "@/type/preload";
import { AudioQueryToJSON } from "@/openapi/models";
import { irodoriRequest } from "@/helpers/irodoriEngine";
import { pcm16ToWav, readIrodoriPcmStream } from "@/helpers/irodoriStream";

/**
 * 生成済み音声のキャッシュ。キーにセリフの文・話者・Irodori の設定が入っているので、
 * 設定を変えたセリフは別のキーになる。合計が上限（設定の audio_cache_mb、既定 25MB）を
 * 超えたら、最後に使ってから一番時間のたったものから捨てる。
 */
const audioBlobCache = new Map<string, Blob>();
let audioBlobCacheBytes = 0;
let audioCacheLimitBytes = IRODORI_DEFAULT_AUDIO_CACHE_MB * 1024 * 1024;
let audioCacheRevision = 0;

/** モデル・辞書など、キーに入らない条件が変わったときに全部捨てる。 */
export function clearAudioCache() {
  audioCacheRevision++;
  audioBlobCache.clear();
  audioBlobCacheBytes = 0;
}

/** キャッシュの上限（MB）。0 なら持たない。 */
export function setAudioCacheLimitMb(megabytes: number) {
  audioCacheLimitBytes = Math.max(0, megabytes) * 1024 * 1024;
  trimAudioCache();
}

function trimAudioCache() {
  for (const [key, blob] of audioBlobCache) {
    if (audioBlobCacheBytes <= audioCacheLimitBytes) break;
    audioBlobCache.delete(key);
    audioBlobCacheBytes -= blob.size;
  }
}

function getCachedAudio(id: string): Blob | undefined {
  const blob = audioBlobCache.get(id);
  if (blob != undefined) {
    // 使った順に並べ直す（Map は入れた順に回る）
    audioBlobCache.delete(id);
    audioBlobCache.set(id, blob);
  }
  return blob;
}

function putCachedAudio(id: string, blob: Blob) {
  const previous = audioBlobCache.get(id);
  if (previous != undefined) {
    audioBlobCache.delete(id);
    audioBlobCacheBytes -= previous.size;
  }
  if (blob.size > audioCacheLimitBytes) return;
  audioBlobCache.set(id, blob);
  audioBlobCacheBytes += blob.size;
  trimAudioCache();
}

/**
 * セリフごとに、最後に使った音声のキャッシュキー。前のセリフから続けて生成する（blockwise）とき、
 * 直前のセリフの音声をキャッシュから引くのに使う。
 */
const lastAudioIds = new Map<AudioKey, string>();

function rememberLineAudio(audioKey: AudioKey | undefined, id: string) {
  if (audioKey != undefined) lastAudioIds.set(audioKey, id);
}

/**
 * 前のセリフから続けて生成するときに聞かせる、直前のセリフの音声。
 *
 * 設定がオン（対応モデルを PyTorch か TensorRT で動かしている）で、直前のセリフが同じ話者で、その音声が
 * キャッシュに残っているときだけ返す。直前のセリフをまだ作っていなければ、ふつうの1行生成になる
 * （前のセリフまでさかのぼって作り直すことはしない）。話者が変わる続きは学習していないので続けない。
 */
export function previousLineContext(
  state: AudioStoreState,
  audioKey: AudioKey,
): IrodoriLineContext | undefined {
  if (!irodoriBlockwise.value) return undefined;
  const index = state.audioKeys.indexOf(audioKey);
  if (index <= 0) return undefined;
  const previousKey = state.audioKeys[index - 1];
  const current = state.audioItems[audioKey];
  const previous = state.audioItems[previousKey];
  if (
    current == undefined ||
    previous == undefined ||
    previous.morphingInfo != undefined ||
    previous.voice.engineId !== current.voice.engineId ||
    previous.voice.styleId !== current.voice.styleId
  )
    return undefined;
  const id = lastAudioIds.get(previousKey);
  if (id == undefined) return undefined;
  const blob = getCachedAudio(id);
  return blob == undefined ? undefined : { id, blob };
}

async function contextAudio(context: IrodoriLineContext | undefined) {
  if (context == undefined) return undefined;
  const bytes = new Uint8Array(await context.blob.arrayBuffer());
  return { dataUrl: `data:audio/wav;base64,${await toBase64(bytes)}` };
}

type Instance = {
  invoke: IEngineConnectorFactoryActionsMapper;
};

/** `synthesis` に渡す引数（通常の生成とストリーミング再生で共有する）。 */
function synthesisParams(
  state: AudioStoreState & SettingStoreState,
  audioItem: AudioItem,
  engineAudioQuery: ReturnType<typeof convertAudioQueryFromEditorToEngine>,
  irodoriContextAudio?: { dataUrl: string },
) {
  const speaker = audioItem.voice.styleId;
  const effectiveSeed =
    audioItem.irodori?.seed === null
      ? null
      : (audioItem.irodori?.seed ?? IRODORI_DEFAULT_SEED);
  return {
    audioQuery: {
      ...engineAudioQuery,
      // The OpenAPI serializer emits this typed field as `irodori_seed`.
      // irodori_seed: effectiveSeed (serialized from the camelCase field)
      irodoriSeed: effectiveSeed,
      // 既定ステップ数はモデル依存（MeanFlow は4、RF は8）。
      irodoriSteps: audioItem.irodori?.steps ?? irodoriDefaultSteps.value,
      irodoriSchedule: audioItem.irodori?.schedule ?? IRODORI_DEFAULT_SCHEDULE,
      irodoriEnglishReading:
        audioItem.irodori?.englishReading ?? IRODORI_DEFAULT_ENGLISH_READING,
      irodoriKanaStyle:
        audioItem.irodori?.kanaStyle ?? IRODORI_DEFAULT_KANA_STYLE,
      irodoriEnglishSpacing:
        audioItem.irodori?.englishSpacing ?? IRODORI_DEFAULT_ENGLISH_SPACING,
      irodoriTokenSplit:
        audioItem.irodori?.tokenSplit ?? IRODORI_DEFAULT_TOKEN_SPLIT,
      irodoriSeconds: audioItem.irodori?.seconds ?? null,
      irodoriCaption: audioItem.irodori?.caption,
      irodoriCaptionStrength:
        audioItem.irodori?.captionStrength ?? IRODORI_DEFAULT_CAPTION_STRENGTH,
      irodoriCfgText: audioItem.irodori?.cfgText ?? IRODORI_DEFAULT_CFG_TEXT,
      irodoriCfgCaption:
        audioItem.irodori?.cfgCaption ?? IRODORI_DEFAULT_CFG_CAPTION,
      irodoriCfgSpeaker:
        audioItem.irodori?.cfgSpeaker ?? IRODORI_DEFAULT_CFG_SPEAKER,
      irodoriCfgRange: audioItem.irodori?.cfgRange ?? IRODORI_DEFAULT_CFG_RANGE,
      irodoriReferenceStrength:
        audioItem.irodori?.referenceStrength ??
        IRODORI_DEFAULT_REFERENCE_STRENGTH,
      irodoriSpeakerStrength:
        audioItem.irodori?.speakerStrength ?? IRODORI_DEFAULT_SPEAKER_STRENGTH,
      irodoriSecondarySpeakerStyleId:
        audioItem.irodori?.secondarySpeakerStyleId ?? undefined,
      irodoriSecondarySpeakerStrength:
        audioItem.irodori?.secondarySpeakerStrength ?? 0.5,
      irodoriAdditionalSpeakers: audioItem.irodori?.additionalSpeakers,
      irodoriReferenceAudio: audioItem.irodori?.referenceAudio,
      irodoriContextAudio,
    },
    speaker,
    enableInterrogativeUpspeak:
      state.experimentalSetting.enableInterrogativeUpspeak,
  };
}

/**
 * エンジンで音声を合成する。音声のキャッシュ機構も備える。
 */
export async function fetchAudioFromAudioItem(
  state: AudioStoreState & SettingStoreState,
  instance: Instance,
  {
    audioItem,
    audioKey,
    context,
  }: {
    audioItem: AudioItem;
    audioKey?: AudioKey;
    context?: IrodoriLineContext;
  },
): Promise<FetchAudioResult> {
  const engineId = audioItem.voice.engineId;
  // モーフィングは前のセリフから続けて生成できない
  if (audioItem.morphingInfo != undefined) context = undefined;

  const [id, audioQuery] = await generateUniqueIdAndQuery(
    state,
    audioItem,
    context?.id,
  );
  if (audioQuery == undefined)
    throw new Error("audioQuery is not defined for audioItem");

  const cached = getCachedAudio(id);
  if (cached != undefined) {
    rememberLineAudio(audioKey, id);
    return { audioQuery, blob: cached };
  }

  const speaker = audioItem.voice.styleId;

  const engineAudioQuery = convertAudioQueryFromEditorToEngine(
    audioQuery,
    state.engineManifests[engineId].defaultSamplingRate,
  );

  let blob: Blob;
  // FIXME: モーフィングが設定で無効化されていてもモーフィングが行われるので気づけるUIを作成する
  if (audioItem.morphingInfo != undefined) {
    if (!isMorphable(state, { audioItem })) throw new NotMorphableError();
    blob = await instance.invoke("synthesisMorphing")({
      audioQuery: engineAudioQuery,
      baseSpeaker: speaker,
      targetSpeaker: audioItem.morphingInfo.targetStyleId,
      morphRate: audioItem.morphingInfo.rate,
    });
  } else {
    blob = await instance.invoke("synthesis")(
      synthesisParams(
        state,
        audioItem,
        engineAudioQuery,
        await contextAudio(context),
      ),
    );
  }
  putCachedAudio(id, blob);
  rememberLineAudio(audioKey, id);
  return { audioQuery, blob };
}

/**
 * 生成しながら音声を受け取る（Irodori エンジンの `/irodori/synthesis_stream`）。
 *
 * キャッシュにあればそれを返す（streamed: false）。ストリーミングできない
 * 条件（モーフィング、古いエンジン）では undefined を返すので、呼び出し側は
 * 通常の生成にフォールバックする。最後まで受け取れたら通常の生成と同じ
 * キャッシュに入れるので、2回目以降の再生は待たない。
 */
export async function streamAudioFromAudioItem(
  state: AudioStoreState & SettingStoreState,
  {
    audioItem,
    audioKey,
    context,
    endpoint,
    signal,
    onPcm,
  }: {
    audioItem: AudioItem;
    audioKey?: AudioKey;
    context?: IrodoriLineContext;
    endpoint: string;
    signal: AbortSignal;
    onPcm: (samples: Int16Array, sampleRate: number) => void | Promise<void>;
  },
): Promise<{ blob: Blob; streamed: boolean } | undefined> {
  const engineId = audioItem.voice.engineId;
  if (audioItem.morphingInfo != undefined) return undefined;
  const [id, audioQuery] = await generateUniqueIdAndQuery(
    state,
    audioItem,
    context?.id,
  );
  if (audioQuery == undefined)
    throw new Error("audioQuery is not defined for audioItem");
  const cached = getCachedAudio(id);
  if (cached != undefined) {
    rememberLineAudio(audioKey, id);
    return { blob: cached, streamed: false };
  }

  const params = synthesisParams(
    state,
    audioItem,
    convertAudioQueryFromEditorToEngine(
      audioQuery,
      state.engineManifests[engineId].defaultSamplingRate,
    ),
    await contextAudio(context),
  );
  const revision = audioCacheRevision;
  const response = await irodoriRequest(
    endpoint,
    `/irodori/synthesis_stream?speaker=${params.speaker}`,
    {
      method: "POST",
      body: AudioQueryToJSON(params.audioQuery),
      timeoutMs: 10 * 60_000,
      signal,
    },
  );
  if (response.status === 404) return undefined;
  if (!response.ok)
    throw new Error(`音声の生成に失敗しました (${response.status})`);

  const parts: Int16Array[] = [];
  let sampleRate = 0;
  await readIrodoriPcmStream(response, {
    onHeader: (header) => {
      sampleRate = header.sampleRate;
    },
    onPcm: async (samples) => {
      parts.push(samples);
      await onPcm(samples, sampleRate);
    },
  });
  const blob = pcm16ToWav(parts, sampleRate);
  if (revision === audioCacheRevision) {
    putCachedAudio(id, blob);
    rememberLineAudio(audioKey, id);
  }
  return { blob, streamed: true };
}

/**
 * セリフの合成条件（合成に送るのと同じ query と話者）。声づくりで「このセリフの声」を
 * 保存するとき、エンジンはこれで直近の合成結果を照合する（voice_design.take_key）。
 */
export async function lineSynthesisRequest(
  state: AudioStoreState & SettingStoreState,
  audioItem: AudioItem,
): Promise<{ speaker: number; query: unknown }> {
  const [, audioQuery] = await generateUniqueIdAndQuery(state, audioItem);
  if (audioQuery == undefined)
    throw new Error("先にこのセリフを生成してください");
  const params = synthesisParams(
    state,
    audioItem,
    convertAudioQueryFromEditorToEngine(
      audioQuery,
      state.engineManifests[audioItem.voice.engineId].defaultSamplingRate,
    ),
  );
  return {
    speaker: params.speaker,
    query: AudioQueryToJSON(params.audioQuery),
  };
}

export async function generateLabFromAudioQuery(
  audioQuery: EditorAudioQuery,
  offset?: number,
) {
  const speedScale = audioQuery.speedScale;

  let labString = "";
  let timestamp = offset ?? 0;

  labString += timestamp.toFixed() + " ";
  timestamp += (audioQuery.prePhonemeLength * 10000000) / speedScale;
  labString += timestamp.toFixed() + " ";
  labString += "pau" + "\n";

  audioQuery.accentPhrases.forEach((accentPhrase) => {
    accentPhrase.moras.forEach((mora) => {
      if (mora.consonantLength != undefined && mora.consonant != undefined) {
        labString += timestamp.toFixed() + " ";
        timestamp += (mora.consonantLength * 10000000) / speedScale;
        labString += timestamp.toFixed() + " ";
        labString += mora.consonant + "\n";
      }
      labString += timestamp.toFixed() + " ";
      timestamp += (mora.vowelLength * 10000000) / speedScale;
      labString += timestamp.toFixed() + " ";
      if (mora.vowel != "N") {
        labString += mora.vowel.toLowerCase() + "\n";
      } else {
        labString += mora.vowel + "\n";
      }
    });
    if (accentPhrase.pauseMora != undefined) {
      labString += timestamp.toFixed() + " ";
      timestamp +=
        (accentPhrase.pauseMora.vowelLength *
          audioQuery.pauseLengthScale *
          10000000) /
        speedScale;
      labString += timestamp.toFixed() + " ";
      labString += accentPhrase.pauseMora.vowel + "\n";
    }
  });

  labString += timestamp.toFixed() + " ";
  timestamp += (audioQuery.postPhonemeLength * 10000000) / speedScale;
  labString += timestamp.toFixed() + " ";
  labString += "pau" + "\n";

  return labString;
}

async function generateUniqueIdAndQuery(
  state: SettingStoreState,
  audioItem: AudioItem,
  /** 前のセリフから続けて生成するときの、直前のセリフの音声のキャッシュキー。 */
  contextId?: string,
): Promise<[string, EditorAudioQuery | undefined]> {
  audioItem = JSON.parse(JSON.stringify(audioItem)) as AudioItem;
  const audioQuery = audioItem.query;
  if (audioQuery != undefined) {
    audioQuery.outputSamplingRate =
      state.engineSettings[audioItem.voice.engineId].outputSamplingRate;
    audioQuery.outputStereo = state.savingSetting.outputStereo;
  }

  const effectiveSeed =
    audioItem.irodori?.seed === null
      ? null
      : (audioItem.irodori?.seed ?? IRODORI_DEFAULT_SEED);
  const cacheNonce = effectiveSeed === null ? crypto.randomUUID() : undefined;
  const id = await generateTempUniqueId([
    audioCacheRevision,
    audioItem.text,
    audioQuery,
    audioItem.voice,
    audioItem.morphingInfo,
    audioItem.irodori,
    cacheNonce,
    state.experimentalSetting.enableInterrogativeUpspeak, // このフラグが違うと、同じAudioQueryで違う音声が生成されるので追加
    // 続けて生成したセリフは、前のセリフの音声が変われば別の音声になる
    ...(contextId != undefined ? [contextId] : []),
  ]);
  return [id, audioQuery];
}

export function isMorphable(
  state: AudioStoreState,
  {
    audioItem,
  }: {
    audioItem: AudioItem;
  },
) {
  if (audioItem.morphingInfo?.targetStyleId == undefined) return false;
  const { engineId, styleId } = audioItem.voice;
  const info =
    state.morphableTargetsInfo[engineId]?.[styleId]?.[
      audioItem.morphingInfo.targetStyleId
    ];
  if (info == undefined) return false;
  return info.isMorphable;
}

class NotMorphableError extends Error {
  constructor() {
    super("モーフィングの設定が無効です。");
  }
}

export function handlePossiblyNotMorphableError(e: unknown) {
  if (e instanceof NotMorphableError) {
    return e.message;
  } else {
    window.backend.logError(e);
    return;
  }
}

/**
 * AudioQueryから音声の長さを計算する(秒)
 */
export function calculateAudioLength(audioQuery: EditorAudioQuery) {
  if (audioQuery.accentPhrases.length === 0) return 0;

  let length = 0;
  length += audioQuery.prePhonemeLength;
  audioQuery.accentPhrases.forEach((accentPhrase) => {
    accentPhrase.moras.forEach((mora) => {
      if (mora.consonantLength != undefined) {
        length += mora.consonantLength;
      }
      length += mora.vowelLength;
    });
    if (accentPhrase.pauseMora != undefined) {
      length +=
        accentPhrase.pauseMora.vowelLength * audioQuery.pauseLengthScale;
    }
  });
  length += audioQuery.postPhonemeLength;
  return length / audioQuery.speedScale;
}
