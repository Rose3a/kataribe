import type {
  AudioItem,
  AudioStoreState,
  EditorAudioQuery,
  FetchAudioResult,
  IEngineConnectorFactoryActionsMapper,
  SettingStoreState,
} from "./type";
import { convertAudioQueryFromEditorToEngine } from "./proxy";
import { generateTempUniqueId } from "./utility";
import {
  IRODORI_DEFAULT_CFG_CAPTION,
  IRODORI_DEFAULT_CFG_SPEAKER,
  IRODORI_DEFAULT_CFG_TEXT,
  IRODORI_DEFAULT_CAPTION_STRENGTH,
  IRODORI_DEFAULT_REFERENCE_STRENGTH,
  IRODORI_DEFAULT_SPEAKER_STRENGTH,
  IRODORI_DEFAULT_SEED,
  IRODORI_DEFAULT_SCHEDULE,
  irodoriDefaultSteps,
} from "@/domain/irodori";

const audioBlobCache: Record<string, Blob> = {};
let audioCacheRevision = 0;

export function clearAudioCache() {
  audioCacheRevision++;
  for (const key of Object.keys(audioBlobCache)) delete audioBlobCache[key];
}

type Instance = {
  invoke: IEngineConnectorFactoryActionsMapper;
};

/**
 * エンジンで音声を合成する。音声のキャッシュ機構も備える。
 */
export async function fetchAudioFromAudioItem(
  state: AudioStoreState & SettingStoreState,
  instance: Instance,
  {
    audioItem,
  }: {
    audioItem: AudioItem;
  },
): Promise<FetchAudioResult> {
  const engineId = audioItem.voice.engineId;

  const [id, audioQuery] = await generateUniqueIdAndQuery(state, audioItem);
  if (audioQuery == undefined)
    throw new Error("audioQuery is not defined for audioItem");

  if (Object.prototype.hasOwnProperty.call(audioBlobCache, id)) {
    const blob = audioBlobCache[id];
    return { audioQuery, blob };
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
    const effectiveSeed =
      audioItem.irodori?.seed === null
        ? null
        : (audioItem.irodori?.seed ?? IRODORI_DEFAULT_SEED);
    blob = await instance.invoke("synthesis")({
      audioQuery: {
        ...engineAudioQuery,
        // The OpenAPI serializer emits this typed field as `irodori_seed`.
        // irodori_seed: effectiveSeed (serialized from the camelCase field)
        irodoriSeed: effectiveSeed,
        // 既定ステップ数はモデル依存（MeanFlow は4、RF は8）。
        irodoriSteps: audioItem.irodori?.steps ?? irodoriDefaultSteps.value,
        irodoriSchedule:
          audioItem.irodori?.schedule ?? IRODORI_DEFAULT_SCHEDULE,
        irodoriSeconds: audioItem.irodori?.seconds ?? null,
        irodoriCaption: audioItem.irodori?.caption,
        irodoriCaptionStrength:
          audioItem.irodori?.captionStrength ??
          IRODORI_DEFAULT_CAPTION_STRENGTH,
        irodoriCfgText: audioItem.irodori?.cfgText ?? IRODORI_DEFAULT_CFG_TEXT,
        irodoriCfgCaption:
          audioItem.irodori?.cfgCaption ?? IRODORI_DEFAULT_CFG_CAPTION,
        irodoriCfgSpeaker:
          audioItem.irodori?.cfgSpeaker ?? IRODORI_DEFAULT_CFG_SPEAKER,
        irodoriReferenceStrength:
          audioItem.irodori?.referenceStrength ??
          IRODORI_DEFAULT_REFERENCE_STRENGTH,
        irodoriSpeakerStrength:
          audioItem.irodori?.speakerStrength ??
          IRODORI_DEFAULT_SPEAKER_STRENGTH,
        irodoriSecondarySpeakerStyleId:
          audioItem.irodori?.secondarySpeakerStyleId ?? undefined,
        irodoriSecondarySpeakerStrength:
          audioItem.irodori?.secondarySpeakerStrength ?? 0.5,
        irodoriAdditionalSpeakers: audioItem.irodori?.additionalSpeakers,
        irodoriReferenceAudio: audioItem.irodori?.referenceAudio,
      },
      speaker,
      enableInterrogativeUpspeak:
        state.experimentalSetting.enableInterrogativeUpspeak,
    });
  }
  audioBlobCache[id] = blob;
  return { audioQuery, blob };
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
