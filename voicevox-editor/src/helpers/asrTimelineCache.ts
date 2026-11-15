/**
 * /irodori/timeline（ASR で測る文字ごとの発話時刻）を、音声ごとに 1 回だけ取りに行くためのキャッシュ。
 *
 * 口パク（CharacterPortrait）と、再生中のセリフを文字ごとに追う表示（AudioCell）が同じ音声で
 * 同時に要求しても、エンジンの ASR は 1 回しか走らない。生成した音声の Blob はエディタが
 * キャッシュして使い回すので、同じ音声を何度再生しても 2 回目からは待ちがない。
 * 失敗（ASR モデルの取得中など）は覚えず、次の再生でもう一度取りに行く。
 */
import { fetchAsrTimeline } from "@/helpers/irodoriEngine";
import type { AsrTimelineResponse } from "@/domain/irodori";
import { createEngineUrl } from "@/domain/url";
import type { EngineId, EngineInfo } from "@/type/preload";

const timelines = new WeakMap<
  Blob,
  Map<string, Promise<AsrTimelineResponse | null>>
>();

export function getAsrTimelineCached(
  endpoint: string,
  text: string,
  audio: Blob,
): Promise<AsrTimelineResponse | null> {
  let byText = timelines.get(audio);
  if (byText == undefined) {
    byText = new Map();
    timelines.set(audio, byText);
  }
  const cached = byText.get(text);
  if (cached != undefined) return cached;
  const request = fetchAsrTimeline(endpoint, text, audio).then((result) => {
    if (result == null) byText.delete(text);
    return result;
  });
  byText.set(text, request);
  return request;
}

/** エンジンの URL（engineInfos と altPortInfos から） */
export function engineEndpoint(
  state: {
    engineInfos: Record<EngineId, EngineInfo>;
    altPortInfos: Record<EngineId, string>;
  },
  engineId: EngineId,
): string | undefined {
  const info = state.engineInfos[engineId];
  if (info == undefined) return undefined;
  return createEngineUrl({
    ...info,
    port: state.altPortInfos[engineId] ?? info.defaultPort,
  });
}
