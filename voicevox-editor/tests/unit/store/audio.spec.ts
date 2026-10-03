import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { store } from "@/store";
import type { AudioItem, EditorAudioQuery } from "@/store/type";
import type { AudioQuery } from "@/openapi";
import {
  AudioKey,
  EngineId,
  PresetKey,
  SpeakerId,
  StyleId,
} from "@/type/preload";
import { cloneWithUnwrapProxy } from "@/helpers/cloneWithUnwrapProxy";
import { resetMockMode, uuid4 } from "@/helpers/random";
import { getEngineManifestMock } from "@/mock/engineMock/manifestMock";

const { fetchAudioQuery } = vi.hoisted(() => ({
  fetchAudioQuery:
    vi.fn<
      (request: { text: string; speaker: number }) => Promise<AudioQuery>
    >(),
}));

vi.mock("@/infrastructures/EngineConnector", () => {
  const factory = { instance: () => ({ audioQuery: fetchAudioQuery }) };
  return {
    OpenAPIEngineConnectorFactory: factory,
    OpenAPIEngineAndMockConnectorFactory: factory,
  };
});

const initialState = cloneWithUnwrapProxy(store.state);

function createQuery(kana: string): EditorAudioQuery {
  return {
    accentPhrases: [],
    speedScale: 1,
    pitchScale: 0,
    intonationScale: 1,
    volumeScale: 1,
    pauseLengthScale: 1,
    prePhonemeLength: 0.1,
    postPhonemeLength: 0.1,
    outputSamplingRate: 48000,
    outputStereo: false,
    kana,
  };
}

function insertAudioItem({
  withPreset = true,
  withQuery = true,
}: { withPreset?: boolean; withQuery?: boolean } = {}) {
  const engineId = EngineId(uuid4());
  const audioKey = AudioKey(uuid4());
  const presetKey = PresetKey(uuid4());
  store.mutations.SET_ENGINE_INFOS({
    engineIds: [engineId],
    engineInfos: [
      {
        uuid: engineId,
        protocol: "http:",
        hostname: "127.0.0.1",
        defaultPort: "50125",
        pathname: "",
        name: "Irodori-TTS",
        version: "1.0.0",
        executionEnabled: false,
        executionFilePath: "",
        executionArgs: [],
        type: "path",
        isDefault: true,
      },
    ],
  });
  store.mutations.SET_ENGINE_MANIFESTS({
    engineManifests: {
      [engineId]: { ...getEngineManifestMock(), name: "Irodori-TTS" },
    },
  });
  store.mutations.SET_PRESET_ITEMS({
    presetItems: {
      [presetKey]: {
        name: "デフォルト",
        speedScale: 1.25,
        pitchScale: 0,
        intonationScale: 1,
        volumeScale: 1,
        pauseLengthScale: 1,
        prePhonemeLength: 0.1,
        postPhonemeLength: 0.1,
      },
    },
  });
  const audioItem: AudioItem = {
    text: "編集前",
    voice: {
      engineId,
      speakerId: SpeakerId(uuid4()),
      styleId: StyleId(1),
    },
    ...(withQuery ? { query: createQuery("ヘンシュウマエ") } : {}),
    ...(withPreset ? { presetKey } : {}),
  };
  store.mutations.INSERT_AUDIO_ITEM({
    audioKey,
    audioItem,
    prevAudioKey: undefined,
  });
  return audioKey;
}

beforeEach(() => {
  store.replaceState(cloneWithUnwrapProxy(initialState));
  resetMockMode();
  fetchAudioQuery.mockReset();
  fetchAudioQuery.mockImplementation(async ({ text }) => ({
    ...createQuery(text),
    outputSamplingRate: 48000,
  }));
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("COMMAND_CHANGE_AUDIO_TEXT（Irodori）", () => {
  test.each([0.75, 1.5])(
    "プリセットから変更した話速 %s を文章の編集後も保持する",
    async (speedScale) => {
      const audioKey = insertAudioItem();
      await store.actions.COMMAND_MULTI_SET_AUDIO_SPEED_SCALE({
        audioKeys: [audioKey],
        speedScale,
      });

      await store.actions.COMMAND_CHANGE_AUDIO_TEXT({
        audioKey,
        text: "編集後",
      });

      expect(store.state.audioItems[audioKey].query).toMatchObject({
        kana: "編集後",
        speedScale,
      });
      expect(fetchAudioQuery).toHaveBeenCalledWith({
        text: "編集後",
        speaker: 1,
        enableKatakanaEnglish: undefined,
      });
    },
  );

  test("プリセットがない場合も変更した話速を保持する", async () => {
    const audioKey = insertAudioItem({ withPreset: false });
    await store.actions.COMMAND_MULTI_SET_AUDIO_SPEED_SCALE({
      audioKeys: [audioKey],
      speedScale: 1.5,
    });

    await store.actions.COMMAND_CHANGE_AUDIO_TEXT({ audioKey, text: "編集後" });

    expect(store.state.audioItems[audioKey].query?.speedScale).toBe(1.5);
  });

  test("クエリの初回取得時にはプリセットの話速を適用する", async () => {
    const audioKey = insertAudioItem({ withQuery: false });

    await store.actions.COMMAND_CHANGE_AUDIO_TEXT({ audioKey, text: "編集後" });

    expect(store.state.audioItems[audioKey].query).toMatchObject({
      kana: "編集後",
      speedScale: 1.25,
    });
  });

  test("文章編集の取り消しとやり直しでも話速を保持する", async () => {
    const audioKey = insertAudioItem();
    await store.actions.COMMAND_MULTI_SET_AUDIO_SPEED_SCALE({
      audioKeys: [audioKey],
      speedScale: 0.75,
    });
    await store.actions.COMMAND_CHANGE_AUDIO_TEXT({ audioKey, text: "編集後" });

    await store.actions.UNDO({ editor: "talk" });
    expect(store.state.audioItems[audioKey]).toMatchObject({
      text: "編集前",
      query: { kana: "ヘンシュウマエ", speedScale: 0.75 },
    });

    await store.actions.REDO({ editor: "talk" });
    expect(store.state.audioItems[audioKey]).toMatchObject({
      text: "編集後",
      query: { kana: "編集後", speedScale: 0.75 },
    });
  });

  test("クエリの取得失敗時にも話速を保持する", async () => {
    const audioKey = insertAudioItem();
    await store.actions.COMMAND_MULTI_SET_AUDIO_SPEED_SCALE({
      audioKeys: [audioKey],
      speedScale: 1.5,
    });
    vi.stubGlobal("backend", { logError: vi.fn() });
    fetchAudioQuery.mockRejectedValueOnce(new Error("engine unavailable"));

    await expect(
      store.actions.COMMAND_CHANGE_AUDIO_TEXT({ audioKey, text: "編集後" }),
    ).rejects.toThrow("engine unavailable");

    expect(store.state.audioItems[audioKey]).toMatchObject({
      text: "編集後",
      query: { kana: "ヘンシュウマエ", speedScale: 1.5 },
    });
  });
});
