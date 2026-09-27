<template>
  <div
    class="character-portrait-wrapper"
    :data-lipsync="lipSyncMode"
    :data-mouth="isPlaying ? mouthShape : 'n'"
  >
    <span class="character-name">{{ characterName }}</span>
    <span v-if="isMultipleEngine" class="character-engine-name">{{
      engineName
    }}</span>
    <SpeakerAvatar
      v-if="useAvatar && characterInfo"
      :name="characterInfo.metas.speakerName"
      :iconPath="
        characterInfo.portraitKind === 'icon' ? styleInfo?.iconPath : undefined
      "
      :text="activeText"
      :playing="isPlaying"
      :getLevel="() => store.getters.AUDIO_PLAYBACK_VOLUME()"
      :getSeconds="
        () => store.getters.ACTIVE_AUDIO_ELEM_CURRENT_TIME_GETTER() ?? 0
      "
      :getDuration="getAudioDurationSeconds"
      :anchors="avatarAnchors"
    />
    <img
      v-else-if="portraitPath"
      :src="displayPortraitPath"
      class="character-portrait"
      :alt="characterName"
    />
    <div v-else class="character-portrait-empty" aria-hidden="true"></div>
    <div v-if="characterInfo?.credit" class="character-credit">
      {{ characterInfo.credit }}
    </div>
    <div v-if="isInitializingSpeaker" class="loading">
      <QSpinner color="primary" size="5rem" :thickness="4" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from "vue";
import SpeakerAvatar from "./SpeakerAvatar.vue";
import {
  BlinkScheduler,
  LipSyncDriver,
  moraTimeline,
  moraTimelineFromAnchors,
  resolveLipSyncOptions,
  type LipSyncMode,
  type LipSyncSpeed,
  type LipSyncBlink,
  type MouthShape,
  type MoraStep,
  type TimelineSource,
} from "@/helpers/portraitLipSync";
import { fetchAsrTimeline } from "@/helpers/irodoriEngine";
import { parseEmotionMarks } from "@/helpers/speakerAvatar";
import type { AsrAnchor, AsrTimelineResponse } from "@/domain/irodori";
import { createEngineUrl } from "@/domain/url";
import { getAudioDurationSeconds, getPlaybackBlob } from "@/store/audioPlayer";
import { useStore } from "@/store";
import type { AudioKey } from "@/type/preload";
import { formatCharacterStyleName } from "@/store/utility";

const store = useStore();

const characterInfo = computed(() => {
  const activeAudioKey: AudioKey | undefined = store.getters.ACTIVE_AUDIO_KEY;
  const audioItem = activeAudioKey
    ? store.state.audioItems[activeAudioKey]
    : undefined;

  const engineId = audioItem?.voice.engineId;
  const styleId = audioItem?.voice.styleId;

  if (
    engineId == undefined ||
    styleId == undefined ||
    !store.state.engineIds.some((id) => id === engineId)
  )
    return undefined;

  return store.getters.CHARACTER_INFO(engineId, styleId);
});

const styleInfo = computed(() => {
  const activeAudioKey = store.getters.ACTIVE_AUDIO_KEY;

  const audioItem = activeAudioKey
    ? store.state.audioItems[activeAudioKey]
    : undefined;

  const styleId = audioItem?.voice.styleId;
  const style = characterInfo.value?.metas.styles.find(
    (style) => style.styleId === styleId,
  );
  return style;
});

const characterName = computed(() => {
  // 初期化前・未選択時
  if (characterInfo.value == undefined) {
    return "台本の行を選択してください";
  }

  const speakerName = characterInfo.value.metas.speakerName;
  const styleName = styleInfo.value?.styleName;
  return styleName
    ? formatCharacterStyleName(speakerName, styleName)
    : speakerName;
});

const engineName = computed(() => {
  const activeAudioKey = store.getters.ACTIVE_AUDIO_KEY;
  const audioItem = activeAudioKey
    ? store.state.audioItems[activeAudioKey]
    : undefined;
  const engineId = audioItem?.voice.engineId ?? store.state.engineIds[0];
  const engineManifest = store.state.engineManifests[engineId];
  const engineInfo = store.state.engineInfos[engineId];
  return engineManifest ? engineManifest.brandName : engineInfo.name;
});

const portraitPath = computed(
  () => styleInfo.value?.portraitPath || characterInfo.value?.portraitPath,
);
/** 専用の立ち絵が無い話者は、立ち絵の代わりに動くアイコンを出す */
const useAvatar = computed(
  () =>
    characterInfo.value != undefined &&
    (characterInfo.value.portraitKind ?? "portrait") !== "portrait",
);
const activeText = computed(() => {
  const activeKey = store.getters.ACTIVE_AUDIO_KEY;
  return activeKey ? (store.state.audioItems[activeKey]?.text ?? "") : "";
});
const mouthPaths = computed(() => styleInfo.value?.mouthPaths);
// 開いた口の絵。無い話者でも母音パーツがあれば『あ』を開閉用に使う。
const mouthOpenPath = computed(
  () => styleInfo.value?.mouthOpenPath ?? styleInfo.value?.mouthPaths?.a,
);
const blinkPath = computed(() => styleInfo.value?.blinkPath);
const isMouthOpen = ref(false);
const isBlinking = ref(false);
const mouthShape = ref<MouthShape>("n");
let mouthTimer: number | undefined;
let blinkTimer: number | undefined;
const isPlaying = computed(() => store.getters.NOW_PLAYING === true);

// 設定（設定 / オプション → 立ち絵と口パク）
const lipSyncMode = computed<LipSyncMode>(() => store.state.lipSyncMode);
const lipSyncSpeed = computed<LipSyncSpeed>(() => store.state.lipSyncSpeed);
const lipSyncBlink = computed<LipSyncBlink>(() => store.state.lipSyncBlink);
const lipSyncTimeline = computed<TimelineSource>(
  () => store.state.lipSyncTimeline,
);
const lipSyncOptions = computed(() =>
  resolveLipSyncOptions({
    mode: lipSyncMode.value,
    speed: lipSyncSpeed.value,
    blink: lipSyncBlink.value,
    timelineSource: lipSyncTimeline.value,
  }),
);
const hasVowelParts = computed(() => {
  const paths = mouthPaths.value;
  return paths != undefined && Object.keys(paths).length > 0;
});

/**
 * 母音モードでは、母音パーツがあればそれを、無ければ従来の口パク画像を使う。
 * 口が閉じているときは通常の立ち絵に戻す。
 */
const displayPortraitPath = computed(() => {
  if (lipSyncMode.value === "off") return portraitPath.value;
  if (isPlaying.value && isMouthOpen.value) {
    if (lipSyncMode.value === "vowel") {
      const part = mouthPaths.value?.[mouthShape.value];
      if (part) return part;
    }
    if (mouthOpenPath.value) return mouthOpenPath.value;
  }
  if (!isPlaying.value && isBlinking.value && blinkPath.value)
    return blinkPath.value;
  return portraitPath.value;
});

// ASR で作ったタイムラインのキャッシュ（同じセリフの再生で再要求しない）
const asrTimelineCache = new WeakMap<Blob, Map<string, MoraStep[]>>();
let asrRequestId = 0;

/** 選択中のセリフのエンジンの URL。 */
function activeEngineEndpoint(): string | undefined {
  const activeKey = store.getters.ACTIVE_AUDIO_KEY;
  const engineId = activeKey
    ? store.state.audioItems[activeKey]?.voice.engineId
    : undefined;
  if (engineId == undefined) return undefined;
  const info = store.state.engineInfos[engineId];
  if (info == undefined) return undefined;
  return createEngineUrl({
    ...info,
    port: store.state.altPortInfos[engineId] ?? info.defaultPort,
  });
}

/** ASR タイムラインを取りに行く。取れなければ null（=推定タイムラインへ戻す）。 */
async function loadAsrTimeline(
  text: string,
  audio: Blob,
): Promise<MoraStep[] | null> {
  const cached = asrTimelineCache.get(audio)?.get(text);
  if (cached != undefined) return cached.length > 0 ? cached : null;
  const endpoint = activeEngineEndpoint();
  if (endpoint == undefined) return null;
  const response: AsrTimelineResponse | null = await fetchAsrTimeline(
    endpoint,
    text,
    audio,
  );
  if (response?.anchors == undefined) {
    return null;
  }
  const steps = moraTimelineFromAnchors(response.anchors);
  const cachedTimelines =
    asrTimelineCache.get(audio) ?? new Map<string, MoraStep[]>();
  cachedTimelines.set(text, steps);
  asrTimelineCache.set(audio, cachedTimelines);
  return steps;
}

// 動くアイコン用：行の途中にある絵文字の切り替え時刻を ASR の文字時刻で合わせる。
// 取れなければ SpeakerAvatar が文字数で按分する
const avatarAnchorCache = new WeakMap<Blob, Map<string, AsrAnchor[]>>();
const avatarAnchors = ref<AsrAnchor[]>();
let avatarRequestId = 0;
watch(
  [isPlaying, useAvatar, activeText, lipSyncTimeline],
  ([playing, avatar, text, timeline]) => {
    const requestId = ++avatarRequestId;
    avatarAnchors.value = undefined;
    if (!playing || !avatar || timeline !== "asr") return;
    const { plain, marks } = parseEmotionMarks(text);
    if (!marks.some((mark) => mark.index > 0)) return;
    const audio = getPlaybackBlob();
    const endpoint = activeEngineEndpoint();
    if (!audio || endpoint == undefined) return;
    const cached = avatarAnchorCache.get(audio)?.get(plain);
    if (cached) {
      avatarAnchors.value = cached;
      return;
    }
    void fetchAsrTimeline(endpoint, plain, audio).then((response) => {
      if (requestId !== avatarRequestId || response?.anchors == undefined)
        return;
      const byText = avatarAnchorCache.get(audio) ?? new Map();
      byText.set(plain, response.anchors);
      avatarAnchorCache.set(audio, byText);
      avatarAnchors.value = response.anchors;
    });
  },
  { immediate: true },
);

watch(
  [
    isPlaying,
    mouthOpenPath,
    lipSyncMode,
    lipSyncSpeed,
    lipSyncTimeline,
    hasVowelParts,
    useAvatar,
    () => store.getters.ACTIVE_AUDIO_KEY,
  ],
  ([playing, mouth]) => {
    const requestId = ++asrRequestId;
    const audio = getPlaybackBlob();
    if (mouthTimer != undefined) {
      window.clearInterval(mouthTimer);
      mouthTimer = undefined;
    }
    isMouthOpen.value = false;
    mouthShape.value = "n";
    // 動くアイコンは音量だけで動くので、口パクのタイマーは使わない
    if (!playing || lipSyncMode.value === "off" || useAvatar.value) return;
    if (lipSyncMode.value === "simple" && !mouth) return;

    const options = lipSyncOptions.value;
    const activeKey = store.getters.ACTIVE_AUDIO_KEY;
    const text = activeKey
      ? (store.state.audioItems[activeKey]?.text ?? "")
      : "";
    const driver = new LipSyncDriver(options, []);
    // 音声の長さはデコード完了後に確定するので、pollごとに遅延生成する
    let timelineBuilt = false;
    const ensureTimeline = () => {
      if (timelineBuilt || text === "") return;
      // ASR のタイムラインが既にあればそれを最優先で使う
      const asr =
        lipSyncTimeline.value === "asr" && audio
          ? asrTimelineCache.get(audio)?.get(text)
          : undefined;
      if (asr != undefined && asr.length > 0) {
        driver.setTimeline(asr);
        timelineBuilt = true;
        return;
      }
      const duration = getAudioDurationSeconds();
      if (duration <= 0) return;
      driver.setTimeline(moraTimeline(text, duration));
      timelineBuilt = true;
    };
    if (lipSyncTimeline.value === "asr" && text !== "" && audio) {
      void loadAsrTimeline(text, audio).then((steps) => {
        // 再生中のセリフが変わっていたら破棄する
        if (requestId !== asrRequestId || steps == null) return;
        driver.setTimeline(steps);
        timelineBuilt = true;
      });
    }
    mouthTimer = window.setInterval(() => {
      ensureTimeline();
      const level = store.getters.AUDIO_PLAYBACK_VOLUME();
      const seconds =
        store.getters.ACTIVE_AUDIO_ELEM_CURRENT_TIME_GETTER() ?? 0;
      const frame = driver.update(level, seconds, performance.now());
      isMouthOpen.value = frame.open;
      mouthShape.value = frame.shape;
    }, options.pollMs);
  },
  { immediate: true },
);
watch(
  [blinkPath, isPlaying, lipSyncBlink],
  ([blink, playing, blinkMode]) => {
    if (blinkTimer != undefined) window.clearTimeout(blinkTimer);
    isBlinking.value = false;
    if (!blink || playing || blinkMode === "off") return;
    const options = lipSyncOptions.value.blink;
    if (options.durationMs <= 0) return;
    const scheduler = new BlinkScheduler(
      options.minMs,
      options.maxMs,
      options.durationMs,
    );
    // 瞬きは全体絵の差し替えなので、発話中は使わない
    const tick = () => {
      if (isPlaying.value) {
        blinkTimer = window.setTimeout(tick, 200);
        return;
      }
      isBlinking.value = scheduler.update(performance.now());
      const wait = isBlinking.value
        ? 40
        : Math.max(120, 2200 + Math.random() * 3200);
      blinkTimer = window.setTimeout(tick, wait);
    };
    blinkTimer = window.setTimeout(tick, 1200 + Math.random() * 2000);
  },
  { immediate: true },
);
onBeforeUnmount(() => {
  ++asrRequestId;
  ++avatarRequestId;
  if (mouthTimer != undefined) window.clearInterval(mouthTimer);
  if (blinkTimer != undefined) window.clearTimeout(blinkTimer);
});

const isInitializingSpeaker = computed(() => {
  const activeAudioKey = store.getters.ACTIVE_AUDIO_KEY;
  return (
    activeAudioKey &&
    store.state.audioKeysWithInitializingSpeaker.includes(activeAudioKey)
  );
});

const isMultipleEngine = computed(() => store.state.engineIds.length > 1);
</script>

<style scoped lang="scss">
@use "@/styles/colors" as colors;

.character-name,
.character-engine-name {
  flex: 0 0 auto;
  align-self: stretch;
  padding: 1px 24px 1px 8px;
  background-color: colors.$background;
  color: colors.$display;
  overflow-wrap: anywhere;
}

.character-portrait-wrapper {
  position: relative;
  display: flex;
  flex-direction: column;
  place-items: center;
  width: 100%;
  height: 100%;
  overflow: hidden;
  background: #fff;
  .character-portrait {
    display: block;
    width: 100%;
    height: auto;
    min-height: 0;
    flex: 1 1 auto;
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    object-position: center center;
  }
  .character-portrait-empty {
    width: 100%;
    height: auto;
    min-height: 0;
    flex: 1 1 auto;
    background: #fff;
  }
  .character-credit {
    flex: 0 1 auto;
    max-height: 18%;
    overflow-y: auto;
    width: 100%;
    box-sizing: border-box;
    padding: 4px 8px;
    color: #555;
    background: #fff;
    font-size: 0.65rem;
    line-height: 1.35;
    white-space: pre-line;
    overflow-wrap: anywhere;
    text-align: center;
  }
  .loading {
    position: absolute;
    width: 100%;
    height: 100%;
    background-color: rgba(colors.$background-rgb, 0.3);
    display: grid;
    justify-content: center;
    align-content: center;
  }
}
</style>
