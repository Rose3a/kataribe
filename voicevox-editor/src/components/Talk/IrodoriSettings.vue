<template>
  <section class="q-pa-md irodori-settings">
    <div v-if="audioItem" class="q-mb-md">
      <div class="text-subtitle1 q-mb-sm">セリフごとの設定</div>
      <div class="row items-center q-gutter-sm q-mb-sm irodori-line-actions">
        <QBtn
          color="primary"
          dense
          icon="play_arrow"
          label="この行だけ生成して再生"
          :loading="linePlaying"
          :disable="locked || !canPlayActiveLine"
          @click="playActiveLine"
        />
      </div>
      <div v-if="!canPlayActiveLine" class="text-caption q-mb-sm">
        セリフを入力すると、この1行だけの音声を生成できます
      </div>
      <div v-if="linePlaying" class="text-caption q-mb-sm">
        この行を生成中。止めたいときは上の「停止」から
      </div>
      <div
        v-if="lineError"
        role="alert"
        class="text-negative text-caption q-mb-sm"
      >
        {{ lineError }}
      </div>
      <div class="row q-col-gutter-sm irodori-cfg-row">
        <QInput
          v-model="cfgTextText"
          outlined
          dense
          type="number"
          label="テキストCFG"
          hint="既定値: 3（0〜20）"
          inputmode="decimal"
          :min="0"
          :max="20"
          :step="0.5"
          :disable="locked"
          class="col-4 irodori-number-input"
          @change="saveLineCfg('cfgText', cfgTextText)"
        />
        <QInput
          v-model="cfgCaptionText"
          outlined
          dense
          type="number"
          label="キャプションCFG"
          hint="既定値: 3（0〜20）"
          inputmode="decimal"
          :min="0"
          :max="20"
          :step="0.5"
          :disable="locked"
          class="col-4 irodori-number-input"
          @change="saveLineCfg('cfgCaption', cfgCaptionText)"
        />
        <QInput
          v-model="cfgSpeakerText"
          outlined
          dense
          type="number"
          label="スピーカーCFG"
          hint="既定値: 5（0〜20）"
          inputmode="decimal"
          :min="0"
          :max="20"
          :step="0.5"
          :disable="locked"
          class="col-4 irodori-number-input"
          @change="saveLineCfg('cfgSpeaker', cfgSpeakerText)"
        />
      </div>
      <QInput
        v-model.number="stepsValue"
        outlined
        dense
        type="number"
        label="ステップ数 (1〜80)"
        :hint="`既定値: ${defaultSteps}（このモデル）`"
        :min="1"
        :max="80"
        :step="1"
        :disable="locked"
        class="q-mb-sm irodori-number-input"
        @change="saveSteps"
      >
        <template #prepend>
          <QIcon
            v-if="showStepsQualityWarning"
            name="warning"
            color="negative"
            size="sm"
            role="img"
            aria-label="音声品質に関する警告"
          >
            <QTooltip>
              ステップ数がデフォルト未満のため、音声の質が低い可能性があります。
            </QTooltip>
          </QIcon>
        </template>
        <template #append>
          <div
            class="irodori-number-stepper"
            role="group"
            aria-label="ステップ数を調整"
          >
            <QBtn
              flat
              dense
              icon="expand_less"
              aria-label="ステップ数を1増やす"
              :disable="locked"
              @click.stop="adjustSteps(1)"
            />
            <QBtn
              flat
              dense
              icon="expand_more"
              aria-label="ステップ数を1減らす"
              :disable="locked"
              @click.stop="adjustSteps(-1)"
            />
          </div>
        </template>
      </QInput>
      <QSelect
        v-model="scheduleValue"
        outlined
        dense
        label="Schedule"
        :options="scheduleModes"
        emitValue
        mapOptions
        :disable="locked"
        class="q-mb-sm"
        @update:modelValue="saveSchedule"
      />
      <QSelect
        v-model="secondsValue"
        outlined
        dense
        label="音声長"
        :options="durations"
        emitValue
        mapOptions
        :disable="locked"
        class="q-mb-sm"
        @update:modelValue="saveSeconds"
      />
      <QInput
        dense
        borderless
        maxlength="5"
        :class="{ disabled: speedScaleSlider.qSliderProps.disable.value }"
        :disable="speedScaleSlider.qSliderProps.disable.value"
        :modelValue="
          speedScaleSlider.state.currentValue.value != undefined
            ? speedScaleSlider.state.currentValue.value.toFixed(2)
            : speedScaleSlider.qSliderProps.min.value.toFixed(2)
        "
        @change="handleSpeedScaleChange"
      >
        <template #before
          ><span class="text-body1 text-display">話速</span></template
        >
      </QInput>
      <QSlider
        dense
        snap
        color="primary"
        trackSize="2px"
        :min="speedScaleSlider.qSliderProps.min.value"
        :max="speedScaleSlider.qSliderProps.max.value"
        :step="speedScaleSlider.qSliderProps.step.value"
        :disable="speedScaleSlider.qSliderProps.disable.value"
        :modelValue="speedScaleSlider.qSliderProps.modelValue.value"
        @update:modelValue="
          speedScaleSlider.qSliderProps['onUpdate:modelValue']
        "
        @change="speedScaleSlider.qSliderProps.onChange"
        @wheel="speedScaleSlider.qSliderProps.onWheel"
        @pan="speedScaleSlider.qSliderProps.onPan"
      />
      <QInput
        v-model="seedText"
        outlined
        dense
        label="シード（空欄＝ランダム）"
        type="number"
        :disable="locked"
        class="q-mb-sm irodori-number-input"
        @update:modelValue="saveSeed"
      >
        <template #append>
          <div
            class="irodori-number-stepper"
            role="group"
            aria-label="シードを調整"
          >
            <QBtn
              flat
              dense
              icon="expand_less"
              aria-label="シードを1増やす"
              :disable="locked"
              @click.stop="adjustSeed(1)"
            />
            <QBtn
              flat
              dense
              icon="expand_more"
              aria-label="シードを1減らす"
              :disable="locked"
              @click.stop="adjustSeed(-1)"
            />
          </div>
        </template>
      </QInput>
      <QFile
        v-model="referenceFile"
        outlined
        dense
        clearable
        accept="audio/*"
        label="音声リファレンス（任意）"
        hint="この行の話者・声質の参考音声。最大10MB"
        :disable="locked"
        class="q-mb-sm"
        @update:modelValue="handleReferenceFileChange"
      />
      <div v-if="referenceAudioName" class="text-caption q-mb-sm">
        適用中: {{ referenceAudioName }}
        <QBtn
          flat
          dense
          label="解除"
          icon="clear"
          :disable="locked"
          @click="clearReferenceAudio"
        />
      </div>
      <div class="irodori-strength-control q-mb-md">
        <div class="irodori-strength-label">
          <span>音声参照の強度</span>
          <span>{{ referenceStrengthValue.toFixed(1) }}</span>
        </div>
        <QSlider
          v-model="referenceStrengthValue"
          dense
          snap
          color="primary"
          trackSize="2px"
          :min="0"
          :max="1"
          :step="0.1"
          :disable="locked || !referenceAudioName"
          aria-label="音声参照の強度"
          @change="saveStrength('referenceStrength', referenceStrengthValue)"
        />
        <div class="text-caption">0で音声参照なし、1で最大</div>
      </div>
      <QInput
        v-model="captionText"
        outlined
        dense
        autogrow
        type="textarea"
        label="キャプション（任意）"
        hint="このセリフの場面・話し方・感情など。空欄なら未指定"
        :maxlength="2000"
        counter
        :disable="locked"
        class="q-mb-sm"
        @update:modelValue="saveCaption"
      />
      <div class="irodori-strength-control">
        <div class="irodori-strength-label">
          <span>指示キャプションの強度</span>
          <span>{{ captionStrengthValue.toFixed(1) }}</span>
        </div>
        <QSlider
          v-model="captionStrengthValue"
          dense
          snap
          color="primary"
          trackSize="2px"
          :min="0"
          :max="1"
          :step="0.1"
          :disable="locked || !captionText.trim()"
          aria-label="指示キャプションの強度"
          @change="saveStrength('captionStrength', captionStrengthValue)"
        />
        <div class="text-caption">0で指示なし、1で最大</div>
      </div>
    </div>
    <div v-if="error" role="alert" class="text-negative text-caption q-mt-sm">
      {{ error }}
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { useStore } from "@/store";
import { createEngineUrl } from "@/domain/url";
import { fetchIrodoriDefaultSteps } from "@/helpers/irodoriEngine";
import {
  clearAudioCache,
  handlePossiblyNotMorphableError,
} from "@/store/audioGenerate";
import {
  IRODORI_DEFAULT_CFG_CAPTION,
  IRODORI_DEFAULT_CFG_SPEAKER,
  IRODORI_DEFAULT_CFG_TEXT,
  IRODORI_DEFAULT_CAPTION_STRENGTH,
  IRODORI_DEFAULT_REFERENCE_STRENGTH,
  IRODORI_DEFAULT_SEED,
  IRODORI_DEFAULT_SCHEDULE,
  irodoriDefaultSteps,
  IRODORI_DEFAULT_STEPS,
} from "@/domain/irodori";
import {
  previewSliderHelper,
  type PreviewSliderHelper,
} from "@/helpers/previewSliderHelper";
import type { EngineManifest } from "@/openapi";
import { SLIDER_PARAMETERS } from "@/store/utility";
import type { AudioKey, EngineId } from "@/type/preload";

const props = defineProps<{ engineId: EngineId; activeAudioKey: AudioKey }>();
const store = useStore();
const error = ref("");
const audioItem = computed(() => store.state.audioItems[props.activeAudioKey]);
const query = computed(() => audioItem.value?.query);
const supportedFeatures = computed(
  () =>
    (store.state.engineIds.some(
      (id) => id === audioItem.value?.voice.engineId,
    ) &&
      audioItem.value != undefined &&
      store.state.engineManifests[audioItem.value.voice.engineId]
        .supportedFeatures) as EngineManifest["supportedFeatures"] | undefined,
);
const selectedAudioKeys = computed(() =>
  store.state.enableMultiSelect
    ? store.getters.SELECTED_AUDIO_KEYS
    : [props.activeAudioKey],
);
// 選択中モデルの既定ステップ数（MeanFlow は4、RF は8）。エンジンの modelInfo から更新する。
// irodori より先に定義する（irodori が init 時にこの値を使うため）。
const defaultSteps = computed(() => irodoriDefaultSteps.value);
watch(
  defaultSteps,
  (value, previous) => {
    irodoriDefaultSteps.value = value;
    if (previous == undefined || value === previous) return;
    // モデルを切り替えたら、既定のままにしていたセリフのステップ数を追随させる。
    for (const audioKey of Object.keys(store.state.audioItems)) {
      const item = store.state.audioItems[audioKey as AudioKey];
      const steps = item?.irodori?.steps;
      if (steps != undefined && steps !== previous) continue;
      void store.actions.COMMAND_SET_IRODORI_SETTINGS({
        audioKey: audioKey as AudioKey,
        irodori: { ...item?.irodori, steps: value },
      });
    }
    clearAudioCache();
  },
  { immediate: true },
);
watch(
  () => props.engineId,
  async (engineId, _previous, onCleanup) => {
    let cancelled = false;
    onCleanup(() => {
      cancelled = true;
    });
    const info = store.state.engineInfos[engineId];
    if (!info) return;
    const steps = await fetchIrodoriDefaultSteps(
      createEngineUrl({
        ...info,
        port: store.state.altPortInfos[engineId] ?? info.defaultPort,
      }),
    );
    if (!cancelled && steps != undefined) irodoriDefaultSteps.value = steps;
  },
  { immediate: true },
);
const irodori = computed(() => {
  const value = audioItem.value?.irodori;
  return {
    ...(value ?? { seed: IRODORI_DEFAULT_SEED }),
    steps: value?.steps ?? defaultSteps.value,
    schedule: value?.schedule ?? IRODORI_DEFAULT_SCHEDULE,
    seconds: value?.seconds ?? null,
    cfgText: value?.cfgText ?? IRODORI_DEFAULT_CFG_TEXT,
    cfgCaption: value?.cfgCaption ?? IRODORI_DEFAULT_CFG_CAPTION,
    cfgSpeaker: value?.cfgSpeaker ?? IRODORI_DEFAULT_CFG_SPEAKER,
    captionStrength: value?.captionStrength ?? IRODORI_DEFAULT_CAPTION_STRENGTH,
    referenceStrength:
      value?.referenceStrength ?? IRODORI_DEFAULT_REFERENCE_STRENGTH,
  };
});
const seedText = ref("");
const stepsValue = ref(IRODORI_DEFAULT_STEPS);
const scheduleValue = ref<"linear" | "sway">(IRODORI_DEFAULT_SCHEDULE);
const secondsValue = ref<number | null>(null);
const captionText = ref("");
const cfgTextText = ref("");
const cfgCaptionText = ref("");
const cfgSpeakerText = ref("");
const captionStrengthValue = ref(IRODORI_DEFAULT_CAPTION_STRENGTH);
const referenceStrengthValue = ref(IRODORI_DEFAULT_REFERENCE_STRENGTH);
watch(
  irodori,
  (value) => {
    seedText.value = value.seed == null ? "" : String(value.seed);
    stepsValue.value = value.steps;
    scheduleValue.value = value.schedule;
    secondsValue.value = value.seconds;
    captionText.value = value.caption ?? "";
    cfgTextText.value = String(value.cfgText);
    cfgCaptionText.value = String(value.cfgCaption);
    cfgSpeakerText.value = String(value.cfgSpeaker);
    captionStrengthValue.value = value.captionStrength;
    referenceStrengthValue.value = value.referenceStrength;
  },
  { immediate: true, deep: true },
);
const referenceFile = ref<File | null>(null);
const referenceAudioName = computed(
  () => irodori.value.referenceAudio?.name ?? "",
);
watch(
  () => props.activeAudioKey,
  () => {
    referenceFile.value = null;
  },
  { immediate: true },
);
type LineCfgKey = "cfgText" | "cfgCaption" | "cfgSpeaker";
const lineCfgDefaults: Record<LineCfgKey, number> = {
  cfgText: IRODORI_DEFAULT_CFG_TEXT,
  cfgCaption: IRODORI_DEFAULT_CFG_CAPTION,
  cfgSpeaker: IRODORI_DEFAULT_CFG_SPEAKER,
};
function saveSteps(input: string | number | null = stepsValue.value) {
  const value = Number(input);
  if (!Number.isInteger(value) || value < 1 || value > 80) {
    error.value = "ステップ数は1〜80の整数だけ入力してください";
    stepsValue.value = irodori.value.steps;
    return;
  }
  error.value = "";
  stepsValue.value = value;
  clearAudioCache();
  void store.actions.COMMAND_SET_IRODORI_SETTINGS({
    audioKey: props.activeAudioKey,
    irodori: { ...irodori.value, steps: value },
  });
}
function adjustSteps(direction: 1 | -1) {
  saveSteps(Math.min(80, Math.max(1, stepsValue.value + direction)));
}
function saveSchedule(value: "linear" | "sway" | null) {
  if (value !== "linear" && value !== "sway") return;
  scheduleValue.value = value;
  clearAudioCache();
  void store.actions.COMMAND_SET_IRODORI_SETTINGS({
    audioKey: props.activeAudioKey,
    irodori: { ...irodori.value, schedule: value },
  });
}
function saveSeconds(value: number | null) {
  if (value != null && (!Number.isFinite(value) || value < 0.1 || value > 60)) {
    error.value = "音声長は自動または0.1〜60秒で指定してください";
    secondsValue.value = irodori.value.seconds;
    return;
  }
  error.value = "";
  secondsValue.value = value;
  clearAudioCache();
  void store.actions.COMMAND_SET_IRODORI_SETTINGS({
    audioKey: props.activeAudioKey,
    irodori: { ...irodori.value, seconds: value },
  });
}
function saveLineCfg(key: LineCfgKey, input: string | number | null) {
  const raw = String(input ?? "").trim();
  if (raw === "") {
    const defaultValue = lineCfgDefaults[key];
    if (key === "cfgText") cfgTextText.value = String(defaultValue);
    if (key === "cfgCaption") cfgCaptionText.value = String(defaultValue);
    if (key === "cfgSpeaker") cfgSpeakerText.value = String(defaultValue);
    saveLineCfg(key, defaultValue);
    return;
  }
  const value = Number(raw);
  if (!Number.isFinite(value) || value < 0 || value > 20) {
    error.value = `${key}は0〜20の数値だけ入力してください`;
    const currentValue = irodori.value[key];
    if (key === "cfgText") cfgTextText.value = String(currentValue);
    if (key === "cfgCaption") cfgCaptionText.value = String(currentValue);
    if (key === "cfgSpeaker") cfgSpeakerText.value = String(currentValue);
    return;
  }
  error.value = "";
  // CFG変更後に古い音声を再利用しない。
  clearAudioCache();
  void store.actions.COMMAND_SET_IRODORI_SETTINGS({
    audioKey: props.activeAudioKey,
    irodori: { ...irodori.value, [key]: Number(value.toFixed(4)) },
  });
}
function saveSeed() {
  const value = seedText.value.trim();
  void store.actions.COMMAND_SET_IRODORI_SETTINGS({
    audioKey: props.activeAudioKey,
    irodori: { ...irodori.value, seed: value === "" ? null : Number(value) },
  });
}
function saveCaption() {
  const value = captionText.value.trim();
  void store.actions.COMMAND_SET_IRODORI_SETTINGS({
    audioKey: props.activeAudioKey,
    irodori: { ...irodori.value, caption: value || undefined },
  });
  clearAudioCache();
}
type StrengthKey = "captionStrength" | "referenceStrength";
function saveStrength(key: StrengthKey, input: number | null) {
  const value = Number(input);
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    error.value = "強度は0〜1の数値だけ指定できます";
    if (key === "captionStrength") {
      captionStrengthValue.value = irodori.value.captionStrength;
    } else {
      referenceStrengthValue.value = irodori.value.referenceStrength;
    }
    return;
  }
  error.value = "";
  const normalized = Number(value.toFixed(1));
  if (key === "captionStrength") captionStrengthValue.value = normalized;
  else referenceStrengthValue.value = normalized;
  clearAudioCache();
  void store.actions.COMMAND_SET_IRODORI_SETTINGS({
    audioKey: props.activeAudioKey,
    irodori: { ...irodori.value, [key]: normalized },
  });
}
function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === "string") resolve(reader.result);
      else reject(new Error("音声リファレンスを読み込めませんでした"));
    };
    reader.onerror = () =>
      reject(
        reader.error ?? new Error("音声リファレンスを読み込めませんでした"),
      );
    reader.readAsDataURL(file);
  });
}
async function handleReferenceFileChange(value: File | File[] | null) {
  const file = Array.isArray(value) ? value[0] : value;
  error.value = "";
  if (file == null) {
    clearReferenceAudio();
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    referenceFile.value = null;
    error.value = "音声リファレンスは10MB以下にしてください";
    return;
  }
  if (file.type && !file.type.startsWith("audio/")) {
    referenceFile.value = null;
    error.value = "音声ファイルを選択してください";
    return;
  }
  try {
    const dataUrl = await readFileAsDataUrl(file);
    void store.actions.COMMAND_SET_IRODORI_SETTINGS({
      audioKey: props.activeAudioKey,
      irodori: {
        ...irodori.value,
        referenceAudio: {
          dataUrl,
          mime: file.type || "audio/wav",
          name: file.name,
        },
      },
    });
    clearAudioCache();
  } catch (cause) {
    referenceFile.value = null;
    error.value = cause instanceof Error ? cause.message : String(cause);
  }
}
function clearReferenceAudio() {
  referenceFile.value = null;
  void store.actions.COMMAND_SET_IRODORI_SETTINGS({
    audioKey: props.activeAudioKey,
    irodori: { ...irodori.value, referenceAudio: undefined },
  });
  clearAudioCache();
}
function adjustSeed(delta: number) {
  const current = Number(seedText.value);
  seedText.value = String((Number.isFinite(current) ? current : 0) + delta);
  saveSeed();
}
// エンジンが返した理由を、その場で読める日本語にまとめる。
// ワーカー落ちはスタックトレースが返るので、原因だけを短く伝える。
function summarizeEngineError(cause: unknown, detail: string): string {
  if (/worker|NoneType|stdin|UnicodeDecodeError|pipe/i.test(detail)) {
    return "エンジンの生成ワーカーが停止しました。セリフを短く分けて試すか、エンジンを再起動してください";
  }
  const head =
    detail
      .split("\n")
      .map((line) => line.trim())
      .find((line) => line !== "" && !/^Traceback/.test(line)) ?? "";
  if (head) return head;
  return cause instanceof Error
    ? cause.message
    : "再生に失敗しました。エンジンの再起動をお試しください。";
}
// 失敗時にエンジンが返したJSONの detail を取り出す。
async function engineErrorDetail(cause: unknown): Promise<string> {
  const response = (cause as { response?: Response } | undefined)?.response;
  if (response == undefined || typeof response.clone !== "function") return "";
  try {
    const body = await response.clone().text();
    const detail = (JSON.parse(body) as { detail?: unknown }).detail;
    return typeof detail === "string" ? detail : "";
  } catch {
    return "";
  }
}
// セリフ1行だけの音声生成と再生。連続再生と違い、対象は選択中の行のみ。
// 停止はツールバーの既存ボタンに任せる。
const linePlaying = ref(false);
const lineError = ref("");
const canPlayActiveLine = computed(
  () => (audioItem.value?.text.trim().length ?? 0) > 0,
);
async function playActiveLine() {
  if (linePlaying.value || locked.value) return;
  if (!canPlayActiveLine.value) {
    lineError.value = "セリフが空欄のため生成できません";
    return;
  }
  linePlaying.value = true;
  lineError.value = "";
  try {
    await store.actions.PLAY_AUDIO({ audioKey: props.activeAudioKey });
  } catch (cause) {
    const detail = await engineErrorDetail(cause);
    const message = handlePossiblyNotMorphableError(cause);
    lineError.value =
      message == undefined ? summarizeEngineError(cause, detail) : message;
  } finally {
    linePlaying.value = false;
  }
}
const speedScaleSlider: PreviewSliderHelper = previewSliderHelper({
  modelValue: () => query.value?.speedScale ?? null,
  disable: () => locked.value || !supportedFeatures.value?.adjustSpeedScale,
  max: SLIDER_PARAMETERS.speedScale.max,
  min: SLIDER_PARAMETERS.speedScale.min,
  step: SLIDER_PARAMETERS.speedScale.step,
  scrollStep: SLIDER_PARAMETERS.speedScale.scrollStep,
  scrollMinStep: SLIDER_PARAMETERS.speedScale.scrollMinStep,
  onChange: (speedScale: number) =>
    store.actions.COMMAND_MULTI_SET_AUDIO_SPEED_SCALE({
      audioKeys: selectedAudioKeys.value,
      speedScale,
    }),
});
const handleSpeedScaleChange = (inputValue: string | number | null) => {
  if (inputValue == null) throw new Error("inputValue is null");
  const speedScale = adjustSliderValue(
    inputValue.toString(),
    SLIDER_PARAMETERS.speedScale.min(),
    SLIDER_PARAMETERS.speedScale.max(),
  );
  return speedScaleSlider.qSliderProps.onChange(speedScale);
};
const adjustSliderValue = (
  inputStr: string,
  minimalVal: number,
  maximamVal: number,
) => {
  const inputNum = Number(convertFullWidthNumbers(inputStr));
  if (Number.isNaN(inputNum)) return minimalVal;
  return Math.min(Math.max(inputNum, minimalVal), maximamVal);
};
const convertFullWidthNumbers = (inputStr: string) => {
  const numberConversionMap = [
    ["０", "0"],
    ["１", "1"],
    ["２", "2"],
    ["３", "3"],
    ["４", "4"],
    ["５", "5"],
    ["６", "6"],
    ["７", "7"],
    ["８", "8"],
    ["９", "9"],
    ["．", "."],
    ["・", "."],
    ["－", "-"],
  ];
  let convertedInputStr = inputStr;
  for (const [pattern, replacement] of numberConversionMap) {
    convertedInputStr = convertedInputStr.replace(
      new RegExp(pattern, "g"),
      replacement,
    );
  }
  return convertedInputStr;
};
const durations = [
  { label: "自動", value: null },
  ...[3, 5, 10, 15, 30].map((value) => ({ label: `${value}秒`, value })),
];
const scheduleModes = [
  { label: "sway", value: "sway" },
  { label: "linear", value: "linear" },
];
const locked = computed(() => store.getters.UI_LOCKED);
const showStepsQualityWarning = computed(
  () => stepsValue.value < defaultSteps.value,
);
</script>

<style scoped>
.irodori-settings {
  border-left: 3px solid #6c63ff;
  background: linear-gradient(
    180deg,
    rgba(108, 99, 255, 0.08),
    transparent 65%
  );
}
.irodori-settings :deep(.q-btn--standard) {
  background: #6c63ff;
  color: #fff;
}
.irodori-settings :deep(.irodori-number-input .q-field__control) {
  padding-right: 0;
}
/*
 * Quasarはヒント用に固定の20pxを確保し（.q-field--with-bottom の padding-bottom）、
 * さらに未フォーカス時に transform: translateY(100%) でヒントを下へずらす。
 * 3列に並ぶCFG欄は幅が狭くヒントが2行に折り返すため、予約された20pxをはみ出して
 * 次の行のラベルに重なっていた。ヒントを通常フローに戻し、フィールド自身が
 * ヒントの高さぶん伸びるようにして、折り返しても重ならないようにする。
 */
.irodori-settings :deep(.q-field--with-bottom) {
  padding-bottom: 0;
}
.irodori-settings :deep(.q-field .q-field__bottom) {
  position: static;
  transform: none;
  min-height: 0;
  padding: 3px 0 2px;
  line-height: 1.25;
  overflow: visible;
}
.irodori-settings :deep(.q-field .q-field__bottom .q-field__messages) {
  line-height: 1.25;
}
.irodori-settings :deep(.irodori-cfg-row .q-field__bottom) {
  padding: 3px 2px 2px;
}
.irodori-settings :deep(.irodori-cfg-row) {
  align-items: stretch;
}
.irodori-strength-control {
  padding: 0 4px;
}
.irodori-strength-label {
  display: flex;
  justify-content: space-between;
  color: rgba(0, 0, 0, 0.72);
  font-size: 0.875rem;
  line-height: 1.25;
}
.irodori-strength-control :deep(.q-slider) {
  margin: 0 4px;
}
.irodori-settings :deep(.irodori-number-input .q-field__native[type="number"]) {
  appearance: textfield;
  -moz-appearance: textfield;
  color-scheme: light;
}
.irodori-settings
  :deep(
    .irodori-number-input
      .q-field__native[type="number"]::-webkit-inner-spin-button
  ),
.irodori-settings
  :deep(
    .irodori-number-input
      .q-field__native[type="number"]::-webkit-outer-spin-button
  ) {
  appearance: none;
  -webkit-appearance: none;
  margin: 0;
}
.irodori-number-stepper {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 2px 0;
}
.irodori-number-stepper :deep(.q-btn) {
  width: 24px;
  min-width: 24px;
  height: 15px;
  min-height: 15px;
  padding: 0;
  border: 1px solid rgba(108, 99, 255, 0.42);
  border-radius: 4px;
  color: #6c63ff;
  background: rgba(108, 99, 255, 0.08);
  transition:
    background-color 120ms ease,
    color 120ms ease,
    transform 120ms ease;
}
.irodori-number-stepper :deep(.q-btn:hover) {
  background: #6c63ff;
  color: #fff;
}
.irodori-number-stepper :deep(.q-btn:active) {
  transform: scale(0.92);
}
.irodori-number-stepper :deep(.q-icon) {
  font-size: 13px;
}
:global(:root[is-dark-theme="true"])
  .irodori-settings
  :deep(.irodori-number-input .q-field__native[type="number"]) {
  color-scheme: dark;
}
:global(:root[is-dark-theme="true"]) .irodori-strength-label {
  color: rgba(255, 255, 255, 0.82);
}
:global(:root[is-dark-theme="true"]) .irodori-number-stepper :deep(.q-btn) {
  border-color: rgba(168, 161, 255, 0.58);
  color: #b0aaff;
  background: rgba(168, 161, 255, 0.12);
}
:global(:root[is-dark-theme="true"])
  .irodori-number-stepper
  :deep(.q-btn:hover) {
  background: #8c83ff;
  color: #1c1c28;
}
</style>
