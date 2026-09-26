<template>
  <section class="irodori-settings q-pa-md">
    <div v-if="audioItem" class="settings-stack">
      <h2 class="settings-title">セリフの設定</h2>

      <QCard flat bordered class="settings-card">
        <div class="settings-card-heading">
          <span>話者と声質</span>
          <small>追加 {{ additionalSpeakersValue.length }}/3人</small>
        </div>
        <div class="settings-card-body">
          <div class="settings-label">基本の話者</div>
          <div class="primary-voice">
            <CharacterButton
              :selected-voice="audioItem.voice"
              :character-infos="store.state.characterInfos[engineId] ?? []"
              :show-label="true"
              :ui-locked="locked"
              @update:selected-voice="selectPrimaryVoice"
            />
          </div>
          <div class="strength-row primary-strength-row">
            <label>話者の強度</label>
            <QSlider
              v-model="speakerStrengthValue"
              dense
              snap
              color="primary"
              :min="0"
              :max="1"
              :step="0.01"
              :disable="locked || !!referenceAudioName"
              aria-label="基本の話者の強度"
              @change="saveStrength('speakerStrength', speakerStrengthValue)"
            />
            <QInput
              :model-value="speakerStrengthValue"
              type="number"
              dense
              outlined
              :min="0"
              :max="1"
              :step="0.01"
              :disable="locked || !!referenceAudioName"
              aria-label="基本の話者の強度の数値"
              @change="saveStrength('speakerStrength', Number($event))"
            />
          </div>
          <QSeparator spaced />
          <div class="settings-label">追加する話者</div>
          <div
            v-for="(entry, index) in additionalSpeakersValue"
            :key="`${entry.styleId}-${index}`"
            class="additional-speaker-row"
          >
            <CharacterButton
              :selected-voice="additionalVoice(entry.styleId)"
              :character-infos="additionalCharacterInfos(index)"
              :emptiable="true"
              :show-label="true"
              :ui-locked="locked || !!referenceAudioName"
              @update:selected-voice="selectAdditionalSpeaker(index, $event)"
            />
            <div class="additional-strength">
              <QSlider
                v-model="entry.strength"
                dense
                snap
                color="primary"
                :min="0"
                :max="1"
                :step="0.01"
                :disable="locked || !!referenceAudioName"
                :aria-label="`追加話者${index + 1}の強度`"
                @change="saveAdditionalStrength(index, entry.strength)"
              />
            </div>
            <QInput
              :model-value="entry.strength"
              type="number"
              dense
              outlined
              :min="0"
              :max="1"
              :step="0.01"
              :disable="locked || !!referenceAudioName"
              :aria-label="`追加話者${index + 1}の強度の数値`"
              @change="saveAdditionalStrength(index, Number($event))"
            />
            <QBtn
              flat
              round
              dense
              icon="close"
              :disable="locked"
              :aria-label="`追加話者${index + 1}を削除`"
              @click="selectAdditionalSpeaker(index, undefined)"
            />
          </div>
          <div
            v-if="additionalSpeakersValue.length < 3"
            class="add-speaker-row"
          >
            <CharacterButton
              :selected-voice="undefined"
              :character-infos="
                additionalCharacterInfos(additionalSpeakersValue.length)
              "
              :emptiable="true"
              :show-label="true"
              placeholder-label="話者を追加"
              :ui-locked="locked || !!referenceAudioName"
              @update:selected-voice="
                selectAdditionalSpeaker(additionalSpeakersValue.length, $event)
              "
            />
          </div>
          <p class="settings-hint">
            基本の話者と追加話者の強度は、このセリフにだけ適用されます。
          </p>
          <p v-if="referenceAudioName" class="settings-hint">
            音声参照を解除すると、話者の強度と追加話者を編集できます。
          </p>
        </div>
      </QCard>

      <QCard flat bordered class="settings-card">
        <div class="settings-card-heading">話し方</div>
        <div class="settings-card-body">
          <div class="strength-row speed-row">
            <label>話速</label>
            <QSlider
              dense
              snap
              color="primary"
              :min="speedScaleSlider.qSliderProps.min.value"
              :max="speedScaleSlider.qSliderProps.max.value"
              :step="speedScaleSlider.qSliderProps.step.value"
              :disable="speedScaleSlider.qSliderProps.disable.value"
              :model-value="speedScaleSlider.qSliderProps.modelValue.value"
              @update:model-value="
                speedScaleSlider.qSliderProps['onUpdate:modelValue']
              "
              @change="speedScaleSlider.qSliderProps.onChange"
              @wheel="speedScaleSlider.qSliderProps.onWheel"
              @pan="speedScaleSlider.qSliderProps.onPan"
            />
            <QInput
              dense
              outlined
              :disable="speedScaleSlider.qSliderProps.disable.value"
              :model-value="
                speedScaleSlider.state.currentValue.value != undefined
                  ? speedScaleSlider.state.currentValue.value.toFixed(2)
                  : speedScaleSlider.qSliderProps.min.value.toFixed(2)
              "
              aria-label="話速の数値"
              @change="handleSpeedScaleChange"
            />
          </div>
          <p v-if="speedTargetCount > 1" class="settings-hint">
            話速は選択中の{{ speedTargetCount }}行すべてに適用されます。
          </p>
          <QFile
            v-model="referenceFile"
            outlined
            dense
            clearable
            accept="audio/*"
            label="音声参照（任意）"
            hint="この行の話者・声質の参考音声。最大10MB"
            :disable="locked || additionalSpeakersValue.length > 0"
            @update:model-value="handleReferenceFileChange"
          />
          <p v-if="additionalSpeakersValue.length > 0" class="settings-hint">
            音声参照は追加話者と同時に使えません。
          </p>
          <div v-if="referenceAudioName" class="row items-center q-gutter-sm">
            <span class="text-caption">適用中: {{ referenceAudioName }}</span>
            <QBtn
              flat
              dense
              label="解除"
              icon="clear"
              :disable="locked"
              @click="clearReferenceAudio"
            />
            <QBtn
              flat
              dense
              :label="referenceAudioPlaying ? '停止' : '再生'"
              :icon="referenceAudioPlaying ? 'stop' : 'play_arrow'"
              :aria-label="
                referenceAudioPlaying ? '音声参照を停止' : '音声参照を再生'
              "
              @click="
                referenceAudioPlaying
                  ? stopReferenceAudio()
                  : playReferenceAudio()
              "
            />
          </div>
          <div v-if="referenceAudioName" class="strength-row">
            <label>音声参照の強度</label>
            <QSlider
              v-model="referenceStrengthValue"
              dense
              snap
              color="primary"
              :min="0"
              :max="1"
              :step="0.1"
              :disable="locked"
              aria-label="音声参照の強度"
              @change="
                saveStrength('referenceStrength', referenceStrengthValue)
              "
            />
            <span class="strength-value">{{
              referenceStrengthValue.toFixed(1)
            }}</span>
          </div>
          <QInput
            v-model="captionText"
            outlined
            dense
            autogrow
            type="textarea"
            label="キャプション（任意）"
            hint="場面・話し方・感情など。空欄なら未指定"
            :maxlength="2000"
            counter
            :disable="locked"
            @update:model-value="saveCaption"
          />
          <div v-if="captionText.trim()" class="strength-row">
            <label>キャプションの強度</label>
            <QSlider
              v-model="captionStrengthValue"
              dense
              snap
              color="primary"
              :min="0"
              :max="1"
              :step="0.1"
              :disable="locked"
              aria-label="キャプションの強度"
              @change="saveStrength('captionStrength', captionStrengthValue)"
            />
            <span class="strength-value">{{
              captionStrengthValue.toFixed(1)
            }}</span>
          </div>
          <QSeparator spaced />
          <div class="settings-label">読み方</div>
          <QSelect
            v-model="englishReadingValue"
            outlined
            dense
            label="英単語・英文"
            :options="englishReadings"
            emit-value
            map-options
            hint="API → エーピーアイ、server → サーバー。ユーザー辞書の登録が優先です"
            :disable="locked"
            @update:model-value="saveEnglishReading"
          />
          <QSelect
            v-model="englishSpacingValue"
            outlined
            dense
            label="英単語の区切り"
            :options="englishSpacings"
            emit-value
            map-options
            :hint="
              englishReadingValue === 'off'
                ? '英字のまま読むときは使われません'
                : 'つなげると、語ごとに区切らず続けて読みます（アイ ラブ ユー → アイラブユー）'
            "
            :disable="locked || englishReadingValue === 'off'"
            @update:model-value="saveEnglishSpacing"
          />
          <QSelect
            v-model="kanaStyleValue"
            outlined
            dense
            label="文中のカタカナ語"
            :options="kanaStyles"
            emit-value
            map-options
            hint="カタカナ語で言いよどむときは、ひらがなにすると改善する場合があります"
            :disable="locked"
            @update:model-value="saveKanaStyle"
          />
        </div>
      </QCard>

      <QExpansionItem
        v-model="advancedExpanded"
        label="詳細設定"
        caption="ステップ・Schedule・音声長・シード・CFG"
        icon="tune"
        header-class="advanced-heading"
        class="settings-card advanced-settings"
      >
        <div class="settings-card-body">
          <QInput
            v-model.number="stepsValue"
            outlined
            dense
            type="number"
            label="ステップ数 (1〜80)"
            :hint="
              showStepsQualityWarning
                ? `既定値 ${defaultSteps} より少ないため、音声の質が下がる場合があります`
                : `既定値: ${defaultSteps}（このモデル）。多いほど高品質・低速`
            "
            :min="1"
            :max="80"
            :step="1"
            :disable="locked"
            class="irodori-number-input"
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
                <QTooltip
                  >ステップ数がデフォルト未満のため、音声の質が低い可能性があります。</QTooltip
                >
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
            emit-value
            map-options
            :hint="
              meanflow
                ? 'MeanFlowモデルでは使われません'
                : 'ステップの刻み方。sway は生成の序盤を細かく刻む（既定）、linear は均等'
            "
            :disable="locked || meanflow"
            @update:model-value="saveSchedule"
          />
          <QSelect
            v-model="secondsValue"
            outlined
            dense
            label="音声長"
            :options="durations"
            emit-value
            map-options
            :disable="locked"
            @update:model-value="saveSeconds"
          />
          <QInput
            v-model="seedText"
            outlined
            dense
            label="シード（空欄＝ランダム）"
            type="number"
            :disable="locked"
            class="irodori-number-input"
            @change="saveSeed"
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
              :disable="locked || meanflow"
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
              :disable="locked || meanflow"
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
              :disable="locked || meanflow"
              class="col-4 irodori-number-input"
              @change="saveLineCfg('cfgSpeaker', cfgSpeakerText)"
            />
          </div>
          <p class="settings-hint">
            <template v-if="meanflow">
              MeanFlowモデルではCFGは使われません。
            </template>
            <template v-else>
              CFGは、テキスト・キャプション・話者にどれだけ忠実に従うかの強さです。
              上げるほど指定に沿いますが、上げすぎると不自然になることがあります。
            </template>
          </p>
        </div>
      </QExpansionItem>

      <div v-if="error" role="alert" class="text-negative text-caption">
        {{ error }}
      </div>
      <div v-if="lineError" role="alert" class="text-negative text-caption">
        {{ lineError }}
      </div>
      <div v-if="!canPlayActiveLine" class="text-caption">
        セリフを入力すると、この行を生成できます
      </div>
      <div v-if="linePlaying" class="text-caption">
        この行を生成中。停止は上のボタンから
      </div>
      <div class="settings-action">
        <QBtn
          color="primary"
          icon="play_arrow"
          label="この行を生成して再生"
          :loading="linePlaying"
          :disable="locked || !canPlayActiveLine"
          class="full-width"
          @click="playActiveLine"
        />
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from "vue";
import { useStore } from "@/store";
import CharacterButton from "@/components/CharacterButton.vue";
import { createEngineUrl } from "@/domain/url";
import { fetchIrodoriModelInfo } from "@/helpers/irodoriEngine";
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
  IRODORI_DEFAULT_SPEAKER_STRENGTH,
  IRODORI_DEFAULT_SEED,
  IRODORI_DEFAULT_SCHEDULE,
  IRODORI_DEFAULT_ENGLISH_READING,
  IRODORI_DEFAULT_KANA_STYLE,
  IRODORI_DEFAULT_ENGLISH_SPACING,
  type IrodoriEnglishReading,
  type IrodoriEnglishSpacing,
  type IrodoriKanaStyle,
  irodoriDefaultSteps,
  irodoriMeanflow,
  IRODORI_DEFAULT_STEPS,
} from "@/domain/irodori";
import {
  previewSliderHelper,
  type PreviewSliderHelper,
} from "@/helpers/previewSliderHelper";
import type { EngineManifest } from "@/openapi";
import { SLIDER_PARAMETERS } from "@/store/utility";
import type { AudioKey, EngineId, Voice } from "@/type/preload";

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
    const modelInfo = await fetchIrodoriModelInfo(
      createEngineUrl({
        ...info,
        port: store.state.altPortInfos[engineId] ?? info.defaultPort,
      }),
    );
    if (cancelled || modelInfo == undefined) return;
    irodoriDefaultSteps.value = modelInfo.defaultSteps;
    irodoriMeanflow.value = modelInfo.meanflow;
  },
  { immediate: true },
);
const irodori = computed(() => {
  const value = audioItem.value?.irodori;
  return {
    ...(value ?? { seed: IRODORI_DEFAULT_SEED }),
    steps: value?.steps ?? defaultSteps.value,
    schedule: value?.schedule ?? IRODORI_DEFAULT_SCHEDULE,
    englishReading: value?.englishReading ?? IRODORI_DEFAULT_ENGLISH_READING,
    kanaStyle: value?.kanaStyle ?? IRODORI_DEFAULT_KANA_STYLE,
    englishSpacing: value?.englishSpacing ?? IRODORI_DEFAULT_ENGLISH_SPACING,
    seconds: value?.seconds ?? null,
    cfgText: value?.cfgText ?? IRODORI_DEFAULT_CFG_TEXT,
    cfgCaption: value?.cfgCaption ?? IRODORI_DEFAULT_CFG_CAPTION,
    cfgSpeaker: value?.cfgSpeaker ?? IRODORI_DEFAULT_CFG_SPEAKER,
    captionStrength: value?.captionStrength ?? IRODORI_DEFAULT_CAPTION_STRENGTH,
    referenceStrength:
      value?.referenceStrength ?? IRODORI_DEFAULT_REFERENCE_STRENGTH,
    speakerStrength: value?.speakerStrength ?? IRODORI_DEFAULT_SPEAKER_STRENGTH,
    additionalSpeakers:
      value?.additionalSpeakers ??
      (value?.secondarySpeakerStyleId != null
        ? [
            {
              styleId: value.secondarySpeakerStyleId,
              strength: value.secondarySpeakerStrength ?? 0.5,
            },
          ]
        : []),
  };
});
type AdditionalSpeaker = { styleId: number; strength: number };
const additionalSpeakersValue = ref<AdditionalSpeaker[]>([]);
async function selectPrimaryVoice(voice: Voice | undefined) {
  if (!voice) return;
  if (
    audioItem.value?.voice.engineId === voice.engineId &&
    audioItem.value.voice.styleId === voice.styleId
  ) return;
  try {
    await store.actions.COMMAND_MULTI_CHANGE_VOICE({
      audioKeys: [props.activeAudioKey],
      voice,
    });
    error.value = "";
    const remaining = additionalSpeakersValue.value.filter(
      (entry) => entry.styleId !== voice.styleId,
    );
    if (remaining.length !== additionalSpeakersValue.value.length) {
      saveAdditionalSpeakers(remaining);
    }
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : String(cause);
  }
}
function additionalCharacterInfos(index: number) {
  const selectedElsewhere = new Set(
    additionalSpeakersValue.value
      .filter((_, entryIndex) => entryIndex !== index)
      .map((entry) => entry.styleId),
  );
  return (store.state.characterInfos[props.engineId] ?? []).filter(
    (character) =>
      character.metas.speakerName !== "話者なし" &&
      character.metas.styles.some(
        (style) =>
          style.styleId !== audioItem.value?.voice.styleId &&
          !selectedElsewhere.has(style.styleId),
      ),
  );
}
function additionalVoice(styleId: number): Voice | undefined {
  const character = (store.state.characterInfos[props.engineId] ?? []).find(
    (item) => item.metas.styles.some((style) => style.styleId === styleId),
  );
  const style = character?.metas.styles.find(
    (item) => item.styleId === styleId,
  );
  if (!character || !style) return undefined;
  return {
    engineId: props.engineId,
    speakerId: character.metas.speakerUuid,
    styleId: style.styleId,
  };
}
const seedText = ref("");
const stepsValue = ref(IRODORI_DEFAULT_STEPS);
const scheduleValue = ref<"linear" | "sway">(IRODORI_DEFAULT_SCHEDULE);
const englishReadingValue = ref<IrodoriEnglishReading>(
  IRODORI_DEFAULT_ENGLISH_READING,
);
const kanaStyleValue = ref<IrodoriKanaStyle>(IRODORI_DEFAULT_KANA_STYLE);
const englishSpacingValue = ref<IrodoriEnglishSpacing>(
  IRODORI_DEFAULT_ENGLISH_SPACING,
);
const secondsValue = ref<number | null>(null);
const captionText = ref("");
const cfgTextText = ref("");
const cfgCaptionText = ref("");
const cfgSpeakerText = ref("");
const captionStrengthValue = ref(IRODORI_DEFAULT_CAPTION_STRENGTH);
const referenceStrengthValue = ref(IRODORI_DEFAULT_REFERENCE_STRENGTH);
const speakerStrengthValue = ref(IRODORI_DEFAULT_SPEAKER_STRENGTH);
const advancedExpanded = ref(false);
watch(
  irodori,
  (value) => {
    seedText.value = value.seed == null ? "" : String(value.seed);
    stepsValue.value = value.steps;
    scheduleValue.value = value.schedule;
    englishReadingValue.value = value.englishReading;
    kanaStyleValue.value = value.kanaStyle;
    englishSpacingValue.value = value.englishSpacing;
    secondsValue.value = value.seconds;
    captionText.value = value.caption ?? "";
    cfgTextText.value = String(value.cfgText);
    cfgCaptionText.value = String(value.cfgCaption);
    cfgSpeakerText.value = String(value.cfgSpeaker);
    captionStrengthValue.value = value.captionStrength;
    referenceStrengthValue.value = value.referenceStrength;
    speakerStrengthValue.value = value.speakerStrength;
    additionalSpeakersValue.value = value.additionalSpeakers.map((entry) => ({
      ...entry,
    }));
  },
  { immediate: true, deep: true },
);
const referenceFile = ref<File | null>(null);
const referenceAudioName = computed(
  () => irodori.value.referenceAudio?.name ?? "",
);
const referenceAudioPlaying = ref(false);
const referenceAudio = ref<HTMLAudioElement | null>(null);
watch(
  () => props.activeAudioKey,
  () => {
    stopReferenceAudio();
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
// エラー表示では内部のキー名ではなく、画面の欄名を使う。
const lineCfgLabels: Record<LineCfgKey, string> = {
  cfgText: "テキストCFG",
  cfgCaption: "キャプションCFG",
  cfgSpeaker: "スピーカーCFG",
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
function saveEnglishReading(value: IrodoriEnglishReading | null) {
  if (value == null || !englishReadings.some((o) => o.value === value)) return;
  englishReadingValue.value = value;
  void store.actions.COMMAND_SET_IRODORI_SETTINGS({
    audioKey: props.activeAudioKey,
    irodori: { ...irodori.value, englishReading: value },
  });
}
function saveEnglishSpacing(value: IrodoriEnglishSpacing | null) {
  if (value == null || !englishSpacings.some((o) => o.value === value)) return;
  englishSpacingValue.value = value;
  void store.actions.COMMAND_SET_IRODORI_SETTINGS({
    audioKey: props.activeAudioKey,
    irodori: { ...irodori.value, englishSpacing: value },
  });
}
function saveKanaStyle(value: IrodoriKanaStyle | null) {
  if (value == null || !kanaStyles.some((o) => o.value === value)) return;
  kanaStyleValue.value = value;
  void store.actions.COMMAND_SET_IRODORI_SETTINGS({
    audioKey: props.activeAudioKey,
    irodori: { ...irodori.value, kanaStyle: value },
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
    error.value = `${lineCfgLabels[key]}は0〜20の数値だけ入力してください`;
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
// 入力確定時（@change）にだけ保存し、1文字ごとに元に戻す履歴を積まない。
function saveSeed() {
  const raw = seedText.value.trim();
  const seed = raw === "" ? null : Number(raw);
  if (seed != null && !Number.isSafeInteger(seed)) {
    error.value = "シードは整数で入力するか、空欄にしてください";
    seedText.value =
      irodori.value.seed == null ? "" : String(irodori.value.seed);
    return;
  }
  error.value = "";
  if (seed === irodori.value.seed) return;
  void store.actions.COMMAND_SET_IRODORI_SETTINGS({
    audioKey: props.activeAudioKey,
    irodori: { ...irodori.value, seed },
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
type StrengthKey = "captionStrength" | "referenceStrength" | "speakerStrength";
function saveAdditionalSpeakers(entries: AdditionalSpeaker[]) {
  if (
    entries.length > 3 ||
    new Set(entries.map((entry) => entry.styleId)).size !== entries.length ||
    entries.some((entry) => entry.styleId === audioItem.value?.voice.styleId)
  ) {
    error.value = "追加話者は重複なしで最大3人までです";
    return;
  }
  additionalSpeakersValue.value = entries;
  error.value = "";
  clearAudioCache();
  void store.actions.COMMAND_SET_IRODORI_SETTINGS({
    audioKey: props.activeAudioKey,
    irodori: {
      ...irodori.value,
      additionalSpeakers: entries,
      secondarySpeakerStyleId: null,
    },
  });
}
function selectAdditionalSpeaker(index: number, voice: Voice | undefined) {
  const entries = additionalSpeakersValue.value.map((entry) => ({ ...entry }));
  if (!voice) {
    if (index < entries.length) entries.splice(index, 1);
  } else if (index === entries.length && entries.length < 3) {
    entries.push({ styleId: voice.styleId, strength: 0.5 });
  } else if (index < entries.length) {
    entries[index].styleId = voice.styleId;
  }
  saveAdditionalSpeakers(entries);
}
function saveAdditionalStrength(index: number, input: number | null) {
  const strength = Number(input);
  if (!Number.isFinite(strength) || strength < 0 || strength > 1) {
    error.value = "強度は0〜1の数値だけ指定できます";
    additionalSpeakersValue.value = irodori.value.additionalSpeakers.map(
      (entry) => ({ ...entry }),
    );
    return;
  }
  const entries = additionalSpeakersValue.value.map((entry) => ({ ...entry }));
  if (!entries[index]) return;
  entries[index].strength = Number(strength.toFixed(2));
  saveAdditionalSpeakers(entries);
}
function saveStrength(key: StrengthKey, input: number | null) {
  const value = Number(input);
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    error.value = "強度は0〜1の数値だけ指定できます";
    if (key === "captionStrength") {
      captionStrengthValue.value = irodori.value.captionStrength;
    } else if (key === "referenceStrength") {
      referenceStrengthValue.value = irodori.value.referenceStrength;
    } else {
      speakerStrengthValue.value = irodori.value.speakerStrength;
    }
    return;
  }
  error.value = "";
  const normalized = Number(value.toFixed(key === "speakerStrength" ? 2 : 1));
  if (key === "captionStrength") captionStrengthValue.value = normalized;
  else if (key === "referenceStrength")
    referenceStrengthValue.value = normalized;
  else speakerStrengthValue.value = normalized;
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
  stopReferenceAudio();
  referenceFile.value = null;
  void store.actions.COMMAND_SET_IRODORI_SETTINGS({
    audioKey: props.activeAudioKey,
    irodori: { ...irodori.value, referenceAudio: undefined },
  });
  clearAudioCache();
}
async function playReferenceAudio() {
  const dataUrl = irodori.value.referenceAudio?.dataUrl;
  if (dataUrl == undefined) return;
  stopReferenceAudio();
  const audio = new Audio(dataUrl);
  referenceAudio.value = audio;
  audio.addEventListener("ended", () => {
    if (referenceAudio.value !== audio) return;
    referenceAudioPlaying.value = false;
    referenceAudio.value = null;
  });
  audio.addEventListener("error", () => {
    if (referenceAudio.value !== audio) return;
    referenceAudioPlaying.value = false;
    referenceAudio.value = null;
    error.value = "音声リファレンスを再生できませんでした";
  });
  try {
    await audio.play();
    referenceAudioPlaying.value = true;
  } catch {
    if (referenceAudio.value === audio) {
      referenceAudio.value = null;
      error.value = "音声リファレンスを再生できませんでした";
    }
  }
}
function stopReferenceAudio() {
  const audio = referenceAudio.value;
  if (audio == null) {
    referenceAudioPlaying.value = false;
    return;
  }
  audio.pause();
  audio.currentTime = 0;
  referenceAudio.value = null;
  referenceAudioPlaying.value = false;
}
onBeforeUnmount(stopReferenceAudio);
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
// 422 は VOICEVOX と同じく [{ loc, msg }] の配列で返る。
async function engineErrorDetail(cause: unknown): Promise<string> {
  const response = (cause as { response?: Response } | undefined)?.response;
  if (response == undefined || typeof response.clone !== "function") return "";
  try {
    const body = await response.clone().text();
    const detail = (JSON.parse(body) as { detail?: unknown }).detail;
    if (Array.isArray(detail)) {
      return detail
        .map((item: { loc?: unknown[]; msg?: string }) =>
          `${item.loc?.at(-1) ?? ""}: ${item.msg ?? ""}`.trim(),
        )
        .join(" / ");
    }
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
const englishReadings: { label: string; value: IrodoriEnglishReading }[] = [
  { label: "カタカナで読む", value: "katakana" },
  { label: "ひらがなで読む", value: "hiragana" },
  { label: "変換しない（英字のまま）", value: "off" },
];
const englishSpacings: { label: string; value: IrodoriEnglishSpacing }[] = [
  { label: "語ごとに区切る", value: "keep" },
  { label: "つなげて読む", value: "join" },
];
const kanaStyles: { label: string; value: IrodoriKanaStyle }[] = [
  { label: "カタカナのまま", value: "katakana" },
  { label: "ひらがなにする", value: "hiragana" },
];
const locked = computed(() => store.getters.UI_LOCKED);
const showStepsQualityWarning = computed(
  () => stepsValue.value < defaultSteps.value,
);
// MeanFlow モデルでは Schedule と CFG がエンジン側で無効化される。
const meanflow = computed(() => irodoriMeanflow.value);
// 話速だけは VOICEVOX 本体と同じく、複数選択中の全行に適用される。
const speedTargetCount = computed(() => selectedAudioKeys.value.length);
</script>

<style scoped>
.irodori-settings {
  border-left: 3px solid var(--color-primary);
  background: var(--color-background);
  color: var(--color-display);
}
.settings-stack {
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.settings-title {
  margin: 0;
  font-size: 1.1rem;
  font-weight: 700;
  line-height: 1.4;
}
.settings-card {
  overflow: hidden;
  border: 1px solid rgba(var(--color-display-rgb), 0.16);
  border-radius: 10px;
  background: var(--color-surface);
  color: var(--color-display);
}
.settings-card-heading {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
  padding: 12px 14px;
  background: rgba(var(--color-primary-rgb), 0.07);
  font-weight: 700;
}
.settings-card-heading small,
.settings-hint {
  color: rgba(var(--color-display-rgb), 0.65);
  font-size: 0.75rem;
  font-weight: 400;
}
.settings-card-body {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 14px;
}
.settings-label {
  font-size: 0.85rem;
  font-weight: 700;
}
.primary-voice {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
  font-weight: 600;
}
.primary-voice img {
  width: 32px;
  height: 32px;
  object-fit: cover;
  border-radius: 5px;
}
.strength-row,
.additional-speaker-row {
  display: grid;
  grid-template-columns: minmax(112px, 1fr) minmax(88px, 1.3fr) 74px 28px;
  align-items: center;
  gap: 8px;
  min-width: 0;
}
.strength-row label {
  font-size: 0.83rem;
}
.strength-row > :deep(.q-slider) {
  min-width: 0;
}
.strength-row > :deep(.q-field),
.additional-speaker-row > :deep(.q-field) {
  width: 74px;
  min-width: 0;
}
.strength-row :deep(.q-field__native),
.additional-speaker-row :deep(.q-field__native) {
  min-width: 0;
  text-align: center;
  font-variant-numeric: tabular-nums;
}
.primary-strength-row,
.speed-row {
  grid-template-columns: minmax(112px, 1fr) minmax(88px, 1.3fr) 74px;
}
.additional-strength {
  min-width: 0;
}
.additional-speaker-row > :deep(.character-button) {
  width: 100%;
}
.add-speaker-row > :deep(.character-button) {
  width: 100%;
  border-style: dashed;
}
.settings-hint {
  margin: 0;
}
.strength-value {
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.advanced-settings {
  border: 1px solid rgba(var(--color-display-rgb), 0.16);
}
.advanced-settings :deep(.q-item) {
  background: rgba(var(--color-primary-rgb), 0.06);
  color: var(--color-display);
}
.settings-action {
  position: sticky;
  bottom: 0;
  z-index: 2;
  padding: 8px 0;
  background: var(--color-background);
}
.settings-action :deep(.q-btn) {
  min-height: 44px;
}
@media (max-width: 520px) {
  .strength-row,
  .additional-speaker-row {
    grid-template-columns: minmax(90px, 1fr) minmax(70px, 1fr) 64px 24px;
    gap: 5px;
  }
  .primary-strength-row,
  .speed-row {
    grid-template-columns: minmax(90px, 1fr) minmax(70px, 1fr) 64px;
  }
  .strength-row > :deep(.q-field),
  .additional-speaker-row > :deep(.q-field) {
    width: 64px;
  }
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
  border: 1px solid rgba(var(--color-primary-rgb), 0.42);
  border-radius: 4px;
  color: var(--color-primary);
  background: rgba(var(--color-primary-rgb), 0.08);
  transition:
    background-color 120ms ease,
    color 120ms ease,
    transform 120ms ease;
}
.irodori-number-stepper :deep(.q-btn:hover) {
  background: var(--color-primary);
  color: var(--color-display-on-primary);
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
:global(:root[is-dark-theme="true"]) .irodori-number-stepper :deep(.q-btn) {
  border-color: rgba(var(--color-primary-rgb), 0.58);
  color: var(--color-primary);
  background: rgba(var(--color-primary-rgb), 0.12);
}
:global(:root[is-dark-theme="true"])
  .irodori-number-stepper
  :deep(.q-btn:hover) {
  background: var(--color-primary);
  color: var(--color-display-on-primary);
}
</style>
