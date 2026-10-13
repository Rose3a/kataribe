<!--
声づくり: このセリフで生成した声を話者として保存する（エンジンの voice_design.py）。
キャプション（と話者なし・任意の話者）で生成して気に入ったら、名前を付けて保存する。
行の話者が声づくりの声なら、ここで名前・アイコンの変更と削除もできる。
-->
<template>
  <div class="voice-design">
    <div class="settings-label">声づくり</div>
    <div v-if="currentVoice" class="current-voice">
      <div v-if="renaming" class="row items-center no-wrap q-gutter-xs">
        <QInput
          v-model="renameText"
          dense
          outlined
          autofocus
          :maxlength="32"
          class="col"
          aria-label="声の新しい名前"
          @keydown.enter="rename"
          @keydown.esc="renaming = false"
        />
        <QBtn flat dense icon="check" aria-label="名前を確定" @click="rename" />
      </div>
      <div v-else class="row items-center no-wrap">
        <span class="col current-name"
          >声づくりの声「{{ currentVoice.name }}」</span
        >
        <QBtn
          flat
          dense
          round
          icon="more_vert"
          :disable="busy || locked"
          aria-label="この声の操作"
        >
          <QMenu>
            <QList dense style="min-width: 160px">
              <QItem v-close-popup clickable @click="startRename">
                <QItemSection>名前を変える</QItemSection>
              </QItem>
              <QItem v-close-popup clickable @click="changeIcon">
                <QItemSection>アイコンを変える</QItemSection>
              </QItem>
              <QItem v-close-popup clickable @click="deleteOpen = true">
                <QItemSection class="text-warning">削除する</QItemSection>
              </QItem>
            </QList>
          </QMenu>
        </QBtn>
      </div>
      <div v-if="currentVoice.caption" class="text-caption current-caption">
        作ったときのキャプション: {{ currentVoice.caption }}
      </div>
    </div>
    <div class="text-caption">
      このセリフで最後に生成した声を、話者として保存できます。キャプションは「落ち着いた男性の声で、低めのトーンでゆっくり読み上げてください。」のように「〜の声で、〜読み上げてください」の形にすると効きやすくなります。
    </div>
    <div class="row items-center q-gutter-sm save-row">
      <QSelect
        v-model="target"
        outlined
        dense
        emitValue
        mapOptions
        :options="targetOptions"
        class="col-12"
        aria-label="保存のしかた"
      />
      <QInput
        v-if="target == null"
        v-model="saveName"
        outlined
        dense
        :maxlength="32"
        placeholder="新しい話者の名前"
        class="col"
        aria-label="新しい話者の名前"
      />
      <QBtn
        color="primary"
        unelevated
        :label="target == null ? '保存' : '差し替え'"
        :loading="busy"
        :disable="!canSave"
        @click="save"
      />
    </div>
    <QCheckbox
      v-if="target == null"
      v-model="switchAfterSave"
      dense
      label="保存したら、このセリフの話者にする"
    />
    <div v-if="hasReferenceAudio" class="text-caption">
      音声参照を使うセリフの声は保存できません。
    </div>
    <div
      v-if="status"
      class="text-caption"
      :class="{ 'text-negative': failed }"
    >
      {{ status }}
    </div>
    <!-- 削除の確認（SHOW_WARNING_DIALOG は reka-ui で、Quasar のダイアログと重なると
         フォーカスを取り合うので、ここでは QDialog を使う） -->
    <QDialog v-model="deleteOpen">
      <QCard style="min-width: 320px; max-width: 480px">
        <QCardSection class="text-h6"
          >「{{ currentVoice?.name }}」を削除しますか？</QCardSection
        >
        <QCardSection class="q-pt-none">
          保存した声のファイルを消します。元に戻せません。この声を使っているセリフは、話者を選び直す必要があります。
        </QCardSection>
        <QCardActions align="right">
          <QBtn v-close-popup flat label="削除しない" />
          <QBtn
            v-close-popup
            outline
            color="warning"
            label="削除する"
            @click="remove"
          />
        </QCardActions>
      </QCard>
    </QDialog>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { useStore } from "@/store";
import { createEngineUrl } from "@/domain/url";
import {
  irodoriRequest,
  refreshIrodoriSpeakers,
} from "@/helpers/irodoriEngine";
import { lineSynthesisRequest } from "@/store/audioGenerate";
import type { AudioKey, EngineId, Voice } from "@/type/preload";

type SavedVoice = {
  id: string;
  name: string;
  caption: string;
  speakerUuid: string;
};

const props = defineProps<{ engineId: EngineId; activeAudioKey: AudioKey }>();
const store = useStore();

const audioItem = computed(() => store.state.audioItems[props.activeAudioKey]);
const locked = computed(() => store.getters.UI_LOCKED);
const endpoint = computed(() => {
  const info = store.state.engineInfos[props.engineId];
  if (!info) return undefined;
  return createEngineUrl({
    ...info,
    port: store.state.altPortInfos[props.engineId] ?? info.defaultPort,
  });
});

const voices = ref<SavedVoice[]>([]);
const target = ref<string | null>(null);
const saveName = ref("");
const switchAfterSave = ref(true);
const renaming = ref(false);
const renameText = ref("");
const deleteOpen = ref(false);
const busy = ref(false);
const status = ref("");
const failed = ref(false);

const hasReferenceAudio = computed(
  () => audioItem.value?.irodori?.referenceAudio != undefined,
);
const currentVoice = computed(() =>
  voices.value.find(
    (voice) => voice.speakerUuid === audioItem.value?.voice.speakerId,
  ),
);
const targetOptions = computed(() => [
  { label: "新しい話者として保存", value: null },
  ...voices.value.map((voice) => ({
    label: `「${voice.name}」の声をこのセリフの声に差し替え`,
    value: voice.id,
  })),
]);
const canSave = computed(
  () =>
    !busy.value &&
    !locked.value &&
    !hasReferenceAudio.value &&
    audioItem.value?.query != undefined &&
    (target.value != null || saveName.value.trim() !== ""),
);

function report(message: string, isError = false) {
  status.value = message;
  failed.value = isError;
}

async function responseError(response: Response) {
  try {
    const body = (await response.json()) as { detail?: string };
    return body.detail ?? `通信エラー (${response.status})`;
  } catch {
    return `通信エラー (${response.status})`;
  }
}

async function post(path: string, body: Record<string, unknown>) {
  if (!endpoint.value) throw new Error("エンジンに接続できません");
  const response = await irodoriRequest(endpoint.value, path, {
    method: "POST",
    body,
    timeoutMs: 600000,
  });
  if (!response.ok) throw new Error(await responseError(response));
  return (await response.json()) as Record<string, unknown>;
}

async function loadVoices() {
  if (!endpoint.value) return;
  const response = await irodoriRequest(endpoint.value, "/irodori/voices");
  if (!response.ok) return;
  voices.value = (await response.json()) as SavedVoice[];
  if (target.value && !voices.value.some((v) => v.id === target.value))
    target.value = null;
}

async function reloadSpeakers() {
  if (!endpoint.value) return;
  await refreshIrodoriSpeakers(endpoint.value);
  await store.actions.LOAD_CHARACTER({ engineId: props.engineId });
  await loadVoices();
}

async function run(task: () => Promise<void>) {
  busy.value = true;
  try {
    await task();
  } catch (cause) {
    report(cause instanceof Error ? cause.message : String(cause), true);
  } finally {
    busy.value = false;
  }
}

function voiceFor(speakerUuid: string): Voice | undefined {
  const character = (store.state.characterInfos[props.engineId] ?? []).find(
    (item) => item.metas.speakerUuid === speakerUuid,
  );
  const style = character?.metas.styles[0];
  if (!character || !style) return undefined;
  return {
    engineId: props.engineId,
    speakerId: character.metas.speakerUuid,
    styleId: style.styleId,
  };
}

async function save() {
  const item = audioItem.value;
  if (!item) return;
  await run(async () => {
    report("保存中…（声のエンコードをしています）");
    const line = await lineSynthesisRequest(store.state, item);
    if (target.value != null) {
      const voice = voices.value.find((v) => v.id === target.value);
      await post("/irodori/voices/update", { id: target.value, ...line });
      await reloadSpeakers();
      report(`「${voice?.name ?? ""}」の声を差し替えました`);
      target.value = null;
      return;
    }
    const name = saveName.value.trim();
    const result = await post("/irodori/voices/save", { name, ...line });
    await reloadSpeakers();
    saveName.value = "";
    const voice =
      typeof result.speakerUuid === "string"
        ? voiceFor(result.speakerUuid)
        : undefined;
    if (switchAfterSave.value && voice) {
      await store.actions.COMMAND_MULTI_CHANGE_VOICE({
        audioKeys: [props.activeAudioKey],
        voice,
      });
      // 声づくりの声は追加話者と組み合わせられない
      const irodori = store.state.audioItems[props.activeAudioKey]?.irodori;
      if (irodori?.additionalSpeakers?.length) {
        await store.actions.COMMAND_SET_IRODORI_SETTINGS({
          audioKey: props.activeAudioKey,
          irodori: { ...irodori, additionalSpeakers: [] },
        });
      }
    }
    report(`「${name}」を話者として保存しました`);
  });
}

function startRename() {
  if (!currentVoice.value) return;
  renameText.value = currentVoice.value.name;
  renaming.value = true;
}

async function rename() {
  const voice = currentVoice.value;
  const name = renameText.value.trim();
  renaming.value = false;
  if (!voice || !name || name === voice.name) return;
  await run(async () => {
    await post("/irodori/voices/update", { id: voice.id, name });
    await reloadSpeakers();
    report(`名前を「${name}」に変えました`);
  });
}

function changeIcon() {
  const voice = currentVoice.value;
  if (!voice) return;
  const input = document.createElement("input");
  input.type = "file";
  input.accept = "image/png,image/jpeg,image/webp";
  input.onchange = () => {
    const file = input.files?.[0];
    if (!file) return;
    if (file.size > 2_000_000) {
      report("アイコンの画像は2MB以下にしてください", true);
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result !== "string") return;
      const icon = reader.result;
      void run(async () => {
        await post("/irodori/voices/update", { id: voice.id, icon });
        await reloadSpeakers();
        report(`「${voice.name}」のアイコンを変えました`);
      });
    };
    reader.readAsDataURL(file);
  };
  input.click();
}

async function remove() {
  const voice = currentVoice.value;
  if (!voice) return;
  await run(async () => {
    await post("/irodori/voices/delete", { id: voice.id });
    await reloadSpeakers();
    report(
      `「${voice.name}」を削除しました。このセリフの話者を選び直してください`,
    );
  });
}

watch(
  () => [props.engineId, props.activeAudioKey],
  () => {
    status.value = "";
    renaming.value = false;
    void loadVoices().catch(() => undefined);
  },
  { immediate: true },
);
</script>

<style scoped lang="scss">
.voice-design {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.settings-label {
  font-size: 0.85rem;
  font-weight: 700;
}
.current-name {
  font-weight: 600;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.current-caption {
  opacity: 0.75;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.save-row {
  margin-left: 0;
}
</style>
