<!--
タイトルバーに配置される、エディタを切り替えるボタン
-->

<template>
  <!-- FIXME: 画面サイズが小さくなると表示が崩れるのを直す -->
  <!-- NOTE: デザインしづらいからQBtnかdivの方が良い -->
  <QBtnToggle
    :modelValue="openedEditor"
    unelevated
    :disable="uiLocked"
    dense
    toggleColor="primary"
    :options="
      isIrodori
        ? [{ label: 'トーク', value: 'talk' }]
        : [
            { label: 'トーク', value: 'talk' },
            { label: 'ソング（非対応）', value: 'song', disable: true },
          ]
    "
    @update:modelValue="switchEditor"
  />
  <QBtn
    v-if="isIrodori"
    dense
    flat
    label="話者マージ"
    :disable="uiLocked || !mixEndpoint"
    @click="mixOpen = true"
  />
  <QDialog
    v-if="isIrodori && mixEngineId && mixEndpoint"
    v-model="mixOpen"
    maximized
  >
    <SpeakerMixEditor
      v-if="mixOpen"
      :engine-id="mixEngineId"
      :endpoint="mixEndpoint"
      @close="mixOpen = false"
    />
  </QDialog>
</template>

<script setup lang="ts">
import { computed, ref } from "vue";
import { useStore } from "@/store";
import type { EditorType } from "@/type/preload";
import { createEngineUrl } from "@/domain/url";
import SpeakerMixEditor from "@/components/SpeakerMixEditor.vue";

const store = useStore();

const openedEditor = computed(() => store.state.openedEditor);
const uiLocked = computed(() => store.getters.UI_LOCKED);
const isIrodori = import.meta.env.VITE_APP_NAME === "voicevox-irodori";
const mixOpen = ref(false);
const mixEngineId = computed(
  () =>
    Object.values(store.state.engineInfos).find((info) => info.isDefault)?.uuid,
);
const mixEndpoint = computed(() => {
  const engineId = mixEngineId.value;
  if (!engineId) return undefined;
  const info = store.state.engineInfos[engineId];
  return createEngineUrl({
    ...info,
    port: store.state.altPortInfos[engineId] ?? info.defaultPort,
  });
});

const switchEditor = async (editor: EditorType) => {
  await store.actions.SET_ROOT_MISC_SETTING({
    key: "openedEditor",
    value: editor,
  });
};
</script>

<style scoped lang="scss">
@use "@/styles/variables" as vars;
@use "@/styles/colors" as colors;
.q-btn-group {
  :deep(.q-btn) {
    padding-left: 0.75rem;
    padding-right: 0.75rem;
  }

  // 選択されているボタンの文字を太字にする
  :deep(.q-btn[aria-pressed="true"]) {
    span {
      font-weight: 700;
      color: colors.$display-on-primary !important;
    }
  }
}
</style>
