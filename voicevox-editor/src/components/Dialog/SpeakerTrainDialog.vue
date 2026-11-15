<template>
  <QDialog
    v-model="dialogOpened"
    maximized
    allowFocusOutside
    transitionShow="jump-up"
    transitionHide="jump-down"
    class="speaker-train-dialog transparent-backdrop"
  >
    <QLayout container view="hHh Lpr fFf" class="bg-background">
      <QPageContainer>
        <QHeader class="q-pa-sm">
          <QToolbar>
            <QToolbarTitle class="text-display"
              >話者学習（音声から話者を作る）</QToolbarTitle
            >
            <QSpace />
            <QBtn
              round
              flat
              icon="close"
              color="display"
              aria-label="話者学習を閉じる"
              @click="dialogOpened = false"
            />
          </QToolbar>
        </QHeader>
        <QPage>
          <div class="container">
            <BaseScrollArea>
              <div
                v-for="engine in engines"
                :key="engine.engineId"
                class="train-card"
              >
                <h5 v-if="engines.length > 1" class="headline">
                  {{ engine.name }}
                </h5>
                <IrodoriSpeakerTraining
                  :endpoint="engine.endpoint"
                  :engineId="engine.engineId"
                  :currentModel="currentModels[engine.engineId]"
                  :disable="locked"
                />
              </div>
            </BaseScrollArea>
          </div>
        </QPage>
      </QPageContainer>
    </QLayout>
  </QDialog>
</template>

<script setup lang="ts">
import { computed, ref, watch } from "vue";
import BaseScrollArea from "@/components/Base/BaseScrollArea.vue";
import IrodoriSpeakerTraining from "@/components/Talk/IrodoriSpeakerTraining.vue";
import { createEngineUrl } from "@/domain/url";
import { fetchIrodoriStatusQuietly } from "@/helpers/irodoriEngine";
import { useStore } from "@/store";
import type { EngineId } from "@/type/preload";

const dialogOpened = defineModel<boolean>("dialogOpened");
const store = useStore();

const engines = computed(() =>
  store.state.engineIds.map((engineId) => {
    const info = store.state.engineInfos[engineId];
    return {
      engineId,
      name: info.name,
      endpoint: createEngineUrl({
        ...info,
        port: store.state.altPortInfos[engineId] ?? info.defaultPort,
      }),
    };
  }),
);
// このダイアログ自身が UI を 1 つロックしているので、それより多いときだけ止める
const locked = computed(() => store.state.uiLockCount > 1);

// ベースモデルの初期値に使う、エディタで選んでいるモデル
const currentModels = ref<Partial<Record<EngineId, string>>>({});
watch(dialogOpened, async (opened) => {
  if (!opened) return;
  for (const engine of engines.value) {
    const status = await fetchIrodoriStatusQuietly(engine.endpoint);
    if (status) currentModels.value[engine.engineId] = status.settings.model;
  }
});
</script>

<style scoped lang="scss">
@use "@/styles/v2/colors" as colors-v2;
@use "@/styles/v2/variables" as vars;
@use "@/styles/v2/mixin" as mixin;

.container {
  position: absolute;
  left: 0;
  right: 0;
  height: 100%;
  background-color: colors-v2.$background;
}

.headline {
  @include mixin.headline-2;
}

:global(.speaker-train-dialog) {
  z-index: vars.$z-index-fullscreen-dialog !important;
}

.train-card {
  margin: auto;
  max-width: 960px;
  padding: vars.$padding-2;
}

.speaker-train-dialog .q-layout-container :deep(.absolute-full) {
  right: 0 !important;
  .scroll {
    left: unset !important;
    right: unset !important;
    width: unset !important;
    max-height: unset;
  }
}
</style>
