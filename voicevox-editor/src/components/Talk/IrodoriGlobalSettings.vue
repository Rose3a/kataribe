<template>
  <section class="q-pa-md">
    <div class="text-subtitle1 q-mb-sm">Irodori-TTS</div>
    <div class="text-caption q-mb-sm">全行共通の生成設定</div>
    <template v-if="settings">
      <QSelect
        v-model="settings.backend"
        outlined
        dense
        label="エンジン"
        :options="backends"
        emitValue
        mapOptions
        :disable="locked"
        class="q-mb-sm"
      />
      <QSelect
        v-model="settings.model"
        outlined
        dense
        label="モデル（ファイル名 または Hugging Face repo_id）"
        :options="modelOptions"
        emit-value
        map-options
        use-input
        fill-input
        hide-selected
        new-value-mode="add-unique"
        hint="プリセットから選択、または .safetensors の絶対パスを入力"
        @new-value="setCustomModel"
        :disable="locked"
        class="q-mb-sm"
      >
        <template #option="scope">
          <QItem v-bind="scope.itemProps">
            <QItemSection>
              <QItemLabel>{{ scope.opt.label }}</QItemLabel>
              <QItemLabel caption>{{ scope.opt.description }}</QItemLabel>
            </QItemSection>
          </QItem>
        </template>
      </QSelect>
      <div v-if="modelInfo" class="text-caption q-mb-sm">
        {{ modelInfo.kind === "hf" ? "Hugging Face" : "ローカル" }} ·
        {{ modelInfo.flowParameterization }} · 既定
        {{ modelInfo.defaultSteps }} ステップ
        <span v-if="modelInfo.kind === 'hf' && !modelInfo.downloaded">
          （未ダウンロード: 適用時に取得します）
        </span>
      </div>
      <div v-if="licenseName" class="text-caption q-mb-sm">
        ライセンス:
        <a
          v-if="licenseUrl"
          :href="licenseUrl"
          target="_blank"
          rel="noopener noreferrer"
        >
          {{ licenseName }}
        </a>
        <span v-else>{{ licenseName }}</span>
      </div>
      <div v-if="modelInfo?.meanflow" class="text-caption q-mb-sm">
        MeanFlow モデル: ステップ数4が既定。ScheduleとCFGは未使用。
      </div>
      <QBtn
        color="primary"
        label="設定を適用"
        :loading="busy"
        :disable="locked"
        @click="apply"
      />
      <QBtn flat label="一覧を更新" :disable="locked" @click="refresh" />
      <div class="text-caption q-mt-sm">{{ status }}</div>
      <div v-if="progress.active" class="q-mt-sm">
        <QLinearProgress :value="progress.percent / 100" rounded />
        <div class="text-caption">
          {{ progress.percent }}% - {{ progress.stage }}
        </div>
      </div>
      <div class="row q-mt-sm">
        <QBtn
          flat
          dense
          label="モデルフォルダ"
          :disable="locked"
          @click="openFolder('models')"
        />
        <QBtn
          flat
          dense
          label="話者フォルダ"
          :disable="locked"
          @click="openFolder('speakers')"
        />
      </div>
      <QExpansionItem
        label="モデル・話者の追加 / セットアップ"
        dense
        class="q-mt-sm"
      >
        <div class="text-caption q-pa-sm" style="overflow-wrap: anywhere">
          <p>
            モデルフォルダ: {{ modelFolder }}<br />一覧内の
            .safetensors、または任意の .safetensors 絶対パスを指定できます。
          </p>
          <p>
            話者: {{ speakerFolder }}<br />*.speaker.safetensors
            を置いて一覧を更新します。
          </p>
          <p>
            切り替え後の初回生成時にモデルを読み込みます。前のエンジンは解放します。
          </p>
          <p>
            TensorRTはモデルに対応したGPU用plan、RadeonはDirectML環境が必要です。
          </p>
          <p>
            初回準備はエンジンフォルダの setup_venv.bat。配置手順は同梱の
            IRODORI_EDITOR.md を参照してください。
          </p>
        </div>
      </QExpansionItem>
    </template>
    <div v-if="error" role="alert" class="text-negative text-caption q-mt-sm">
      {{ error }}
    </div>
    <QBtn
      v-if="!settings"
      flat
      label="接続を再試行"
      :disable="locked"
      @click="refresh"
    />
  </section>
</template>
<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import { useStore } from "@/store";
import { createEngineUrl } from "@/domain/url";
import { clearAudioCache } from "@/store/audioGenerate";
import { IRODORI_DEFAULT_STEPS, irodoriDefaultSteps } from "@/domain/irodori";
import type {
  IrodoriModelInfo as ModelInfo,
  IrodoriSettings as Settings,
  IrodoriStatus as Status,
} from "@/domain/irodori";
import {
  fetchIrodoriStatus,
  forgetIrodoriSession,
  openIrodoriFolder,
  refreshIrodoriSpeakers,
  saveIrodoriSettings,
} from "@/helpers/irodoriEngine";
import type { EngineId } from "@/type/preload";
const props = defineProps<{ engineId: EngineId }>();
const store = useStore();
const modelInfo = ref<ModelInfo>();
const defaultSteps = computed(
  () =>
    modelInfo.value?.defaultSteps ??
    irodoriDefaultSteps.value ??
    IRODORI_DEFAULT_STEPS,
);
const settings = ref<Settings>();
watch(defaultSteps, (value) => {
  irodoriDefaultSteps.value = value;
});
const modelFolder = ref("");
const speakerFolder = ref("");
const status = ref("");
const error = ref("");
const busy = ref(false);
const progress = ref<Status["progress"]>({
  active: false,
  percent: 0,
  stage: "idle",
});
// オプション自身が持つロックを除き、生成中や他の処理中は変更を防ぐ。
const locked = computed(
  () =>
    busy.value ||
    store.state.uiLockCount > (store.state.isSettingDialogOpen ? 1 : 0),
);
const backendDefinitions = [
  { label: "CPU / PyTorch", value: "cpu" },
  { label: "NVIDIA / CUDA", value: "cuda" },
  { label: "NVIDIA / TensorRT", value: "trt" },
  { label: "AMD / DirectML", value: "radeon" },
];
const availableBackends = ref<Record<string, boolean>>({ cpu: true });
const backends = computed(() =>
  backendDefinitions.filter(
    (item) => availableBackends.value[item.value] === true,
  ),
);
type ModelOption = {
  label: string;
  value: string;
  description: string;
};
const modelOptions = ref<ModelOption[]>([
  {
    label: "Irodori-TTS v4.1 Small（既定・8ステップ）",
    value: "Aratako/Irodori-TTS-v4.1-Small",
    description: "RFモデル / MIT",
  },
  {
    label: "Irodori-TTS v4.1 Small MF（4ステップ）",
    value: "Aratako/Irodori-TTS-v4.1-Small-MF",
    description: "MeanFlowモデル / MIT",
  },
  {
    label: "Irodori-TTS v4.1 Anime（8ステップ）",
    value: "phasefield-audio/Irodori-TTS-v4.1-Anime",
    description: "Anime fine-tune / MIT",
  },
]);
function ensureModelOption(source: string) {
  const value = source.trim();
  if (!value || modelOptions.value.some((option) => option.value === value)) {
    return;
  }
  modelOptions.value.push({
    label: value,
    value,
    description: "カスタムモデル",
  });
}
const licenseName = computed(
  () =>
    modelInfo.value?.license ??
    (modelOptions.value.some(
      (option) => option.value === settings.value?.model,
    )
      ? "MIT"
      : undefined),
);
const licenseUrl = computed(
  () =>
    modelInfo.value?.licenseUrl ??
    (settings.value?.model.includes("/")
      ? `https://huggingface.co/${settings.value.model}`
      : undefined),
);
function setCustomModel(
  value: string,
  done: () => void,
) {
  const trimmed = value.trim();
  if (!trimmed || !settings.value) return done();
  // QSelect の emit-value と new-value-mode の組み合わせでは、プリセット外の
  // 文字列が次の描画で失われることがある。選択肢と v-model を明示的に更新する。
  ensureModelOption(trimmed);
  settings.value.model = trimmed;
  done();
}
const endpoint = computed(() => {
  const info = store.state.engineInfos[props.engineId];
  return createEngineUrl({
    ...info,
    port: store.state.altPortInfos[props.engineId] ?? info.defaultPort,
  });
});
/** 設定の保存（save=true）または状態の取得。 */
async function loadSettings(save: boolean): Promise<Status> {
  const current = settings.value;
  return save && current != undefined
    ? await saveIrodoriSettings(endpoint.value, current)
    : await fetchIrodoriStatus(endpoint.value);
}

async function openFolder(folder: "models" | "speakers") {
  try {
    await openIrodoriFolder(endpoint.value, folder);
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : String(cause);
  }
}

async function run(save: boolean, refreshSpeakers = false) {
  if (busy.value) return;
  busy.value = true;
  store.mutations.LOCK_UI();
  error.value = "";
  try {
    const result = await loadSettings(save);
    settings.value = result.settings;
    ensureModelOption(result.settings.model);
    progress.value = result.progress;
    modelInfo.value = result.modelInfo ?? modelInfo.value;
    availableBackends.value = result.availableBackends ?? { cpu: true };
    modelFolder.value = result.modelFolder;
    speakerFolder.value = result.speakerFolder;
    if (refreshSpeakers) {
      await refreshIrodoriSpeakers(endpoint.value);
      await store.actions.LOAD_CHARACTER({ engineId: props.engineId });
    }
    if (save || refreshSpeakers) clearAudioCache();
    status.value = result.loaded
      ? "モデル読込済み"
      : "準備OK · 初回生成時にモデルを読み込みます";
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : String(cause);
  } finally {
    busy.value = false;
    store.mutations.UNLOCK_UI();
  }
}
async function pollStatus() {
  if (busy.value) return;
  try {
    const result = await fetchIrodoriStatus(endpoint.value, 10000);
    progress.value = result.progress;
    if (result.modelInfo) modelInfo.value = result.modelInfo;
  } catch {
    // The engine can be restarting while the editor remains open.
  }
}
const apply = () => run(true);
const refresh = () => run(false, true);
let pollTimer: ReturnType<typeof setInterval> | undefined;
onMounted(() => {
  pollTimer = setInterval(() => void pollStatus(), 1500);
});
onUnmounted(() => {
  if (pollTimer) clearInterval(pollTimer);
});
watch(
  () => props.engineId,
  () => {
    forgetIrodoriSession();
    settings.value = undefined;
    void run(false);
  },
  { immediate: true },
);
</script>
