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
          <QItemLabel
            v-if="
              scope.opt.group &&
              scope.opt.group !== modelOptions[scope.index - 1]?.group
            "
            header
            class="q-pb-xs"
          >
            {{ scope.opt.group }}
          </QItemLabel>
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
        <span v-if="modelInfo.quantization">
          · 量子化 {{ modelInfo.quantization }}
        </span>
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
      <div v-if="selectedModelOption?.note" class="text-caption q-mb-sm">
        {{ selectedModelOption.note }}
      </div>
      <div v-if="modelInfo?.meanflow" class="text-caption q-mb-sm">
        MeanFlow モデル: ScheduleとCFGは未使用。ステップ数はモデル自身の既定が{{
          modelInfo.modelDefaultSteps ?? 4
        }}ですが、下の既定ステップ数に従います。
      </div>
      <QInput
        v-model.number="defaultStepsSetting"
        outlined
        dense
        type="number"
        :min="IRODORI_MIN_STEPS"
        :max="IRODORI_MAX_STEPS"
        label="既定ステップ数（全行共通・1〜80）"
        hint="ステップ数を指定していないセリフに使います。多いほど高品質・低速。「設定を適用」で反映されます。"
        :disable="locked"
        class="q-mb-md"
      />
      <QSelect
        v-model="tokenSplitScope"
        outlined
        dense
        label="語彙分割辞書の対象（読めない語の対策）"
        :options="tokenSplitScopes"
        emitValue
        mapOptions
        :disable="locked"
        class="q-mb-sm"
      />
      <div class="text-caption q-mb-sm">
        辞書は Small
        系のトークナイザ（modernbert-ja）の出現度で作ってあります。Large
        は日本語の語句を細かく割るので、同じ語句が読めない問題は起きにくく、既定では
        Small 系だけに当てます。{{ tokenSplitNote }}
      </div>
      <QToggle
        v-model="streamPlayback"
        dense
        label="ストリーミング再生（生成の完了を待たずに再生を始める）"
        :disable="locked"
      />
      <div class="text-caption q-mb-sm">
        長いセリフで再生が始まるまでの待ちが短くなります（トグルを切り替えるとすぐ保存されます）。
        参照音声を使うセリフは、透かしを入れるため生成後にまとめて再生します。ストリーミング中はクリック位置からの再生や口パクは使えず、再生し終えると通常の再生になります。
      </div>
      <div v-if="hasUnappliedChanges" class="text-caption text-warning q-mb-sm">
        未適用の変更があります。「設定を適用」を押すと反映されます。
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
        v-model="storageOpen"
        label="モデル・キャッシュの容量（確認 / 削除）"
        dense
        class="q-mt-sm"
      >
        <IrodoriStorageManager
          v-if="storageOpen"
          class="q-pa-sm"
          :endpoint
          :disable="locked"
          @deleted="run(false)"
        />
      </QExpansionItem>
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
            v4-Large 用の話者は、同じ話者のフォルダに
            名前.1280.speaker.safetensors（例:
            tsukuyomi.1280.speaker.safetensors）として置くと、同じ話者・同じ画像のまま
            Large 選択時に自動で切り替わります。
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
import IrodoriStorageManager from "./IrodoriStorageManager.vue";
import { useStore } from "@/store";
import { createEngineUrl } from "@/domain/url";
import { clearAudioCache } from "@/store/audioGenerate";
import {
  IRODORI_DEFAULT_STEPS,
  IRODORI_DEFAULT_TOKEN_SPLIT_SCOPE,
  IRODORI_MAX_STEPS,
  IRODORI_MIN_STEPS,
  irodoriDefaultSteps,
  irodoriMeanflow,
  irodoriStreamPlayback,
} from "@/domain/irodori";
import type {
  IrodoriModelInfo as ModelInfo,
  IrodoriSettings as Settings,
  IrodoriStatus as Status,
  IrodoriTokenSplitScope,
  IrodoriTokenSplitState,
} from "@/domain/irodori";
import {
  fetchIrodoriStatus,
  forgetIrodoriSession,
  irodoriRequest,
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
// エンジンに保存済みの設定。画面上の選択と比べて未適用の変更を知らせる。
const appliedSettings = ref<Settings>();
const hasUnappliedChanges = computed(
  () =>
    settings.value != undefined &&
    appliedSettings.value != undefined &&
    (settings.value.backend !== appliedSettings.value.backend ||
      settings.value.model !== appliedSettings.value.model ||
      (settings.value.default_steps ?? IRODORI_DEFAULT_STEPS) !==
        (appliedSettings.value.default_steps ?? IRODORI_DEFAULT_STEPS) ||
      (settings.value.token_split_scope ??
        IRODORI_DEFAULT_TOKEN_SPLIT_SCOPE) !==
        (appliedSettings.value.token_split_scope ??
          IRODORI_DEFAULT_TOKEN_SPLIT_SCOPE)),
);
watch(defaultSteps, (value) => {
  irodoriDefaultSteps.value = value;
});
watch(modelInfo, (value) => {
  if (value) irodoriMeanflow.value = value.meanflow;
});
// ストリーミング再生。切り替えるとすぐエンジンへ保存する（モデルの再読み込みは起きない）。
const streamPlayback = computed<boolean>({
  get: () => settings.value?.stream_playback === true,
  set: (value) => {
    if (!settings.value) return;
    settings.value.stream_playback = value;
    irodoriStreamPlayback.value = value;
    irodoriRequest(endpoint.value, "/irodori/settings", {
      method: "POST",
      body: { stream_playback: value },
    })
      .then(async (response) => {
        if (!response.ok) throw new Error(await response.text());
      })
      .catch((cause: unknown) => {
        error.value = cause instanceof Error ? cause.message : String(cause);
      });
  },
});
// 全行共通の既定ステップ数。範囲外や空欄は 1〜80 に丸める（適用で保存）。
const defaultStepsSetting = computed<number>({
  get: () => settings.value?.default_steps ?? IRODORI_DEFAULT_STEPS,
  set: (value) => {
    if (!settings.value) return;
    const steps = Math.round(Number(value));
    settings.value.default_steps = Number.isFinite(steps)
      ? Math.min(IRODORI_MAX_STEPS, Math.max(IRODORI_MIN_STEPS, steps))
      : IRODORI_DEFAULT_STEPS;
  },
});
const tokenSplit = ref<IrodoriTokenSplitState>();
const tokenSplitScope = computed<IrodoriTokenSplitScope>({
  get: () =>
    settings.value?.token_split_scope ?? IRODORI_DEFAULT_TOKEN_SPLIT_SCOPE,
  set: (value) => {
    if (settings.value) settings.value.token_split_scope = value;
  },
});
const tokenSplitScopes: { label: string; value: IrodoriTokenSplitScope }[] = [
  { label: "Small 系のみ（既定）", value: "small" },
  { label: "Small と Large の両方", value: "all" },
  { label: "使わない（どのモデルにも当てない）", value: "none" },
];
// 選んだ対象で、読み込み中（または選択中）のモデルに辞書が効くか。
const tokenSplitNote = computed(() => {
  const state = tokenSplit.value;
  if (!state) return "";
  const active =
    tokenSplitScope.value !== "none" &&
    (tokenSplitScope.value === "all" ||
      !state.modelTokenizer ||
      state.modelTokenizer === state.dictionaryTokenizer);
  return active
    ? "いまのモデルには辞書が効きます（セリフごとの設定が「使う」のとき）。"
    : "いまのモデルには辞書を当てません。";
});
const modelFolder = ref("");
// 開いたときだけ一覧を取る（フォルダサイズの集計に数秒かかる）。
const storageOpen = ref(false);
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
  group?: string;
  license?: string;
  note?: string;
  unsupportedBackends?: string[];
};
const modelOptions = ref<ModelOption[]>([
  {
    label: "Irodori-TTS v4.1 Small（既定・8ステップ）",
    value: "Aratako/Irodori-TTS-v4.1-Small",
    group: "Small 系（軽量・おすすめ）",
    description: "RFモデル / MIT",
    license: "MIT",
  },
  {
    label: "Irodori-TTS v4.1 Small MF（4ステップ）",
    value: "Aratako/Irodori-TTS-v4.1-Small-MF",
    group: "Small 系（軽量・おすすめ）",
    description: "MeanFlowモデル / MIT",
    license: "MIT",
  },
  {
    label: "Irodori-TTS v4.1 Small Yomi Tech（難読漢字・技術用語の読み改善）",
    value: "j-llm/Irodori-TTS-v4.1-Small-Yomi-Tech-tuned",
    group: "Small 系（軽量・おすすめ）",
    description: "RFモデル / 重み・コード MIT、読み辞書 CC BY-SA 4.0",
    license: "MIT（重み・コード）/ CC BY-SA 4.0（読み辞書: JMdict © EDRDG）",
    note: "v4.1 Small に、難読漢字の読み（Yomi）と英語・技術用語の読み（Tech）の改善を加えたモデルです。埋め込みの読み辞書は JMdict（© EDRDG, CC BY-SA 4.0）を含みます。読み間違いが減るだけで、声質は v4.1 Small と同じです。技術用語の読み替えは作者の推論スクリプト側の処理のため、このアプリでは Yomi の改善が中心です。",
  },
  {
    label: "Irodori-TTS v4.1 Anime（8ステップ）",
    value: "phasefield-audio/Irodori-TTS-v4.1-Anime",
    group: "Small 系（軽量・おすすめ）",
    description: "Anime fine-tune / MIT",
    license: "MIT",
  },
  {
    label: "Irodori-TTS v4 Large INT4（TensorRT 対応・省VRAM）",
    value: "Aratako/Irodori-TTS-v4-Large-Quantized/int4-weight-only",
    group: "Large 系（高品質・VRAM多め）",
    description:
      "RFモデル 3.3B・INT4量子化 / CUDA・TensorRT / Gemma Terms of Use",
    license: "Gemma Terms of Use",
    note: "約2.8GBをダウンロードします。VRAMは約3.5〜5GBで、生成は PyTorch の INT4 より速くなります。NVIDIA Ampere 以降（RTX 30 シリーズ以降）が必要です。TensorRT では、初回だけ INT4 用の plan を作ります（数分、作業用に約4GBの空きが必要）。既存の話者ファイル（v4.1 Small 用）は使えないため、「話者なし」か参照音声、または 名前.1280.speaker.safetensors を使います。Gemma の利用規約と禁止用途ポリシーにも従ってください。",
  },
  {
    label: "Irodori-TTS v4 Large INT8（Large の推奨）",
    value: "Aratako/Irodori-TTS-v4-Large-Quantized/int8-weight-only",
    group: "Large 系（高品質・VRAM多め）",
    description: "RFモデル 3.3B・INT8量子化 / CUDA専用 / Gemma Terms of Use",
    license: "Gemma Terms of Use",
    note: "約3.6GBをダウンロードします。VRAMは6GB以上を推奨。音質は元の Large とほぼ同じで、生成は bf16 版より1〜2割遅くなります。NVIDIA / CUDA 専用（TensorRT は不可）。既存の話者ファイル（v4.1 Small 用）は使えないため、「話者なし」か参照音声で生成します。Gemma の利用規約と禁止用途ポリシーにも従ってください。",
    unsupportedBackends: ["trt"],
  },
  {
    label: "Irodori-TTS v4 Large（bf16・TensorRT向け）",
    value: "Aratako/Irodori-TTS-v4-Large",
    group: "Large 系（高品質・VRAM多め）",
    description: "RFモデル 3.3B / Gemma Terms of Use",
    license: "Gemma Terms of Use",
    note: "約13GBをダウンロードします。VRAMは8GB以上、初回の TensorRT 変換には RAM 16GB 以上を推奨。既存の話者ファイル（v4.1 Small 用）は使えないため、「話者なし」か参照音声で生成します。Gemma の利用規約と禁止用途ポリシーにも従ってください。",
  },
  {
    label: "Irodori-TTS 500M v3（旧版・読み比べ用）",
    value: "Aratako/Irodori-TTS-500M-v3",
    group: "旧版（v3）",
    description:
      "RFモデル 500M・スタイル指示（絵文字・キャプション）なし / MIT",
    license: "MIT",
    note: "約1GBをダウンロードします。トークナイザが llm-jp-3 で、v4 系（modernbert-ja）とは語句の割り方が違います。語彙分割辞書は modernbert-ja 用のため、このモデルには当てません。既存の話者ファイル（768次元）はそのまま使えます。",
  },
]);
// 量子化モデルは TensorRT に変換できないので、選んだときは CUDA（無ければ CPU）へ切り替える。
watch(
  () => settings.value?.model,
  (model) => {
    const option = modelOptions.value.find((item) => item.value === model);
    const current = settings.value;
    if (!current || !option?.unsupportedBackends?.includes(current.backend))
      return;
    current.backend = availableBackends.value.cuda === true ? "cuda" : "cpu";
  },
);
function ensureModelOption(source: string) {
  const value = source.trim();
  if (!value || modelOptions.value.some((option) => option.value === value)) {
    return;
  }
  modelOptions.value.push({
    label: value,
    value,
    group: "カスタム",
    description: "カスタムモデル",
  });
}
const selectedModelOption = computed(() =>
  modelOptions.value.find((option) => option.value === settings.value?.model),
);
const licenseName = computed(
  () => modelInfo.value?.license ?? selectedModelOption.value?.license,
);
const licenseUrl = computed(
  () =>
    modelInfo.value?.licenseUrl ??
    (settings.value?.model.includes("/")
      ? `https://huggingface.co/${settings.value.model}`
      : undefined),
);
function setCustomModel(value: string, done: () => void) {
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
    appliedSettings.value = { ...result.settings };
    irodoriStreamPlayback.value = result.settings.stream_playback === true;
    ensureModelOption(result.settings.model);
    progress.value = result.progress;
    modelInfo.value = result.modelInfo ?? modelInfo.value;
    tokenSplit.value = result.tokenSplit ?? tokenSplit.value;
    availableBackends.value = result.availableBackends ?? { cpu: true };
    modelFolder.value = result.modelFolder;
    speakerFolder.value = result.speakerFolder;
    if (refreshSpeakers) {
      // モデル変更の保存時点でエンジンは話者一覧を作り直している。再スキャンが
      // 失敗しても、話者一覧の読み直しは必ず行い、古いモデル用の話者を残さない。
      try {
        await refreshIrodoriSpeakers(endpoint.value);
      } finally {
        await store.actions.LOAD_CHARACTER({ engineId: props.engineId });
      }
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
    if (result.tokenSplit) tokenSplit.value = result.tokenSplit;
  } catch {
    // The engine can be restarting while the editor remains open.
  }
}
// モデルが変わると使える話者（埋め込み次元）も変わるので、話者一覧も読み直す。
const apply = () =>
  run(true, settings.value?.model !== appliedSettings.value?.model);
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
    appliedSettings.value = undefined;
    void run(false);
  },
  { immediate: true },
);
</script>
