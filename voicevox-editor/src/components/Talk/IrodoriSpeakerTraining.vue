<template>
  <div class="speaker-train">
    <p class="text-caption q-mb-sm">
      キャラの音声から話者（16
      トークンの話者埋め込み）を作ります。モデル本体は変えずに話者ファイルだけを学習するので、TensorRT
      でもそのまま使えます。音声と同じ名前の .txt
      があればセリフとして使い、無い音声は Haas-ja で書き起こします。1〜30
      秒の音声を合計 10 分以上入れるのがおすすめです。
    </p>
    <p class="text-caption text-warning q-mb-sm">
      使えるのは、話者本人または権利者から許可を得た音声だけです（README
      の「Speaker Inversion と話者埋め込み」を参照）。
    </p>
    <template v-if="options">
      <QInput
        v-model="name"
        dense
        outlined
        label="話者の名前"
        hint="話者一覧に出る名前です（1〜32文字）。"
        :disable="running || disable"
        class="q-mb-md"
      />
      <QInput
        v-model="dataDir"
        dense
        outlined
        label="音声のフォルダ"
        hint="「選ぶ」でフォルダを選ぶ画面が開きます（パスを貼り付けても構いません。例: …\voices\キャラ名）。下のフォルダも探します（.wav .flac .ogg .mp3）。"
        :disable="running || disable"
        class="q-mb-md"
      >
        <template #append>
          <QBtn
            flat
            dense
            noCaps
            icon="folder_open"
            label="選ぶ"
            :loading="picking"
            :disable="running || disable"
            @click="pickFolder"
          />
        </template>
      </QInput>
      <QSelect
        v-model="baseModel"
        dense
        outlined
        emitValue
        mapOptions
        :options="baseModelOptions"
        label="ベースモデル"
        hint="この話者を使うモデルを選びます。v4.1 Small と同じ構造のモデルだけ出ます（量子化版・MeanFlow・Large は不可）。"
        :disable="running || disable"
        class="q-mb-md"
      />
      <QToggle
        v-model="learnSpeed"
        label="話速も学習する"
        :disable="running || disable"
      />
      <div class="text-caption q-mb-sm q-ml-sm">
        オンでキャラの話す速さ・間の取り方も覚えます（ゆっくり話すキャラ向け）。オフだと話速は音声を参照したときと同じくらいになります。
      </div>
      <QToggle
        v-model="verifyText"
        label="付属のテキストも書き起こしで確かめる"
        :disable="running || disable || !options.haasAvailable"
      />
      <div class="text-caption q-mb-sm q-ml-sm">
        音声と大きく違うテキスト（別のセリフ・性別の分岐など）は、書き起こしに置き換えます。
      </div>
      <div class="row items-center q-gutter-x-md q-mb-md">
        <QBtnToggle
          v-model="device"
          dense
          noCaps
          toggleColor="primary"
          :options="deviceOptions"
          :disable="running || disable"
        />
        <QInput
          v-model.number="steps"
          dense
          outlined
          type="number"
          :min="50"
          :max="5000"
          label="ステップ数"
          style="width: 9em"
          :disable="running || disable"
        />
      </div>
      <div class="text-caption q-mb-sm">{{ timeHint }}</div>
      <QToggle
        v-if="exists"
        v-model="overwrite"
        :label="`同じ名前の学習した話者「${name.trim()}」を上書きする`"
        :disable="running || disable"
      />
      <div
        v-if="!options.haasAvailable"
        class="text-caption text-warning q-mb-sm"
      >
        書き起こし（Haas-ja）が入っていません。テキストの無い音声を使うには、エンジンフォルダの
        bat\haas_setup.bat を一度実行してください（約 1.2GB のモデルと、別の
        Python 環境を入れます）。
      </div>
      <div class="row q-gutter-x-sm q-mt-sm">
        <QBtn
          v-if="!running"
          color="primary"
          icon="school"
          label="学習を始める"
          :disable="disable || !canStart"
          :loading="starting"
          @click="start"
        />
        <QBtn
          v-else
          outline
          color="negative"
          icon="stop"
          label="中止"
          @click="cancel"
        />
        <QBtn
          flat
          icon="refresh"
          aria-label="選べるモデルを更新"
          :disable="running || disable"
          @click="loadOptions"
        />
      </div>
    </template>
    <div v-else-if="loading" class="text-caption">読み込み中…</div>

    <div v-if="status && status.state !== 'idle'" class="q-mt-md">
      <QLinearProgress
        :value="status.percent / 100"
        rounded
        :color="stateColor"
        class="q-mb-xs"
      />
      <div class="text-caption">
        {{ stateLabel }} · {{ Math.round(status.percent) }}% ·
        {{ status.message }}
        <template v-if="status.state === 'running' && status.detail.vram_gb">
          · VRAM {{ status.detail.vram_gb }} GB</template
        >
      </div>
      <div
        v-if="status.state === 'done' && status.result"
        class="text-caption q-mt-sm result"
      >
        「{{ status.result.name }}」を保存しました（話者一覧に追加済み）。<br />
        使った音声 {{ status.result.data.used }} 本・{{
          status.result.data.used_minutes
        }}
        分（書き起こし
        {{ status.result.data.transcribed }} 本、テキストを置き換え
        {{ status.result.data.text_replaced_by_asr }} 本、短すぎ
        {{ status.result.data.too_short }} 本・長すぎ
        {{ status.result.data.too_long }} 本は除外）。<br />
        話速: ベース（話者なし）の {{ paceLabel }}。学習
        {{ Math.round(status.result.metrics.train_seconds) }} 秒。<br />
        {{ status.result.speaker_file }}
      </div>
      <div
        v-if="status.state === 'error'"
        role="alert"
        class="text-caption text-negative q-mt-sm"
        style="white-space: pre-wrap"
      >
        {{ status.error }}
      </div>
      <QExpansionItem
        v-if="status.log.length > 0"
        dense
        label="ログ"
        class="q-mt-sm"
      >
        <pre class="log">{{ status.log.join("\n") }}</pre>
      </QExpansionItem>
    </div>
    <div
      v-if="error"
      role="alert"
      class="text-caption text-negative q-mt-sm"
      style="white-space: pre-wrap"
    >
      {{ error }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import { useStore } from "@/store";
import type {
  IrodoriSpeakerTrainOptions as Options,
  IrodoriSpeakerTrainStatus as TrainStatus,
} from "@/domain/irodori";
import {
  cancelSpeakerTrain,
  fetchSpeakerTrainOptions,
  fetchSpeakerTrainStatus,
  pickIrodoriFolder,
  refreshIrodoriSpeakers,
  startSpeakerTrain,
} from "@/helpers/irodoriEngine";
import type { EngineId } from "@/type/preload";

const props = defineProps<{
  endpoint: string;
  engineId: EngineId;
  /** いまエディタで選んでいるモデル（ベースの初期値にする）。 */
  currentModel?: string;
  disable?: boolean;
}>();
const store = useStore();

const options = ref<Options>();
const status = ref<TrainStatus>();
const loading = ref(false);
const starting = ref(false);
const picking = ref(false);
const error = ref("");

const name = ref("");
const dataDir = ref("");
const baseModel = ref("");
const learnSpeed = ref(true);
const verifyText = ref(true);
const device = ref<"cuda" | "cpu">("cuda");
const steps = ref(500);
const overwrite = ref(false);

const running = computed(() => status.value?.state === "running");
const exists = computed(() =>
  (options.value?.existing ?? []).includes(name.value.trim()),
);
const canStart = computed(
  () =>
    name.value.trim().length > 0 &&
    dataDir.value.trim().length > 0 &&
    baseModel.value.length > 0 &&
    Number.isInteger(steps.value) &&
    steps.value >= 50 &&
    steps.value <= 5000 &&
    (!exists.value || overwrite.value),
);
const baseModelOptions = computed(() =>
  (options.value?.baseModels ?? []).map((model) => ({
    label: model.source === "local" ? `models\\${model.label}` : model.label,
    value: model.path,
  })),
);
const deviceOptions = computed(() => [
  {
    label: "GPU（CUDA）",
    value: "cuda",
    disable: options.value?.cudaAvailable !== true,
  },
  { label: "CPU", value: "cpu" },
]);
const timeHint = computed(() =>
  device.value === "cuda"
    ? "GPU: 500 ステップで数分（VRAM は約 2GB）。書き起こしと音声の符号化に別に数分かかります。"
    : "CPU: 短いセリフ中心なら 500 ステップで 10〜15 分、長い音声が多いと 30 分以上かかります。",
);
const stateLabel = computed(
  () =>
    ({
      idle: "",
      running: "学習中",
      done: "完了",
      error: "失敗",
      cancelled: "中止",
    })[status.value?.state ?? "idle"],
);
const stateColor = computed(() =>
  status.value?.state === "error"
    ? "negative"
    : status.value?.state === "cancelled"
      ? "grey-6"
      : "primary",
);
const paceLabel = computed(() => {
  const pace = status.value?.result?.metrics.pace_vs_base;
  return pace == undefined ? "-" : `${pace.toFixed(2)} 倍の長さ`;
});

function message(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause);
}

/** エディタで選んでいるモデルと同じものを、ベースの初期値にする。 */
function pickBaseModel(value: Options) {
  if (value.baseModels.some((model) => model.path === baseModel.value)) return;
  const current = (props.currentModel ?? "").replace(/\\/g, "/");
  const match = value.baseModels.find(
    (model) =>
      current.length > 0 &&
      (model.label === current ||
        model.path.replace(/\\/g, "/").endsWith(current)),
  );
  baseModel.value = (match ?? value.baseModels[0])?.path ?? "";
}

async function loadOptions() {
  loading.value = true;
  error.value = "";
  try {
    const value = await fetchSpeakerTrainOptions(props.endpoint);
    options.value = value;
    pickBaseModel(value);
    if (!value.cudaAvailable) device.value = "cpu";
    if (!value.haasAvailable) verifyText.value = false;
  } catch (cause) {
    error.value = message(cause);
  } finally {
    loading.value = false;
  }
}

let wasRunning = false;
async function poll() {
  try {
    const value = await fetchSpeakerTrainStatus(props.endpoint);
    status.value = value;
    if (wasRunning && value.state === "done") {
      // 新しい話者を一覧に出す（エンジンは学習の終わりに一覧を作り直している）
      try {
        await refreshIrodoriSpeakers(props.endpoint);
      } finally {
        await store.actions.LOAD_CHARACTER({ engineId: props.engineId });
      }
      void loadOptions();
    }
    wasRunning = value.state === "running";
  } catch {
    // エンジンの再起動中など。次の問い合わせで戻る
  }
}

async function start() {
  starting.value = true;
  error.value = "";
  try {
    status.value = await startSpeakerTrain(props.endpoint, {
      name: name.value.trim(),
      dataDir: dataDir.value.trim().replace(/^"|"$/g, ""),
      baseModel: baseModel.value,
      speed: learnSpeed.value ? "learn" : "keep",
      device: device.value,
      steps: steps.value,
      verifyText: verifyText.value,
      overwrite: overwrite.value,
    });
    wasRunning = true;
  } catch (cause) {
    error.value = message(cause);
  } finally {
    starting.value = false;
  }
}

/** エンジンの PC でフォルダを選ぶ画面を開く（ブラウザからは PC のパスが取れないため）。 */
async function pickFolder() {
  picking.value = true;
  error.value = "";
  try {
    const picked = await pickIrodoriFolder(
      props.endpoint,
      dataDir.value.trim(),
    );
    if (picked != undefined) dataDir.value = picked;
  } catch (cause) {
    error.value = message(cause);
  } finally {
    picking.value = false;
  }
}

async function cancel() {
  try {
    status.value = await cancelSpeakerTrain(props.endpoint);
  } catch (cause) {
    error.value = message(cause);
  }
}

let timer: ReturnType<typeof setInterval> | undefined;
onMounted(() => {
  void loadOptions();
  void poll();
  timer = setInterval(() => void poll(), 1500);
});
onUnmounted(() => {
  if (timer) clearInterval(timer);
});
watch(
  () => props.endpoint,
  () => {
    void loadOptions();
    void poll();
  },
);
</script>

<style scoped lang="scss">
.result,
.log {
  overflow-wrap: anywhere;
}
.log {
  max-height: 16em;
  overflow: auto;
  font-size: 11px;
  white-space: pre-wrap;
  margin: 0;
}
</style>
