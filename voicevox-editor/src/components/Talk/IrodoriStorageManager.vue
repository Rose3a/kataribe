<template>
  <div class="storage">
    <div class="row items-center q-gutter-x-sm q-mb-sm">
      <div class="text-caption col">
        <template v-if="storage">
          合計 {{ formatBytes(storage.totalBytes) }} · 古いキャッシュ
          {{ formatBytes(staleBytes) }}
        </template>
        <template v-else-if="loading">読み込み中…</template>
      </div>
      <QBtn
        flat
        round
        icon="refresh"
        :loading
        :disable="disable || deleting"
        aria-label="一覧を更新"
        @click="load"
      />
    </div>
    <div v-if="storage?.identifying" class="text-caption q-mb-sm">
      TensorRT plan がどのモデル用かを照合しています（初回のみ・1分ほど）…
    </div>
    <div v-if="storage && staleIds.length > 0" class="q-mb-sm">
      <QBtn
        v-if="confirmId !== BULK"
        outline
        color="warning"
        icon="delete_sweep"
        :label="`古いキャッシュをまとめて削除（${staleIds.length}件・${formatBytes(staleBytes)}）`"
        :disable="disable || deleting"
        @click="confirmId = BULK"
      />
      <div v-else class="row items-center q-gutter-x-sm">
        <span class="text-caption">
          もう使われない TensorRT plan などを削除します。よろしいですか？
        </span>
        <QBtn
          color="negative"
          icon="delete"
          label="削除"
          :loading="deleting"
          @click="remove(staleIds)"
        />
        <QBtn flat label="やめる" @click="confirmId = ''" />
      </div>
    </div>
    <div v-if="message" class="text-caption text-positive q-mb-sm">
      {{ message }}
    </div>
    <div
      v-if="error"
      role="alert"
      class="text-caption text-negative q-mb-sm"
      style="white-space: pre-wrap"
    >
      {{ error }}
    </div>
    <template v-for="group in groups" :key="group.title">
      <div v-if="group.entries.length > 0" class="q-mb-sm">
        <div class="text-caption text-weight-bold q-mt-sm">
          {{ group.title }}（{{ formatBytes(group.bytes) }}）
        </div>
        <QList dense separator>
          <QItem
            v-for="entry in group.entries"
            :key="entry.id"
            class="q-px-none"
          >
            <QItemSection>
              <QItemLabel class="entry-label">{{ entry.label }}</QItemLabel>
              <QItemLabel caption class="entry-caption">
                {{ entry.detail
                }}<template v-if="entry.modified != undefined">
                  · {{ formatDate(entry.modified) }}</template
                >
              </QItemLabel>
              <QItemLabel v-if="entry.note" caption class="entry-caption">
                {{ entry.note }}
              </QItemLabel>
              <QItemLabel
                v-if="confirmId === entry.id"
                class="row items-center q-gutter-sm q-mt-xs"
              >
                <span class="text-caption">削除しますか？</span>
                <QBtn
                  color="negative"
                  icon="delete"
                  label="削除"
                  :loading="deleting"
                  @click="remove([entry.id])"
                />
                <QBtn flat label="やめる" @click="confirmId = ''" />
              </QItemLabel>
            </QItemSection>
            <QItemSection side top class="items-end">
              <QItemLabel caption>{{ formatBytes(entry.bytes) }}</QItemLabel>
              <QBadge
                :color="STATUS[entry.status].color"
                :label="STATUS[entry.status].label"
                class="q-mt-xs"
              />
              <QBtn
                v-if="entry.deletable && confirmId !== entry.id"
                outline
                color="negative"
                icon="delete"
                label="削除"
                class="q-mt-sm"
                :disable="disable || deleting"
                :aria-label="`${entry.label}（${entry.detail}）を削除`"
                @click="confirmId = entry.id"
              />
            </QItemSection>
          </QItem>
        </QList>
      </div>
    </template>
  </div>
</template>
<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from "vue";
import type {
  IrodoriStorageEntry as Entry,
  IrodoriStorageStatus as Storage,
} from "@/domain/irodori";
import {
  deleteIrodoriStorage,
  fetchIrodoriStorage,
} from "@/helpers/irodoriEngine";

const props = defineProps<{ endpoint: string; disable?: boolean }>();
const emit = defineEmits<{ deleted: [] }>();

const BULK = "__bulk__";
const STATUS: Record<Entry["status"], { label: string; color: string }> = {
  in_use: { label: "使用中", color: "primary" },
  unused: { label: "未使用", color: "grey-7" },
  stale: { label: "古い", color: "warning" },
  required: { label: "必須", color: "grey-5" },
};

const storage = ref<Storage>();
const loading = ref(false);
const deleting = ref(false);
const error = ref("");
const message = ref("");
const confirmId = ref("");

const GROUPS: { title: string; kinds: Entry["kind"][] }[] = [
  { title: "モデル（Hugging Face からダウンロード）", kinds: ["hf"] },
  { title: "モデル（models フォルダ）", kinds: ["local"] },
  {
    title: "TensorRT 変換キャッシュ",
    kinds: ["trt", "codec", "trt-build", "codec-build", "legacy"],
  },
  { title: "その他", kinds: ["asr"] },
];
const groups = computed(() =>
  GROUPS.map((group) => {
    const entries = (storage.value?.entries ?? []).filter((entry) =>
      group.kinds.includes(entry.kind),
    );
    return {
      title: group.title,
      entries,
      bytes: entries.reduce((sum, entry) => sum + entry.bytes, 0),
    };
  }),
);
const staleEntries = computed(() =>
  (storage.value?.entries ?? []).filter(
    (entry) => entry.status === "stale" && entry.deletable,
  ),
);
const staleIds = computed(() => staleEntries.value.map((entry) => entry.id));
const staleBytes = computed(() =>
  staleEntries.value.reduce((sum, entry) => sum + entry.bytes, 0),
);

function formatBytes(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(0)} MB`;
  return `${Math.round(bytes / 1e3)} KB`;
}

function formatDate(seconds: number): string {
  return new Date(seconds * 1000).toLocaleDateString("ja-JP");
}

async function load() {
  if (loading.value) return;
  loading.value = true;
  error.value = "";
  try {
    storage.value = await fetchIrodoriStorage(props.endpoint);
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : String(cause);
  } finally {
    loading.value = false;
  }
}

async function remove(ids: string[]) {
  if (deleting.value || ids.length === 0) return;
  deleting.value = true;
  error.value = "";
  message.value = "";
  try {
    const result = await deleteIrodoriStorage(props.endpoint, ids);
    storage.value = result;
    message.value = `${formatBytes(result.freedBytes ?? 0)} を空けました`;
    emit("deleted");
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : String(cause);
    // 一部だけ消えた可能性があるので一覧を取り直す。
    await load();
  } finally {
    deleting.value = false;
    confirmId.value = "";
  }
}

// 照合中はハッシュ計算が終わるまで、ときどき一覧を取り直す。
let timer: ReturnType<typeof setInterval> | undefined;
onMounted(() => {
  void load();
  timer = setInterval(() => {
    if (storage.value?.identifying && !deleting.value) void load();
  }, 5000);
});
onUnmounted(() => {
  if (timer) clearInterval(timer);
});
</script>
<style scoped lang="scss">
.entry-label,
.entry-caption {
  overflow-wrap: anywhere;
}
</style>
