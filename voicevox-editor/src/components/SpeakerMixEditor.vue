<template>
  <div class="mix-editor">
    <header class="studio-header">
      <div class="studio-heading">
        <div class="eyebrow">KATARIBE / VOICE LAB</div>
        <h1>話者マージ</h1>
        <div class="header-subtitle">
          16トークンの声を、ひとつずつ組み立てる
        </div>
      </div>
      <div class="header-actions">
        <span class="draft-indicator"
          ><span class="draft-dot" /> 下書きを自動保存</span
        >
        <QBtn
          flat
          dense
          icon="close"
          aria-label="閉じる"
          @click="emit('close')"
        />
      </div>
    </header>
    <main class="studio-body">
      <div class="row items-center q-gutter-sm q-mb-md">
        <span class="metric"
          ><strong>{{ assignedCount }}</strong
          ><span>/ 16 配置</span></span
        >
        <span class="metric"
          ><strong>{{ overlapCount }}</strong
          ><span>列が重なり</span></span
        >
        <span v-if="overlapCount > 0" class="overlap-hint"
          >重ねすぎると声が平均化します</span
        >
        <QSpace />
        <QBtn
          flat
          dense
          icon="restart_alt"
          label="全列を0に戻す"
          :disable="busy || assignedCount === 0"
          @click="clearAll"
        />
      </div>
      <div
        v-if="recipes.length"
        class="recipe-row row items-center q-gutter-sm q-mb-md"
      >
        <QSelect
          v-model="selectedRecipe"
          dense
          outlined
          emit-value
          map-options
          label="保存済みの配合"
          :options="
            recipes.map((recipe) => ({
              label: recipe.name,
              value: recipe.name,
            }))
          "
          style="min-width: 240px"
        />
        <QBtn
          outline
          dense
          label="配合を開く"
          :disable="!selectedRecipe || busy"
          @click="loadRecipe"
        />
      </div>
      <div class="studio-workspace">
        <aside class="studio-library">
          <div class="panel-kicker">SOURCE</div>
          <h2>話者ライブラリ</h2>
          <p>列へドラッグして配置。＋で選択中の列に追加。</p>
          <QInput
            v-model="speakerSearch"
            dense
            outlined
            clearable
            placeholder="話者を探す"
            class="speaker-search"
          >
            <template #prepend><QIcon name="search" size="18px" /></template>
          </QInput>
          <div class="speaker-list">
            <div
              v-for="speaker in filteredSpeakers"
              :key="speaker.id"
              class="speaker-chip"
              draggable="true"
              @dragstart="dragSpeaker($event, speaker.id)"
            >
              <span
                class="speaker-avatar"
                :style="{ borderColor: colorFor(speakerIndex(speaker.id)) }"
                >{{ speakerLabel(speaker.name).slice(0, 1) }}</span
              >
              <span class="speaker-name" :title="speaker.name">{{
                speakerLabel(speaker.name)
              }}</span>
              <QBtn
                flat
                dense
                size="sm"
                icon="add"
                :aria-label="`${speaker.name}をトークン${selectedToken + 1}へ配置`"
                @click="addSpeaker(speaker.id, selectedToken)"
              />
            </div>
          </div>
        </aside>
        <section class="studio-canvas">
          <div class="canvas-heading">
            <div>
              <div class="panel-kicker">COMPOSITION</div>
              <h2>トークン配置</h2>
            </div>
            <span>縦 = 強度 · 色の幅 = 配合比</span>
          </div>
          <div class="token-scroll">
            <div class="token-grid">
              <div
                v-for="(entries, index) in tokens"
                :key="index"
                class="token-column"
                :class="{ selected: selectedToken === index }"
                tabindex="0"
                role="button"
                :aria-label="`トークン${index + 1}、強度${Math.round(columnGain(entries) * 100)}%、話者${entries.length}件`"
                @click="selectedToken = index"
                @keydown.enter="selectedToken = index"
                @keydown.up.prevent="nudgeColumnGain(index, 0.05)"
                @keydown.down.prevent="nudgeColumnGain(index, -0.05)"
                @dragover.prevent
                @drop.prevent="dropSpeaker($event, index)"
              >
                <div
                  class="column-space"
                  @pointerdown="setColumnGain(index, $event)"
                  @pointermove="moveColumnGain(index, $event)"
                >
                  <div
                    class="column-bar"
                    :style="{ height: `${columnGain(entries) * 100}%` }"
                  >
                    <div v-if="columnGain(entries) > 0" class="vertex-dot" />
                    <div
                      v-for="entry in entries.filter(
                        (item) => item.strength > 0,
                      )"
                      :key="entry.speaker"
                      class="bar-piece"
                      :style="{
                        width: `${(entry.strength / columnTotal(entries)) * 100}%`,
                        background: colorFor(speakerIndex(entry.speaker)),
                      }"
                      :title="`${speakerName(entry.speaker)}: ${entry.strength.toFixed(2)}`"
                    />
                  </div>
                </div>
                <div class="token-label">
                  {{ index + 1
                  }}<span
                    v-if="index === 0"
                    title="発話長の予測にも関わる場合があります"
                    >*</span
                  >
                </div>
              </div>
            </div>
          </div>
          <div class="canvas-footnote">
            列を選択して詳細を編集。棒の上端を上下にドラッグして強度を調整できます。*
            先頭トークンは発話長にも関わる場合があります。
          </div>
        </section>
        <aside class="studio-inspector">
          <div class="panel-kicker">INSPECTOR</div>
          <div class="inspector-content row q-gutter-md q-mt-md">
            <div class="col-12 col-md-6">
              <div class="text-subtitle1 q-mb-sm">
                トークン {{ selectedToken + 1 }}
              </div>
              <div
                v-if="selectedEntries.length === 0"
                class="text-caption q-mb-sm"
              >
                未配置（すべて0）
              </div>
              <div
                v-for="entry in selectedEntries"
                :key="entry.speaker"
                class="row items-center q-gutter-sm q-mb-sm"
              >
                <div class="entry-label">{{ speakerName(entry.speaker) }}</div>
                <QSlider
                  class="col"
                  :model-value="entry.strength"
                  :min="0"
                  :max="1"
                  :step="0.01"
                  @update:model-value="setStrength(entry, $event)"
                />
                <QInput
                  class="entry-value"
                  :model-value="entry.strength"
                  type="number"
                  dense
                  outlined
                  :min="0"
                  :max="1"
                  :step="0.01"
                  suffix="/ 1"
                  aria-label="話者の強度"
                  @change="setStrength(entry, Number($event))"
                />
                <QBtn
                  flat
                  dense
                  icon="star"
                  title="この話者を主役にする"
                  @click="makePrimary(entry.speaker)"
                />
                <QBtn
                  flat
                  dense
                  icon="delete"
                  title="配置を削除"
                  @click="removeSpeaker(entry.speaker)"
                />
              </div>
              <div
                v-if="
                  selectedEntries.length > 1 && columnTotal(selectedEntries) > 0
                "
                class="text-caption"
              >
                比率:
                {{
                  selectedEntries
                    .map(
                      (entry) =>
                        `${speakerName(entry.speaker)} ${Math.round((entry.strength / columnTotal(selectedEntries)) * 100)}%`,
                    )
                    .join(" / ")
                }}
              </div>
              <QBtn
                flat
                dense
                label="この列を0に戻す"
                :disable="selectedEntries.length === 0"
                @click="tokens[selectedToken] = []"
              />
              <div class="text-caption q-mt-sm">
                「主役にする」は選んだ話者を100%、残りの合計強度を25%に調整します。
              </div>
            </div>
            <div class="col-12 col-md-5">
              <div class="text-subtitle1 q-mb-sm">範囲操作</div>
              <div class="range-controls">
                <QSelect
                  v-model="rangeSpeaker"
                  class="range-speaker"
                  dense
                  outlined
                  emit-value
                  map-options
                  label="話者"
                  :options="
                    speakers.map((speaker) => ({
                      label: speakerLabel(speaker.name),
                      value: speaker.id,
                    }))
                  "
                />
                <div class="range-bounds">
                  <QInput
                    v-model.number="rangeStart"
                    dense
                    outlined
                    type="number"
                    label="開始"
                    :min="1"
                    :max="16"
                  />
                  <QInput
                    v-model.number="rangeEnd"
                    dense
                    outlined
                    type="number"
                    label="終了"
                    :min="1"
                    :max="16"
                  />
                </div>
                <QBtn
                  color="primary"
                  label="配置"
                  :disable="!rangeSpeaker"
                  @click="applyRange"
                />
              </div>
              <div class="range-actions">
                <QBtn
                  outline
                  dense
                  label="選択列を範囲へ複製"
                  @click="copyToRange"
                />
                <QBtn flat dense label="範囲を0に戻す" @click="clearRange" />
              </div>
            </div>
          </div>
        </aside>
      </div>
      <section class="studio-bottom">
        <div class="studio-preview">
          <div class="panel-kicker">AUDITION</div>
          <h2>声を聴き比べる</h2>
          <div class="preview-controls">
            <QInput
              v-model="previewText"
              class="preview-text"
              outlined
              label="試聴するセリフ"
            />
            <QBtn
              color="primary"
              icon="play_arrow"
              label="現在の配合で試聴"
              :loading="busy"
              :disable="busy || !previewText.trim()"
              @click="preview"
            />
          </div>
          <div v-if="previewHistory.length" class="q-mt-md">
            <div class="text-subtitle1">試聴履歴</div>
            <div v-if="previewIsStale" class="text-caption text-warning">
              現在の配合またはセリフは、最後に試聴した状態から変わっています。
            </div>
            <div
              v-for="(item, index) in previewHistory"
              :key="item.id"
              class="preview-row row items-center q-gutter-sm q-mt-sm"
            >
              <QBadge
                :label="`#${previewHistory.length - index}`"
                color="primary"
              />
              <span class="preview-summary">{{ item.summary }}</span>
              <audio :src="item.url" controls preload="none" />
              <QBtn
                flat
                dense
                icon="settings_backup_restore"
                title="この配合を戻す"
                @click="restorePreview(item)"
              />
            </div>
          </div>
        </div>
        <div class="studio-save">
          <div class="panel-kicker">EXPORT</div>
          <h2>話者として保存</h2>
          <div class="save-controls">
            <QInput
              v-model="saveName"
              class="save-name"
              outlined
              label="保存名（英数字・_・-）"
              hint="元話者は上書きせず、新しい話者として保存します"
            />
            <QBtn
              outline
              color="primary"
              label="新しい話者として保存"
              :loading="busy"
              :disable="busy || !validSaveName || assignedCount === 0"
              @click="save"
            />
          </div>
          <div
            v-if="saveName && !validSaveName"
            class="text-caption text-negative q-mt-sm"
          >
            保存名は半角英数字・_・- の1〜48文字にしてください。
          </div>
        </div>
        <div
          v-if="status"
          class="q-mt-md"
          :class="error ? 'text-negative' : 'text-positive'"
          role="status"
        >
          {{ status }}
        </div>
      </section>
    </main>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import {
  irodoriRequest,
  refreshIrodoriSpeakers,
} from "@/helpers/irodoriEngine";
import { useStore } from "@/store";
import type { EngineId } from "@/type/preload";

type Speaker = { id: string; name: string };
type Contribution = { speaker: string; strength: number };
type Recipe = { name: string; tokens: unknown };
type PreviewItem = {
  id: number;
  url: string;
  tokens: Contribution[][];
  text: string;
  summary: string;
  fingerprint: string;
};
const props = defineProps<{ endpoint: string; engineId: EngineId }>();
const emit = defineEmits<{ close: [] }>();
const store = useStore();
const speakers = ref<Speaker[]>([]);
const speakerSearch = ref("");
const recipes = ref<Recipe[]>([]);
const selectedRecipe = ref<string | null>(null);
const tokens = ref<Contribution[][]>(Array.from({ length: 16 }, () => []));
const selectedToken = ref(0);
const selectedEntries = computed(() => tokens.value[selectedToken.value]);
const rangeSpeaker = ref<string | null>(null);
const rangeStart = ref(1);
const rangeEnd = ref(16);
const previewText = ref("こんにちは。声の変化を試しています。");
const saveName = ref("");
const previewHistory = ref<PreviewItem[]>([]);
const previewSequence = ref(0);
const busy = ref(false);
const status = ref("");
const error = ref(false);
const draftLoaded = ref(false);
const draftKey = `irodori-speaker-mix:${props.engineId}`;
const colorFor = (index: number) =>
  `hsl(${(Math.max(0, index) * 137.508 + 12) % 360} 70% 59%)`;
const speakerIndex = (id: string) =>
  speakers.value.findIndex((speaker) => speaker.id === id);
const speakerLabel = (name: string) =>
  name.replace(/^orenoyome_dlc_/, "").replaceAll("_", " · ");
const speakerName = (id: string) =>
  speakerLabel(speakers.value.find((speaker) => speaker.id === id)?.name ?? id);
const filteredSpeakers = computed(() => {
  const query = speakerSearch.value.trim().toLocaleLowerCase();
  return query
    ? speakers.value.filter(
        (speaker) =>
          speaker.name.toLocaleLowerCase().includes(query) ||
          speaker.id.toLocaleLowerCase().includes(query),
      )
    : speakers.value;
});
const columnTotal = (entries: Contribution[]) =>
  entries.reduce((sum, entry) => sum + entry.strength, 0);
const columnGain = (entries: Contribution[]) =>
  Math.max(0, ...entries.map((entry) => entry.strength));
const assignedCount = computed(
  () => tokens.value.filter((entries) => columnGain(entries) > 0).length,
);
const overlapCount = computed(
  () =>
    tokens.value.filter(
      (entries) => entries.filter((entry) => entry.strength > 0).length > 1,
    ).length,
);
const validSaveName = computed(() =>
  /^[A-Za-z0-9_-]{1,48}$/.test(saveName.value.trim()),
);
const currentFingerprint = computed(() =>
  JSON.stringify({ tokens: tokens.value, text: previewText.value }),
);
const previewIsStale = computed(
  () =>
    previewHistory.value.length > 0 &&
    previewHistory.value[0].fingerprint !== currentFingerprint.value,
);
const setStrength = (entry: Contribution, value: number | null) => {
  const number = Number(value);
  if (!Number.isFinite(number)) return;
  entry.strength = Math.round(Math.max(0, Math.min(1, number)) * 100) / 100;
};

function addSpeaker(id: string, token: number) {
  if (!speakers.value.some((speaker) => speaker.id === id)) return;
  const entries = tokens.value[token];
  if (!entries.some((entry) => entry.speaker === id))
    entries.push({ speaker: id, strength: 1 });
  selectedToken.value = token;
}
function clearAll() {
  tokens.value = Array.from({ length: 16 }, () => []);
  selectedToken.value = 0;
}
function sanitizedTokens(value: unknown): Contribution[][] | null {
  if (!Array.isArray(value) || value.length !== 16) return null;
  return value.map((entries) => {
    if (!Array.isArray(entries)) return [];
    const seen = new Set<string>();
    return entries
      .filter((entry): entry is Contribution => {
        if (
          typeof entry !== "object" ||
          entry === null ||
          typeof entry.speaker !== "string" ||
          typeof entry.strength !== "number" ||
          !Number.isFinite(entry.strength) ||
          entry.strength < 0 ||
          entry.strength > 1 ||
          seen.has(entry.speaker) ||
          !speakers.value.some((speaker) => speaker.id === entry.speaker)
        )
          return false;
        seen.add(entry.speaker);
        return true;
      })
      .map((entry) => ({ speaker: entry.speaker, strength: entry.strength }));
  });
}
function loadRecipe() {
  const recipe = recipes.value.find(
    (item) => item.name === selectedRecipe.value,
  );
  if (!recipe) return;
  const normalized = sanitizedTokens(recipe.tokens);
  if (!normalized) {
    error.value = true;
    status.value = "配合データを読み込めませんでした";
    return;
  }
  tokens.value = normalized;
  saveName.value = recipe.name;
  error.value = false;
  status.value = `「${recipe.name}」の配合を開きました`;
}
async function fetchRecipes() {
  const response = await irodoriRequest(props.endpoint, "/irodori/mix/recipes");
  if (!response.ok) throw new Error(await responseError(response));
  recipes.value = (await response.json()) as Recipe[];
}
function makePrimary(id: string) {
  const entries = selectedEntries.value;
  const others = entries.filter((entry) => entry.speaker !== id);
  const otherTotal = columnTotal(others);
  for (const entry of entries) {
    if (entry.speaker === id) entry.strength = 1;
    else
      entry.strength =
        otherTotal > 0 ? (entry.strength * 0.25) / otherTotal : 0;
  }
}
function removeSpeaker(id: string) {
  tokens.value[selectedToken.value] = selectedEntries.value.filter(
    (entry) => entry.speaker !== id,
  );
}
function dragSpeaker(event: DragEvent, id: string) {
  event.dataTransfer?.setData("text/plain", id);
}
function dropSpeaker(event: DragEvent, token: number) {
  const id = event.dataTransfer?.getData("text/plain");
  if (id) addSpeaker(id, token);
}
function setColumnGain(token: number, event: PointerEvent) {
  const entries = tokens.value[token];
  if (entries.length === 0) return;
  if (event.type === "pointerdown")
    (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
  const bounds = (event.currentTarget as HTMLElement).getBoundingClientRect();
  const next = Math.max(
    0,
    Math.min(1, 1 - (event.clientY - bounds.top) / bounds.height),
  );
  updateColumnGain(token, next);
}
function updateColumnGain(token: number, next: number) {
  const entries = tokens.value[token];
  if (entries.length === 0) return;
  const current = columnGain(entries);
  if (current === 0) entries[0].strength = next;
  else
    for (const entry of entries)
      entry.strength = Math.max(
        0,
        Math.min(1, (entry.strength * next) / current),
      );
  selectedToken.value = token;
}
function nudgeColumnGain(token: number, delta: number) {
  updateColumnGain(
    token,
    Math.max(0, Math.min(1, columnGain(tokens.value[token]) + delta)),
  );
}
function moveColumnGain(token: number, event: PointerEvent) {
  if (event.buttons & 1) setColumnGain(token, event);
}
function applyRange() {
  if (!rangeSpeaker.value) return;
  const bounds = rangeBounds();
  if (!bounds) return;
  for (let token = bounds[0]; token <= bounds[1]; token++)
    addSpeaker(rangeSpeaker.value, token);
}
function rangeBounds(): [number, number] | null {
  const start = Math.floor(Number(rangeStart.value));
  const end = Math.floor(Number(rangeEnd.value));
  if (!Number.isFinite(start) || !Number.isFinite(end)) return null;
  return [
    Math.max(0, Math.min(15, Math.min(start, end) - 1)),
    Math.max(0, Math.min(15, Math.max(start, end) - 1)),
  ];
}
function copyToRange() {
  const bounds = rangeBounds();
  if (!bounds) return;
  const source = selectedEntries.value.map((entry) => ({ ...entry }));
  for (let token = bounds[0]; token <= bounds[1]; token++)
    tokens.value[token] = source.map((entry) => ({ ...entry }));
}
function clearRange() {
  const bounds = rangeBounds();
  if (!bounds) return;
  for (let token = bounds[0]; token <= bounds[1]; token++)
    tokens.value[token] = [];
}
function restorePreview(item: PreviewItem) {
  tokens.value = item.tokens.map((entries) =>
    entries.map((entry) => ({ ...entry })),
  );
  previewText.value = item.text;
  status.value = `試聴 #${item.id} の配合を戻しました`;
  error.value = false;
}
function payload() {
  return {
    tokens: tokens.value.map((entries) =>
      entries.map(({ speaker, strength }) => ({ speaker, strength })),
    ),
  };
}
async function responseError(response: Response) {
  try {
    const body = (await response.json()) as { detail?: string };
    return body.detail ?? `通信エラー (${response.status})`;
  } catch {
    return `通信エラー (${response.status})`;
  }
}
async function preview() {
  const snapshot = payload();
  const text = previewText.value;
  const fingerprint = currentFingerprint.value;
  const placed = snapshot.tokens.filter(
    (entries) => columnGain(entries) > 0,
  ).length;
  const overlaps = snapshot.tokens.filter(
    (entries) => entries.filter((entry) => entry.strength > 0).length > 1,
  ).length;
  busy.value = true;
  status.value = "音声を生成中…";
  error.value = false;
  try {
    const response = await irodoriRequest(
      props.endpoint,
      "/irodori/mix/preview",
      {
        method: "POST",
        body: { ...snapshot, text },
        timeoutMs: 300000,
      },
    );
    if (!response.ok) throw new Error(await responseError(response));
    const item: PreviewItem = {
      id: ++previewSequence.value,
      url: URL.createObjectURL(await response.blob()),
      tokens: snapshot.tokens.map((entries) =>
        entries.map((entry) => ({ ...entry })),
      ),
      text,
      summary: `${placed}列を配置・${overlaps}列で重なり`,
      fingerprint,
    };
    previewHistory.value.unshift(item);
    if (previewHistory.value.length > 5) {
      const removed = previewHistory.value.pop();
      if (removed) URL.revokeObjectURL(removed.url);
    }
    status.value = "生成しました";
  } catch (cause) {
    error.value = true;
    status.value =
      cause instanceof Error ? cause.message : "試聴に失敗しました";
  } finally {
    busy.value = false;
  }
}
async function save() {
  const name = saveName.value.trim();
  const snapshot = payload();
  busy.value = true;
  status.value = "保存中…";
  error.value = false;
  try {
    const response = await irodoriRequest(props.endpoint, "/irodori/mix/save", {
      method: "POST",
      body: { ...snapshot, name },
      timeoutMs: 60000,
    });
    if (!response.ok) throw new Error(await responseError(response));
    status.value = `話者「${name}」を保存しました`;
    try {
      await refreshIrodoriSpeakers(props.endpoint);
      await store.actions.LOAD_CHARACTER({ engineId: props.engineId });
      const speakerResponse = await irodoriRequest(
        props.endpoint,
        "/irodori/mix/speakers",
      );
      if (speakerResponse.ok)
        speakers.value = (await speakerResponse.json()) as Speaker[];
      await fetchRecipes();
    } catch {
      status.value += "。一覧の更新は、トーク画面から実行してください";
    }
  } catch (cause) {
    error.value = true;
    status.value =
      cause instanceof Error ? cause.message : "保存に失敗しました";
  } finally {
    busy.value = false;
  }
}
onMounted(async () => {
  try {
    const response = await irodoriRequest(
      props.endpoint,
      "/irodori/mix/speakers",
    );
    if (!response.ok) throw new Error(await responseError(response));
    speakers.value = (await response.json()) as Speaker[];
    rangeSpeaker.value = speakers.value[0]?.id ?? null;
    await fetchRecipes().catch(() => {
      recipes.value = [];
    });
    try {
      const saved = localStorage.getItem(draftKey);
      if (saved) {
        const draft = JSON.parse(saved) as { tokens?: unknown; text?: unknown };
        const normalized = sanitizedTokens(draft.tokens);
        if (normalized) tokens.value = normalized;
        if (typeof draft.text === "string") previewText.value = draft.text;
      }
    } catch {
      /* Corrupt local draft should not prevent editing. */
    }
    draftLoaded.value = true;
  } catch (cause) {
    error.value = true;
    status.value =
      cause instanceof Error ? cause.message : "話者一覧を取得できませんでした";
  }
});
watch(
  [tokens, previewText],
  () => {
    if (!draftLoaded.value) return;
    try {
      localStorage.setItem(
        draftKey,
        JSON.stringify({ tokens: tokens.value, text: previewText.value }),
      );
    } catch {
      /* The editor still works when local storage is unavailable. */
    }
  },
  { deep: true },
);
onUnmounted(() => {
  for (const item of previewHistory.value) URL.revokeObjectURL(item.url);
});
</script>

<style scoped>
.mix-editor {
  --mix-line: rgba(var(--color-display-rgb), 0.13);
  --mix-muted: rgba(var(--color-display-rgb), 0.62);
  --mix-tint: rgba(var(--color-primary-rgb), 0.1);
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  color: var(--color-display);
  background: var(--color-background);
}
.studio-header {
  flex: none;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
  padding: 18px 28px;
  border-bottom: 1px solid var(--mix-line);
  background: var(--color-surface);
}
.eyebrow,
.panel-kicker {
  color: var(--color-primary);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.12em;
}
.studio-heading h1 {
  margin: 2px 0 0;
  font-size: 24px;
  line-height: 1.25;
  font-weight: 700;
}
.header-subtitle {
  color: var(--mix-muted);
  font-size: 12px;
  margin-top: 3px;
}
.header-actions {
  display: flex;
  align-items: center;
  gap: 16px;
}
.draft-indicator {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  color: var(--mix-muted);
  font-size: 12px;
  white-space: nowrap;
}
.draft-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--color-primary);
}
.studio-body {
  flex: 1;
  min-height: 0;
  overflow: auto;
  padding: 22px 26px 30px;
}
.metric {
  display: inline-flex;
  align-items: baseline;
  gap: 5px;
  padding-right: 18px;
  border-right: 1px solid var(--mix-line);
  font-size: 12px;
  color: var(--mix-muted);
}
.metric strong {
  color: var(--color-display);
  font-size: 20px;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
}
.overlap-hint {
  color: var(--color-warning);
  font-size: 12px;
}
.recipe-row {
  padding: 6px 0 14px;
}
.studio-workspace {
  display: grid;
  grid-template-columns: minmax(190px, 240px) minmax(0, 1fr) minmax(
      270px,
      320px
    );
  gap: 14px;
  align-items: stretch;
  min-height: 460px;
}
.studio-library,
.studio-canvas,
.studio-inspector,
.studio-preview,
.studio-save {
  min-width: 0;
  border: 1px solid var(--mix-line);
  border-radius: 14px;
  background: var(--color-surface);
}
.studio-library,
.studio-inspector {
  padding: 18px;
}
.studio-canvas {
  padding: 18px 18px 12px;
  background:
    linear-gradient(135deg, var(--mix-tint), transparent 32%),
    var(--color-surface);
}
.studio-workspace h2,
.studio-bottom h2 {
  margin: 4px 0 12px;
  font-size: 16px;
  line-height: 1.35;
  font-weight: 700;
}
.studio-library p {
  margin: 0 0 14px;
  color: var(--mix-muted);
  font-size: 12px;
  line-height: 1.5;
}
.speaker-search {
  margin-bottom: 12px;
}
.speaker-list {
  display: flex;
  flex-direction: column;
  gap: 7px;
  max-height: 390px;
  overflow-y: auto;
  padding-right: 2px;
}
.speaker-chip {
  display: flex;
  align-items: center;
  width: 100%;
  min-width: 0;
  gap: 9px;
  padding: 7px 8px;
  border: 1px solid var(--mix-line);
  border-radius: 10px;
  background: var(--color-background);
  cursor: grab;
  transition:
    border-color 0.15s,
    background 0.15s;
}
.speaker-chip:hover {
  border-color: var(--color-primary);
  background: var(--mix-tint);
}
.speaker-avatar {
  display: inline-grid;
  flex: none;
  place-items: center;
  width: 30px;
  height: 30px;
  border-radius: 9px;
  border: 2px solid;
  background: var(--color-surface);
  color: var(--color-display);
  font-size: 13px;
  font-weight: 700;
}
.speaker-name {
  min-width: 0;
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 12px;
}
.range-controls {
  display: grid;
  gap: 8px;
}
.range-speaker {
  min-width: 0;
  width: 100%;
}
.range-bounds {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}
.range-actions {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  margin-top: 10px;
}
.canvas-heading {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: end;
}
.canvas-heading span,
.canvas-footnote {
  color: var(--mix-muted);
  font-size: 11px;
  line-height: 1.5;
}
.canvas-footnote {
  margin-top: 14px;
}
.token-scroll {
  overflow-x: auto;
  padding-bottom: 8px;
}
.token-grid {
  display: grid;
  min-width: 760px;
  grid-template-columns: repeat(16, minmax(40px, 1fr));
  gap: 2px;
}
.token-column {
  cursor: pointer;
  padding: 4px;
  border: 0;
  border-radius: 8px;
}
.token-column:hover {
  background: rgba(var(--color-primary-rgb), 0.06);
}
.token-column.selected {
  border: 0;
  background: var(--mix-tint);
  box-shadow: inset 0 0 0 1px rgba(var(--color-primary-rgb), 0.5);
}
.token-column:focus-visible {
  outline: 2px solid var(--color-primary);
  outline-offset: 2px;
}
.column-space {
  display: flex;
  align-items: end;
  height: 285px;
  touch-action: none;
  background: repeating-linear-gradient(
    to top,
    var(--mix-line) 0,
    var(--mix-line) 1px,
    transparent 1px,
    transparent 25%
  );
}
.column-bar {
  display: flex;
  width: 100%;
  min-height: 0;
  position: relative;
  border-radius: 5px 5px 0 0;
}
.vertex-dot {
  position: absolute;
  top: -5px;
  left: calc(50% - 5px);
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--color-display);
  border: 2px solid var(--color-surface);
  box-shadow: 0 1px 5px rgba(0, 0, 0, 0.2);
}
.bar-piece {
  height: 100%;
}
.token-label {
  text-align: center;
  padding-top: 5px;
  color: var(--mix-muted);
  font-size: 12px;
  font-variant-numeric: tabular-nums;
}
.token-column.selected .token-label {
  color: var(--color-display);
  font-weight: 700;
}
.inspector-content {
  display: block;
  margin: 0 !important;
}
.inspector-content > div {
  width: 100%;
  max-width: 100%;
  margin: 0 0 22px !important;
}
.inspector-content > div + div {
  padding-top: 18px;
  border-top: 1px solid var(--mix-line);
}
.inspector-content .entry-label {
  width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.inspector-content .entry-value {
  width: 94px;
  text-align: right;
  color: var(--color-display);
  font-variant-numeric: tabular-nums;
}
.studio-bottom {
  display: grid;
  grid-template-columns: minmax(0, 1.3fr) minmax(280px, 0.7fr);
  gap: 14px;
  margin-top: 14px;
}
.studio-preview,
.studio-save {
  padding: 18px;
}
.preview-controls {
  display: flex;
  align-items: end;
  gap: 12px;
}
.preview-text {
  flex: 1;
  min-width: 220px;
}
.save-controls {
  display: grid;
  gap: 12px;
  margin-top: 22px;
}
.save-name {
  min-width: 0;
}
.save-controls :deep(.q-btn) {
  justify-self: start;
}
.preview-row {
  border: 0;
  border-top: 1px solid var(--mix-line);
  border-radius: 0;
  padding: 10px 0 0;
}
.preview-row audio {
  min-width: 230px;
  max-width: 100%;
  height: 36px;
}
.preview-summary {
  color: var(--mix-muted);
  font-size: 12px;
}
.studio-bottom > [role="status"] {
  grid-column: 1 / -1;
  margin: 0 !important;
}
@media (max-width: 1180px) {
  .studio-workspace {
    grid-template-columns: 200px minmax(0, 1fr);
  }
  .studio-inspector {
    grid-column: 1 / -1;
  }
  .inspector-content {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
  }
  .inspector-content > div + div {
    padding-top: 0;
    border-top: 0;
  }
}
@media (max-width: 760px) {
  .studio-header {
    padding: 14px 16px;
  }
  .draft-indicator,
  .header-subtitle {
    display: none;
  }
  .studio-body {
    padding: 16px;
  }
  .studio-workspace,
  .studio-bottom {
    grid-template-columns: 1fr;
  }
  .studio-inspector {
    grid-column: auto;
  }
  .inspector-content {
    display: block;
  }
  .inspector-content > div + div {
    padding-top: 18px;
    border-top: 1px solid var(--mix-line);
  }
  .speaker-list {
    max-height: 160px;
  }
  .speaker-avatar {
    width: 34px;
    height: 34px;
  }
  .preview-controls {
    display: grid;
  }
}
</style>
