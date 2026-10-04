<template>
  <QDialog
    v-model="dialogOpened"
    maximized
    transitionShow="jump-up"
    transitionHide="jump-down"
    class="setting-dialog transparent-backdrop"
    persistent
  >
    <QLayout>
      <QPageContainer>
        <QHeader class="q-pa-sm">
          <QToolbar>
            <QToolbarTitle class="text-display"
              >読めない語句の辞書（語彙分割辞書）</QToolbarTitle
            >
            <QSpace />
            <QBtn
              round
              flat
              icon="close"
              color="display"
              aria-label="辞書を閉じる"
              @click="closeDialog"
            />
          </QToolbar>
        </QHeader>
        <BaseNavigationView>
          <template #sidebar>
            <div class="list-header">
              <BaseToggleGroup v-model="listKind" type="single">
                <BaseToggleGroupItem
                  :label="`自分の登録 ${userEntries.length}`"
                  value="user"
                />
                <BaseToggleGroupItem
                  :label="`自動 ${autoEntries.length}`"
                  value="auto"
                />
              </BaseToggleGroup>
              <BaseButton
                label="追加"
                icon="add"
                :disabled="busy"
                @click="selectNew"
              />
            </div>
            <BaseTextField
              v-model="filter"
              placeholder="絞り込み（単語・書き換え・メモ）"
              ariaLabel="絞り込み"
            />
            <div class="list">
              <BaseListItem
                v-for="entry in shownEntries"
                :key="entry.surface"
                :selected="current?.surface === entry.surface && !isNew"
                @click="select(entry)"
              >
                <div class="listitem">
                  <span class="listitem-surface">{{ entry.surface }}</span>
                  <span class="listitem-text">
                    {{
                      entry.text === entry.surface ? "（未設定）" : entry.text
                    }}
                  </span>
                </div>
              </BaseListItem>
            </div>
          </template>

          <div class="detail">
            <BaseScrollArea>
              <div v-if="current != undefined" class="inner">
                <h2 class="title">
                  {{ isNew ? "新しい語句の追加" : current.surface }}
                </h2>
                <div v-if="current.note" class="note">{{ current.note }}</div>
                <div v-if="current.method" class="note">
                  自動の登録（{{ current.method }}、OK率
                  {{ current.ok_before }} → {{ current.ok_after }}）。
                  保存すると自分の登録として上書きします。
                </div>

                <div class="form-row">
                  <h3 class="headline">語句</h3>
                  <BaseTextField
                    v-model="surface"
                    ariaLabel="語句"
                    :readonly="!isNew"
                    :hasError="surface.trim() === ''"
                  >
                    <template #error>語句は必須です。</template>
                  </BaseTextField>
                  <TokenView
                    label="そのまま渡るトークン"
                    :tokens="tokenView?.results[0]?.tokens"
                  />
                </div>

                <div class="form-row">
                  <h3 class="headline">書き換え</h3>
                  <div>
                    書いたとおりにモデルへ渡します。漢字・ひらがな・カタカナが使えます。
                    「|」は見えない区切り（間を入れずにトークンを分ける）、
                    「[ZW]」はゼロ幅スペース、空白は区切って読みます。
                    語句と同じにすると何もしません（自動の登録を止めるときにも使えます）。
                  </div>
                  <BaseTextField
                    v-model="text"
                    ariaLabel="書き換え"
                    :hasError="text.trim() === ''"
                  >
                    <template #error>書き換えは必須です。</template>
                  </BaseTextField>
                  <TokenView
                    label="書き換えで渡るトークン"
                    :tokens="tokenView?.results[1]?.tokens"
                  />
                  <div v-if="tokenView?.source === 'fallback'" class="note">
                    モデルが未読み込みのため、既定のトークナイザで分けています。
                  </div>
                </div>

                <div class="form-row">
                  <h3 class="headline">メモ</h3>
                  <BaseTextField v-model="note" ariaLabel="メモ" />
                </div>

                <div class="form-row">
                  <h3 class="headline">聞き比べ</h3>
                  <div>
                    台本の {X}
                    に語句を入れて読ませます。どちらも辞書は当てずに、書いたとおりに合成します。
                    （声: {{ voice.name }}）
                  </div>
                  <BaseTextField v-model="carrier" ariaLabel="台本" />
                  <div class="buttons">
                    <BaseButton
                      :label="playing === 'original' ? '停止' : '原文を再生'"
                      :icon="playing === 'original' ? 'stop' : 'play_arrow'"
                      :disabled="
                        busy || (playing != undefined && playing !== 'original')
                      "
                      @click="togglePlay('original')"
                    />
                    <BaseButton
                      :label="playing === 'rewrite' ? '停止' : '書き換えを再生'"
                      :icon="playing === 'rewrite' ? 'stop' : 'play_arrow'"
                      :disabled="
                        busy ||
                        text.trim() === '' ||
                        (playing != undefined && playing !== 'rewrite')
                      "
                      @click="togglePlay('rewrite')"
                    />
                  </div>
                </div>
              </div>
              <div v-else class="inner empty">
                左の一覧から語句を選ぶか、「追加」で新しく登録してください。
              </div>
            </BaseScrollArea>
            <footer v-if="current != undefined" class="footer">
              <BaseButton
                v-if="isUserEntry"
                label="削除"
                variant="danger"
                :disabled="busy"
                @click="remove"
              />
              <BaseButton
                label="保存"
                variant="primary"
                :disabled="
                  busy || !dirty || surface.trim() === '' || text.trim() === ''
                "
                @click="save"
              />
            </footer>
          </div>
        </BaseNavigationView>
      </QPageContainer>
    </QLayout>
  </QDialog>
</template>

<script setup lang="ts">
import { computed, ref, watch } from "vue";
import TokenView from "./TokenView.vue";
import BaseButton from "@/components/Base/BaseButton.vue";
import BaseListItem from "@/components/Base/BaseListItem.vue";
import BaseNavigationView from "@/components/Base/BaseNavigationView.vue";
import BaseScrollArea from "@/components/Base/BaseScrollArea.vue";
import BaseTextField from "@/components/Base/BaseTextField.vue";
import BaseToggleGroup from "@/components/Base/BaseToggleGroup.vue";
import BaseToggleGroupItem from "@/components/Base/BaseToggleGroupItem.vue";
import type { IrodoriTokenView, TokenSplitEntry } from "@/domain/irodori";
import { createEngineUrl } from "@/domain/url";
import {
  deleteTokenSplit,
  fetchTokenView,
  listTokenSplit,
  putTokenSplit,
} from "@/helpers/irodoriEngine";
import { useStore } from "@/store";
import { UnreachableError } from "@/type/utility";

const dialogOpened = defineModel<boolean>("dialogOpened", { default: false });
const store = useStore();

const userEntries = ref<TokenSplitEntry[]>([]);
const autoEntries = ref<TokenSplitEntry[]>([]);
const listKind = ref<"user" | "auto">("user");
const filter = ref("");
const busy = ref(false);

const current = ref<TokenSplitEntry | undefined>();
const isNew = ref(false);
const surface = ref("");
const text = ref("");
const note = ref("");
const carrier = ref("これは{X}です。");
const tokenView = ref<IrodoriTokenView | undefined>();
const playing = ref<"original" | "rewrite" | undefined>();

const shownEntries = computed(() => {
  const source =
    listKind.value === "user" ? userEntries.value : autoEntries.value;
  const query = filter.value.trim();
  if (query === "") return source;
  return source.filter((entry) =>
    [entry.surface, entry.text, entry.note ?? ""].some((value) =>
      value.includes(query),
    ),
  );
});
const isUserEntry = computed(
  () =>
    !isNew.value &&
    userEntries.value.some((e) => e.surface === current.value?.surface),
);
const dirty = computed(
  () =>
    current.value != undefined &&
    (isNew.value ||
      !isUserEntry.value ||
      text.value !== current.value.text ||
      note.value !== (current.value.note ?? "")),
);

// 聞き比べの声はつくよみちゃん（いなければ先頭のキャラクター）
const PREFERRED_SPEAKER = "つくよみちゃん";
const voice = computed(() => {
  const infos = store.getters.USER_ORDERED_CHARACTER_INFOS("talk");
  if (infos == undefined || infos.length === 0)
    throw new UnreachableError("assert USER_ORDERED_CHARACTER_INFOS");
  const metas = (
    infos.find((info) => info.metas.speakerName === PREFERRED_SPEAKER) ??
    infos[0]
  ).metas;
  const { engineId, styleId } = metas.styles[0];
  return {
    engineId,
    speakerId: metas.speakerUuid,
    styleId,
    name: metas.speakerName,
  };
});
const endpoint = computed(() => {
  const { engineId } = voice.value;
  const info = store.state.engineInfos[engineId];
  if (info == undefined) return undefined;
  return createEngineUrl({
    ...info,
    port: store.state.altPortInfos[engineId] ?? info.defaultPort,
  });
});

const showError = (title: string, error: unknown) => {
  void store.actions.SHOW_ALERT_DIALOG({
    title,
    message: error instanceof Error ? error.message : String(error),
  });
};

const load = async () => {
  if (endpoint.value == undefined) return;
  try {
    const result = await listTokenSplit(endpoint.value);
    userEntries.value = result.user;
    autoEntries.value = result.auto;
  } catch (error) {
    showError("辞書を読み込めませんでした", error);
  }
};
watch(dialogOpened, (opened) => {
  if (opened) void load();
});

const confirmDiscard = async () => {
  if (!dirty.value || current.value == undefined) return true;
  if (!isNew.value && !isUserEntry.value) return true; // 自動の登録を眺めているだけ
  const result = await store.actions.SHOW_WARNING_DIALOG({
    title: "保存していない変更があります",
    message: "変更を破棄しますか？",
    actionName: "破棄する",
    cancel: "戻る",
  });
  return result === "OK";
};

const edit = (entry: TokenSplitEntry, asNew: boolean) => {
  current.value = entry;
  isNew.value = asNew;
  surface.value = entry.surface;
  text.value = entry.text;
  note.value = entry.note ?? "";
};
const select = async (entry: TokenSplitEntry) => {
  if (!(await confirmDiscard())) return;
  edit(entry, false);
};
const selectNew = async () => {
  if (!(await confirmDiscard())) return;
  edit({ surface: "", text: "" }, true);
};

const save = async () => {
  if (endpoint.value == undefined) return;
  busy.value = true;
  try {
    const saved = await putTokenSplit(endpoint.value, {
      surface: surface.value,
      text: text.value,
      note: note.value,
    });
    await load();
    listKind.value = "user";
    edit(saved, false);
  } catch (error) {
    showError("保存できませんでした", error);
  } finally {
    busy.value = false;
  }
};

const remove = async () => {
  if (endpoint.value == undefined || current.value == undefined) return;
  const result = await store.actions.SHOW_WARNING_DIALOG({
    title: "この語句を削除しますか？",
    message: `「${current.value.surface}」を自分の登録から削除します。`,
    actionName: "削除する",
    isWarningColorButton: true,
    cancel: "削除しない",
  });
  if (result !== "OK") return;
  busy.value = true;
  try {
    await deleteTokenSplit(endpoint.value, current.value.surface);
    current.value = undefined;
    await load();
  } catch (error) {
    showError("削除できませんでした", error);
  } finally {
    busy.value = false;
  }
};

// トークンの分け方（入力が止まって少ししてから問い合わせる）
let tokenTimer: ReturnType<typeof setTimeout> | undefined;
let tokenRequest = 0;
watch([surface, text], () => {
  clearTimeout(tokenTimer);
  tokenTimer = setTimeout(() => {
    const requestId = ++tokenRequest;
    if (endpoint.value == undefined) return;
    void fetchTokenView(endpoint.value, [surface.value, text.value]).then(
      (result) => {
        if (requestId === tokenRequest) tokenView.value = result;
      },
    );
  }, 300);
});

/** 辞書の表記（| と [ZW]）を合成に渡す文字へ。 */
function decodeNotation(value: string): string {
  return value.replaceAll("|", "⁣").replaceAll("[ZW]", "​");
}

const togglePlay = async (which: "original" | "rewrite") => {
  if (playing.value != undefined) {
    void store.actions.STOP_AUDIO();
    return;
  }
  // 合成時と同じく、書き換えた語の前後に見えない区切りを入れる（文頭・文末は不要）
  const rewritten = decodeNotation(text.value).replace(/^⁣+|⁣+$/g, "");
  const word =
    which === "original" || rewritten === surface.value
      ? surface.value
      : `⁣${rewritten}⁣`;
  const script = (
    carrier.value.includes("{X}") ? carrier.value.replaceAll("{X}", word) : word
  ).replace(/^⁣+|⁣+$/g, "");
  playing.value = which;
  try {
    const audioItem = await store.actions.GENERATE_AUDIO_ITEM({
      text: script,
      voice: {
        engineId: voice.value.engineId,
        speakerId: voice.value.speakerId,
        styleId: voice.value.styleId,
      },
    });
    // 書いたとおりに聞くため、語彙分割辞書は当てない
    audioItem.irodori = { ...audioItem.irodori, tokenSplit: "off" };
    const { blob } = await store.actions.FETCH_AUDIO_FROM_AUDIO_ITEM({
      audioItem,
    });
    await store.actions.PLAY_AUDIO_BLOB({ audioBlob: blob });
  } catch (error) {
    showError("生成に失敗しました", error);
  } finally {
    playing.value = undefined;
  }
};

const closeDialog = async () => {
  if (!(await confirmDiscard())) return;
  if (playing.value != undefined) void store.actions.STOP_AUDIO();
  dialogOpened.value = false;
};
</script>

<style lang="scss" scoped>
@use "@/styles/v2/colors" as colors;
@use "@/styles/v2/variables" as vars;
@use "@/styles/v2/mixin" as mixin;

.list-header {
  display: flex;
  gap: vars.$gap-1;
  align-items: center;
  justify-content: space-between;
  margin-bottom: vars.$padding-1;
}

.list {
  display: flex;
  flex-direction: column;
  width: 280px;
  margin-top: vars.$padding-1;
}

.listitem {
  display: flex;
  flex-direction: column;
  align-items: start;
  overflow: hidden;
  width: 100%;
}

.listitem-surface,
.listitem-text {
  width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.listitem-text {
  font-size: 0.75rem;
  color: colors.$display-sub;
}

.detail {
  display: flex;
  flex-flow: column;
  height: 100%;
}

.inner {
  max-width: 960px;
  margin: auto;
  width: 100%;
  display: flex;
  flex-direction: column;
  padding: vars.$padding-2;
  gap: vars.$gap-2;
}

.empty {
  color: colors.$display-sub;
}

.title {
  @include mixin.headline-1;
  word-break: break-all;
}

.headline {
  @include mixin.headline-2;
}

.form-row {
  display: flex;
  flex-flow: column;
  gap: vars.$gap-1;
}

.note {
  color: colors.$display-sub;
}

.buttons {
  display: flex;
  gap: vars.$gap-1;
}

.footer {
  padding: vars.$padding-2;
  display: flex;
  justify-content: flex-end;
  gap: vars.$gap-1;
}
</style>
