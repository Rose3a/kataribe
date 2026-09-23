<template>
  <QBtn
    ref="buttonRef"
    flat
    class="q-pa-none character-button"
    :class="{ 'with-label': showLabel, opaque: loading }"
    :disable="uiLocked"
    aria-haspopup="menu"
    :aria-label="selectedStyleInfo ? selectedVoiceInfoText : '話者を選択'"
  >
    <!-- q-imgだとdisableのタイミングで点滅する -->
    <div class="icon-container">
      <img
        v-if="selectedStyleInfo != undefined"
        class="q-pa-none q-ma-none"
        :src="selectedStyleInfo.iconPath"
        :alt="selectedVoiceInfoText"
      />
      <QAvatar v-else-if="!emptiable" rounded size="2rem" color="primary"
        ><span color="text-display-on-primary">?</span></QAvatar
      >
      <QIcon v-else name="person_add" size="2rem" aria-hidden="true" />
    </div>
    <span v-if="showLabel" class="character-button-label">
      {{ selectedCharacter ? selectedVoiceInfoText : placeholderLabel }}
    </span>
    <div v-if="loading" class="loading">
      <QSpinner color="primary" size="1.6rem" :thickness="7" />
    </div>
    <QMenu
      class="character-menu character-picker-menu"
      style="width: min(90vw, 27rem); min-width: min(90vw, 20rem)"
      transitionShow="none"
      transitionHide="none"
      :max-height="maxMenuHeight"
      @beforeShow="onMenuBeforeShow"
    >
      <QTabs
        v-if="folderTabs.length > 0"
        :model-value="activeFolder"
        dense
        align="left"
        outside-arrows
        mobile-arrows
        active-color="primary"
        indicator-color="primary"
        class="speaker-folder-tabs"
        @update:model-value="selectFolder"
      >
        <QTab
          v-if="hasUngroupedCharacters"
          :name="ROOT_FOLDER_TAB"
          label="その他"
        />
        <QTab
          v-for="folder in folderTabs"
          :key="folder"
          :name="folder"
          :label="folderTabLabel(folder)"
          :title="folder"
          class="speaker-folder-tab"
        />
      </QTabs>
      <QList class="character-item-container">
        <QItem
          v-if="selectedStyleInfo == undefined && !emptiable"
          class="warning-item row no-wrap items-center"
        >
          <span class="text-warning vertical-middle"
            >有効なスタイルが選択されていません</span
          >
        </QItem>
        <QItem
          v-if="characterInfos.length === 0"
          class="warning-item row no-wrap items-center"
        >
          <span class="text-warning vertical-middle"
            >選択可能なスタイルがありません</span
          >
        </QItem>
        <QItem v-if="emptiable" class="to-unselect-item q-pa-none">
          <QBtn
            v-close-popup
            flat
            noCaps
            class="full-width"
            :class="selectedCharacter == undefined && 'selected-background'"
            @click="$emit('update:selectedVoice', undefined)"
          >
            <span>選択解除</span>
          </QBtn>
        </QItem>
        <QVirtualScroll
          :items="visibleCharacterInfos"
          :virtual-scroll-item-size="48"
          class="speaker-virtual-list"
          v-slot="{ item: characterInfo, index: characterIndex }"
        >
          <QItem
            :key="characterInfo.metas.speakerUuid"
            class="q-pa-none"
            :class="isSelectedItem(characterInfo) && 'selected-character-item'"
          >
            <QBtnGroup flat class="col full-width">
              <QBtn
                v-close-popup
                flat
                noCaps
                class="col-grow speaker-name-button"
                :aria-label="characterInfo.metas.speakerName"
                :title="characterInfo.metas.speakerName"
                @click="onSelectSpeaker(characterInfo.metas.speakerUuid)"
                @mouseover="reassignSubMenuOpen(-1)"
                @mouseleave="reassignSubMenuOpen.cancel()"
              >
                <QAvatar rounded size="2rem" class="q-mr-md">
                  <QImg
                    v-if="characterInfo"
                    noSpinner
                    noTransition
                    :ratio="1"
                    :src="
                      getDefaultStyleWrapper(characterInfo.metas.speakerUuid)
                        .iconPath
                    "
                  />
                  <QAvatar
                    v-if="
                      showEngineInfo && characterInfo.metas.styles.length < 2
                    "
                    class="engine-icon"
                    rounded
                  >
                    <img
                      :src="
                        engineIcons[
                          getDefaultStyleWrapper(
                            characterInfo.metas.speakerUuid,
                          ).engineId
                        ]
                      "
                    />
                  </QAvatar>
                </QAvatar>
                <div class="speaker-name">
                  {{ characterInfo.metas.speakerName }}
                </div>
              </QBtn>
              <!-- スタイルが2つ以上あるものだけ、スタイル選択ボタンを表示する-->
              <template v-if="characterInfo.metas.styles.length >= 2">
                <QSeparator vertical />

                <div
                  class="flex items-center q-px-sm q-py-none cursor-pointer"
                  :class="
                    subMenuOpenFlags[characterIndex] && 'selected-background'
                  "
                  role="application"
                  :aria-label="`${characterInfo.metas.speakerName}のスタイル、マウスオーバーするか、右矢印キーを押してスタイル選択を表示できます`"
                  tabindex="0"
                  @mouseover="reassignSubMenuOpen(characterIndex)"
                  @mouseleave="reassignSubMenuOpen.cancel()"
                  @keyup.right="reassignSubMenuOpen(characterIndex)"
                >
                  <QIcon name="keyboard_arrow_right" color="grey-6" size="sm" />
                  <QMenu
                    v-model="subMenuOpenFlags[characterIndex]"
                    noParentEvent
                    anchor="top end"
                    self="top start"
                    transitionShow="none"
                    transitionHide="none"
                    class="character-menu"
                  >
                    <QList style="min-width: max-content">
                      <QItem
                        v-for="(style, styleIndex) in characterInfo.metas
                          .styles"
                        :key="styleIndex"
                        v-close-popup
                        clickable
                        activeClass="selected-style-item"
                        :active="
                          selectedVoice != undefined &&
                          style.styleId === selectedVoice.styleId
                        "
                        :aria-pressed="
                          selectedVoice != undefined &&
                          style.styleId === selectedVoice.styleId
                        "
                        role="button"
                        @click="
                          $emit('update:selectedVoice', {
                            engineId: style.engineId,
                            speakerId: characterInfo.metas.speakerUuid,
                            styleId: style.styleId,
                          })
                        "
                      >
                        <QAvatar rounded size="2rem" class="q-mr-md">
                          <QImg
                            noSpinner
                            noTransition
                            :ratio="1"
                            :src="
                              characterInfo.metas.styles[styleIndex].iconPath
                            "
                          />
                          <QAvatar
                            v-if="showEngineInfo"
                            rounded
                            class="engine-icon"
                          >
                            <img
                              :src="
                                engineIcons[
                                  characterInfo.metas.styles[styleIndex]
                                    .engineId
                                ]
                              "
                            />
                          </QAvatar>
                        </QAvatar>
                        <QItemSection v-if="style.styleName"
                          >{{ characterInfo.metas.speakerName }}（{{
                            style.styleName
                          }}）</QItemSection
                        >
                        <QItemSection v-else>{{
                          characterInfo.metas.speakerName
                        }}</QItemSection>
                      </QItem>
                    </QList>
                  </QMenu>
                </div>
              </template>
            </QBtnGroup>
          </QItem>
        </QVirtualScroll>
      </QList>
    </QMenu>
  </QBtn>
</template>

<script setup lang="ts">
import { debounce, QBtn } from "quasar";
import { computed, type Ref, ref } from "vue";
import { useStore } from "@/store";
import type { CharacterInfo, SpeakerId, Voice } from "@/type/preload";
import { formatCharacterStyleName } from "@/store/utility";
import { useEngineIcons } from "@/composables/useEngineIcons";

const props = withDefaults(
  defineProps<{
    characterInfos: CharacterInfo[];
    loading?: boolean;
    selectedVoice: Voice | undefined;
    showEngineInfo?: boolean;
    emptiable?: boolean;
    showLabel?: boolean;
    placeholderLabel?: string;
    uiLocked: boolean;
  }>(),
  {
    loading: false,
    showEngineInfo: false,
    emptiable: false,
    showLabel: false,
    placeholderLabel: "話者を選択",
  },
);

const emit = defineEmits({
  "update:selectedVoice": (selectedVoice: Voice | undefined) => {
    return (
      selectedVoice == undefined ||
      (typeof selectedVoice.engineId === "string" &&
        typeof selectedVoice.speakerId === "string" &&
        typeof selectedVoice.styleId === "number")
    );
  },
});

const store = useStore();

const selectedCharacter = computed(() => {
  const selectedVoice = props.selectedVoice;
  if (selectedVoice == undefined) return undefined;
  const character = props.characterInfos.find(
    (characterInfo) =>
      characterInfo.metas.speakerUuid === selectedVoice?.speakerId &&
      characterInfo.metas.styles.some(
        (style) =>
          style.engineId === selectedVoice.engineId &&
          style.styleId === selectedVoice.styleId,
      ),
  );
  return character;
});

const selectedVoiceInfoText = computed(() => {
  if (!selectedCharacter.value) {
    return "キャラクター未選択";
  }

  const speakerName = selectedCharacter.value.metas.speakerName;
  if (!selectedStyleInfo.value) {
    return speakerName;
  }

  const styleName = selectedStyleInfo.value.styleName;
  return formatCharacterStyleName(speakerName, styleName);
});

const isSelectedItem = (characterInfo: CharacterInfo) =>
  selectedCharacter.value != undefined &&
  characterInfo.metas.speakerUuid ===
    selectedCharacter.value?.metas.speakerUuid;

const selectedStyleInfo = computed(() => {
  const selectedVoice = props.selectedVoice;
  const style = selectedCharacter.value?.metas.styles.find(
    (style) =>
      style.engineId === selectedVoice?.engineId &&
      style.styleId === selectedVoice.styleId,
  );
  return style;
});

const engineIcons = useEngineIcons(() => store.state.engineManifests);

// A folder becomes a tab only when it contains more than one speaker.  Single
// speaker folders keep the established, flat list behaviour.
const ROOT_FOLDER_TAB = "__irodori_root__";
const activeFolder = ref(ROOT_FOLDER_TAB);
const folderTabs = computed(() => {
  const counts = new Map<string, number>();
  for (const characterInfo of props.characterInfos) {
    const folder = characterInfo.metas.irodoriFolder;
    if (folder) counts.set(folder, (counts.get(folder) ?? 0) + 1);
  }
  return [...counts]
    .filter(([, count]) => count >= 2)
    .map(([folder]) => folder);
});

const tabbedFolders = computed(() => new Set(folderTabs.value));
const folderIsTabbed = (folder: string | undefined) =>
  folder != undefined && tabbedFolders.value.has(folder);

const folderTabLabel = (folder: string) => {
  const maxLength = 12;
  if (folder.length <= maxLength) return folder;

  // Keep a suffix so similarly named folders such as EMBEDDINGS_0921 and
  // EMBEDDINGS_0922 remain distinguishable at a glance.
  const suffixLength = 3;
  return `${folder.slice(0, maxLength - suffixLength - 1)}…${folder.slice(-suffixLength)}`;
};

const hasUngroupedCharacters = computed(() =>
  props.characterInfos.some(
    (characterInfo) => !folderIsTabbed(characterInfo.metas.irodoriFolder),
  ),
);

const visibleCharacterInfos = computed(() =>
  activeFolder.value === ROOT_FOLDER_TAB
    ? props.characterInfos.filter(
        (characterInfo) => !folderIsTabbed(characterInfo.metas.irodoriFolder),
      )
    : props.characterInfos.filter(
        (characterInfo) =>
          characterInfo.metas.irodoriFolder === activeFolder.value,
      ),
);

const characterInfoBySpeaker = computed(() => {
  const infos = new Map<SpeakerId, CharacterInfo>();
  for (const info of props.characterInfos) {
    if (!infos.has(info.metas.speakerUuid)) {
      infos.set(info.metas.speakerUuid, info);
    }
  }
  return infos;
});
const defaultStyleIdBySpeaker = computed(() => {
  const styles = new Map<SpeakerId, number>();
  for (const defaultStyle of store.state.defaultStyleIds) {
    if (!styles.has(defaultStyle.speakerUuid)) {
      styles.set(defaultStyle.speakerUuid, defaultStyle.defaultStyleId);
    }
  }
  return styles;
});
const getDefaultStyleWrapper = (speakerUuid: SpeakerId) => {
  const characterInfo = characterInfoBySpeaker.value.get(speakerUuid);
  const defaultStyleId = defaultStyleIdBySpeaker.value.get(speakerUuid);
  const style =
    characterInfo?.metas.styles.find(
      (item) => item.styleId === defaultStyleId,
    ) ?? characterInfo?.metas.styles[0];
  if (style == undefined) throw new Error("defaultStyle == undefined");
  return style;
};

const onSelectSpeaker = (speakerUuid: SpeakerId) => {
  const style = getDefaultStyleWrapper(speakerUuid);
  emit("update:selectedVoice", {
    engineId: style.engineId,
    speakerId: speakerUuid,
    styleId: style.styleId,
  });
};

const subMenuOpenFlags = ref(
  [...Array(props.characterInfos.length)].map(() => false),
);

const reassignSubMenuOpen = debounce((idx: number) => {
  if (subMenuOpenFlags.value[idx]) return;
  const arr = [...Array(props.characterInfos.length)].map(() => false);
  arr[idx] = true;
  subMenuOpenFlags.value = arr;
}, 100);

const selectFolder = (folder: string | number | null) => {
  if (typeof folder !== "string") return;
  activeFolder.value = folder;
  reassignSubMenuOpen(-1);
};

// 高さを制限してメニューが下方向に展開されるようにする
const buttonRef: Ref<InstanceType<typeof QBtn> | undefined> = ref();
const heightLimit = "65vh"; // QMenuのデフォルト値
const maxMenuHeight = ref(heightLimit);
const updateMenuHeight = () => {
  if (buttonRef.value == undefined)
    throw new Error("buttonRef.value == undefined");
  const el = buttonRef.value.$el;
  if (!(el instanceof Element)) throw new Error("!(el instanceof Element)");
  const buttonRect = el.getBoundingClientRect();
  // QMenuは展開する方向のスペースが不足している場合、自動的に展開方向を変更してしまうためmax-heightで制限する。
  // AudioDetailよりボタンが下に来ることはないのでその最低高185pxに余裕を持たせた170pxを最小の高さにする。
  // pxで指定するとウインドウサイズ変更に追従できないので ウインドウの高さの96% - ボタンの下端の座標 でメニューの高さを決定する。
  maxMenuHeight.value = `max(170px, min(${heightLimit}, calc(96vh - ${buttonRect.bottom}px)))`;
};

// Opening the menu again returns to the selected speaker's folder instead of
// the top of one long list.  This is also useful after changing a line's voice.
const onMenuBeforeShow = () => {
  const selectedFolder = selectedCharacter.value?.metas.irodoriFolder;
  const defaultFolder = hasUngroupedCharacters.value
    ? ROOT_FOLDER_TAB
    : (folderTabs.value[0] ?? ROOT_FOLDER_TAB);
  activeFolder.value =
    selectedFolder && folderIsTabbed(selectedFolder)
      ? selectedFolder
      : defaultFolder;
  reassignSubMenuOpen(-1);
  updateMenuHeight();
};
</script>

<style scoped lang="scss">
@use "@/styles/colors" as colors;

.character-button {
  border: solid 1px;
  border-color: colors.$primary;
  font-size: 0;
  height: fit-content;

  background: colors.$background;

  &.with-label {
    min-width: 0;
    width: 100%;
    font-size: 0.875rem;

    :deep(.q-btn__content) {
      justify-content: flex-start;
      flex-wrap: nowrap;
      min-width: 0;
    }
  }

  .character-button-label {
    margin-left: 8px;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .icon-container {
    height: 2rem;
    width: 2rem;

    img {
      max-height: 100%;
      max-width: 100%;
      object-fit: scale-down;
    }
  }

  .loading {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    margin: auto;
    background-color: rgba(colors.$background-rgb, 0.74);
    display: grid;
    justify-content: center;
    align-content: center;

    svg {
      filter: drop-shadow(0 0 1px colors.$background);
    }
  }
}

.opaque {
  opacity: 1 !important;
}

.character-menu {
  // Keep the popup stable when tabs have speakers with differently sized names.
  width: min(90vw, 27rem);

  .speaker-folder-tabs {
    min-width: 0;
    max-width: 100%;
    border-bottom: 1px solid rgba(colors.$primary-rgb, 0.2);

    .speaker-folder-tab {
      flex: 0 0 6rem;
      max-width: 6rem;
      min-width: 0;
      overflow: hidden;
    }

    :deep(.q-tab__label) {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
  }

  .character-item-container {
    display: flex;
    flex-direction: column;
    min-width: 0;
    width: 100%;
  }

  .speaker-virtual-list {
    max-height: 55vh;
    overflow: auto;
  }

  .q-item {
    color: colors.$display;
  }

  .q-btn-group {
    min-width: 0;

    > .q-btn:first-child > :deep(.q-btn__content) {
      justify-content: flex-start;
      min-width: 0;
      width: 100%;
    }

    > div:last-child:hover {
      background-color: rgba(colors.$primary-rgb, 0.1);
    }
  }

  .speaker-name-button {
    min-width: 0;
  }

  .speaker-name {
    flex: 1 1 0;
    min-width: 0;
    overflow: hidden;
    text-align: left;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .warning-item {
    order: -3;
  }
  .to-unselect-item {
    order: -2;
  }

  .selected-character-item,
  .selected-style-item,
  .selected-background {
    background-color: rgba(colors.$primary-rgb, 0.2);
  }

  .engine-icon {
    position: absolute;
    width: 13px;
    height: 13px;
    bottom: -6px;
    right: -6px;
  }
}

</style>

<style lang="scss">
// QMenu is teleported outside the component, so its layout rules are unscoped.
.character-picker-menu {
  width: min(90vw, 27rem);
  min-width: min(90vw, 20rem);

  .speaker-folder-tabs {
    min-width: 0;
    max-width: 100%;
  }

  .speaker-folder-tab {
    flex: 0 0 6rem;
    max-width: 6rem;
    min-width: 0;
    overflow: hidden;
  }

  .speaker-folder-tabs .q-tab__label,
  .speaker-folder-tab .q-tab__content,
  .speaker-name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .speaker-name-button,
  .q-btn-group,
  .q-btn-group > .q-btn:first-child > .q-btn__content,
  .character-item-container {
    min-width: 0;
  }

  .character-item-container,
  .q-btn-group > .q-btn:first-child > .q-btn__content {
    width: 100%;
  }

  .speaker-virtual-list,
  .speaker-virtual-list .q-item,
  .speaker-virtual-list .q-btn-group {
    width: 100%;
  }

  .speaker-name {
    flex: 1 1 0;
    min-width: 0;
    text-align: left;
  }
}
</style>
