<template>
  <QDialog
    v-model="dialogOpened"
    maximized
    transitionShow="jump-up"
    transitionHide="jump-down"
    class="help-dialog transparent-backdrop"
  >
    <QLayout container view="hHh Lpr lff">
      <QPageContainer>
        <QHeader class="q-pa-sm">
          <QToolbar>
            <QToolbarTitle class="text-display">
              ヘルプ /
              {{ selectedPage.parent ? selectedPage.parent + " / " : ""
              }}{{ selectedPage.name }}
            </QToolbarTitle>
            <QBtn
              v-if="selectedPage.shouldShowOpenLogDirectoryButton"
              unelevated
              color="toolbar-button"
              textColor="toolbar-button-display"
              class="text-no-wrap text-bold q-mr-sm"
              @click="openLogDirectory"
            >
              ログフォルダを開く
            </QBtn>
            <!-- close button -->
            <QBtn
              round
              flat
              icon="close"
              color="display"
              aria-label="ヘルプを閉じる"
              @click="dialogOpened = false"
            />
          </QToolbar>
        </QHeader>
        <BaseNavigationView>
          <template #sidebar>
            <template v-for="(page, pageIndex) of pagedata" :key="pageIndex">
              <BaseListItem
                v-if="page.type === 'item'"
                :selected="selectedPageIndex === pageIndex"
                @click="selectedPageIndex = pageIndex"
              >
                {{ page.name }}
              </BaseListItem>
              <div v-else-if="page.type === 'separator'" class="list-label">
                {{ page.name }}
              </div>
            </template>
          </template>
          <QTabPanels v-model="selectedPageIndex">
            <QTabPanel
              v-for="(page, pageIndex) of pagedata"
              :key="pageIndex"
              :name="pageIndex"
              class="q-pa-none"
            >
              <Component
                :is="page.component"
                v-if="page.type === 'item'"
                v-bind="page.props"
              />
            </QTabPanel>
          </QTabPanels>
        </BaseNavigationView>
      </QPageContainer>
    </QLayout>
  </QDialog>
</template>

<script setup lang="ts">
import { computed, ref } from "vue";
import type { Component } from "vue";
import MarkdownView from "./HelpMarkdownViewSection.vue";
import OssLicense from "./HelpOssLicenseSection.vue";
import BaseListItem from "@/components/Base/BaseListItem.vue";
import BaseNavigationView from "@/components/Base/BaseNavigationView.vue";
import { useStore } from "@/store";
import type { OssLicenseInfo } from "@/domain/staticAssets";
import {
  irodoriEthicsNoticeSource,
  irodoriEthicsNoticeTerms,
} from "@/domain/irodoriEthicsNotice";

type PageItem = {
  type: "item";
  name: string;
  parent?: string;
  component: Component;
  props?: Record<string, unknown>;
  shouldShowOpenLogDirectoryButton?: boolean;
};
type PageSeparator = {
  type: "separator";
  name: string;
};
type PageData = PageItem | PageSeparator;

const dialogOpened = defineModel<boolean>("dialogOpened", { default: false });
const isIrodoriFork = import.meta.env.VITE_APP_NAME === "voicevox-irodori";

const store = useStore();

// エディタのOSSライセンス取得
const licenses = ref<OssLicenseInfo[]>();
void store.actions.GET_OSS_LICENSES().then((obj) => (licenses.value = obj));

const howToUse = ref<string>("");
void store.actions.GET_HOW_TO_USE_TEXT().then((obj) => (howToUse.value = obj));

const ossCommunityInfos = ref<string>("");
void store.actions
  .GET_OSS_COMMUNITY_INFOS()
  .then((obj) => (ossCommunityInfos.value = obj));

const qAndA = ref<string>("");
void store.actions.GET_Q_AND_A_TEXT().then((obj) => (qAndA.value = obj));

const contact = ref<string>("");
void store.actions.GET_CONTACT_TEXT().then((obj) => (contact.value = obj));

// アプリに同梱した情報と、起動中の各エンジンが返す情報を一つにまとめる。
// 同一項目は本文まで一致するときだけ重複を除く。
const allLicenses = computed<OssLicenseInfo[]>(() => {
  const entries = [
    ...(licenses.value ?? []),
    ...store.getters.GET_SORTED_ENGINE_INFOS.flatMap(
      (engineInfo) =>
        store.state.engineManifests[engineInfo.uuid]?.dependencyLicenses ?? [],
    ),
  ];
  const seen = new Set<string>();
  return entries.filter((license) => {
    const key = [
      license.name,
      license.version ?? "",
      license.license ?? "",
      license.text,
    ].join("\u0000");
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
});

const pagedata = computed(() => {
  const data: PageData[] = [
    {
      type: "item",
      name: "kataribeについて",
      component: MarkdownView,
      props: {
        markdown: ossCommunityInfos.value,
      },
    },
    {
      type: "item",
      name: "使い方",
      component: MarkdownView,
      props: {
        markdown: howToUse.value,
      },
    },
    ...(isIrodoriFork
      ? [
          {
            type: "item" as const,
            name: "利用上の注意",
            component: MarkdownView,
            props: {
              markdown: `${irodoriEthicsNoticeTerms}\n\n${irodoriEthicsNoticeSource}`,
            },
          },
        ]
      : []),
    {
      type: "item",
      name: "ライセンス・クレジット",
      component: OssLicense,
      props: {
        licenses: allLicenses.value,
      },
    },

    {
      type: "item",
      name: "よくあるご質問",
      component: MarkdownView,
      props: {
        markdown: qAndA.value,
      },
    },
    {
      type: "item",
      name: "お問い合わせ",
      component: MarkdownView,
      props: {
        markdown: contact.value,
      },
      shouldShowOpenLogDirectoryButton: true,
    },
  ];
  return data;
});

const selectedPageIndex = ref(0);

const selectedPage = computed(() => {
  if (pagedata.value[selectedPageIndex.value].type == "item") {
    return pagedata.value[selectedPageIndex.value] as PageItem;
  } else {
    throw new Error("selectedPageにはPageItem型の値を指定してください。");
  }
});

const openLogDirectory = () => window.backend.openLogDirectory();
</script>

<style scoped lang="scss">
@use "@/styles/v2/colors" as colors;
@use "@/styles/v2/variables" as vars;

.list-label {
  padding: vars.$padding-2;
  padding-bottom: vars.$padding-1;
  color: colors.$display-sub;
}

.help-dialog .q-layout-container :deep(.absolute-full) {
  right: 0 !important;
  .scroll {
    left: unset !important;
    right: unset !important;
    width: unset !important;
    max-height: unset;
  }
}

.q-tab-panels {
  display: contents;
}
</style>
