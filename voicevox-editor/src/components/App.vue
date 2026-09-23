<template>
  <ErrorBoundary>
    <TooltipProvider disableHoverableContent :delayDuration="500">
      <MenuBar
        v-if="openedEditor != undefined"
        :subMenuData
        :editor="openedEditor"
      />
      <KeepAlive>
        <Component
          :is="openedEditor == 'talk' ? TalkEditor : SingEditor"
          v-if="openedEditor != undefined"
          :key="openedEditor"
          :isEnginesReady
          :isProjectFileLoaded
          :startupError
          :startupStage
        />
      </KeepAlive>
      <IrodoriEthicsDialog
        v-if="isIrodoriFork"
        v-model:dialogOpened="isIrodoriEthicsDialogOpen"
        @reject="rejectIrodoriEthics"
        @accept="acceptIrodoriEthics"
      />
      <SpeakerPolicyReviewDialog
        v-if="isIrodoriFork"
        v-model:dialogOpened="isSpeakerPolicyReviewDialogOpen"
        :speakers="pendingSpeakerPolicyReviews"
        @defer="deferSpeakerPolicyReview"
        @accept="acceptSpeakerPolicyReview"
      />
      <AllDialog :isEnginesReady />
    </TooltipProvider>
  </ErrorBoundary>
</template>

<script setup lang="ts">
import { watch, onMounted, ref, computed, toRaw, watchEffect } from "vue";
import { useGtm } from "@gtm-support/vue-gtm";
import { TooltipProvider } from "reka-ui";
import { useCommonMenuBarData } from "./Menu/MenuBar/useCommonMenuBarData";
import TalkEditor from "@/components/Talk/TalkEditor.vue";
import SingEditor from "@/components/Sing/SingEditor.vue";
import type { EngineId } from "@/type/preload";
import ErrorBoundary from "@/components/ErrorBoundary.vue";
import { useStore } from "@/store";
import { useHotkeyManager } from "@/plugins/hotkeyPlugin";
import AllDialog from "@/components/Dialog/AllDialog.vue";
import IrodoriEthicsDialog from "@/components/Dialog/AcceptDialog/IrodoriEthicsDialog.vue";
import SpeakerPolicyReviewDialog, {
  type SpeakerPolicyReview,
} from "@/components/Dialog/SpeakerPolicyReviewDialog.vue";
import { IRODORI_ETHICS_NOTICE_VERSION } from "@/domain/irodoriEthicsNotice";
import MenuBar from "@/components/Menu/MenuBar/MenuBar.vue";
import { useMenuBarData as useTalkMenuBarData } from "@/components/Talk/menuBarData";
import { useMenuBarData as useSingMenuBarData } from "@/components/Sing/menuBarData";
import { setFontToCss, setThemeToCss } from "@/domain/dom";
import { concatMenuBarData } from "@/components/Menu/MenuBar/menuBarData";
import { isElectron } from "@/helpers/platform";
import { useElectronMenuBarData } from "@/backend/electron/renderer/menuBarData";
import { removeNullableAndBoolean } from "@/helpers/arrayHelper";

const store = useStore();
const isIrodoriFork = import.meta.env.VITE_APP_NAME === "voicevox-irodori";

// TODO: useMenuBarData系の関数をcomposableじゃなくする
const commonMenuBarData = useCommonMenuBarData(store);
const talkMenuBarData = useTalkMenuBarData(store);
const singMenuBarData = useSingMenuBarData(store);
const electronMenuBarData = useElectronMenuBarData(store);

const subMenuData = computed(() =>
  concatMenuBarData(
    removeNullableAndBoolean([
      commonMenuBarData,
      store.state.openedEditor === "talk" && talkMenuBarData,
      store.state.openedEditor === "song" && singMenuBarData,
      isElectron && electronMenuBarData,
    ]),
  ),
);

const openedEditor = computed(() => store.state.openedEditor);

const isIrodoriEthicsDialogOpen = ref(false);
const isSpeakerPolicyReviewDialogOpen = ref(false);
const pendingSpeakerPolicyReviews = ref<SpeakerPolicyReview[]>([]);

const policyFingerprint = (policy: string) => {
  let hash = 2166136261;
  for (const character of policy) {
    hash ^= character.codePointAt(0) ?? 0;
    hash = Math.imul(hash, 16777619);
  }
  return `${policy.length}:${(hash >>> 0).toString(16)}`;
};

const getIrodoriSpeakerPolicyReviews = (): SpeakerPolicyReview[] =>
  [...store.getters.GET_ALL_CHARACTER_INFOS.values()]
    .filter((characterInfo) =>
      characterInfo.metas.styles.some(
        (style) =>
          store.state.engineManifests[style.engineId]?.brandName ===
          "Irodori-TTS",
      ),
    )
    .map((characterInfo) => {
      // エンジンの既存 API は Markdown の改行として空白を補うため、
      // 確認画面では credit.txt 本来の改行に戻して表示する。
      const policy = characterInfo.metas.policy.replaceAll("  \n", "\n").trim();
      return {
        id: characterInfo.metas.speakerUuid,
        name: characterInfo.metas.speakerName,
        policy,
        fingerprint: policyFingerprint(policy || "(credit.txt missing)"),
      };
    });

const openSpeakerPolicyReviewIfNeeded = () => {
  if (
    !isIrodoriFork ||
    !isEnginesReady.value ||
    isIrodoriEthicsDialogOpen.value ||
    isSpeakerPolicyReviewDialogOpen.value
  ) {
    return;
  }

  const reviews = getIrodoriSpeakerPolicyReviews().filter(
    (review) =>
      store.state.reviewedIrodoriSpeakerPolicies[review.id] !==
      review.fingerprint,
  );
  if (reviews.length > 0) {
    pendingSpeakerPolicyReviews.value = reviews;
    isSpeakerPolicyReviewDialogOpen.value = true;
  }
};

const acceptIrodoriEthics = () => {
  void store.actions.SET_ROOT_MISC_SETTING({
    key: "irodoriEthicsNoticeVersion",
    value: IRODORI_ETHICS_NOTICE_VERSION,
  });
  isIrodoriEthicsDialogOpen.value = false;
  openSpeakerPolicyReviewIfNeeded();
};

const rejectIrodoriEthics = () => {
  isIrodoriEthicsDialogOpen.value = false;
  void store.actions.CHECK_EDITED_AND_NOT_SAVE({ nextAction: "close" });
};

const acceptSpeakerPolicyReview = () => {
  const reviewedSpeakerPolicies = {
    ...store.state.reviewedIrodoriSpeakerPolicies,
  };
  for (const review of pendingSpeakerPolicyReviews.value) {
    reviewedSpeakerPolicies[review.id] = review.fingerprint;
  }
  void store.actions.SET_ROOT_MISC_SETTING({
    key: "reviewedIrodoriSpeakerPolicies",
    value: reviewedSpeakerPolicies,
  });
  isSpeakerPolicyReviewDialogOpen.value = false;
};

const deferSpeakerPolicyReview = () => {
  isSpeakerPolicyReviewDialogOpen.value = false;
};

watch(
  () =>
    getIrodoriSpeakerPolicyReviews()
      .map((review) => `${review.id}:${review.fingerprint}`)
      .join("|"),
  () => {
    if (
      store.state.irodoriEthicsNoticeVersion === IRODORI_ETHICS_NOTICE_VERSION
    ) {
      openSpeakerPolicyReviewIfNeeded();
    }
  },
);

// Google Tag Manager
const gtm = useGtm();
watch(
  () => store.state.acceptRetrieveTelemetry,
  (acceptRetrieveTelemetry) => {
    gtm?.enable(!isIrodoriFork && acceptRetrieveTelemetry === "Accepted");
  },
  { immediate: true },
);

// フォントの制御用パラメータを変更する
watchEffect(() => {
  setFontToCss(store.state.editorFont);
});

// エディタの切り替えを監視してショートカットキーの設定を変更する
watchEffect(
  () => {
    if (openedEditor.value) {
      hotkeyManager.onEditorChange(openedEditor.value);
    }
  },
  { flush: "post" },
);

// テーマの変更を監視してCSS変数を変更する
watchEffect(() => {
  const theme = store.state.availableThemes.find((value) => {
    return value.name == store.state.currentTheme;
  });
  if (theme == undefined) {
    // NOTE: Vuexが初期化されていない場合はまだテーマが読み込まれていないので無視
    if (store.state.isVuexReady) {
      throw Error(`Theme not found: ${store.state.currentTheme}`);
    } else {
      return;
    }
  }
  setThemeToCss(theme);
});

// ソングの再生デバイスを同期
watchEffect(() => {
  void store.actions.APPLY_DEVICE_ID_TO_AUDIO_CONTEXT({
    device: store.state.savingSetting.audioOutputDevice,
  });
});

// ソフトウェアを初期化
const { hotkeyManager } = useHotkeyManager();
const isEnginesReady = ref(false);
const isProjectFileLoaded = ref<boolean | "waiting">("waiting");
const startupError = ref("");
const startupStage = ref("設定を読み込み中・・・");
const startupRetryKey = "irodori-editor-startup-retries";
const maxStartupRetries = 3;
onMounted(async () => {
  try {
    const queryString = window.location.search;
    const urlParams = new URLSearchParams(queryString);

    await store.actions.INIT_VUEX();

    // ショートカットキーの設定を登録
    const hotkeySettings = store.state.hotkeySettings;
    hotkeyManager.load(structuredClone(toRaw(hotkeySettings)));

    // エンジンの初期化開始

    // エンジン情報取得
    await store.actions.PULL_AND_INIT_ENGINE_INFOS();

    // URLパラメータに従ってマルチエンジンをオフにする
    const isMultiEngineOffMode =
      urlParams.get("isMultiEngineOffMode") === "true";
    void store.actions.SET_IS_MULTI_ENGINE_OFF_MODE(isMultiEngineOffMode);

    // マルチエンジンオフモードのときはデフォルトエンジンだけにする
    let engineIds: EngineId[];
    if (isMultiEngineOffMode) {
      const main = Object.values(store.state.engineInfos).find(
        (engine) => engine.isDefault,
      );
      if (!main) {
        throw new Error("No default engine found");
      }
      engineIds = [main.uuid];
    } else {
      engineIds = store.state.engineIds;
    }
    await store.actions.LOAD_USER_CHARACTER_ORDER();
    startupStage.value = "話者情報を読み込み中・・・";
    const engineStart = await store.actions.POST_ENGINE_START({
      engineIds,
      onCharacterProgress: (_engineId, completed, total) => {
        startupStage.value = `話者情報を読み込み中・・・ ${completed}/${total}`;
      },
    });
    if (!engineStart.success) {
      throw new Error("エンジンの準備が完了しませんでした");
    }

    // 辞書を同期
    startupStage.value = "辞書を同期中・・・";
    await store.actions.SYNC_ALL_USER_DICT();

    isEnginesReady.value = true;

    if (isIrodoriFork) {
      if (
        store.state.irodoriEthicsNoticeVersion !== IRODORI_ETHICS_NOTICE_VERSION
      ) {
        isIrodoriEthicsDialogOpen.value = true;
      } else {
        openSpeakerPolicyReviewIfNeeded();
      }
    }

    // エンジン起動後にダイアログを開く
    void store.actions.SET_DIALOG_OPEN({
      isAcceptRetrieveTelemetryDialogOpen:
        !isIrodoriFork && store.state.acceptRetrieveTelemetry === "Unconfirmed",
      isAcceptTermsDialogOpen:
        !isIrodoriFork &&
        import.meta.env.MODE !== "development" &&
        store.state.acceptTerms !== "Accepted",
    });

    // プロジェクトファイルが指定されていればロード
    const projectFilePath = await store.actions.GET_INITIAL_PROJECT_FILE_PATH();
    if (projectFilePath != undefined) {
      isProjectFileLoaded.value = await store.actions.LOAD_PROJECT_FILE({
        type: "path",
        filePath: projectFilePath,
      });
    } else {
      isProjectFileLoaded.value = false;
    }
    if (isIrodoriFork) {
      try {
        sessionStorage.removeItem(startupRetryKey);
      } catch {
        // Storage may be disabled by the browser; startup itself succeeded.
      }
    }
  } catch (error) {
    window.backend.logError(
      error,
      `Editor startup failed: ${startupStage.value}`,
    );
    startupError.value = error instanceof Error ? error.message : String(error);
    if (isIrodoriFork) {
      try {
        const previous = Number(sessionStorage.getItem(startupRetryKey) ?? 0);
        if (Number.isFinite(previous) && previous < maxStartupRetries) {
          const attempt = previous + 1;
          sessionStorage.setItem(startupRetryKey, String(attempt));
          startupStage.value += `・自動再試行 ${attempt}/${maxStartupRetries}`;
          window.setTimeout(() => window.location.reload(), 1000 * attempt);
        }
      } catch {
        // Keep the visible error and manual reload when storage is unavailable.
      }
    }
  }
});
</script>
