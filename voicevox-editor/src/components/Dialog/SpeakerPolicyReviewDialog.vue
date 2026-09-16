<template>
  <QDialog
    v-model="dialogOpened"
    persistent
    maximized
    class="transparent-backdrop"
  >
    <QLayout container view="hHh Lpr lff" class="bg-background">
      <QHeader class="q-py-sm">
        <QToolbar>
          <QToolbarTitle class="text-display">話者の利用条件</QToolbarTitle>
          <QSpace />
          <QBtn
            unelevated
            label="あとで確認"
            color="toolbar-button"
            textColor="toolbar-button-display"
            class="text-no-wrap q-mr-md text-bold"
            @click="$emit('defer')"
          />
          <QBtn
            unelevated
            label="確認しました"
            color="toolbar-button"
            textColor="toolbar-button-display"
            class="text-no-wrap text-bold"
            @click="$emit('accept')"
          />
        </QToolbar>
      </QHeader>

      <QPageContainer>
        <QPage class="page q-pa-md">
          <div class="content">
            <p>
              新しく検出または条件が更新された話者の利用条件です。話者ごとに条件が異なるため、使用前に確認してください。
            </p>
            <QCard v-for="speaker in speakers" :key="speaker.id" flat bordered>
              <QCardSection>
                <div class="text-h6">{{ speaker.name }}</div>
              </QCardSection>
              <QSeparator />
              <QCardSection v-if="speaker.policy" class="policy-text">
                {{ speaker.policy }}
              </QCardSection>
              <QCardSection v-else class="text-warning">
                <strong>credit.txt が見つかりません。</strong><br />
                この話者の利用条件は確認できません。配布元のページ・同梱資料を確認してください。
              </QCardSection>
            </QCard>
          </div>
        </QPage>
      </QPageContainer>
    </QLayout>
  </QDialog>
</template>

<script setup lang="ts">
export type SpeakerPolicyReview = {
  id: string;
  name: string;
  policy: string;
  fingerprint: string;
};

const dialogOpened = defineModel<boolean>("dialogOpened", { default: false });

defineProps<{
  speakers: SpeakerPolicyReview[];
}>();

defineEmits<{
  defer: [];
  accept: [];
}>();
</script>

<style scoped lang="scss">
@use "@/styles/v2/colors" as colors;

.page {
  background-color: colors.$background;
  color: colors.$display;
}

.content {
  max-width: 960px;
  margin: auto;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.policy-text {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
</style>
