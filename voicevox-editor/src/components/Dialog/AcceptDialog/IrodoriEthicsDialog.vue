<template>
  <AcceptDialog
    v-model:dialogOpened="dialogOpened"
    title="Irodori-TTS を使う前の確認"
    rejectLabel="使用せずに終了"
    acceptLabel="確認して使用を開始"
    heading="利用前に確認してください"
    :terms
    persistent
    @reject="$emit('reject')"
    @accept="$emit('accept')"
  >
    <p>
      Irodori-TTS
      のモデル利用上の注意に加え、話者・参照音声・画像にはそれぞれ個別の利用条件があります。利用を始める前に、次の内容を確認してください。
    </p>
  </AcceptDialog>
</template>

<script setup lang="ts">
import AcceptDialog from "./AcceptDialog.vue";
import {
  irodoriEthicsNoticeSource,
  irodoriEthicsNoticeTerms,
} from "@/domain/irodoriEthicsNotice";

const dialogOpened = defineModel<boolean>("dialogOpened", { default: false });

defineEmits<{
  reject: [];
  accept: [];
}>();

const terms = `${irodoriEthicsNoticeTerms}\n\n${irodoriEthicsNoticeSource}`;
</script>
