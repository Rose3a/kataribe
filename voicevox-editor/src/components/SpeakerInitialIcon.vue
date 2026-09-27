<template>
  <div
    class="speaker-initial-icon"
    :class="{ wide: look.initials.length > 1 }"
    :style="{ '--hue': look.hue }"
    role="img"
    :aria-label="name"
  >
    <span aria-hidden="true">{{ look.initials }}</span>
  </div>
</template>

<script setup lang="ts">
/**
 * 画像の無い話者のアイコン。話者名から色と頭文字を決めるので、
 * 立ち絵欄（SpeakerAvatar）・セリフ一覧・話者選択で同じ見た目になる。
 * 四角く描くので、画像アイコンと同じく枠いっぱいに収まる（丸くするのは置き場所の側）。
 */
import { computed } from "vue";
import { avatarLook } from "@/helpers/speakerAvatar";

const props = defineProps<{ name: string }>();
const look = computed(() => avatarLook(props.name));
</script>

<style scoped lang="scss">
.speaker-initial-icon {
  width: 100%;
  height: 100%;
  container-type: size;
  display: grid;
  place-items: center;
  overflow: hidden;
  color: #fff;
  font-weight: 700;
  line-height: 1;
  background: linear-gradient(
    135deg,
    hsl(var(--hue) 70% 60%),
    hsl(calc(var(--hue) + 40) 62% 40%)
  );
  span {
    font-size: 42cqmin;
    text-shadow: 0 2px 8px rgb(0 0 0 / 0.25);
  }
  &.wide span {
    font-size: 30cqmin;
    letter-spacing: -0.02em;
  }
}
</style>
