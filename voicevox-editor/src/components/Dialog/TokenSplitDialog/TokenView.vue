<template>
  <div class="token-view">
    <span class="label">{{ props.label }}</span>
    <span v-if="props.tokens == undefined" class="empty">—</span>
    <template v-else>
      <template v-for="(token, index) in props.tokens" :key="index">
        <span
          v-if="'split' in token"
          class="split"
          title="見えない区切り（ここでトークンを分ける）"
          >|</span
        >
        <span
          v-else
          class="token"
          :class="{ rare: token.rare, dictionary: token.dictionary }"
          :title="describe(token)"
          >{{ display(token.text) }}</span
        >
      </template>
      <span v-if="props.tokens.some((t) => 'rare' in t && t.rare)" class="hint">
        色付きは学習の少ないトークン
      </span>
    </template>
  </div>
</template>

<script setup lang="ts">
import type { IrodoriToken } from "@/domain/irodori";

const props = defineProps<{
  label: string;
  tokens: IrodoriToken[] | undefined;
}>();

type Token = Exclude<IrodoriToken, { split: true }>;

function display(text: string): string {
  return text.replaceAll(" ", "␣").replaceAll("​", "[ZW]");
}

function describe(token: Token): string {
  const notes = [`ID ${token.id}`, `出現度 ${token.score}`];
  if (token.rare) notes.push("学習の少ないトークン");
  if (token.dictionary) notes.push("語彙分割辞書で自動的に分ける");
  return notes.join("・");
}
</script>

<style lang="scss" scoped>
@use "@/styles/v2/colors" as colors;
@use "@/styles/v2/variables" as vars;

.token-view {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 4px;
}

.label {
  color: colors.$display-sub;
  margin-right: vars.$gap-1;
}

.token {
  padding: 0 6px;
  border: 1px solid colors.$border;
  border-radius: vars.$radius-1;
  background-color: colors.$surface;
  white-space: pre;
}

.rare {
  border-color: colors.$warning;
  color: colors.$display-warning;
}

.dictionary {
  text-decoration: underline dotted;
}

.split {
  color: colors.$display-sub;
  font-weight: bold;
}

.empty,
.hint {
  color: colors.$display-sub;
}

.hint {
  margin-left: vars.$gap-1;
  font-size: 0.85em;
}
</style>
