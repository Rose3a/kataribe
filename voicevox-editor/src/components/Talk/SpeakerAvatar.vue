<template>
  <div
    ref="stageEl"
    class="speaker-avatar"
    :style="{ '--accent': accent }"
    :data-emotion="state.emotion"
    :data-motion="look.motion"
  >
    <div ref="moverEl" class="mover">
      <div ref="ringEl" class="ring"></div>
      <div ref="faceEl" class="face">
        <img v-if="iconPath" :src="iconPath" alt="" />
        <SpeakerInitialIcon v-else :name />
      </div>
      <div class="tint" :style="{ background: tint }"></div>
      <div class="blush" :style="{ opacity: effect.blush ? 0.9 : 0 }">
        <i></i><i></i>
      </div>
      <div class="badge" :class="{ on: badge !== '' }">{{ badge }}</div>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * 立ち絵の無い話者の「動くアイコン」。音量で動き、セリフの絵文字で感情の演出を切り替える。
 * 動きは requestAnimationFrame で直接 style を書き、Vue の再描画は感情が変わったときだけにする。
 */
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import {
  EMOTION_EFFECTS,
  Follow,
  avatarLook,
  emotionAt,
  parseEmotionMarks,
  smoothNoise,
  type EmotionState,
} from "@/helpers/speakerAvatar";
import type { AsrAnchor } from "@/domain/irodori";
import SpeakerInitialIcon from "@/components/SpeakerInitialIcon.vue";

const props = defineProps<{
  name: string;
  /** アイコン画像。無ければ名前から色と頭文字を作る */
  iconPath?: string;
  text: string;
  playing: boolean;
  /** 再生中の音量（RMS）・再生位置・長さ（秒） */
  getLevel: () => number;
  getSeconds: () => number;
  getDuration: () => number;
  /** ASR の文字時刻（あれば絵文字の切り替え時刻に使う） */
  anchors?: readonly AsrAnchor[];
}>();

const stageEl = ref<HTMLElement>();
const moverEl = ref<HTMLElement>();
const ringEl = ref<HTMLElement>();
const faceEl = ref<HTMLElement>();

const look = computed(() => avatarLook(props.name));
const parsed = computed(() => parseEmotionMarks(props.text));
const state = ref<EmotionState>({ emotion: "neutral", intensity: 0 });
const effect = computed(() => EMOTION_EFFECTS[state.value.emotion]);
const badge = computed(() => effect.value.badge ?? "");
const tint = computed(() => effect.value.tint ?? "transparent");

// アクセント色：アイコンがあれば彩度の高い画素の平均、無ければ名前の色
const imageAccent = ref<string>();
const accent = computed(
  () => imageAccent.value ?? `hsl(${look.value.hue} 70% 55%)`,
);
watch(
  () => props.iconPath,
  async (src) => {
    imageAccent.value = undefined;
    if (!src) return;
    try {
      const image = new Image();
      image.crossOrigin = "anonymous";
      image.src = src;
      await image.decode();
      const canvas = document.createElement("canvas");
      canvas.width = canvas.height = 24;
      const context = canvas.getContext("2d", { willReadFrequently: true });
      if (!context) return;
      context.drawImage(image, 0, 0, 24, 24);
      const data = context.getImageData(0, 0, 24, 24).data;
      let r = 0,
        g = 0,
        b = 0,
        weight = 0;
      for (let i = 0; i < data.length; i += 4) {
        const max = Math.max(data[i], data[i + 1], data[i + 2]);
        const min = Math.min(data[i], data[i + 1], data[i + 2]);
        const w = ((max - min) / 255) * (data[i + 3] / 255);
        r += data[i] * w;
        g += data[i + 1] * w;
        b += data[i + 2] * w;
        weight += w;
      }
      if (src === props.iconPath && weight > 0.5)
        imageAccent.value = `rgb(${Math.round(r / weight)} ${Math.round(g / weight)} ${Math.round(b / weight)})`;
    } catch {
      // 別オリジンで読めない等。名前の色のまま使う
    }
  },
  { immediate: true },
);

// ---- 動き ----
const level = new Follow(9, 0.8);
const y = new Follow(4.5, 0.45);
const scaleX = new Follow(6, 0.35, 1);
const scaleY = new Follow(6, 0.35, 1);
const rotation = new Follow(2.5, 0.7);
const size = new Follow(3, 0.6, 1);
let lowSince = 0;
let lastOnset = 0;
let nextParticle = 0;
let frameId: number | undefined;
let lastTime = performance.now();
const seed = Math.random() * 100;

function currentEmotion(): EmotionState {
  const seconds = props.playing ? props.getSeconds() : 0;
  return emotionAt(parsed.value, seconds, props.getDuration(), props.anchors);
}

function onEmotionChange(next: EmotionState) {
  const e = EMOTION_EFFECTS[next.emotion];
  // 切り替えの瞬間の一押し（ばねに勢いを与えるだけ。動きはばねが作る）
  if (e.pop) {
    size.kick(2 * next.intensity);
    y.kick(-240 * next.intensity);
  }
  if (e.hop) y.kick(-110 * next.intensity);
  if (e.shake) rotation.kick(40);
}

function spawnParticle(glyph: string, motion: "fall" | "rise" | undefined) {
  const stage = stageEl.value;
  const mover = moverEl.value;
  if (!stage || !mover || stage.childElementCount > 24) return;
  const s = stage.getBoundingClientRect();
  const m = mover.getBoundingClientRect();
  const particle = document.createElement("div");
  particle.className = "particle";
  particle.textContent = glyph;
  const ratioY =
    motion === "fall" ? 0.25 + Math.random() * 0.3 : Math.random() * 0.6;
  particle.style.left = `${m.left - s.left + (0.1 + Math.random() * 0.8) * m.width}px`;
  particle.style.top = `${m.top - s.top + ratioY * m.height}px`;
  stage.append(particle);
  const dx = (Math.random() - 0.5) * 80;
  const dy =
    motion === "fall" ? 40 + Math.random() * 40 : -40 - Math.random() * 30;
  particle.animate(
    [
      { transform: "translate(0,0) scale(.4)", opacity: 0 },
      { opacity: 1, offset: 0.2 },
      {
        transform: `translate(${dx}px,${dy}px) rotate(${(Math.random() - 0.5) * 60}deg) scale(1)`,
        opacity: 0,
      },
    ],
    { duration: 900 + Math.random() * 600, easing: "cubic-bezier(.2,.7,.3,1)" },
  ).onfinish = () => particle.remove();
}

function frame(now: number) {
  frameId = requestAnimationFrame(frame);
  const dt = Math.min(0.05, (now - lastTime) / 1000);
  lastTime = now;
  const time = now / 1000;

  const next = currentEmotion();
  if (
    next.emotion !== state.value.emotion ||
    Math.abs(next.intensity - state.value.intensity) > 0.01
  ) {
    if (next.emotion !== state.value.emotion) onEmotionChange(next);
    state.value = next;
  }
  const e = effect.value;
  const k = state.value.intensity;
  // 音量（RMS）を 0〜1 に。台詞の声はおおむね RMS 0.02〜0.2
  const raw = props.playing
    ? Math.min(1, Math.max(0, (props.getLevel() - 0.01) * 5.5))
    : 0;
  const L = level.update(dt, raw);

  // 声の出だし（少し静かだった後に大きくなった瞬間）で跳ねる・伸びる
  if (raw < 0.12) lowSince ||= time;
  if (
    raw > 0.3 &&
    lowSince &&
    time - lowSince > 0.05 &&
    time - lastOnset > 0.11
  ) {
    lastOnset = time;
    lowSince = 0;
    const hop = (look.value.motion === "bounce" ? 60 : 22) * (e.hop ?? 1);
    y.kick(-hop * (0.6 + raw));
    if (look.value.motion === "jelly") {
      scaleY.kick(1.6);
      scaleX.kick(-1.0);
    }
  }

  const breathe = Math.sin((time * 2 * Math.PI) / 4 + seed) * 0.012;
  const floatY = e.float ? Math.sin(time * 1.3 + seed) * 5 * k : 0;
  const ty = y.update(dt, floatY + (e.droop ?? 0) * k);
  const tilt =
    (e.tilt ?? 0) * k +
    (e.sway ? Math.sin(time * 0.9) * 5 * k : 0) +
    smoothNoise(time * 0.35 + seed) * 1.5;
  const rot = rotation.update(dt, tilt);
  let s = size.update(dt, (e.shrink ?? 1) * (props.playing ? 1.04 : 1));
  let sx = scaleX.update(dt, 1);
  let sy = scaleY.update(dt, 1);
  if (look.value.motion === "pulse") s *= 1 + L * 0.08;
  if (look.value.motion === "jelly") {
    sy *= 1 + L * 0.07;
    sx *= 1 - L * 0.035;
  }
  const shake = (e.shake ?? 0) * k * (0.4 + L);
  const tremble = (e.tremble ?? 0) * k;
  const ox =
    smoothNoise(time * 38 + seed) * 6 * shake +
    smoothNoise(time * 22 + 5) * 2.2 * tremble;
  const oy =
    smoothNoise(time * 41 + 9) * 3 * shake +
    smoothNoise(time * 25 + 3) * 1.5 * tremble;

  if (moverEl.value)
    moverEl.value.style.transform = `translate(${ox.toFixed(2)}px, ${(ty + oy).toFixed(2)}px) rotate(${rot.toFixed(2)}deg) scale(${(s * sx).toFixed(4)}, ${(s * sy * (1 + breathe)).toFixed(4)})`;
  if (ringEl.value) {
    ringEl.value.style.setProperty(
      "--glow",
      (props.playing ? 0.2 + L * 1.1 : 0).toFixed(3),
    );
    ringEl.value.style.opacity = (
      props.playing ? 0.55 + L * 0.45 : 0.3
    ).toFixed(3);
    ringEl.value.style.borderColor = e.ringColor ?? "";
  }
  if (faceEl.value)
    faceEl.value.style.filter = e.desaturate
      ? `saturate(${(1 - e.desaturate * k).toFixed(3)})`
      : "";

  // 粒子：話している間ほど多い。停止中は控えめ
  if (e.particles && e.rate && time > nextParticle) {
    spawnParticle(
      e.particles[Math.floor(Math.random() * e.particles.length)],
      e.particleMotion,
    );
    nextParticle = time + 1 / (e.rate * (props.playing ? 0.5 + L * 1.5 : 0.35));
  }
}

onMounted(() => {
  lastTime = performance.now();
  frameId = requestAnimationFrame(frame);
});
onBeforeUnmount(() => {
  if (frameId != undefined) cancelAnimationFrame(frameId);
});
</script>

<style scoped lang="scss">
.speaker-avatar {
  position: relative;
  flex: 1 1 auto;
  align-self: stretch;
  min-height: 0;
  display: grid;
  place-items: center;
  overflow: hidden;
  container-type: size;
  background: radial-gradient(
    ellipse at 50% 40%,
    color-mix(in srgb, var(--accent) 16%, #fff),
    #fff 70%
  );
}

.mover {
  --size: min(58cqw, 58cqh);
  position: relative;
  width: var(--size);
  height: var(--size);
  will-change: transform;
}

// 立ち絵欄のアイコンは全話者丸で統一する
.ring,
.face,
.tint {
  border-radius: 50%;
}

.ring {
  position: absolute;
  inset: -6px;
  border: 3px solid var(--accent);
  opacity: 0.3;
  box-shadow: 0 0 calc(var(--glow, 0) * 36px) calc(var(--glow, 0) * 5px)
    var(--accent);
}

.face {
  position: absolute;
  inset: 0;
  overflow: hidden;
  background: #eee;
  box-shadow: 0 4px 14px rgb(0 0 0 / 0.12);
  img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }
}

.tint {
  position: absolute;
  inset: 0;
  pointer-events: none;
  transition: background 0.4s;
}

.blush {
  position: absolute;
  left: 14%;
  right: 14%;
  top: 58%;
  height: 14%;
  display: flex;
  justify-content: space-between;
  pointer-events: none;
  transition: opacity 0.35s;
  i {
    width: 30%;
    border-radius: 50%;
    background: radial-gradient(#ff6b8a, #ff6b8a00 70%);
  }
}

.badge {
  position: absolute;
  right: -6%;
  top: -8%;
  font-size: calc(var(--size) * 0.2);
  transform: scale(0);
  transition: transform 0.25s cubic-bezier(0.3, 1.8, 0.5, 1);
  filter: drop-shadow(0 2px 3px rgb(0 0 0 / 0.3));
  &.on {
    transform: scale(1);
  }
}

:deep(.particle) {
  position: absolute;
  pointer-events: none;
  font-size: 18px;
  will-change: transform, opacity;
}

@media (prefers-reduced-motion: reduce) {
  .mover {
    transform: none !important;
  }
}
</style>
