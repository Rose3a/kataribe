<template>
  <div
    ref="root"
    class="train-chart"
    :class="{ compact }"
    tabindex="0"
    role="img"
    :aria-label
    @keydown="onKey"
    @blur="leave"
  >
    <div class="chart-head">
      <div class="chart-title">{{ title }}</div>
      <div v-if="legend.length > 1" class="legend">
        <span v-for="item in legend" :key="item.key" class="legend-item">
          <svg width="16" height="8" aria-hidden="true">
            <line
              x1="1"
              y1="4"
              x2="15"
              y2="4"
              :stroke="item.color"
              stroke-width="2"
              stroke-linecap="round"
              :stroke-dasharray="isDashed(item.variant) ? '4 3' : undefined"
            />
            <circle
              v-if="hasDots(item.variant)"
              cx="8"
              cy="4"
              r="2.5"
              :fill="item.color"
            />
          </svg>
          {{ item.label }}
        </span>
      </div>
    </div>
    <div class="plot" :style="{ height: `${height}px` }">
      <svg
        v-if="width > 0"
        :width
        :height
        class="plot-svg"
        @pointermove="onPointer"
        @pointerleave="leave"
      >
        <defs>
          <clipPath :id="clipId">
            <rect
              :x="M.left"
              :y="M.top - 4"
              :width="innerWidth"
              :height="innerHeight + 8"
            />
          </clipPath>
        </defs>
        <!-- 目盛り（横線）と縦軸の数 -->
        <g v-if="hasData" class="grid">
          <line
            v-for="tick in yTicks"
            :key="`y${tick}`"
            :x1="M.left"
            :x2="width - M.right"
            :y1="yPos(tick)"
            :y2="yPos(tick)"
          />
        </g>
        <g v-if="hasData" class="axis-text">
          <text
            v-for="tick in yTicks"
            :key="`yl${tick}`"
            :x="M.left - 6"
            :y="yPos(tick)"
            text-anchor="end"
            dominant-baseline="middle"
          >
            {{ formatTick(tick) }}
          </text>
          <text
            v-for="tick in xTicks"
            :key="`xl${tick}`"
            :x="xPos(tick)"
            :y="height - 4"
            text-anchor="middle"
          >
            {{ tick }}
          </text>
        </g>
        <line
          class="baseline"
          :x1="M.left"
          :x2="width - M.right"
          :y1="M.top + innerHeight"
          :y2="M.top + innerHeight"
        />
        <!-- 線 -->
        <g :clip-path="`url(#${clipId})`">
          <path
            v-for="line in paths"
            :key="line.key"
            :d="line.d"
            fill="none"
            :stroke="line.color"
            :stroke-width="line.variant === 'faint' ? 1 : 2"
            :stroke-opacity="line.variant === 'faint' ? 0.28 : 1"
            :stroke-dasharray="isDashed(line.variant) ? '5 4' : undefined"
            stroke-linejoin="round"
            stroke-linecap="round"
          />
          <template v-for="line in paths" :key="`m${line.key}`">
            <template v-if="hasDots(line.variant)">
              <circle
                v-for="(dot, i) in line.dots"
                :key="i"
                :cx="dot.x"
                :cy="dot.y"
                r="4"
                :fill="line.color"
                class="marker"
              />
            </template>
          </template>
        </g>
        <!-- 十字線 -->
        <g v-if="hover">
          <line
            class="crosshair"
            :x1="hover.px"
            :x2="hover.px"
            :y1="M.top"
            :y2="M.top + innerHeight"
          />
          <circle
            v-for="row in hover.rows.filter((r) => r.py != undefined)"
            :key="row.key"
            :cx="row.px"
            :cy="row.py"
            r="4.5"
            :fill="row.color"
            class="marker"
          />
        </g>
      </svg>
      <div v-if="!hasData" class="empty">{{ emptyText }}</div>
      <div
        v-if="hover && active"
        class="tooltip"
        :style="tooltipStyle"
        aria-hidden="true"
      >
        <div class="tooltip-head">ステップ {{ hover.x }}</div>
        <div v-for="row in hover.rows" :key="row.key" class="tooltip-row">
          <svg width="12" height="8">
            <line
              x1="1"
              y1="4"
              x2="11"
              y2="4"
              :stroke="row.color"
              stroke-width="2"
              stroke-linecap="round"
              :stroke-dasharray="isDashed(row.variant) ? '3 2' : undefined"
            />
          </svg>
          <span class="tooltip-value">{{ format(row.y) }}</span>
          <span class="tooltip-label"
            >{{ row.label
            }}<template v-if="row.x !== hover.x"
              >（{{ row.x }}）</template
            ></span
          >
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, useId } from "vue";

const props = withDefaults(
  defineProps<{
    title: string;
    series: ChartSeries[];
    /** 横軸の右端（全ステップ数）。学習中も最後までの幅で描く。 */
    xMax?: number;
    height?: number;
    logY?: boolean;
    format?: (value: number) => string;
    emptyText?: string;
    compact?: boolean;
  }>(),
  {
    xMax: 0,
    height: 220,
    logY: false,
    format: formatNumber,
    emptyText: "まだ記録がありません",
    compact: false,
  },
);
/** 十字線のステップ（グラフ同士でそろえる）。 */
const hoverX = defineModel<number | null>("hoverX", { default: null });

const M = computed(() => ({
  left: props.compact ? 44 : 52,
  right: 12,
  top: 8,
  bottom: 22,
}));
const clipId = `train-chart-${useId()}`;
const root = ref<HTMLElement>();
const width = ref(0);
const active = ref(false);
let observer: ResizeObserver | undefined;
onMounted(() => {
  observer = new ResizeObserver((entries) => {
    width.value = Math.floor(entries[0].contentRect.width);
  });
  if (root.value) observer.observe(root.value);
});
onUnmounted(() => observer?.disconnect());

const innerWidth = computed(() =>
  Math.max(10, width.value - M.value.left - M.value.right),
);
const innerHeight = computed(() =>
  Math.max(10, props.height - M.value.top - M.value.bottom),
);

const usable = (y: number) => Number.isFinite(y) && (!props.logY || y > 0);
const visible = computed(() =>
  props.series.map((s) => ({
    ...s,
    points: s.points.filter((p) => usable(p.y)),
  })),
);
const hasData = computed(() => visible.value.some((s) => s.points.length > 0));
const legend = computed(() =>
  visible.value.filter((s) => s.variant !== "faint"),
);

const xDomain = computed<[number, number]>(() => {
  let hi = props.xMax;
  for (const s of visible.value)
    for (const p of s.points) hi = Math.max(hi, p.x);
  return [0, Math.max(1, hi)];
});

/** 縦軸の範囲。薄い線（平滑化前）は外れ値で軸が広がらないよう、他の線があれば範囲に入れない（はみ出しは切る）。 */
const yDomain = computed<[number, number]>(() => {
  const main = visible.value.filter((s) => s.variant !== "faint");
  const source = main.some((s) => s.points.length > 0) ? main : visible.value;
  let lo = Infinity;
  let hi = -Infinity;
  for (const s of source)
    for (const p of s.points) {
      lo = Math.min(lo, p.y);
      hi = Math.max(hi, p.y);
    }
  if (!Number.isFinite(lo)) return props.logY ? [0.1, 1] : [0, 1];
  if (props.logY) {
    const a = Math.log10(lo);
    const b = Math.log10(hi);
    const pad = Math.max((b - a) * 0.08, 0.02);
    return [10 ** (a - pad), 10 ** (b + pad)];
  }
  if (hi === lo) {
    const pad = Math.abs(hi) * 0.1 || 1;
    return [lo - pad, hi + pad];
  }
  const pad = (hi - lo) * 0.08;
  return [lo - pad, hi + pad];
});

const xPos = (x: number) =>
  M.value.left +
  ((x - xDomain.value[0]) / (xDomain.value[1] - xDomain.value[0])) *
    innerWidth.value;
function yPos(y: number): number {
  const [lo, hi] = yDomain.value;
  const t = props.logY
    ? (Math.log10(y) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))
    : (y - lo) / (hi - lo);
  return M.value.top + (1 - t) * innerHeight.value;
}

function niceStep(range: number, count: number): number {
  const raw = range / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  return (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
}
function linearTicks(lo: number, hi: number, count: number): number[] {
  const step = niceStep(hi - lo, count);
  const ticks: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step)
    ticks.push(Number(v.toPrecision(12)));
  return ticks;
}
const yTicks = computed(() => {
  const [lo, hi] = yDomain.value;
  const count = props.compact ? 3 : 4;
  if (props.logY) {
    const ticks: number[] = [];
    for (
      let e = Math.floor(Math.log10(lo));
      e <= Math.ceil(Math.log10(hi));
      e++
    )
      for (const m of [1, 2, 5]) {
        const v = m * 10 ** e;
        if (v >= lo && v <= hi) ticks.push(Number(v.toPrecision(12)));
      }
    if (ticks.length >= 2) return ticks;
  }
  return linearTicks(lo, hi, count).filter((v) => !props.logY || v > 0);
});
const xTicks = computed(() => {
  const [lo, hi] = xDomain.value;
  const count = Math.max(2, Math.floor(innerWidth.value / 90));
  return linearTicks(lo, hi, count).filter((v) => Number.isInteger(v));
});

const paths = computed(() =>
  visible.value
    .filter((s) => s.points.length > 0)
    .map((s) => {
      const coords = s.points.map((p) => ({ x: xPos(p.x), y: yPos(p.y) }));
      return {
        key: s.key,
        color: s.color,
        variant: s.variant ?? "line",
        d: coords
          .map(
            (c, i) =>
              `${i === 0 ? "M" : "L"}${c.x.toFixed(1)},${c.y.toFixed(1)}`,
          )
          .join(""),
        dots: coords,
      };
    }),
);

/** 全部の線のステップ（十字線が止まる位置）。 */
const steps = computed(() => {
  const set = new Set<number>();
  for (const s of visible.value) for (const p of s.points) set.add(p.x);
  return [...set].sort((a, b) => a - b);
});
function nearest<T extends { x: number }>(
  items: T[],
  x: number,
): T | undefined {
  let best: T | undefined;
  for (const item of items)
    if (best == undefined || Math.abs(item.x - x) < Math.abs(best.x - x))
      best = item;
  return best;
}

const hover = computed(() => {
  const x = hoverX.value;
  if (x == undefined || !hasData.value) return undefined;
  const rows = visible.value
    .filter((s) => s.variant !== "faint" && s.points.length > 0)
    .map((s) => {
      const p = nearest(s.points, x);
      if (p == undefined) return undefined;
      // 点の少ない線（検証）は、十字線から遠い点に印を付けない（値は表に出す）
      const near = Math.abs(xPos(p.x) - xPos(x)) <= 6;
      return {
        key: s.key,
        label: s.label,
        color: s.color,
        variant: s.variant,
        x: p.x,
        y: p.y,
        px: xPos(p.x),
        py: near ? yPos(p.y) : undefined,
      };
    })
    .filter((row) => row != undefined);
  return { x, px: xPos(x), rows };
});

const tooltipStyle = computed(() => {
  const px = hover.value?.px ?? 0;
  const right = px > width.value * 0.6;
  return right
    ? { right: `${width.value - px + 12}px`, top: `${M.value.top}px` }
    : { left: `${px + 12}px`, top: `${M.value.top}px` };
});

function onPointer(event: PointerEvent) {
  if (steps.value.length === 0) return;
  const rect = (event.currentTarget as SVGElement).getBoundingClientRect();
  const [lo, hi] = xDomain.value;
  const x =
    lo +
    ((event.clientX - rect.left - M.value.left) / innerWidth.value) * (hi - lo);
  const snapped = nearest(
    steps.value.map((s) => ({ x: s })),
    x,
  );
  active.value = true;
  hoverX.value = snapped?.x ?? null;
}
function leave() {
  active.value = false;
  hoverX.value = null;
}
/** キーボードでも十字線を動かせる（← → で前後の記録、Home / End で端）。 */
function onKey(event: KeyboardEvent) {
  const list = steps.value;
  if (list.length === 0) return;
  const index = hoverX.value == undefined ? -1 : list.indexOf(hoverX.value);
  let next: number | undefined;
  if (event.key === "ArrowRight") next = Math.min(list.length - 1, index + 1);
  else if (event.key === "ArrowLeft")
    next = index < 0 ? list.length - 1 : Math.max(0, index - 1);
  else if (event.key === "Home") next = 0;
  else if (event.key === "End") next = list.length - 1;
  else if (event.key === "Escape") return leave();
  if (next == undefined) return;
  event.preventDefault();
  active.value = true;
  hoverX.value = list[next];
}

const ariaLabel = computed(() => {
  const parts = legend.value
    .filter((s) => s.points.length > 0)
    .map(
      (s) => `${s.label} 最後 ${props.format(s.points[s.points.length - 1].y)}`,
    );
  return `${props.title}のグラフ。${parts.join("、") || props.emptyText}。← → で値を読めます。`;
});

function formatTick(value: number): string {
  return props.format(value);
}
function isDashed(variant?: ChartVariant): boolean {
  return variant === "dashed" || variant === "dashed-dots";
}
function hasDots(variant?: ChartVariant): boolean {
  return variant === "dots" || variant === "dashed-dots";
}
</script>

<script lang="ts">
export type ChartVariant = "line" | "faint" | "dashed" | "dots" | "dashed-dots";
export type ChartSeries = {
  key: string;
  label: string;
  /** 線の色（CSS の色。var(--chart-1) など）。 */
  color: string;
  points: { x: number; y: number }[];
  /** line = 実線 / faint = 薄い線（平滑化前の値）/ dashed = 比べる学習 / dots = 点つき。 */
  variant?: ChartVariant;
};

/** 4 桁の有効数字。とても小さい・大きい数は指数で（1e-4 など）。 */
export function formatNumber(value: number): string {
  if (!Number.isFinite(value)) return "-";
  if (value === 0) return "0";
  const abs = Math.abs(value);
  if (abs < 1e-3 || abs >= 1e5)
    return value.toExponential(1).replace(/\.0e/, "e").replace("e+", "e");
  return String(Number(value.toPrecision(4)));
}
</script>

<style scoped lang="scss">
@use "@/styles/v2/variables" as vars;

.train-chart {
  // グラフの色（明るいテーマ）。系列 1 = 青、系列 2 = オレンジ、比べる学習 = 灰色
  --chart-1: #2a78d6;
  --chart-2: #eb6834;
  --chart-compare: #898781;
  --chart-grid: rgba(36, 38, 38, 0.1);
  --chart-baseline: rgba(36, 38, 38, 0.28);
  --chart-text: var(--color-v2-display-sub);
  --chart-surface: var(--color-v2-surface);
  position: relative;
  outline: none;
  border-radius: vars.$radius-1;
  &:focus-visible {
    box-shadow: 0 0 0 2px var(--color-v2-primary);
  }
}
:global(:root[is-dark-theme="true"]) .train-chart {
  --chart-1: #3987e5;
  --chart-2: #d95926;
  --chart-compare: #a3a29b;
  --chart-grid: rgba(255, 255, 255, 0.08);
  --chart-baseline: rgba(255, 255, 255, 0.22);
}

.chart-head {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 4px 16px;
  margin-bottom: 4px;
}
.chart-title {
  font-weight: 600;
  font-size: 13px;
  color: var(--color-v2-display);
}
.compact .chart-title {
  font-size: 12px;
}
.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 2px 12px;
  font-size: 12px;
  color: var(--color-v2-display-sub);
}
.legend-item {
  display: inline-flex;
  align-items: center;
  gap: 4px;
}

.plot {
  position: relative;
}
.plot-svg {
  display: block;
  overflow: visible;
  touch-action: none;
}
.grid line {
  stroke: var(--chart-grid);
  stroke-width: 1;
  shape-rendering: crispEdges;
}
.baseline {
  stroke: var(--chart-baseline);
  stroke-width: 1;
  shape-rendering: crispEdges;
}
.axis-text text {
  fill: var(--chart-text);
  font-size: 11px;
  font-variant-numeric: tabular-nums;
}
.marker {
  stroke: var(--chart-surface);
  stroke-width: 2;
}
.crosshair {
  stroke: var(--chart-baseline);
  stroke-width: 1;
}
.empty {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--color-v2-display-sub);
  font-size: 12px;
  pointer-events: none;
}
.tooltip {
  position: absolute;
  z-index: 2;
  min-width: 140px;
  padding: 6px 10px;
  border-radius: vars.$radius-1;
  background: var(--color-v2-surface);
  border: 1px solid var(--color-v2-border);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
  font-size: 12px;
  pointer-events: none;
  white-space: nowrap;
}
.tooltip-head {
  color: var(--color-v2-display-sub);
  margin-bottom: 2px;
}
.tooltip-row {
  display: flex;
  align-items: center;
  gap: 6px;
}
.tooltip-value {
  font-weight: 600;
  color: var(--color-v2-display);
  font-variant-numeric: tabular-nums;
}
.tooltip-label {
  color: var(--color-v2-display-sub);
}
</style>
