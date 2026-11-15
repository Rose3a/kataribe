/**
 * 再生中のセリフを文字ごとに追う演出（AudioCell）のための、文字ごとの時刻。
 *
 * ASR の anchors は、句読点・空白を除き、カタカナをひらがなにしたセリフの 1 文字ごとに並んでいる
 * （irodori-tts/wrapper/asr_timeline.py の normalize と同じ）。ここで元のセリフの文字へ戻す。
 */
import type { AsrAnchor } from "@/domain/irodori";

/** asr_timeline.py の _PUNCTUATION と同じ。時刻を持たない文字 */
const UNTIMED = new Set([
  ..." 　、。，．！？!?,.・:;；「」『』（）()[]{}〈〉《》…‥~〜ー-―—’‘“”\"'+=*/\\|@#$%^&_`",
]);
const MIN_SECONDS = 0.06;
const MAX_SECONDS = 0.35;

export type CharTime = {
  char: string;
  /** 発話の始まり（秒） */
  t: number;
  /** 長さ（秒） */
  d: number;
};

const isTimed = (char: string) => !(UNTIMED.has(char) || /\s/.test(char));

/** 1 トークンに複数文字が入って同じ時刻になった並びを、次の時刻まで等分する */
function spreadSameStarts(starts: number[], lastEnd: number): number[] {
  const out = [...starts];
  let i = 0;
  while (i < out.length) {
    let j = i;
    while (j + 1 < out.length && out[j + 1] === out[i]) j++;
    if (j > i) {
      const stop = Math.max(
        j + 1 < out.length ? out[j + 1] : lastEnd,
        out[i] + 0.03 * (j - i + 1),
      );
      const step = (stop - out[i]) / (j - i + 1);
      for (let k = 1; k <= j - i; k++) out[i + k] = out[i] + step * k;
    }
    i = j + 1;
  }
  return out;
}

function toCharTimes(
  text: string,
  starts: number[],
  lastEnd: number,
): CharTime[] {
  const chars: CharTime[] = [];
  let k = 0;
  let previousEnd = starts[0] ?? 0;
  for (const char of text) {
    if (!isTimed(char) || k >= starts.length) {
      chars.push({ char, t: previousEnd, d: 0.12 });
      continue;
    }
    const start = starts[k];
    const end = k + 1 < starts.length ? starts[k + 1] : lastEnd;
    k++;
    chars.push({
      char,
      t: start,
      d: Math.min(Math.max(end - start, MIN_SECONDS), MAX_SECONDS),
    });
    previousEnd = Math.max(previousEnd, Math.min(end, start + MAX_SECONDS));
  }
  return chars;
}

/** ASR の実測から */
export function charTimesFromAnchors(
  text: string,
  anchors: readonly AsrAnchor[],
): CharTime[] {
  if (anchors.length === 0) return [];
  const lastEnd = anchors[anchors.length - 1].end;
  const starts = spreadSameStarts(
    anchors.map((anchor) => anchor.start),
    lastEnd,
  );
  return toCharTimes(text, starts, lastEnd);
}

/** ASR が使えないときの推定（声の出ている文字を、音声の長さに均等に並べる） */
export function estimateCharTimes(text: string, duration: number): CharTime[] {
  const count = [...text].filter(isTimed).length;
  if (count === 0 || duration <= 0) return [];
  const begin = duration * 0.04;
  const span = duration * 0.92;
  const starts = Array.from(
    { length: count },
    (_, i) => begin + (span * i) / count,
  );
  return toCharTimes(text, starts, begin + span);
}
