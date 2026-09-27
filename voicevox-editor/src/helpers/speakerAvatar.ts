/**
 * 立ち絵の無い話者のための「動くアイコン」。
 *
 * 顔を描かずに、音量に合わせた動き・色・記号・粒子で「話している」「感情」を見せる。
 * 口パクのように人の顔を真似ないので、不自然さ（不気味の谷）が出にくい。
 * 見た目（色・形・動き方）は話者名から決めるので、画像が無くても話者ごとに違って見える。
 */

/** 感情の種類。セリフ中の絵文字（Irodori の絵文字アノテーション）から決める。 */
export type AvatarEmotion =
  | "neutral"
  | "joy"
  | "relaxed"
  | "anger"
  | "sad"
  | "surprise"
  | "worried"
  | "shy"
  | "smug"
  | "jitome"
  | "scared"
  | "sleepy";

/** 絵文字 → 感情。models/EMOJI_ANNOTATIONS.md の感情グループに対応する。 */
export const EMOJI_EMOTION: Readonly<Record<string, AvatarEmotion>> = {
  "😊": "joy",
  "😆": "joy",
  "🤭": "joy",
  "🫶": "relaxed",
  "😌": "relaxed",
  "😠": "anger",
  "😭": "sad",
  "😲": "surprise",
  "😱": "scared",
  "😰": "scared",
  "😟": "worried",
  "🥺": "worried",
  "🙏": "worried",
  "😖": "worried",
  "🫣": "shy",
  "😎": "smug",
  "😏": "smug",
  "🙄": "jitome",
  "😪": "sleepy",
  "🥱": "sleepy",
};

/** 感情ごとの演出。値は「強さ 1」のときの量。 */
export type EmotionEffect = {
  /** 右上に出す記号 */
  badge?: string;
  /** 周りに出す粒子と、1秒あたりの数 */
  particles?: readonly string[];
  rate?: number;
  /** 粒子が落ちる（涙）/ 上る（寝息）。既定は上に散る */
  particleMotion?: "fall" | "rise";
  /** アイコンに重ねる色 */
  tint?: string;
  /** 光の輪の色（既定は話者のアクセント色） */
  ringColor?: string;
  /** 彩度を下げる量（0〜1） */
  desaturate?: number;
  /** 首かしげ（度）。負は逆向き */
  tilt?: number;
  /** 下がる量（px）。負は上がる */
  droop?: number;
  /** 声の出だしで跳ねる量の倍率 */
  hop?: number;
  /** 震え（怒りは声に合わせて、怯え・心配は常に） */
  shake?: number;
  tremble?: number;
  /** ゆっくり左右に揺れる / 浮かぶ */
  sway?: boolean;
  float?: boolean;
  /** 頬の赤み */
  blush?: boolean;
  /** 縮こまる（大きさの倍率） */
  shrink?: number;
  /** 切り替わった瞬間に跳ねる */
  pop?: boolean;
};

export const EMOTION_EFFECTS: Readonly<Record<AvatarEmotion, EmotionEffect>> = {
  neutral: {},
  joy: {
    badge: "✨",
    particles: ["✨", "♪"],
    rate: 2.5,
    hop: 1.5,
    tint: "#ffd86b22",
  },
  relaxed: { particles: ["🌸"], rate: 0.8, float: true, tint: "#ffc0d018" },
  anger: { badge: "💢", shake: 1, tint: "#ff303033", ringColor: "#ff4a4a" },
  sad: {
    badge: "💧",
    particles: ["💧"],
    rate: 1,
    particleMotion: "fall",
    droop: 8,
    tint: "#3a6bff2e",
    desaturate: 0.55,
  },
  surprise: { badge: "❗", pop: true, hop: 1.2 },
  worried: { badge: "💦", tremble: 0.5, tint: "#6aa0ff1c" },
  shy: {
    badge: "💗",
    particles: ["💗"],
    rate: 1,
    blush: true,
    tilt: 7,
    tint: "#ff7aa01e",
  },
  smug: { badge: "✨", tilt: -6, droop: -4 },
  jitome: { badge: "💬", tilt: 3 },
  scared: {
    badge: "💦",
    tremble: 1.2,
    tint: "#9cc8ff30",
    desaturate: 0.3,
    shrink: 0.94,
  },
  sleepy: {
    badge: "💤",
    particles: ["💤"],
    rate: 0.6,
    particleMotion: "rise",
    sway: true,
    droop: 4,
    tint: "#8a7cff1e",
  },
};

/** ASR が文字を数えるときに落とす記号（irodori-tts/wrapper/asr_timeline.py と同じ）。 */
const ASR_PUNCTUATION = new Set([
  ..." 　、。，．！？!?,.・:;；「」『』（）()[]{}〈〉《》…‥~〜ー-―—’‘“”\"'+=*/\\|@#$%^&_`",
]);

const isEmoji = (grapheme: string) =>
  /\p{Extended_Pictographic}/u.test(grapheme);

export type EmotionMark = {
  /** 何文字目（ASR の数え方）の直前か */
  index: number;
  emotion: AvatarEmotion;
  intensity: number;
};

export type EmotionMarks = {
  /** 絵文字を除いた文。ASR に渡す */
  plain: string;
  marks: EmotionMark[];
  /** ASR の数え方での文字数 */
  charCount: number;
};

/**
 * セリフから絵文字の位置と感情を取り出す。同じ絵文字を重ねると強くなる（😭😭）。
 * 感情に対応しない絵文字（話し方・効果音）は無視する。
 */
export function parseEmotionMarks(text: string): EmotionMarks {
  const segmenter = new Intl.Segmenter("ja", { granularity: "grapheme" });
  let plain = "";
  let index = 0;
  const marks: EmotionMark[] = [];
  for (const { segment } of segmenter.segment(text)) {
    if (isEmoji(segment)) {
      const emotion = EMOJI_EMOTION[[...segment][0]];
      if (emotion == undefined) continue;
      const last = marks[marks.length - 1];
      if (last && last.index === index && last.emotion === emotion) {
        last.intensity = Math.min(1, last.intensity + 0.2);
      } else {
        marks.push({ index, emotion, intensity: 0.65 });
      }
      continue;
    }
    plain += segment;
    if (!ASR_PUNCTUATION.has(segment) && segment.trim() !== "") index++;
  }
  return { plain, marks, charCount: index };
}

export type EmotionState = { emotion: AvatarEmotion; intensity: number };

/**
 * 再生位置での感情。ASR の文字時刻があれば「直後の文字の発話開始」、無ければ文字数で按分する。
 * leadSeconds だけ早めに切り替える（表情は声より少し先に変わると自然に見える）。
 */
export function emotionAt(
  parsed: EmotionMarks,
  seconds: number,
  durationSeconds: number,
  anchors?: readonly { start: number }[],
  leadSeconds = 0.15,
): EmotionState {
  let state: EmotionState = { emotion: "neutral", intensity: 0 };
  for (const mark of parsed.marks) {
    const time =
      mark.index === 0
        ? 0
        : anchors && anchors.length > 0
          ? (anchors[mark.index]?.start ?? durationSeconds)
          : durationSeconds > 0
            ? (mark.index / Math.max(1, parsed.charCount)) * durationSeconds
            : // 音声がまだ無い（長さ不明）なら、行の途中の絵文字は来ない扱い
              Infinity;
    if (time <= seconds + leadSeconds) state = mark;
  }
  return { emotion: state.emotion, intensity: state.intensity };
}

/** 話者ごとの見た目。名前から決めるので、同じ話者はいつも同じ見た目になる。 */
export type AvatarLook = {
  hue: number;
  initials: string;
  motion: "bounce" | "pulse" | "jelly";
};

function nameHash(name: string): number {
  let hash = 2166136261;
  for (const char of name) {
    hash ^= char.codePointAt(0) ?? 0;
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return hash;
}

/** 頭文字。漢字・かなを含めば1文字、英字なら大文字2文字。 */
export function avatarInitials(name: string): string {
  const trimmed = name.trim();
  if (trimmed === "") return "?";
  const japanese = trimmed.match(
    /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}]/u,
  );
  if (japanese) return japanese[0];
  const latin = trimmed.replace(/[^A-Za-z0-9]/g, "");
  return (latin.slice(0, 2) || [...trimmed][0]).toUpperCase();
}

export function avatarLook(name: string): AvatarLook {
  const hash = nameHash(name);
  return {
    hue: hash % 360,
    initials: avatarInitials(name),
    motion: (["bounce", "pulse", "jelly"] as const)[(hash >> 12) % 3],
  };
}

/**
 * 2次の追従（ばね＋ダンパ）。f は速さ[Hz]、z は減衰（1 で行き過ぎなし、1 未満で少し弾む）。
 * 指数平滑と違って動き出しがなめらかで、kick で勢いを与えると自然に弾んで戻る。
 */
export class Follow {
  private k1 = 0;
  private k2 = 0;
  value: number;
  private velocity = 0;

  constructor(frequency: number, damping = 1, initial = 0) {
    this.set(frequency, damping);
    this.value = initial;
  }

  set(frequency: number, damping: number) {
    this.k1 = damping / (Math.PI * frequency);
    this.k2 = 1 / (2 * Math.PI * frequency) ** 2;
  }

  update(dt: number, target: number): number {
    // 大きな dt（タブ復帰直後など）でも発散させない
    const k2 = Math.max(
      this.k2,
      (dt * dt) / 2 + (dt * this.k1) / 2,
      dt * this.k1,
    );
    this.value += dt * this.velocity;
    this.velocity +=
      (dt * (target - this.value - this.k1 * this.velocity)) / k2;
    return this.value;
  }

  kick(velocity: number) {
    this.velocity += velocity;
  }
}

/** なめらかな乱数（-1〜1）。周期が見えないゆらぎに使う。 */
export function smoothNoise(t: number): number {
  const hash = (n: number) => {
    const s = Math.sin(n * 127.1 + 311.7) * 43758.5453;
    return s - Math.floor(s);
  };
  const i = Math.floor(t);
  const f = t - i;
  const u = f * f * (3 - 2 * f);
  return (hash(i) * 2 - 1) * (1 - u) + (hash(i + 1) * 2 - 1) * u;
}
