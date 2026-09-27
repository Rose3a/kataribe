import { describe, expect, it } from "vitest";
import {
  Follow,
  avatarInitials,
  avatarLook,
  emotionAt,
  parseEmotionMarks,
} from "@/helpers/speakerAvatar";

describe("parseEmotionMarks", () => {
  it("絵文字の位置を ASR と同じ数え方（記号・空白を除く）で記録する", () => {
    const parsed = parseEmotionMarks("😊やった、ね！😲えっ");
    expect(parsed.plain).toBe("やった、ね！えっ");
    expect(parsed.charCount).toBe(6);
    expect(parsed.marks).toEqual([
      { index: 0, emotion: "joy", intensity: 0.65 },
      { index: 4, emotion: "surprise", intensity: 0.65 },
    ]);
  });

  it("同じ絵文字を重ねると強くなる", () => {
    const [mark] = parseEmotionMarks("😭😭😭ひどい").marks;
    expect(mark.emotion).toBe("sad");
    expect(mark.intensity).toBeCloseTo(1);
  });

  it("感情に対応しない絵文字（話し方・効果音）は無視する", () => {
    expect(parseEmotionMarks("👂ないしょ📢").marks).toEqual([]);
  });
});

describe("emotionAt", () => {
  const parsed = parseEmotionMarks("😊あいう😠えお");

  it("ASR が無ければ文字数で按分した時刻で切り替える", () => {
    // 5文字中3文字目の後 → 2秒の音声なら 1.2 秒
    expect(emotionAt(parsed, 0.9, 2, undefined, 0).emotion).toBe("joy");
    expect(emotionAt(parsed, 1.25, 2, undefined, 0).emotion).toBe("anger");
  });

  it("ASR の文字時刻があれば直後の文字の発話開始で切り替える", () => {
    const anchors = [0.1, 0.3, 0.5, 1.6, 1.8].map((start) => ({ start }));
    expect(emotionAt(parsed, 1.4, 2, anchors, 0).emotion).toBe("joy");
    expect(emotionAt(parsed, 1.6, 2, anchors, 0).emotion).toBe("anger");
  });

  it("表情は声より少し先に切り替わる", () => {
    const anchors = [0.1, 0.3, 0.5, 1.6, 1.8].map((start) => ({ start }));
    expect(emotionAt(parsed, 1.5, 2, anchors, 0.15).emotion).toBe("anger");
  });

  it("音声の長さが分からないときは行頭の絵文字だけを使う", () => {
    expect(emotionAt(parsed, 0, 0).emotion).toBe("joy");
  });

  it("絵文字が無ければ通常", () => {
    expect(emotionAt(parseEmotionMarks("こんにちは"), 1, 2)).toEqual({
      emotion: "neutral",
      intensity: 0,
    });
  });
});

describe("avatarLook", () => {
  it("同じ名前はいつも同じ見た目になる", () => {
    expect(avatarLook("Zunko")).toEqual(avatarLook("Zunko"));
    expect(avatarLook("Zunko").hue).not.toBe(avatarLook("Aris").hue);
  });

  it("頭文字は日本語なら1文字、英字なら大文字2文字", () => {
    expect(avatarInitials("つくよみちゃん")).toBe("つ");
    expect(avatarInitials("nikke_アリス_羊宮妃那")).toBe("ア");
    expect(avatarInitials("Zunko")).toBe("ZU");
    expect(avatarInitials("")).toBe("?");
  });
});

describe("Follow", () => {
  it("目標値に収束し、大きな dt でも発散しない", () => {
    const follow = new Follow(4, 0.6);
    for (let i = 0; i < 300; i++) follow.update(1 / 60, 1);
    expect(follow.value).toBeCloseTo(1, 3);
    const big = new Follow(9, 0.3);
    for (let i = 0; i < 20; i++) big.update(0.5, 1);
    expect(Number.isFinite(big.value)).toBe(true);
    expect(Math.abs(big.value - 1)).toBeLessThan(1);
  });
});
