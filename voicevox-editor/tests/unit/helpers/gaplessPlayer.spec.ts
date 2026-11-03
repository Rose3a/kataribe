import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { GaplessPlayer } from "@/helpers/gaplessPlayer";

// 時計を進められる AudioContext のまね。音の長さは Blob のバイト数（秒）にする。
class FakeSource {
  buffer: { duration: number } | undefined;
  onended: (() => void) | null = null;
  startedAt: number | undefined;
  offset = 0;
  connect() {}
  start(at: number, offset = 0) {
    this.startedAt = at;
    this.offset = offset;
  }
  stop() {}
}

class FakeContext {
  static last: FakeContext | undefined;
  currentTime = 0;
  closed = false;
  sources: FakeSource[] = [];
  destination = {};
  constructor() {
    FakeContext.last = this;
  }
  decodeAudioData(data: ArrayBuffer) {
    return Promise.resolve({ duration: data.byteLength });
  }
  createBufferSource() {
    const source = new FakeSource();
    this.sources.push(source);
    return source;
  }
  close() {
    this.closed = true;
    return Promise.resolve();
  }
  /** seconds 秒進め、鳴り終えた音の onended を呼ぶ。 */
  async advance(seconds: number) {
    const target = this.currentTime + seconds;
    while (this.currentTime < target - 1e-9) {
      const step = Math.min(0.01, target - this.currentTime);
      this.currentTime += step;
      await vi.advanceTimersByTimeAsync(step * 1000);
      for (const source of this.sources) {
        const end =
          (source.startedAt ?? Infinity) +
          (source.buffer?.duration ?? 0) -
          source.offset;
        if (end <= this.currentTime + 1e-9 && source.onended) {
          const onended = source.onended;
          source.onended = null;
          onended();
        }
      }
    }
  }
}

const seconds = (n: number) => new Blob([new Uint8Array(n)]);

describe("GaplessPlayer", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.stubGlobal("AudioContext", FakeContext);
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("次の行を前の行の終わりの時刻ちょうどに予約する", async () => {
    const player = new GaplessPlayer();
    const starts: number[] = [];
    const first = player.play(seconds(2), {
      onStart: (_, startAt) => starts.push(startAt),
    });
    await vi.advanceTimersByTimeAsync(0);
    const context = FakeContext.last!;
    let firstDone = false;
    void first.then(() => (firstDone = true));

    await context.advance(1.6);
    expect(firstDone).toBe(false);
    await context.advance(0.2);
    // 終わる少し前に「鳴り終わった」と返る
    expect(await first).toBe(true);

    void player.play(seconds(3), {
      onStart: (_, startAt) => starts.push(startAt),
    });
    await vi.advanceTimersByTimeAsync(0);
    const [a, b] = context.sources;
    expect(b.startedAt).toBeCloseTo(a.startedAt! + 2);
    await context.advance(1);
    expect(starts).toHaveLength(2);
    expect(starts[1]).toBeCloseTo(starts[0] + 2);
  });

  it("途中から鳴らすときは offset 分だけ短くつなぐ", async () => {
    const player = new GaplessPlayer();
    void player.play(seconds(4), { offset: 1.5 });
    await vi.advanceTimersByTimeAsync(0);
    void player.play(seconds(1));
    await vi.advanceTimersByTimeAsync(0);
    const [a, b] = FakeContext.last!.sources;
    expect(a.offset).toBe(1.5);
    expect(b.startedAt).toBeCloseTo(a.startedAt! + 2.5);
  });

  it("idle は最後の音が鳴り終えるまで待つ", async () => {
    const player = new GaplessPlayer();
    const played = player.play(seconds(1));
    await vi.advanceTimersByTimeAsync(0);
    const context = FakeContext.last!;
    await context.advance(0.8);
    expect(await played).toBe(true);
    let idle = false;
    void player.idle().then(() => (idle = true));
    await context.advance(0.1);
    expect(idle).toBe(false);
    await context.advance(0.2);
    expect(idle).toBe(true);
  });

  it("止めると待っている再生は false で返り、以後は鳴らさない", async () => {
    const player = new GaplessPlayer();
    const played = player.play(seconds(2));
    await vi.advanceTimersByTimeAsync(0);
    const context = FakeContext.last!;
    player.stop();
    expect(await played).toBe(false);
    expect(await player.idle()).toBe(false);
    expect(await player.play(seconds(1))).toBe(false);
    expect(context.closed).toBe(true);
  });
});
