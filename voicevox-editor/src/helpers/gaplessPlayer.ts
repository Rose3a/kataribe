/**
 * 連続再生で、1行ずつの音声を WebAudio の時刻予約で隙間なくつなぐ。
 *
 * audio 要素に WAV を差し替えて鳴らし直すと、行の切り替えごとに約 60ms 途切れていた。
 * 次の行は前の行の終わりの時刻ちょうどに予約する。呼び出し側（ContinuousPlayer）は
 * 前の行が鳴り終わってから次を渡すので、終わる LOOKAHEAD_SECONDS 前に
 * 「鳴り終わった」と返して次を受け取る。
 */
const LOOKAHEAD_SECONDS = 0.3;
const START_DELAY_SECONDS = 0.04;

export class GaplessPlayer {
  private context: AudioContext | undefined;
  private contextReady: Promise<AudioContext> | undefined;
  private nextTime = 0;
  private readonly sources = new Set<AudioBufferSourceNode>();
  private readonly timers = new Set<number>();
  private readonly pending = new Set<(completed: boolean) => void>();
  private idleWaiters: ((completed: boolean) => void)[] = [];
  private stopped = false;

  constructor(private readonly sinkId?: string) {}

  /** AudioContext の時刻（秒）。onStart の startAt と同じ基準。 */
  now(): number {
    return this.context?.currentTime ?? 0;
  }

  /**
   * 予約済みの音の続きに鳴らす。offset 秒目から鳴らす（途中のアクセント句からの再生）。
   * onStart は実際に鳴り始めるときに呼ぶ。startAt は行の頭に当たる AudioContext の時刻。
   * 終わる少し前に true、止められたら false で解決する。
   */
  async play(
    blob: Blob,
    {
      offset = 0,
      onStart,
    }: {
      offset?: number;
      onStart?: (buffer: AudioBuffer, startAt: number) => void;
    } = {},
  ): Promise<boolean> {
    if (this.stopped) return false;
    const context = await this.ensureContext();
    if (this.stopped) return false;
    const buffer = await context.decodeAudioData(await blob.arrayBuffer());
    if (this.stopped) return false;

    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);
    // 前の行が鳴り終わっていたら（生成が間に合わなかった）、いま鳴らせる時刻から。
    const at = Math.max(
      this.nextTime,
      context.currentTime + START_DELAY_SECONDS,
    );
    const start = Math.min(Math.max(0, offset), buffer.duration);
    source.start(at, start);
    const end = at + (buffer.duration - start);
    this.nextTime = end;
    this.sources.add(source);
    source.onended = () => {
      this.sources.delete(source);
      if (this.sources.size === 0) this.resolveIdle(true);
    };
    this.at(at, () => onStart?.(buffer, at - start));

    return new Promise<boolean>((resolve) => {
      this.pending.add(resolve);
      this.at(end - LOOKAHEAD_SECONDS, () => {
        this.pending.delete(resolve);
        resolve(true);
      });
    });
  }

  /** 予約した音を鳴らし終えるまで待つ。止められたら false。 */
  idle(): Promise<boolean> {
    if (this.stopped) return Promise.resolve(false);
    if (this.sources.size === 0) return Promise.resolve(true);
    return new Promise((resolve) => this.idleWaiters.push(resolve));
  }

  stop(): void {
    if (this.stopped) return;
    this.stopped = true;
    for (const source of this.sources) {
      source.onended = null;
      try {
        source.stop();
      } catch {
        // まだ鳴らしていない・鳴らし終えたものは無視
      }
    }
    this.sources.clear();
    for (const resolve of this.pending) resolve(false);
    this.pending.clear();
    this.resolveIdle(false);
    this.close();
  }

  close(): void {
    for (const timer of this.timers) window.clearTimeout(timer);
    this.timers.clear();
    const context = this.context;
    this.context = undefined;
    this.contextReady = undefined;
    void context?.close().catch(() => undefined);
  }

  private ensureContext(): Promise<AudioContext> {
    this.contextReady ??= (async () => {
      const context = new AudioContext();
      if (this.sinkId != undefined && context.setSinkId != undefined) {
        await context.setSinkId(this.sinkId).catch(() => undefined);
      }
      // 用意している間に止められた
      if (this.stopped) void context.close().catch(() => undefined);
      else this.context = context;
      return context;
    })();
    return this.contextReady;
  }

  /** AudioContext の時刻 time に fn を呼ぶ。 */
  private at(time: number, fn: () => void): void {
    const delay = Math.max(0, (time - this.now()) * 1000);
    const timer = window.setTimeout(() => {
      this.timers.delete(timer);
      if (!this.stopped) fn();
    }, delay);
    this.timers.add(timer);
  }

  private resolveIdle(completed: boolean): void {
    const waiters = this.idleWaiters;
    this.idleWaiters = [];
    for (const resolve of waiters) resolve(completed);
  }
}
