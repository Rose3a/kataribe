/**
 * 届いた PCM（16bit モノラル）を、届いた順に途切れなくつないで再生する。
 *
 * 生成が終わるのを待たずに鳴らすためのもので、WebAudio の時刻予約で
 * チャンクをつなぐ。最初のチャンクだけ少し先に予約して、後続が届くまでの
 * 揺れを吸収する。
 */
const START_DELAY_SECONDS = 0.04;

export class PcmStreamPlayer {
  private context: AudioContext | undefined;
  private nextTime = 0;
  private lastSource: AudioBufferSourceNode | undefined;
  private finished = false;
  private stopped = false;
  private resolveEnded!: (completed: boolean) => void;
  /** 全チャンクを鳴らし終えたら true、止められたら false で解決する。 */
  readonly ended = new Promise<boolean>((resolve) => {
    this.resolveEnded = resolve;
  });
  /** 再生が始まった（最初のチャンクを予約した）時刻。performance.now() 基準。 */
  startedAt: number | undefined;

  constructor(private readonly sinkId?: string) {}

  async push(samples: Int16Array, sampleRate: number): Promise<void> {
    if (this.stopped || samples.length === 0) return;
    if (this.context == undefined) {
      this.context = new AudioContext({ sampleRate });
      if (this.sinkId != undefined && this.context.setSinkId != undefined) {
        await this.context.setSinkId(this.sinkId).catch(() => undefined);
      }
      this.nextTime = this.context.currentTime + START_DELAY_SECONDS;
      this.startedAt = performance.now();
    }
    const context = this.context;
    const buffer = context.createBuffer(1, samples.length, sampleRate);
    const channel = buffer.getChannelData(0);
    for (let i = 0; i < samples.length; i++) channel[i] = samples[i] / 32768;
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);
    // 後続が間に合わなかったら（underrun）、いま鳴らせる時刻から続ける。
    const at = Math.max(this.nextTime, context.currentTime);
    source.start(at);
    this.nextTime = at + buffer.duration;
    this.lastSource = source;
    source.onended = () => {
      if (this.finished && this.lastSource === source) this.close(true);
    };
  }

  /** これ以上チャンクは来ない。残りを鳴らし終えたら ended が解決する。 */
  finish(): void {
    this.finished = true;
    if (this.lastSource == undefined) this.close(true);
  }

  stop(): void {
    if (this.stopped) return;
    this.stopped = true;
    this.close(false);
  }

  private close(completed: boolean): void {
    this.stopped = true;
    const context = this.context;
    this.context = undefined;
    void context?.close().catch(() => undefined);
    this.resolveEnded(completed);
  }
}
