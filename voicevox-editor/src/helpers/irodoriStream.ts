/**
 * POST /irodori/synthesis_stream の応答（フレーム列）を読む。
 *
 * 各フレームは little-endian の uint32 長さ + 本体。最初が JSON ヘッダ、
 * 続いて PCM（s16le）、長さ0で終わり。長さの最上位ビットが立っていれば
 * 本体はエラーメッセージ（ヘッダ送信後に生成が失敗した場合）。
 */
export type IrodoriPcmHeader = {
  sampleRate: number;
  channels: number;
  format: string;
};

const ERROR_FLAG = 0x80000000;

export async function readIrodoriPcmStream(
  response: Response,
  handlers: {
    onHeader: (header: IrodoriPcmHeader) => void;
    onPcm: (samples: Int16Array) => void | Promise<void>;
  },
): Promise<void> {
  if (response.body == undefined) throw new Error("stream body is empty");
  const reader = response.body.getReader();
  let pending = new Uint8Array(0);
  let headerSeen = false;
  const append = (chunk: Uint8Array) => {
    const joined = new Uint8Array(pending.length + chunk.length);
    joined.set(pending);
    joined.set(chunk, pending.length);
    pending = joined;
  };
  for (;;) {
    const { done, value } = await reader.read();
    if (value != undefined) append(value);
    for (;;) {
      if (pending.length < 4) break;
      const view = new DataView(
        pending.buffer,
        pending.byteOffset,
        pending.byteLength,
      );
      const word = view.getUint32(0, true);
      const isError = (word & ERROR_FLAG) !== 0;
      const length = word & ~ERROR_FLAG;
      if (pending.length < 4 + length) break;
      const body = pending.slice(4, 4 + length);
      pending = pending.slice(4 + length);
      if (isError) throw new Error(new TextDecoder().decode(body));
      if (!headerSeen) {
        headerSeen = true;
        handlers.onHeader(
          JSON.parse(new TextDecoder().decode(body)) as IrodoriPcmHeader,
        );
        continue;
      }
      if (length === 0) return;
      // s16le。偶数バイトで切れているので Int16Array にそのまま見立てられる。
      const aligned = body.buffer.slice(
        body.byteOffset,
        body.byteOffset + body.byteLength,
      );
      await handlers.onPcm(new Int16Array(aligned));
    }
    if (done) throw new Error("stream ended before the completion frame");
  }
}

/** モノラル s16le の PCM を WAV にする。 */
export function pcm16ToWav(parts: Int16Array[], sampleRate: number): Blob {
  const samples = parts.reduce((sum, part) => sum + part.length, 0);
  const header = new ArrayBuffer(44);
  const view = new DataView(header);
  const text = (offset: number, value: string) => {
    for (let i = 0; i < value.length; i++)
      view.setUint8(offset + i, value.charCodeAt(i));
  };
  text(0, "RIFF");
  view.setUint32(4, 36 + samples * 2, true);
  text(8, "WAVE");
  text(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  text(36, "data");
  view.setUint32(40, samples * 2, true);
  return new Blob([header, ...parts.map((part) => part.slice().buffer)], {
    type: "audio/wav",
  });
}
