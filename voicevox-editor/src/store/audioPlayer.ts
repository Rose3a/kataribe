/**
 * HTMLAudioElement周りの音声再生・停止などを担当する。
 */
import {
  createVolumeEnvelope,
  ENVELOPE_SECONDS,
} from "@/helpers/portraitLipSync";
import { createPartialStore } from "./vuex";
import type { AudioPlayerStoreState, AudioPlayerStoreTypes } from "./type";
import type { AudioKey } from "@/type/preload";
import { showAlertDialog } from "@/components/Dialog/Dialog";

// ユニットテストが落ちるのを回避するための遅延読み込み
const getAudioElement = (() => {
  let audioElement: HTMLAudioElement | undefined = undefined;
  return () => {
    if (audioElement == undefined) {
      audioElement = new Audio();
    }
    return audioElement;
  };
})();

// ストリーミング再生中の音声（WebAudio）。STOP_AUDIO で止められるようにしておく。
let activeAudioStream: { stop: () => void } | undefined;
export function setActiveAudioStream(
  stream: { stop: () => void } | undefined,
  onlyIfCurrent?: { stop: () => void },
) {
  if (onlyIfCurrent != undefined && activeAudioStream !== onlyIfCurrent) return;
  activeAudioStream = stream;
}

let playbackBlob: Blob | undefined;
export function getPlaybackBlob(): Blob | undefined {
  return playbackBlob;
}

let volumeEnvelope: Float32Array | undefined;
let envelopeVersion = 0;
let audioDurationSeconds = 0;

/** 直近に読み込んだ音声の長さ(秒)。口パクの母音タイムライン配分に使う。 */
export function getAudioDurationSeconds(): number {
  return audioDurationSeconds;
}

async function prepareVolumeEnvelope(blob: Blob) {
  const version = ++envelopeVersion;
  volumeEnvelope = undefined;
  audioDurationSeconds = 0;
  if (!window.AudioContext) return;
  let context: AudioContext | undefined;
  try {
    context = new AudioContext();
    const audio = await context.decodeAudioData(await blob.arrayBuffer());
    if (version === envelopeVersion) {
      volumeEnvelope = createVolumeEnvelope(audio);
      audioDurationSeconds = audio.duration;
    }
  } catch {
    // Unsupported audio can still play normally; leave the portrait closed.
  } finally {
    await context?.close().catch(() => undefined);
  }
}

export const audioPlayerStoreState: AudioPlayerStoreState = {
  nowPlayingAudioKey: undefined,
};

export const audioPlayerStore = createPartialStore<AudioPlayerStoreTypes>({
  ACTIVE_AUDIO_ELEM_CURRENT_TIME_GETTER: {
    getter: (state) => {
      return () =>
        state._activeAudioKey != undefined
          ? getAudioElement().currentTime
          : undefined;
    },
  },

  AUDIO_PLAYBACK_VOLUME: {
    getter: () => () => {
      const audio = getAudioElement();
      if (audio.paused || audio.ended || audio.muted || audio.readyState < 3)
        return 0;
      return (
        (volumeEnvelope?.[Math.floor(audio.currentTime / ENVELOPE_SECONDS)] ??
          0) * audio.volume
      );
    },
  },

  NOW_PLAYING: {
    getter(state, getters) {
      const activeAudioKey = getters.ACTIVE_AUDIO_KEY;
      return (
        activeAudioKey != undefined &&
        activeAudioKey === state.nowPlayingAudioKey
      );
    },
  },

  SET_AUDIO_NOW_PLAYING: {
    mutation(
      state,
      { audioKey, nowPlaying }: { audioKey: AudioKey; nowPlaying: boolean },
    ) {
      state.nowPlayingAudioKey = nowPlaying ? audioKey : undefined;
    },
  },

  SET_AUDIO_SOURCE: {
    mutation(_, { audioBlob }: { audioBlob: Blob }) {
      playbackBlob = audioBlob;
      getAudioElement().src = URL.createObjectURL(audioBlob);
      void prepareVolumeEnvelope(audioBlob);
    },
  },

  PLAY_AUDIO_PLAYER: {
    async action(
      { state, mutations },
      { offset, audioKey }: { offset?: number; audioKey?: AudioKey },
    ) {
      const audioElement = getAudioElement();

      if (offset != undefined) {
        audioElement.currentTime = offset;
      }

      // 一部ブラウザではsetSinkIdが実装されていないので、その環境では無視する
      if (audioElement.setSinkId) {
        audioElement
          .setSinkId(state.savingSetting.audioOutputDevice)
          .catch((err: unknown) => {
            const stop = () => {
              audioElement.pause();
              audioElement.removeEventListener("canplay", stop);
            };
            audioElement.addEventListener("canplay", stop);
            void showAlertDialog({
              title: "エラー",
              message: "再生デバイスが見つかりません",
            });
            throw err;
          });
      }

      // 再生終了時にresolveされるPromiseを返す
      const played = async () => {
        if (audioKey) {
          mutations.SET_AUDIO_NOW_PLAYING({ audioKey, nowPlaying: true });
        }
      };
      audioElement.addEventListener("play", played);

      let paused: () => void;
      const audioPlayPromise = new Promise<boolean>((resolve) => {
        paused = () => {
          resolve(audioElement.ended);
        };
        audioElement.addEventListener("pause", paused);
      }).finally(async () => {
        audioElement.removeEventListener("play", played);
        audioElement.removeEventListener("pause", paused);
        if (audioKey) {
          mutations.SET_AUDIO_NOW_PLAYING({ audioKey, nowPlaying: false });
        }
      });

      void audioElement.play();

      return audioPlayPromise;
    },
  },

  STOP_AUDIO: {
    // 停止中でも呼び出して問題ない
    action() {
      // PLAY_ でonpause時の処理が設定されているため、pauseするだけで良い
      getAudioElement().pause();
      activeAudioStream?.stop();
    },
  },
});
