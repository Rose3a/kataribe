import { z } from "zod";

import {
  engineIdSchema,
  noteIdSchema,
  presetKeySchema,
  speakerIdSchema,
  styleIdSchema,
} from "@/type/preload";

// トーク系のスキーマ
export const moraSchema = z.object({
  text: z.string(),
  vowel: z.string(),
  vowelLength: z.number(),
  pitch: z.number(),
  consonant: z.string().optional(),
  consonantLength: z.number().optional(),
});

export const accentPhraseSchema = z.object({
  moras: z.array(moraSchema),
  accent: z.number(),
  pauseMora: moraSchema.optional(),
  isInterrogative: z.boolean().optional(),
});

export const audioQuerySchema = z.object({
  accentPhrases: z.array(accentPhraseSchema),
  speedScale: z.number(),
  pitchScale: z.number(),
  intonationScale: z.number(),
  volumeScale: z.number(),
  pauseLengthScale: z.number(),
  prePhonemeLength: z.number(),
  postPhonemeLength: z.number(),
  outputSamplingRate: z.union([z.number(), z.literal("engineDefault")]),
  outputStereo: z.boolean(),
  kana: z.string().optional(),
});

export const morphingInfoSchema = z.object({
  rate: z.number(),
  targetEngineId: engineIdSchema,
  targetSpeakerId: speakerIdSchema,
  targetStyleId: styleIdSchema,
});

export const audioItemSchema = z.object({
  text: z.string(),
  voice: z.object({
    engineId: engineIdSchema,
    speakerId: speakerIdSchema,
    styleId: styleIdSchema,
  }),
  query: audioQuerySchema.optional(),
  presetKey: presetKeySchema.optional(),
  morphingInfo: morphingInfoSchema.optional(),
  // prettier-ignore
  irodori: z.object({
    seed: z.number().nullable().optional(),
    steps: z.number().int().min(1).max(80).optional(),
    schedule: z.enum(["linear", "sway"]).optional(),
    seconds: z.number().min(0.1).max(60).nullable().optional(),
    caption: z.string().max(2000).optional(),
    captionStrength: z.number().min(0).max(1).optional(),
    cfgText: z.number().min(0).max(20).optional(),
    cfgCaption: z.number().min(0).max(20).optional(),
    cfgSpeaker: z.number().min(0).max(20).optional(),
    referenceStrength: z.number().min(0).max(1).optional(),
    speakerStrength: z.number().min(0).max(1).optional(),
    // Previous projects may still contain one additional speaker.
    secondarySpeakerStyleId: z.number().int().nullable().optional(),
    secondarySpeakerStrength: z.number().min(0).max(1).optional(),
    additionalSpeakers: z.array(z.object({
      styleId: z.number().int(),
      strength: z.number().min(0).max(1),
    })).max(3).optional(),
    // prettier-ignore
    referenceAudio: z.object({
      dataUrl: z.string(),
      mime: z.string().optional(),
      name: z.string().optional(),
    }).optional(),
  }).optional(),
});

// ソング系のスキーマ
export const tempoSchema = z.object({
  position: z.number(),
  bpm: z.number(),
});

export const timeSignatureSchema = z.object({
  measureNumber: z.number(),
  beats: z.number(),
  beatType: z.number(),
});

export const noteSchema = z.object({
  id: noteIdSchema,
  position: z.number(),
  duration: z.number(),
  noteNumber: z.number(),
  lyric: z.string().nullable(), // 歌詞未入力のときはnull
});

export const singerSchema = z.object({
  engineId: engineIdSchema,
  styleId: styleIdSchema,
});

export const singingTeacherSchema = z.object({
  // TODO: 歌い方設定UIをまずは実装するため、現状はシンガーと歌い方教師が
  // 同一エンジンを使用する前提で、歌い方教師にはstyleIdのみを保持している。
  // 実際のマルチエンジン環境でどう扱うかは未調査で、また型として妥当かも仮でしかないため、仕様を整理してから修正する
  styleId: styleIdSchema,
});

export const phonemeTimingEditSchema = z.object({
  phonemeIndexInNote: z.number(), // ノート内での音素の順番
  offsetSeconds: z.number(), // 単位は秒
});

/** 元のボリュームからのdB変化量。データが無いところはnull。 */
export const volumeEditValueSchema = z.number().nullable();

export const trackSchema = z.object({
  name: z.string(),
  singer: singerSchema.optional(),
  singingTeacher: singingTeacherSchema.optional(),
  keyRangeAdjustment: z.number(), // 音域調整量
  volumeRangeAdjustment: z.number(), // 声量調整量
  notes: z.array(noteSchema),
  pitchEditData: z.array(z.number()), // 値の単位はHzで、データが無いところはVALUE_INDICATING_NO_DATAの値
  volumeEditData: z.array(volumeEditValueSchema),
  phonemeTimingEditData: z.map(noteIdSchema, z.array(phonemeTimingEditSchema)), // 音素タイミングの編集データはノートと紐づけて保持

  solo: z.boolean(),
  mute: z.boolean(),
  gain: z.number(),
  pan: z.number(),
});
