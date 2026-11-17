<template>
  <div class="speaker-train">
    <div class="intro">
      <p class="q-mb-xs">
        キャラの音声から話者（16
        トークンの話者埋め込み）を作ります。モデル本体は変えずに話者ファイルだけを学習するので、TensorRT
        でもそのまま使えます。音声と同じ名前の .txt
        があればセリフとして使い、無い音声は Haas-ja で書き起こします。1〜30
        秒の音声を合計 10 分以上入れるのがおすすめです。
      </p>
      <p class="text-warning q-mb-none">
        使えるのは、話者本人または権利者から許可を得た音声だけです（README
        の「Speaker Inversion と話者埋め込み」を参照）。
      </p>
    </div>

    <div class="layout">
      <!-- ───────── 左: 設定 ───────── -->
      <div class="settings-col">
        <div v-if="!options" class="panel text-caption">
          {{ loading ? "読み込み中…" : "エンジンから設定を取れませんでした。" }}
        </div>
        <template v-else>
          <section class="panel">
            <h3 class="panel-title"><QIcon name="folder" />データ</h3>
            <QInput
              v-model="name"
              dense
              outlined
              label="話者の名前"
              hint="話者一覧に出る名前です（1〜32文字）。"
              :disable="locked"
              class="q-mb-md"
            />
            <QInput
              v-model="dataDir"
              dense
              outlined
              label="音声のフォルダ"
              hint="下のフォルダも探します（.wav .flac .ogg .mp3）。パスの貼り付けも可。"
              :disable="locked"
              class="q-mb-md"
            >
              <template #append>
                <QBtn
                  flat
                  dense
                  noCaps
                  icon="folder_open"
                  label="選ぶ"
                  :loading="picking"
                  :disable="locked"
                  @click="pickFolder"
                />
              </template>
            </QInput>
            <QSelect
              v-model="baseModel"
              dense
              outlined
              emitValue
              mapOptions
              :options="baseModelOptions"
              label="ベースモデル"
              hint="v4.1 Small と同じ構造のモデルだけ出ます（量子化版・MeanFlow・Large は不可）。"
              :disable="locked"
            >
              <template #after>
                <QBtn
                  flat
                  round
                  dense
                  icon="refresh"
                  aria-label="選べるモデルを更新"
                  :disable="locked"
                  @click="loadOptions"
                />
              </template>
            </QSelect>
            <QToggle
              v-if="exists"
              v-model="overwrite"
              class="q-mt-sm"
              :label="`同じ名前の学習した話者「${name.trim()}」を上書きする`"
              :disable="locked"
            />
          </section>

          <section class="panel">
            <div class="panel-title-row">
              <h3 class="panel-title"><QIcon name="tune" />学習の設定</h3>
              <QSpace />
              <QBtnDropdown
                v-if="options.runs.length > 0"
                flat
                dense
                noCaps
                size="sm"
                icon="history"
                label="過去の設定"
                :disable="locked"
              >
                <QList dense>
                  <QItem
                    v-for="run in options.runs"
                    :key="run.name"
                    v-close-popup
                    clickable
                    @click="loadSettingsFrom(run.name)"
                  >
                    <QItemSection>
                      <QItemLabel>{{ run.name }}</QItemLabel>
                      <QItemLabel caption>{{ runCaption(run) }}</QItemLabel>
                    </QItemSection>
                  </QItem>
                </QList>
              </QBtnDropdown>
              <QBtn
                flat
                dense
                noCaps
                size="sm"
                icon="restart_alt"
                label="既定に戻す"
                :disable="locked || isDefault"
                @click="resetSettings"
              />
            </div>
            <div v-if="notice" class="text-caption text-positive q-mb-sm">
              {{ notice }}
            </div>

            <div class="field-label">使う装置</div>
            <QBtnToggle
              v-model="device"
              dense
              noCaps
              spread
              toggleColor="primary"
              :options="deviceOptions"
              :disable="locked"
              class="q-mb-xs device-toggle"
            />
            <div class="text-caption field-note q-mb-md">{{ timeHint }}</div>

            <div class="fields">
              <QInput
                v-model.number="form.steps"
                dense
                outlined
                type="number"
                label="ステップ数"
                hint="学習の回数"
                :error="!!errors.steps"
                :errorMessage="errors.steps"
                :disable="locked"
              />
              <QInput
                v-model.number="form.lr"
                dense
                outlined
                type="number"
                step="any"
                label="学習率"
                :hint="`既定 ${formatNumber(defaults.lr)}`"
                :error="!!errors.lr"
                :errorMessage="errors.lr"
                :disable="locked"
              />
              <QSelect
                v-model="form.lrSchedule"
                dense
                outlined
                emitValue
                mapOptions
                :options="scheduleOptions"
                label="学習率の変え方"
                :disable="locked"
              />
              <QInput
                v-model.number="form.warmup"
                dense
                outlined
                type="number"
                label="ウォームアップ"
                hint="0 から上げる回数"
                :error="!!errors.warmup"
                :errorMessage="errors.warmup"
                :disable="locked"
              />
              <QInput
                v-model.number="form.batch"
                dense
                outlined
                type="number"
                label="バッチ"
                hint="0 = 自動"
                :error="!!errors.batch"
                :errorMessage="errors.batch"
                :disable="locked"
              />
              <QInput
                v-model.number="form.durWeight"
                dense
                outlined
                type="number"
                step="any"
                label="時間予測の重み"
                :hint="learnSpeed ? '話速を覚える強さ' : '話速オフでは不要'"
                :error="!!errors.durWeight"
                :errorMessage="errors.durWeight"
                :disable="locked || !learnSpeed"
              />
            </div>
            <QToggle
              v-model="learnSpeed"
              label="話速も学習する"
              :disable="locked"
              class="q-mt-sm"
            />
            <div class="text-caption field-note q-mb-md">
              オンでキャラの話す速さ・間の取り方も覚えます（ゆっくり話すキャラ向け）。オフだと話速は音声を参照したときと同じくらいです。
            </div>

            <h4 class="sub-title">検証</h4>
            <div class="text-caption field-note q-mb-sm">
              学習に使わない音声を少し取り分けて、決まった条件で loss
              を測ります（{{ options.valMinTotal }} 本未満なら取り分けません）。
            </div>
            <div class="fields">
              <QInput
                v-model.number="form.valPercent"
                dense
                outlined
                type="number"
                step="any"
                label="検証に回す割合（%）"
                hint="0 で検証しない"
                :error="!!errors.valPercent"
                :errorMessage="errors.valPercent"
                :disable="locked"
              />
              <QInput
                v-model.number="form.valEvery"
                dense
                outlined
                type="number"
                label="検証の間隔"
                hint="0 = 自動"
                :error="!!errors.valEvery"
                :errorMessage="errors.valEvery"
                :disable="locked || form.valPercent === 0"
              />
              <QSelect
                v-model="form.keep"
                dense
                outlined
                emitValue
                mapOptions
                :options="keepOptions"
                label="保存する話者"
                :disable="locked || form.valPercent === 0"
              />
              <QInput
                v-model.number="form.seed"
                dense
                outlined
                type="number"
                label="シード"
                hint="分け方・順番"
                :error="!!errors.seed"
                :errorMessage="errors.seed"
                :disable="locked"
              />
            </div>
            <QExpansionItem
              dense
              denseToggle
              label="設定の説明"
              icon="help_outline"
              class="help q-mt-sm"
            >
              <dl class="help-list">
                <dt>ステップ数</dt>
                <dd>
                  学習を回す回数です。多いほど音声に近づきますが、検証 loss
                  が下がらなくなったら、それ以上は時間の無駄です。
                </dd>
                <dt>学習率</dt>
                <dd>
                  1 回に話者を動かす大きさ（AdamW）。大きいと速く動きますが loss
                  が揺れ、小さいと進みが遅くなります。
                </dd>
                <dt>学習率の変え方・ウォームアップ</dt>
                <dd>
                  一定はずっと同じ学習率。cosine は終わりに向けて 0
                  近くまで下げ、最後を落ち着かせます。ウォームアップは最初の数ステップで
                  0 から上げます。
                </dd>
                <dt>バッチ</dt>
                <dd>
                  1 回に使う音声の本数。0 は自動（GPU 16・CPU 4）。大きいほど
                  loss が安定しますが、VRAM と時間が増えます。
                </dd>
                <dt>時間予測の重み</dt>
                <dd>
                  話速（時間予測）の誤差をどれだけ話者に反映するか。話速も学習するときだけ使います。
                </dd>
                <dt>検証</dt>
                <dd>
                  学習の loss は毎回 t
                  と雑音が変わるので揺れます。検証は取り分けた音声を同じ
                  t・同じ雑音で測るので、良くなっているかを比べられます。間隔の
                  0 は自動（GPU 約 20 回・CPU 約 10 回）。別の学習と検証 loss
                  を比べられるのは、音声・シード・割合が同じとき（同じ検証の音声になるとき）です。
                </dd>
                <dt>保存する話者</dt>
                <dd>
                  最後のステップか、検証 loss
                  が一番低かったときの話者を保存します（検証がないときは最後）。
                </dd>
                <dt>シード</dt>
                <dd>同じ値なら、検証の分け方と学習の順番が同じになります。</dd>
              </dl>
            </QExpansionItem>
          </section>

          <section class="panel">
            <h3 class="panel-title"><QIcon name="subject" />テキスト</h3>
            <QToggle
              v-model="verifyText"
              label="付属のテキストも書き起こしで確かめる"
              :disable="locked || !options.haasAvailable"
            />
            <div class="text-caption field-note">
              音声と大きく違うテキスト（別のセリフ・性別の分岐など）は、書き起こしに置き換えます。
            </div>
            <div
              v-if="!options.haasAvailable"
              class="text-caption text-warning q-mt-sm"
            >
              書き起こし（Haas-ja）が入っていません。テキストの無い音声を使うには、エンジンフォルダの
              bat\haas_setup.bat を一度実行してください（約 1.2GB
              のモデルと、別の Python 環境を入れます）。
            </div>
          </section>

          <QBtn
            v-if="!running"
            unelevated
            color="primary"
            textColor="display-on-primary"
            icon="school"
            label="学習を始める"
            class="full-width start-btn"
            :disable="disable || !canStart"
            :loading="starting"
            @click="start"
          />
          <div
            v-if="!running && startBlocker"
            class="text-caption field-note q-mt-xs"
          >
            {{ startBlocker }}
          </div>
        </template>
        <div
          v-if="error"
          role="alert"
          class="text-caption text-negative q-mt-sm"
          style="white-space: pre-wrap"
        >
          {{ error }}
        </div>
      </div>

      <!-- ───────── 右: 学習の様子 ───────── -->
      <div class="monitor-col">
        <section class="panel">
          <div class="panel-title-row">
            <h3 class="panel-title"><QIcon name="insights" />学習の様子</h3>
            <QBadge
              v-if="stateLabel"
              :color="stateColor"
              :textColor="stateTextColor"
              class="q-ml-sm"
              >{{ stateLabel }}</QBadge
            >
            <QSpace />
            <template v-if="running">
              <QBtn
                outline
                dense
                noCaps
                color="primary"
                icon="save"
                label="ここで終えて保存"
                class="q-mr-sm"
                :disable="status?.stage !== 'train' || status?.stopRequested"
                @click="stopEarly"
              >
                <QTooltip
                  >今の話者を保存して学習を終えます（中止と違って結果が残ります）</QTooltip
                >
              </QBtn>
              <QBtn
                outline
                dense
                noCaps
                color="negative"
                icon="stop"
                label="中止"
                @click="cancel"
              />
            </template>
          </div>

          <ol class="stages" aria-label="学習の段階">
            <li
              v-for="(stage, i) in STAGES"
              :key="stage.id"
              :class="`stage-${stageState(i)}`"
            >
              <QIcon :name="stageIcon(i)" size="16px" />
              <span>{{ stage.label }}</span>
            </li>
          </ol>

          <template v-if="status && status.state !== 'idle'">
            <QLinearProgress
              :value="status.percent / 100"
              rounded
              size="6px"
              :color="stateColor"
              class="q-mt-sm q-mb-xs"
            />
            <div class="text-caption status-line">
              <span
                >{{ Math.round(status.percent) }}% · {{ status.message }}</span
              >
              <span v-if="elapsedLabel" class="field-note">{{
                elapsedLabel
              }}</span>
            </div>
          </template>
          <div v-else class="text-caption field-note q-mt-sm">
            左で設定して「学習を始める」を押すと、ここに loss などが出ます。
          </div>

          <div v-if="tiles.length > 0" class="tiles">
            <div v-for="tile in tiles" :key="tile.label" class="tile">
              <div class="tile-label">{{ tile.label }}</div>
              <div class="tile-value">
                {{ tile.value
                }}<span v-if="tile.unit" class="tile-unit">{{
                  tile.unit
                }}</span>
              </div>
              <div v-if="tile.sub" class="tile-sub">{{ tile.sub }}</div>
            </div>
          </div>
        </section>

        <section class="panel">
          <div class="chart-toolbar">
            <div class="toolbar-item smoothing">
              <span class="field-note">平滑化</span>
              <QSlider
                v-model="smoothing"
                :min="0"
                :max="0.99"
                :step="0.01"
                dense
                class="smoothing-slider"
                aria-label="学習の線の平滑化"
              />
              <span class="smoothing-value">{{ smoothing.toFixed(2) }}</span>
            </div>
            <QToggle
              v-model="logY"
              dense
              label="縦軸を対数に"
              class="toolbar-item"
            />
            <QSpace />
            <QSelect
              v-model="compareName"
              dense
              outlined
              emitValue
              mapOptions
              clearable
              :options="compareOptions"
              label="比べる学習"
              class="compare-select"
              :disable="compareOptions.length === 0"
            >
              <template #no-option>
                <QItem
                  ><QItemSection
                    >loss の記録がある学習はまだありません</QItemSection
                  ></QItem
                >
              </template>
            </QSelect>
          </div>
          <div
            v-if="compareRecord"
            class="text-caption field-note q-mb-sm compare-note"
          >
            比べる「{{ compareRecord.name }}」:
            {{ recordSummary(compareRecord) }}
            <QBtn
              flat
              dense
              noCaps
              size="sm"
              icon="input"
              label="この設定を読み込む"
              :disable="locked"
              @click="applyRecord(compareRecord)"
            />
          </div>
          <div v-if="compareError" class="text-caption text-negative q-mb-sm">
            {{ compareError }}
          </div>

          <IrodoriTrainChart
            v-model:hoverX="hoverStep"
            title="loss（RF・速度場の誤差）"
            :series="lossSeries"
            :xMax
            :height="260"
            :logY
            emptyText="学習が始まると loss が出ます"
          />
          <div class="small-charts">
            <IrodoriTrainChart
              v-model:hoverX="hoverStep"
              compact
              title="時間予測の誤差（秒）"
              :series="durSeries"
              :xMax
              :height="150"
              :logY
              :format="(v) => formatNumber(v)"
            />
            <IrodoriTrainChart
              v-model:hoverX="hoverStep"
              compact
              title="学習率"
              :series="lrSeries"
              :xMax
              :height="150"
            />
            <IrodoriTrainChart
              v-model:hoverX="hoverStep"
              compact
              title="勾配ノルム"
              :series="gradSeries"
              :xMax
              :height="150"
              :logY
            />
            <IrodoriTrainChart
              v-model:hoverX="hoverStep"
              compact
              title="初期値（ゼロショット）からの変化（%）"
              :series="driftSeries"
              :xMax
              :height="150"
            />
          </div>
          <div class="text-caption field-note q-mt-sm">
            青 = 学習、オレンジ = 検証（ステップ 0
            は学習前のゼロショット）。学習の線は平滑化しています（薄い線は平滑化前）。グラフにポインターを置くか、フォーカスして
            ← → で値を読めます。
          </div>
        </section>

        <section
          v-if="status?.state === 'done' && status.result"
          class="panel result"
        >
          <h3 class="panel-title"><QIcon name="task_alt" />結果</h3>
          <dl class="result-list">
            <dt>保存した話者</dt>
            <dd>
              「{{ status.result.name }}」（話者一覧に追加済み）·
              {{ savedLabel }}
            </dd>
            <template v-if="resultMetrics.val_loss_init != undefined">
              <dt>検証 loss</dt>
              <dd>
                ゼロショット {{ formatNumber(resultMetrics.val_loss_init) }} →
                保存した話者
                {{ formatNumber(resultMetrics.val_loss_saved ?? NaN) }}
                <span class="field-note"
                  >（{{
                    percentChange(
                      resultMetrics.val_loss_init,
                      resultMetrics.val_loss_saved,
                    )
                  }}・最小
                  {{
                    formatNumber(resultMetrics.val_loss_best ?? NaN)
                  }}、ステップ {{ resultMetrics.best_step }}）</span
                >
              </dd>
              <dt>時間予測の誤差（検証）</dt>
              <dd>
                {{ formatNumber(resultMetrics.val_dur_mae_sec_init ?? NaN) }} 秒
                →
                {{ formatNumber(resultMetrics.val_dur_mae_sec_saved ?? NaN) }}
                秒
              </dd>
            </template>
            <dt>学習 loss（最後の 1 割の平均）</dt>
            <dd>{{ formatNumber(resultMetrics.final_train_loss ?? NaN) }}</dd>
            <dt>話速</dt>
            <dd>ベース（話者なし）の {{ paceLabel }}</dd>
            <dt>データ</dt>
            <dd>
              使った音声 {{ status.result.data.used }} 本・{{
                status.result.data.used_minutes
              }}
              分<template v-if="status.result.data.val"
                >（うち検証 {{ status.result.data.val }} 本）</template
              >。書き起こし
              {{ status.result.data.transcribed }} 本、テキストを置き換え
              {{ status.result.data.text_replaced_by_asr }} 本、短すぎ
              {{ status.result.data.too_short }} 本・長すぎ
              {{ status.result.data.too_long }} 本は除外。
            </dd>
            <dt>学習時間</dt>
            <dd>
              {{ formatDuration(resultMetrics.train_seconds)
              }}<template v-if="resultMetrics.peak_vram_gb">
                · VRAM 最大 {{ resultMetrics.peak_vram_gb }} GB</template
              >
            </dd>
            <dt>ファイル</dt>
            <dd class="path">{{ status.result.speaker_file }}</dd>
          </dl>
        </section>
        <section v-if="status?.state === 'error'" class="panel" role="alert">
          <h3 class="panel-title text-negative">
            <QIcon name="error" />失敗しました
          </h3>
          <div class="text-caption" style="white-space: pre-wrap">
            {{ status.error }}
          </div>
        </section>

        <section v-if="hasDetails" class="panel details">
          <QExpansionItem
            v-if="valRows.length > 0"
            dense
            label="検証の記録（表）"
            :caption="`${valRows.length} 回`"
          >
            <QMarkupTable dense flat class="val-table">
              <thead>
                <tr>
                  <th class="text-right">ステップ</th>
                  <th class="text-right">検証 loss</th>
                  <th class="text-right">ゼロショット比</th>
                  <th class="text-right">時間予測の誤差（秒）</th>
                  <th class="text-right">学習 loss（平滑化）</th>
                  <th class="text-right">学習率</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="row in valRows"
                  :key="row.step"
                  :class="{ best: row.isBest }"
                >
                  <td class="text-right">{{ row.step }}</td>
                  <td class="text-right">
                    {{ formatNumber(row.loss)
                    }}<QIcon
                      v-if="row.isBest"
                      name="star"
                      size="14px"
                      class="q-ml-xs"
                      aria-label="最小"
                    />
                  </td>
                  <td class="text-right">{{ row.change }}</td>
                  <td class="text-right">{{ formatNumber(row.dur) }}</td>
                  <td class="text-right">{{ row.train }}</td>
                  <td class="text-right">{{ row.lr }}</td>
                </tr>
              </tbody>
            </QMarkupTable>
          </QExpansionItem>
          <QExpansionItem
            v-if="configLines.length > 0"
            dense
            label="実際の設定"
            caption="自動の値を決めた後"
          >
            <ul class="config-list">
              <li v-for="item in configLines" :key="item">{{ item }}</li>
            </ul>
          </QExpansionItem>
          <QExpansionItem
            v-if="(status?.log.length ?? 0) > 0"
            dense
            label="ログ"
          >
            <pre class="log">{{ status?.log.join("\n") }}</pre>
          </QExpansionItem>
        </section>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import IrodoriTrainChart, {
  formatNumber,
  type ChartSeries,
} from "./IrodoriTrainChart.vue";
import { useStore } from "@/store";
import type {
  IrodoriSpeakerTrainNumberKey as NumberKey,
  IrodoriSpeakerTrainOptions as Options,
  IrodoriSpeakerTrainPoint as TrainPoint,
  IrodoriSpeakerTrainRecord as TrainRecord,
  IrodoriSpeakerTrainRequest as TrainRequest,
  IrodoriSpeakerTrainRun as TrainRun,
  IrodoriSpeakerTrainSettings as Settings,
  IrodoriSpeakerTrainStatus as TrainStatus,
  IrodoriSpeakerTrainValPoint as ValPoint,
} from "@/domain/irodori";
import {
  cancelSpeakerTrain,
  fetchSpeakerTrainOptions,
  fetchSpeakerTrainRecord,
  fetchSpeakerTrainStatus,
  pickIrodoriFolder,
  refreshIrodoriSpeakers,
  startSpeakerTrain,
  stopSpeakerTrain,
} from "@/helpers/irodoriEngine";
import type { EngineId } from "@/type/preload";

const props = defineProps<{
  endpoint: string;
  engineId: EngineId;
  /** いまエディタで選んでいるモデル（ベースの初期値にする）。 */
  currentModel?: string;
  disable?: boolean;
}>();
const store = useStore();

// エンジンから既定値が取れるまでの値（speaker_train.py の既定と同じ）
const FALLBACK: Settings = {
  steps: 500,
  lr: 0.01,
  lrSchedule: "constant",
  warmup: 0,
  batch: 0,
  durWeight: 1,
  valFrac: 0.05,
  valEvery: 0,
  keep: "last",
  seed: 0,
};
const INT_KEYS = new Set<FormNumberKey>([
  "steps",
  "warmup",
  "batch",
  "valEvery",
  "seed",
]);
// speaker_train.LIMITS と同じ範囲
const FALLBACK_LIMITS: Options["limits"] = {
  steps: [50, 5000],
  lr: [1e-5, 1],
  warmup: [0, 2000],
  batch: [0, 64],
  durWeight: [0, 10],
  valFrac: [0, 0.3],
  valEvery: [0, 5000],
  seed: [0, 2 ** 31 - 1],
};
const STAGES = [
  { id: "scan", label: "音声を集める", match: ["scan"] },
  { id: "transcribe", label: "書き起こし", match: ["transcribe"] },
  { id: "encode", label: "符号化", match: ["encode", "prepare"] },
  { id: "load", label: "読み込み", match: ["load"] },
  { id: "train", label: "学習", match: ["train"] },
  { id: "save", label: "保存", match: ["save", "done"] },
] as const;

/** 画面の入力欄（検証の割合だけ % で持つ）。 */
type Form = Omit<Settings, "valFrac"> & { valPercent: number };
type FormNumberKey = Exclude<NumberKey, "valFrac"> | "valPercent";

const options = ref<Options>();
const status = ref<TrainStatus>();
const loading = ref(false);
const starting = ref(false);
const picking = ref(false);
const error = ref("");
const notice = ref("");

const name = ref("");
const dataDir = ref("");
const baseModel = ref("");
const learnSpeed = ref(true);
const verifyText = ref(true);
const device = ref<"cuda" | "cpu">("cuda");
const overwrite = ref(false);
const form = ref<Form>(toForm(FALLBACK));
let formInitialized = false;

// グラフ
const smoothing = ref(0.8);
const logY = ref(false);
const hoverStep = ref<number | null>(null);
const compareName = ref<string | null>(null);
const compareRecord = ref<TrainRecord>();
const compareError = ref("");
const now = ref(Date.now() / 1000);

function toForm(value: Settings): Form {
  const { valFrac, ...rest } = value;
  return { ...rest, valPercent: Number((valFrac * 100).toFixed(4)) };
}
function toSettings(value: Form): Settings {
  const { valPercent, ...rest } = value;
  return { ...rest, valFrac: valPercent / 100 };
}

const defaults = computed<Settings>(() => {
  const value = options.value?.defaults;
  if (!value) return FALLBACK;
  const { speed: _speed, device: _device, ...rest } = value;
  return { ...FALLBACK, ...rest };
});
const running = computed(() => status.value?.state === "running");
const locked = computed(() => running.value || !!props.disable);
const exists = computed(() =>
  (options.value?.existing ?? []).includes(name.value.trim()),
);

// ───────── 入力の確かめ
const limitsFor = (key: FormNumberKey): [number, number] | undefined => {
  if (key === "valPercent") {
    const lim = options.value?.limits.valFrac;
    return lim && [lim[0] * 100, lim[1] * 100];
  }
  return options.value?.limits[key];
};
const errors = computed(() => {
  const out: Partial<Record<FormNumberKey, string>> = {};
  const keys: FormNumberKey[] = [
    "steps",
    "lr",
    "warmup",
    "batch",
    "durWeight",
    "valPercent",
    "valEvery",
    "seed",
  ];
  for (const key of keys) {
    const value: unknown = form.value[key];
    const lim = limitsFor(key);
    if (typeof value !== "number" || !Number.isFinite(value))
      out[key] = "数を入れてください";
    else if (INT_KEYS.has(key) && !Number.isInteger(value))
      out[key] = "整数を入れてください";
    else if (lim && (value < lim[0] || value > lim[1]))
      out[key] = `範囲 ${formatNumber(lim[0])}〜${formatNumber(lim[1])}`;
    else if (key === "warmup" && value >= form.value.steps)
      out[key] = "ステップ数より小さく";
  }
  return out;
});
const startBlocker = computed(() => {
  if (name.value.trim().length === 0) return "話者の名前を入れてください。";
  if (dataDir.value.trim().length === 0)
    return "音声のフォルダを選んでください。";
  if (baseModel.value.length === 0) return "ベースモデルを選んでください。";
  if (Object.keys(errors.value).length > 0)
    return "学習の設定に範囲外の値があります。";
  if (exists.value && !overwrite.value)
    return "同じ名前の話者があります。別の名前にするか、上書きを選んでください。";
  return "";
});
const canStart = computed(() => startBlocker.value === "");
const isDefault = computed(
  () =>
    JSON.stringify(toSettings(form.value)) ===
      JSON.stringify(toSettings(toForm(defaults.value))) && learnSpeed.value,
);

// ───────── 選択肢
const baseModelOptions = computed(() =>
  (options.value?.baseModels ?? []).map((model) => ({
    label: model.source === "local" ? `models\\${model.label}` : model.label,
    value: model.path,
  })),
);
const deviceOptions = computed(() => [
  {
    label: "GPU（CUDA）",
    value: "cuda",
    disable: options.value?.cudaAvailable !== true,
  },
  { label: "CPU", value: "cpu" },
]);
const scheduleOptions = [
  { label: "一定", value: "constant" },
  { label: "cosine（最後に向けて下げる）", value: "cosine" },
];
const keepOptions = [
  { label: "最後のステップ", value: "last" },
  { label: "検証 loss が一番低いとき", value: "best" },
];
const compareOptions = computed(() =>
  (options.value?.runs ?? [])
    .filter((run) => run.hasHistory)
    .map((run) => ({
      label: `${run.name}（${run.created ?? "?"}）`,
      value: run.name,
    })),
);
const timeHint = computed(() =>
  device.value === "cuda"
    ? "GPU: 500 ステップで数分（VRAM は約 2GB）。書き起こしと音声の符号化に別に数分かかります。"
    : "CPU: 短いセリフ中心なら 500 ステップで 10〜15 分、長い音声が多いと 30 分以上かかります。",
);

// ───────── 状態の表示
const stateLabel = computed(
  () =>
    ({
      idle: "",
      running: status.value?.stopRequested ? "保存して終えます" : "学習中",
      done: "完了",
      error: "失敗",
      cancelled: "中止",
    })[status.value?.state ?? "idle"],
);
const stateColor = computed(() =>
  status.value?.state === "error"
    ? "negative"
    : status.value?.state === "cancelled"
      ? "grey-6"
      : "primary",
);
const stateTextColor = computed(() =>
  stateColor.value === "primary" ? "display-on-primary" : "white",
);
const stageIndex = computed(() =>
  STAGES.findIndex((stage) =>
    (stage.match as readonly string[]).includes(status.value?.stage ?? ""),
  ),
);
function stageState(i: number): "done" | "active" | "error" | "pending" {
  const state = status.value?.state;
  if (state === "done") return "done";
  if (state == undefined || state === "idle") return "pending";
  if (i < stageIndex.value) return "done";
  if (i === stageIndex.value)
    return state === "running"
      ? "active"
      : state === "error"
        ? "error"
        : "pending";
  return "pending";
}
function stageIcon(i: number): string {
  return {
    done: "check_circle",
    active: "radio_button_checked",
    error: "error",
    pending: "radio_button_unchecked",
  }[stageState(i)];
}
const elapsedLabel = computed(() => {
  const value = status.value;
  if (!value?.startedAt) return "";
  const end = value.finishedAt ?? now.value;
  const parts = [`経過 ${formatDuration(end - value.startedAt)}`];
  if (
    value.state === "running" &&
    value.stage === "train" &&
    value.detail.eta_sec != undefined
  )
    parts.push(`残り約 ${formatDuration(value.detail.eta_sec)}`);
  return parts.join(" · ");
});

// ───────── 記録
const trainPts = computed<TrainPoint[]>(
  () => status.value?.history?.train ?? [],
);
const valPts = computed<ValPoint[]>(() => status.value?.history?.val ?? []);
const cmpTrain = computed<TrainPoint[]>(
  () => compareRecord.value?.history?.train ?? [],
);
const cmpVal = computed<ValPoint[]>(
  () => compareRecord.value?.history?.val ?? [],
);
const totalSteps = computed(
  () => status.value?.config?.steps ?? status.value?.params?.steps ?? 0,
);
const xMax = computed(() =>
  Math.max(
    trainPts.value.length > 0 || running.value ? totalSteps.value : 0,
    compareRecord.value?.steps ?? 0,
  ),
);

/** TensorBoard と同じ、立ち上がりを補正した指数移動平均。 */
function smooth(values: number[], weight: number): number[] {
  if (weight <= 0) return values;
  let last = 0;
  return values.map((value, i) => {
    last = last * weight + (1 - weight) * value;
    return last / (1 - weight ** (i + 1));
  });
}
function line<T extends { step: number }>(
  points: T[],
  pick: (point: T) => number,
  weight = 0,
): { x: number; y: number }[] {
  const ys = smooth(points.map(pick), weight);
  return points.map((point, i) => ({ x: point.step, y: ys[i] }));
}
const nonEmpty = (series: ChartSeries[]) =>
  series.filter((s) => s.points.length > 0);
const compareLabel = computed(
  () => `比べる「${compareRecord.value?.name ?? ""}」`,
);

const lossSeries = computed(() =>
  nonEmpty([
    ...(smoothing.value > 0
      ? [
          {
            key: "train-raw",
            label: "学習（平滑化前）",
            color: "var(--chart-1)",
            points: line(trainPts.value, (p) => p.loss),
            variant: "faint" as const,
          },
        ]
      : []),
    {
      key: "train",
      label: smoothing.value > 0 ? "学習（平滑化）" : "学習",
      color: "var(--chart-1)",
      points: line(trainPts.value, (p) => p.loss, smoothing.value),
    },
    {
      key: "val",
      label: "検証",
      color: "var(--chart-2)",
      points: line(valPts.value, (p) => p.loss),
      variant: "dots",
    },
    {
      key: "cmp-train",
      label: `${compareLabel.value} 学習`,
      color: "var(--chart-compare)",
      points: line(cmpTrain.value, (p) => p.loss, smoothing.value),
      variant: "dashed",
    },
    {
      key: "cmp-val",
      label: `${compareLabel.value} 検証`,
      color: "var(--chart-compare)",
      points: line(cmpVal.value, (p) => p.loss),
      variant: "dashed-dots",
    },
  ]),
);
const durSeries = computed(() =>
  nonEmpty([
    {
      key: "train",
      label: "学習",
      color: "var(--chart-1)",
      points: line(trainPts.value, (p) => p.dur_mae_sec, smoothing.value),
    },
    {
      key: "val",
      label: "検証",
      color: "var(--chart-2)",
      points: line(valPts.value, (p) => p.dur_mae_sec),
      variant: "dots",
    },
    {
      key: "cmp",
      label: `${compareLabel.value} ${cmpVal.value.length > 0 ? "検証" : "学習"}`,
      color: "var(--chart-compare)",
      points:
        cmpVal.value.length > 0
          ? line(cmpVal.value, (p) => p.dur_mae_sec)
          : line(cmpTrain.value, (p) => p.dur_mae_sec, smoothing.value),
      variant: cmpVal.value.length > 0 ? "dashed-dots" : "dashed",
    },
  ]),
);
function trainOnly(pick: (point: TrainPoint) => number, weight = 0) {
  return nonEmpty([
    {
      key: "train",
      label: "学習",
      color: "var(--chart-1)",
      points: line(trainPts.value, pick, weight),
    },
    {
      key: "cmp",
      label: compareLabel.value,
      color: "var(--chart-compare)",
      points: line(cmpTrain.value, pick, weight),
      variant: "dashed",
    },
  ]);
}
const lrSeries = computed(() => trainOnly((p) => p.lr));
const gradSeries = computed(() => trainOnly((p) => p.grad, smoothing.value));
const driftSeries = computed(() => trainOnly((p) => p.drift * 100));

// ───────── 数のタイル
const smoothedLoss = computed(() => {
  const values = smooth(
    trainPts.value.map((p) => p.loss),
    smoothing.value,
  );
  return values.at(-1);
});
const bestVal = computed(() =>
  valPts.value.reduce<ValPoint | undefined>(
    (best, p) => (best == undefined || p.loss < best.loss ? p : best),
    undefined,
  ),
);
const tiles = computed(() => {
  const last = trainPts.value.at(-1);
  const lastVal = valPts.value.at(-1);
  const firstVal = valPts.value[0];
  const config = status.value?.config;
  const out: { label: string; value: string; unit?: string; sub?: string }[] =
    [];
  if (!last && !lastVal) return out;
  out.push({
    label: "ステップ",
    value: `${status.value?.detail.step ?? last?.step ?? 0}`,
    unit: ` / ${totalSteps.value}`,
    sub:
      status.value?.result?.metrics.stopped_at != undefined
        ? "途中で終えて保存"
        : undefined,
  });
  if (smoothedLoss.value != undefined)
    out.push({
      label: "学習 loss",
      value: formatNumber(smoothedLoss.value),
      sub:
        smoothing.value > 0
          ? `平滑化 ${smoothing.value.toFixed(2)}`
          : "平滑化なし",
    });
  if (lastVal)
    out.push({
      label: "検証 loss",
      value: formatNumber(lastVal.loss),
      sub:
        firstVal && lastVal !== firstVal
          ? `ゼロショット比 ${percentChange(firstVal.loss, lastVal.loss)}`
          : "学習前（ゼロショット）",
    });
  if (bestVal.value && valPts.value.length > 1)
    out.push({
      label: "検証 loss の最小",
      value: formatNumber(bestVal.value.loss),
      sub: `ステップ ${bestVal.value.step}`,
    });
  const dur = lastVal?.dur_mae_sec ?? last?.dur_mae_sec;
  if (dur != undefined)
    out.push({
      label: "時間予測の誤差",
      value: formatNumber(dur),
      unit: " 秒",
      sub: lastVal ? "検証" : "学習",
    });
  if (last) {
    out.push({
      label: "学習率",
      value: formatNumber(last.lr),
      sub:
        config?.lr_schedule === "cosine"
          ? "cosine"
          : config?.warmup
            ? `ウォームアップ ${config.warmup}`
            : "一定",
    });
    out.push({ label: "勾配ノルム", value: formatNumber(last.grad) });
    out.push({
      label: "初期値からの変化",
      value: formatNumber(last.drift * 100),
      unit: " %",
    });
    out.push({
      label: "速さ",
      value: formatNumber(last.steps_per_sec),
      unit: " step/秒",
      sub: config ? `バッチ ${config.batch}` : undefined,
    });
  }
  const vram =
    status.value?.result?.metrics.peak_vram_gb ?? status.value?.detail.vram_gb;
  if (vram != undefined)
    out.push({ label: "VRAM（最大）", value: `${vram}`, unit: " GB" });
  return out;
});

// ───────── 結果・表・実際の設定
const resultMetrics = computed(
  () =>
    status.value?.result?.metrics ?? {
      actual_over_predicted: NaN,
      pace_vs_base: NaN,
      train_seconds: 0,
    },
);
const paceLabel = computed(() => {
  const pace = resultMetrics.value.pace_vs_base;
  return Number.isFinite(pace) ? `${pace.toFixed(2)} 倍の長さ` : "-";
});
const savedLabel = computed(() => {
  const m = resultMetrics.value;
  if (m.saved_step == undefined) return "";
  const why =
    status.value?.config?.keep === "best" && m.best_step === m.saved_step
      ? "検証 loss が一番低いとき"
      : m.stopped_at != undefined
        ? "途中で終えたとき"
        : "最後";
  return `ステップ ${m.saved_step} の話者（${why}）`;
});
const valRows = computed(() => {
  const smoothed = line(trainPts.value, (p) => p.loss, smoothing.value);
  const first = valPts.value[0];
  return valPts.value.map((p) => {
    const nearestTrain = smoothed.reduce<{ x: number; y: number } | undefined>(
      (best, q) =>
        best == undefined || Math.abs(q.x - p.step) < Math.abs(best.x - p.step)
          ? q
          : best,
      undefined,
    );
    const lr = trainPts.value.find((q) => q.step >= p.step)?.lr;
    return {
      step: p.step,
      loss: p.loss,
      dur: p.dur_mae_sec,
      change: first && p !== first ? percentChange(first.loss, p.loss) : "—",
      train:
        p.step === 0 || nearestTrain == undefined
          ? "—"
          : formatNumber(nearestTrain.y),
      lr: p.step === 0 || lr == undefined ? "—" : formatNumber(lr),
      isBest: bestVal.value?.step === p.step,
    };
  });
});
const configLines = computed(() => {
  const c = status.value?.config;
  if (!c) return [];
  return [
    `バッチ ${c.batch} · ${c.dtype} · 勾配チェックポイント ${c.grad_ckpt > 0 ? "あり" : "なし"}`,
    `学習に使う音声 ${c.train_clips} 本 · 検証 ${c.val_clips} 本${c.val_clips > 0 ? `（${c.val_points} 点、${c.val_every} ステップごと）` : "（検証なし）"}`,
    `学習率 ${formatNumber(c.lr)} · ${c.lr_schedule === "cosine" ? "cosine" : "一定"}${c.warmup > 0 ? ` · ウォームアップ ${c.warmup}` : ""}`,
    `時間予測の重み ${formatNumber(c.dur_weight)} · 保存 ${c.keep === "best" ? "検証 loss 最小" : "最後"} · シード ${c.seed}`,
    `学習するパラメーター ${c.params.toLocaleString()} 個（16 トークン × 768） · 記録は ${c.report_every} ステップごとの平均`,
  ];
});
const hasDetails = computed(
  () =>
    valRows.value.length > 0 ||
    configLines.value.length > 0 ||
    (status.value?.log.length ?? 0) > 0,
);

// ───────── 書式
function percentChange(from: number, to: number | undefined): string {
  if (to == undefined || !Number.isFinite(from) || from === 0) return "-";
  const value = ((to - from) / from) * 100;
  return `${value > 0 ? "+" : value < 0 ? "−" : "±"}${Math.abs(value).toFixed(1)}%`;
}
function formatDuration(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s} 秒`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m} 分 ${s % 60} 秒`;
  return `${Math.floor(m / 60)} 時間 ${m % 60} 分`;
}
function runCaption(run: TrainRun): string {
  const parts = [run.created ?? ""];
  if (run.steps != undefined) parts.push(`${run.steps} ステップ`);
  if (run.lr != undefined) parts.push(`学習率 ${formatNumber(run.lr)}`);
  if (run.metrics.val_loss_saved != undefined)
    parts.push(`検証 loss ${formatNumber(run.metrics.val_loss_saved)}`);
  return parts.filter(Boolean).join(" · ");
}
function recordSummary(record: TrainRecord): string {
  const s = record.settings;
  const parts = [
    `${s?.steps ?? record.steps} ステップ`,
    `学習率 ${formatNumber(s?.lr ?? record.lr)}`,
  ];
  if (s) {
    parts.push(s.lr_schedule === "cosine" ? "cosine" : "一定");
    if (s.warmup > 0) parts.push(`ウォームアップ ${s.warmup}`);
    if (s.batch > 0) parts.push(`バッチ ${s.batch}`);
  }
  if (record.metrics.val_loss_saved != undefined)
    parts.push(`検証 loss ${formatNumber(record.metrics.val_loss_saved)}`);
  return parts.join(" · ");
}
function message(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause);
}

// ───────── 操作
/** エディタで選んでいるモデルと同じものを、ベースの初期値にする。 */
function pickBaseModel(value: Options) {
  if (value.baseModels.some((model) => model.path === baseModel.value)) return;
  const current = (props.currentModel ?? "").replace(/\\/g, "/");
  const match = value.baseModels.find(
    (model) =>
      current.length > 0 &&
      (model.label === current ||
        model.path.replace(/\\/g, "/").endsWith(current)),
  );
  baseModel.value = (match ?? value.baseModels[0])?.path ?? "";
}

async function loadOptions() {
  loading.value = true;
  error.value = "";
  try {
    const raw: Partial<Options> = await fetchSpeakerTrainOptions(
      props.endpoint,
    );
    // 古いエンジン（学習の設定・過去の学習を返さない）でも画面が壊れないようにそろえる
    const value: Options = {
      ...(raw as Options),
      limits: raw.limits ?? FALLBACK_LIMITS,
      runs: raw.runs ?? [],
      valMinTotal: raw.valMinTotal ?? 24,
    };
    options.value = value;
    pickBaseModel(value);
    if (!value.cudaAvailable) device.value = "cpu";
    if (!value.haasAvailable) verifyText.value = false;
    if (!formInitialized) {
      form.value = toForm(defaults.value);
      formInitialized = true;
    }
  } catch (cause) {
    error.value = message(cause);
  } finally {
    loading.value = false;
  }
}

function resetSettings() {
  form.value = toForm(defaults.value);
  learnSpeed.value = true;
  notice.value = "学習の設定を既定に戻しました。";
}

/** 過去の学習（.train.json）の設定を入力欄に入れる。古い記録は分かる項目だけ。 */
function applyRecord(record: TrainRecord) {
  const s = record.settings;
  if (s) {
    form.value = toForm({
      steps: s.steps,
      lr: s.lr,
      lrSchedule: s.lr_schedule,
      warmup: s.warmup,
      batch: s.batch,
      durWeight: s.dur_weight,
      valFrac: s.val_frac,
      valEvery: s.val_every,
      keep: s.keep,
      seed: s.seed,
    });
    learnSpeed.value = s.speed === "learn";
  } else {
    form.value = toForm({
      ...defaults.value,
      steps: record.steps,
      lr: record.lr,
      durWeight: record.dur_weight || defaults.value.durWeight,
    });
    learnSpeed.value = record.speed !== "keep";
  }
  formInitialized = true;
  notice.value = `「${record.name}」の設定を読み込みました${s ? "" : "（古い記録なので、ステップ数・学習率・話速だけ）"}。`;
}

async function loadSettingsFrom(runName: string) {
  error.value = "";
  try {
    applyRecord(await fetchSpeakerTrainRecord(props.endpoint, runName));
  } catch (cause) {
    error.value = message(cause);
  }
}

/** 開き直したとき、動いている（終わった）学習の設定を入力欄に戻す。 */
function restoreFromParams(params: TrainRequest) {
  name.value = params.name;
  dataDir.value = params.dataDir;
  if (params.baseModel) baseModel.value = params.baseModel;
  learnSpeed.value = params.speed === "learn";
  verifyText.value = params.verifyText;
  if (params.device !== "auto") device.value = params.device;
  if (params.lr != undefined) {
    form.value = toForm({ ...defaults.value, ...pickSettings(params) });
    formInitialized = true;
  }
}
function pickSettings(params: TrainRequest): Settings {
  return {
    steps: params.steps,
    lr: params.lr,
    lrSchedule: params.lrSchedule,
    warmup: params.warmup,
    batch: params.batch,
    durWeight: params.durWeight,
    valFrac: params.valFrac,
    valEvery: params.valEvery,
    keep: params.keep,
    seed: params.seed,
  };
}

let wasRunning = false;
let firstPoll = true;
async function poll() {
  now.value = Date.now() / 1000;
  try {
    const value = await fetchSpeakerTrainStatus(props.endpoint);
    status.value = value;
    if (firstPoll && value.params && value.state !== "idle")
      restoreFromParams(value.params);
    firstPoll = false;
    if (wasRunning && value.state === "done") {
      // 新しい話者を一覧に出す（エンジンは学習の終わりに一覧を作り直している）
      try {
        await refreshIrodoriSpeakers(props.endpoint);
      } finally {
        await store.actions.LOAD_CHARACTER({ engineId: props.engineId });
      }
      void loadOptions();
    }
    wasRunning = value.state === "running";
  } catch {
    // エンジンの再起動中など。次の問い合わせで戻る
  }
}

async function start() {
  starting.value = true;
  error.value = "";
  notice.value = "";
  try {
    status.value = await startSpeakerTrain(props.endpoint, {
      name: name.value.trim(),
      dataDir: dataDir.value.trim().replace(/^"|"$/g, ""),
      baseModel: baseModel.value,
      speed: learnSpeed.value ? "learn" : "keep",
      device: device.value,
      verifyText: verifyText.value,
      overwrite: overwrite.value,
      ...toSettings(form.value),
    });
    wasRunning = true;
    hoverStep.value = null;
  } catch (cause) {
    error.value = message(cause);
  } finally {
    starting.value = false;
  }
}

/** エンジンの PC でフォルダを選ぶ画面を開く（ブラウザからは PC のパスが取れないため）。 */
async function pickFolder() {
  picking.value = true;
  error.value = "";
  try {
    const picked = await pickIrodoriFolder(
      props.endpoint,
      dataDir.value.trim(),
    );
    if (picked != undefined) dataDir.value = picked;
  } catch (cause) {
    error.value = message(cause);
  } finally {
    picking.value = false;
  }
}

async function stopEarly() {
  try {
    status.value = await stopSpeakerTrain(props.endpoint);
  } catch (cause) {
    error.value = message(cause);
  }
}

async function cancel() {
  try {
    status.value = await cancelSpeakerTrain(props.endpoint);
  } catch (cause) {
    error.value = message(cause);
  }
}

watch(compareName, async (value) => {
  compareError.value = "";
  compareRecord.value = undefined;
  if (!value) return;
  try {
    const record = await fetchSpeakerTrainRecord(props.endpoint, value);
    if (compareName.value === value) compareRecord.value = record;
  } catch (cause) {
    compareError.value = message(cause);
  }
});
watch(learnSpeed, () => (notice.value = ""));

let timer: ReturnType<typeof setInterval> | undefined;
onMounted(() => {
  void loadOptions();
  void poll();
  timer = setInterval(() => void poll(), 1500);
});
onUnmounted(() => {
  if (timer) clearInterval(timer);
});
watch(
  () => props.endpoint,
  () => {
    firstPoll = true;
    void loadOptions();
    void poll();
  },
);
</script>

<style scoped lang="scss">
@use "@/styles/v2/variables" as vars;

.speaker-train {
  display: flex;
  flex-direction: column;
  gap: vars.$gap-2;
}

.intro {
  font-size: 12px;
  line-height: 1.6;
  color: var(--color-v2-display-sub);
  .text-warning {
    color: var(--color-v2-display-warning) !important;
  }
}

.layout {
  display: grid;
  grid-template-columns: minmax(340px, 400px) minmax(0, 1fr);
  gap: vars.$gap-2;
  align-items: start;
}
@media (max-width: 1023px) {
  .layout {
    grid-template-columns: minmax(0, 1fr);
  }
}
.settings-col,
.monitor-col {
  display: flex;
  flex-direction: column;
  gap: vars.$gap-2;
  min-width: 0;
}

.panel {
  background: var(--color-v2-surface);
  border: 1px solid var(--color-v2-border);
  border-radius: vars.$radius-2;
  padding: vars.$padding-2;
}
.panel-title-row {
  display: flex;
  align-items: center;
  gap: 4px;
  margin-bottom: vars.$gap-1;
  .panel-title {
    margin-bottom: 0;
  }
}
.panel-title {
  display: flex;
  align-items: center;
  gap: 6px;
  margin: 0 0 vars.$gap-2;
  font-size: 15px;
  font-weight: 600;
  line-height: 1.4;
  color: var(--color-v2-display);
  .q-icon {
    font-size: 18px;
    color: var(--color-v2-display-sub);
  }
}
.sub-title {
  margin: 0 0 4px;
  font-size: 13px;
  font-weight: 600;
  line-height: 1.4;
  color: var(--color-v2-display);
}
.field-label {
  font-size: 12px;
  color: var(--color-v2-display-sub);
  margin-bottom: 4px;
}
.field-note {
  color: var(--color-v2-display-sub);
}
.fields {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 4px 12px;
}
.help :deep(.q-item) {
  padding-left: 0;
  font-size: 12px;
  color: var(--color-v2-display-sub);
}
.help-list {
  margin: 0;
  font-size: 12px;
  line-height: 1.6;
  dt {
    font-weight: 600;
    color: var(--color-v2-display);
  }
  dd {
    margin: 0 0 6px;
    color: var(--color-v2-display-sub);
  }
}
.device-toggle {
  border: 1px solid var(--color-v2-border);
}
.start-btn {
  min-height: 44px;
  font-weight: 600;
}

// 段階
.stages {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 16px;
  margin: 4px 0 0;
  padding: 0;
  list-style: none;
  font-size: 12px;
  li {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    color: var(--color-v2-display-sub);
  }
  .stage-done .q-icon {
    color: var(--color-v2-display-sub);
  }
  .stage-active {
    color: var(--color-v2-display);
    font-weight: 600;
    .q-icon {
      color: var(--color-v2-primary);
      animation: pulse 1.2s ease-in-out infinite;
    }
  }
  .stage-error {
    color: var(--color-v2-display-warning);
  }
}
@keyframes pulse {
  50% {
    opacity: 0.35;
  }
}
@media (prefers-reduced-motion: reduce) {
  .stages .stage-active .q-icon {
    animation: none;
  }
}
.status-line {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: 4px 12px;
  overflow-wrap: anywhere;
}

// 数のタイル
.tiles {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(128px, 1fr));
  gap: vars.$gap-1;
  margin-top: vars.$gap-2;
}
.tile {
  padding: 8px 10px;
  border-radius: vars.$radius-1;
  background: var(--color-v2-background-alt);
  min-width: 0;
}
.tile-label {
  font-size: 11px;
  color: var(--color-v2-display-sub);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.tile-value {
  font-size: 18px;
  font-weight: 600;
  line-height: 1.3;
  color: var(--color-v2-display);
}
.tile-unit {
  font-size: 12px;
  font-weight: 400;
  color: var(--color-v2-display-sub);
}
.tile-sub {
  font-size: 11px;
  color: var(--color-v2-display-sub);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

// グラフ
.chart-toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px 16px;
  margin-bottom: vars.$gap-1;
}
.toolbar-item {
  font-size: 12px;
}
.smoothing {
  display: flex;
  align-items: center;
  gap: 8px;
}
.smoothing-slider {
  width: 120px;
}
.smoothing-value {
  width: 3em;
  font-variant-numeric: tabular-nums;
  font-size: 12px;
}
.compare-select {
  min-width: 220px;
}
.compare-note {
  overflow-wrap: anywhere;
}
.small-charts {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: vars.$gap-2;
  margin-top: vars.$gap-2;
}
@media (max-width: 700px) {
  .small-charts,
  .fields {
    grid-template-columns: minmax(0, 1fr);
  }
}

// 結果
.result-list {
  display: grid;
  grid-template-columns: max-content minmax(0, 1fr);
  gap: 4px 16px;
  margin: 0;
  font-size: 13px;
  dt {
    color: var(--color-v2-display-sub);
  }
  dd {
    margin: 0;
  }
  .path {
    overflow-wrap: anywhere;
    font-size: 12px;
  }
}
.details {
  padding: vars.$padding-1;
}
.val-table {
  background: transparent;
  font-variant-numeric: tabular-nums;
  tr.best td {
    font-weight: 600;
  }
}
.config-list {
  margin: 0;
  padding: 0 16px 8px 32px;
  font-size: 12px;
  line-height: 1.7;
}
.log {
  max-height: 16em;
  overflow: auto;
  font-size: 11px;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  margin: 0;
  padding: 0 16px 8px;
}
</style>
