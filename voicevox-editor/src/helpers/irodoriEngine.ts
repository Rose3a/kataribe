/**
 * Irodori エンジン（ローカルの HTTP サーバ）との通信をまとめる。
 *
 * Irodori 固有の口はすべてこのファイルを通す。エンジンは起動ごとに作られる
 * セッショントークンを要求するので、取得・保持・エンジン再起動時の取り直し・
 * タイムアウトをここに集めてある。
 */
import type {
  AsrTimelineResponse,
  IrodoriModelInfo,
  IrodoriSession,
  IrodoriSettings,
  IrodoriStatus,
  IrodoriStorageStatus,
  IrodoriTokenView,
  TokenSplitEntry,
} from "@/domain/irodori";

const SESSION_TIMEOUT_MS = 5000;
const DEFAULT_TIMEOUT_MS = 30000;
/** 口パクのタイムラインは、初回だけASRモデルの取得を待つことがある。 */
const TIMELINE_TIMEOUT_MS = 60000;

const sessionTokens = new Map<string, string>();

/** 保持しているトークンを忘れる（エンジンの再起動や 403 のとき）。 */
export function forgetIrodoriSession(endpoint?: string): void {
  if (endpoint == undefined) sessionTokens.clear();
  else sessionTokens.delete(endpoint);
}

/** セッショントークンを取る。force で取り直す。 */
export async function irodoriSessionToken(
  endpoint: string,
  force = false,
): Promise<string> {
  if (force) sessionTokens.delete(endpoint);
  const cached = sessionTokens.get(endpoint);
  if (cached != undefined) return cached;
  const response = await fetch(`${endpoint}/irodori/session`, {
    signal: AbortSignal.timeout(SESSION_TIMEOUT_MS),
  });
  if (!response.ok)
    throw new Error(`engine session unavailable (${response.status})`);
  const { token } = (await response.json()) as IrodoriSession;
  sessionTokens.set(endpoint, token);
  return token;
}

export type IrodoriRequestInit = {
  method?: "GET" | "POST";
  body?: unknown;
  timeoutMs?: number;
};

/**
 * Irodori 固有APIを叩く。
 *
 * トークンを付けて送り、403（エンジンが再起動してトークンが変わった場合）は
 * 新しいトークンで一度だけやり直す。
 */
export async function irodoriRequest(
  endpoint: string,
  path: string,
  init: IrodoriRequestInit = {},
): Promise<Response> {
  const {
    method = "GET",
    body,
    timeoutMs = DEFAULT_TIMEOUT_MS,
  } = init;
  const send = async (token: string) =>
    await fetch(`${endpoint}${path}`, {
      method,
      headers: {
        ...(body !== undefined ? { "Content-Type": "application/json" } : {}),
        "X-Irodori-Session": token,
      },
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal: AbortSignal.timeout(timeoutMs),
    });
  let response = await send(await irodoriSessionToken(endpoint));
  if (response.status === 403) {
    response = await send(await irodoriSessionToken(endpoint, true));
  }
  return response;
}

async function readStatus(response: Response): Promise<IrodoriStatus> {
  if (!response.ok) {
    throw new Error(
      `設定の通信に失敗しました (${response.status}): ${await response.text()}`,
    );
  }
  return (await response.json()) as IrodoriStatus;
}

/** /irodori/settings から現在の状態を取る。 */
export async function fetchIrodoriStatus(
  endpoint: string,
  timeoutMs = DEFAULT_TIMEOUT_MS,
): Promise<IrodoriStatus> {
  return await readStatus(
    await irodoriRequest(endpoint, "/irodori/settings", { timeoutMs }),
  );
}

/** /irodori/settings に設定を保存する。 */
export async function saveIrodoriSettings(
  endpoint: string,
  settings: IrodoriSettings,
  timeoutMs = DEFAULT_TIMEOUT_MS,
): Promise<IrodoriStatus> {
  return await readStatus(
    await irodoriRequest(endpoint, "/irodori/settings", {
      method: "POST",
      body: settings,
      timeoutMs,
    }),
  );
}

/** モデル／話者フォルダをエクスプローラで開く。 */
export async function openIrodoriFolder(
  endpoint: string,
  folder: "models" | "speakers",
): Promise<void> {
  const response = await irodoriRequest(endpoint, `/irodori/open-${folder}`, {
    method: "POST",
  });
  if (!response.ok) throw new Error("フォルダを開けませんでした");
}

async function readStorage(response: Response): Promise<IrodoriStorageStatus> {
  if (!response.ok) {
    let detail = await response.text();
    try {
      detail = (JSON.parse(detail) as { detail?: string }).detail ?? detail;
    } catch {
      // 本文が JSON でなければそのまま表示する。
    }
    throw new Error(detail);
  }
  return (await response.json()) as IrodoriStorageStatus;
}

/** ダウンロード済みモデルと TensorRT キャッシュの一覧を取る。 */
export async function fetchIrodoriStorage(
  endpoint: string,
): Promise<IrodoriStorageStatus> {
  return await readStorage(
    await irodoriRequest(endpoint, "/irodori/storage", { timeoutMs: 60000 }),
  );
}

/** 一覧の項目を削除する。削除後の一覧と空いた容量を返す。 */
export async function deleteIrodoriStorage(
  endpoint: string,
  ids: string[],
): Promise<IrodoriStorageStatus> {
  return await readStorage(
    await irodoriRequest(endpoint, "/irodori/storage/delete", {
      method: "POST",
      body: { ids },
      timeoutMs: 120000,
    }),
  );
}

/** 話者一覧の再スキャンは話者数が多いと数十秒かかる。 */
const REFRESH_TIMEOUT_MS = 120000;

/** 話者一覧を作り直す。 */
export async function refreshIrodoriSpeakers(
  endpoint: string,
  timeoutMs = REFRESH_TIMEOUT_MS,
): Promise<void> {
  const response = await irodoriRequest(endpoint, "/refresh", { timeoutMs });
  if (!response.ok) throw new Error("話者一覧の更新に失敗しました");
}

/**
 * セリフの文字ごとの発話時刻をエンジンから取る。
 *
 * 再生対象の音声そのものを送信し、別の生成結果との取り違えを防ぐ。
 * 口パクが付かないだけなので、失敗は null で返す。
 */
export async function fetchAsrTimeline(
  endpoint: string,
  text: string,
  audio: Blob,
  timeoutMs = TIMELINE_TIMEOUT_MS,
): Promise<AsrTimelineResponse | null> {
  try {
    const wav = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        if (typeof reader.result === "string")
          resolve(reader.result.split(",")[1]);
        else reject(new Error("音声を読み込めませんでした"));
      };
      reader.onerror = () =>
        reject(reader.error ?? new Error("音声を読み込めませんでした"));
      reader.readAsDataURL(audio);
    });
    const response = await irodoriRequest(endpoint, "/irodori/timeline", {
      method: "POST",
      body: { text, wav },
      timeoutMs,
    });
    if (!response.ok) return null;
    const result = (await response.json()) as AsrTimelineResponse;
    if (!result.available || !result.anchors || result.anchors.length === 0)
      return null;
    return result;
  } catch {
    return null;
  }
}

/** 文がモデルにどう分けて渡るか（辞書画面の表示用）。失敗したら undefined。 */
export async function fetchTokenView(
  endpoint: string,
  texts: string[],
): Promise<IrodoriTokenView | undefined> {
  try {
    const response = await irodoriRequest(endpoint, "/irodori/tokenize", {
      method: "POST",
      body: { texts },
      timeoutMs: SESSION_TIMEOUT_MS,
    });
    if (!response.ok) return undefined;
    return (await response.json()) as IrodoriTokenView;
  } catch {
    return undefined;
  }
}

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = await response.text();
    try {
      detail = (JSON.parse(detail) as { detail?: string }).detail ?? detail;
    } catch {
      // 本文が JSON でなければそのまま出す
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

/** 語彙分割辞書の一覧（自分の登録と自動の登録）。 */
export async function listTokenSplit(
  endpoint: string,
): Promise<{ user: TokenSplitEntry[]; auto: TokenSplitEntry[] }> {
  return await readJson(
    await irodoriRequest(endpoint, "/irodori/token_split/list", {
      method: "POST",
      body: {},
    }),
  );
}

/** 自分の登録に足す（同じ語句なら上書き）。 */
export async function putTokenSplit(
  endpoint: string,
  entry: { surface: string; text: string; note: string },
): Promise<TokenSplitEntry> {
  return await readJson(
    await irodoriRequest(endpoint, "/irodori/token_split/put", {
      method: "POST",
      body: entry,
    }),
  );
}

/** 自分の登録から消す。 */
export async function deleteTokenSplit(
  endpoint: string,
  surface: string,
): Promise<void> {
  await readJson(
    await irodoriRequest(endpoint, "/irodori/token_split/delete", {
      method: "POST",
      body: { surface },
    }),
  );
}

/** 行設定の既定値（ステップ数・MeanFlowか）を、現在選ばれているモデルに合わせる。 */
export async function fetchIrodoriModelInfo(
  endpoint: string,
): Promise<IrodoriModelInfo | undefined> {
  try {
    const result = await fetchIrodoriStatus(endpoint, SESSION_TIMEOUT_MS);
    return result.modelInfo;
  } catch {
    return undefined;
  }
}
