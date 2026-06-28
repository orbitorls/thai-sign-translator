/**
 * API client for the Thai Sign Language Translator backend.
 * All requests target relative URLs so they work via the Vite dev proxy
 * and directly when served by FastAPI in production.
 */

import {
  ConsentScope,
  CONSENT_VERSION,
  feedbackSessionId,
  getOrCreateUserId,
  hasConsentScope,
} from "../privacy/consentStorage";

const API_BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");
const MOCK_MODE_HEADER = { "X-Mock-Mode": "true" } as const;

function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

export type ConsentSource = "consent_modal" | "settings_toggle" | "api" | "withdrawal";

export interface ModelInfo {
  id: string;
  label_th: string;
  label_en: string;
  architecture: string;
  available: boolean;
  default: boolean;
}

export interface ModelsResponse {
  models: ModelInfo[];
  default: string;
}

export interface TranslateResult {
  sentence: string;
  score: number;
  model: string;
  latency_ms?: number;
  warning?: string | null;
  token_score?: number | null;
  landmark_quality?: number | null;
}

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly detail: string
  ) {
    super(`API error ${status}: ${detail}`);
    this.name = "ApiError";
  }
}

async function _handleResponse<T>(resp: Response): Promise<T> {
  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`;
    try {
      const body = await resp.json();
      detail = body.detail ?? detail;
    } catch {
      // ignore parse error
    }
    throw new ApiError(resp.status, detail);
  }
  return resp.json() as Promise<T>;
}

export async function getModels(): Promise<ModelsResponse> {
  const resp = await fetch(apiUrl("/models"), {
    headers: MOCK_MODE_HEADER,
  });
  return _handleResponse<ModelsResponse>(resp);
}

export interface TranslateParams {
  frames: number[][][];
  model?: string;
  featureSchema?: string;
  maxLen?: number;
  signal?: AbortSignal;
}

export interface SupportedPhrasesResult {
  phrases: string[];
  total: number;
  note: string;
}

export async function getSupportedPhrases(): Promise<SupportedPhrasesResult> {
  const resp = await fetch(apiUrl("/supported-phrases"));
  return _handleResponse<SupportedPhrasesResult>(resp);
}

export async function translate(params: TranslateParams): Promise<TranslateResult> {
  const { frames, model, featureSchema = "raw_mediapipe_543x3", maxLen = 128, signal } = params;
  const resp = await fetch(apiUrl("/translate"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...MOCK_MODE_HEADER,
      ...privacyHeaders(),
    },
    signal,
    body: JSON.stringify({
      frames,
      feature_schema: featureSchema,
      model: model ?? null,
      max_len: maxLen,
    }),
  });
  return _handleResponse<TranslateResult>(resp);
}

interface PendingRealtimeRequest {
  resolve: (value: TranslateResult) => void;
  reject: (reason?: unknown) => void;
  timer: ReturnType<typeof window.setTimeout>;
}

let realtimeSocket: WebSocket | null = null;
let realtimeConnecting: Promise<WebSocket> | null = null;
const pendingRealtime = new Map<string, PendingRealtimeRequest>();

function realtimeUrl(): string {
  if (API_BASE) {
    const url = new URL(API_BASE);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    url.pathname = "/ws/translate";
    url.search = "";
    url.hash = "";
    return url.toString();
  }
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/ws/translate`;
}

function rejectPending(reason: unknown): void {
  for (const pending of pendingRealtime.values()) {
    window.clearTimeout(pending.timer);
    pending.reject(reason);
  }
  pendingRealtime.clear();
}

function getRealtimeSocket(): Promise<WebSocket> {
  if (realtimeSocket?.readyState === WebSocket.OPEN) {
    return Promise.resolve(realtimeSocket);
  }
  if (realtimeConnecting) return realtimeConnecting;

  realtimeConnecting = new Promise((resolve, reject) => {
    const ws = new WebSocket(realtimeUrl());
    const timer = window.setTimeout(() => {
      ws.close();
      reject(new Error("realtime WebSocket connection timed out"));
    }, 3000);

    ws.onopen = () => {
      window.clearTimeout(timer);
      realtimeSocket = ws;
      realtimeConnecting = null;
      resolve(ws);
    };
    ws.onerror = () => {
      window.clearTimeout(timer);
      realtimeConnecting = null;
      reject(new Error("realtime WebSocket connection failed"));
    };
    ws.onclose = () => {
      if (realtimeSocket === ws) realtimeSocket = null;
      realtimeConnecting = null;
      rejectPending(new Error("realtime WebSocket closed"));
    };
    ws.onmessage = (event) => {
      let body: any;
      try {
        body = JSON.parse(event.data);
      } catch {
        return;
      }
      const requestId = String(body.request_id ?? "");
      const pending = pendingRealtime.get(requestId);
      if (!pending) return;
      window.clearTimeout(pending.timer);
      pendingRealtime.delete(requestId);
      if (body.type === "error") {
        pending.reject(new ApiError(Number(body.code ?? 500), String(body.detail ?? "Realtime error")));
      } else {
        pending.resolve({
          sentence: String(body.sentence ?? ""),
          score: Number(body.score ?? 0),
          model: String(body.model ?? ""),
          latency_ms: typeof body.latency_ms === "number" ? body.latency_ms : undefined,
          warning: body.warning ?? null,
          token_score: typeof body.token_score === "number" ? body.token_score : null,
          landmark_quality:
            typeof body.landmark_quality === "number" ? body.landmark_quality : null,
        });
      }
    };
  });

  return realtimeConnecting;
}

export function closeRealtimeSocket(): void {
  const ws = realtimeSocket;
  realtimeSocket = null;
  realtimeConnecting = null;
  rejectPending(new DOMException("Realtime request aborted", "AbortError"));
  ws?.close();
}

export async function translateRealtime(params: TranslateParams): Promise<TranslateResult> {
  const { frames, model, featureSchema = "raw_mediapipe_543x3", maxLen = 128, signal } = params;
  if (signal?.aborted) {
    throw new DOMException("Realtime request aborted", "AbortError");
  }

  const ws = await getRealtimeSocket();
  if (signal?.aborted) {
    throw new DOMException("Realtime request aborted", "AbortError");
  }
  const requestId = `${Date.now()}-${Math.random().toString(36).slice(2)}`;

  return new Promise<TranslateResult>((resolve, reject) => {
    const timer = window.setTimeout(() => {
      pendingRealtime.delete(requestId);
      reject(new Error("realtime WebSocket request timed out"));
    }, 15000);

    const abort = () => {
      window.clearTimeout(timer);
      pendingRealtime.delete(requestId);
      reject(new DOMException("Realtime request aborted", "AbortError"));
    };
    signal?.addEventListener("abort", abort, { once: true });

    pendingRealtime.set(requestId, {
      resolve: (value) => {
        signal?.removeEventListener("abort", abort);
        resolve(value);
      },
      reject: (reason) => {
        signal?.removeEventListener("abort", abort);
        reject(reason);
      },
      timer,
    });

    try {
      ws.send(JSON.stringify({
        request_id: requestId,
        frames,
        feature_schema: featureSchema,
        model: model ?? null,
        max_len: maxLen,
        mock_mode: true,
        user_id: getOrCreateUserId(),
        service_consent: hasConsentScope("service"),
      }));
    } catch (e) {
      window.clearTimeout(timer);
      pendingRealtime.delete(requestId);
      signal?.removeEventListener("abort", abort);
      reject(e);
    }
  });
}

function privacyHeaders(): Record<string, string> {
  const headers: Record<string, string> = {
    "X-User-Id": getOrCreateUserId(),
    "X-Session-Id": feedbackSessionId(),
  };
  if (hasConsentScope("service")) {
    headers["X-Service-Consent"] = "true";
  }
  return headers;
}

function feedbackHeaders(): HeadersInit {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...privacyHeaders(),
  };
  if (hasConsentScope("model_improvement")) {
    headers["X-Feedback-Consent"] = "true";
  }
  return headers;
}

function videoHeaders(): HeadersInit {
  const headers: Record<string, string> = {
    ...privacyHeaders(),
  };
  if (hasConsentScope("video_research")) {
    headers["X-Video-Consent"] = "true";
  }
  return headers;
}

export interface ConsentStatusResponse {
  consent_version: string;
  scopes: Record<string, boolean>;
}

export async function syncConsentScope(params: {
  scope: ConsentScope;
  granted: boolean;
  source?: ConsentSource;
}): Promise<void> {
  const resp = await fetch(apiUrl("/privacy/consent"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-User-Id": getOrCreateUserId(),
    },
    body: JSON.stringify({
      scope: params.scope,
      granted: params.granted,
      consent_version: CONSENT_VERSION,
      source: params.source ?? "api",
    }),
  });
  await _handleResponse(resp);
}

export async function getConsentStatus(): Promise<ConsentStatusResponse> {
  const resp = await fetch(apiUrl("/privacy/consent/status"), {
    headers: { "X-User-Id": getOrCreateUserId() },
  });
  return _handleResponse<ConsentStatusResponse>(resp);
}

export async function deleteUserData(): Promise<number> {
  const resp = await fetch(apiUrl("/privacy/delete-data"), {
    method: "POST",
    headers: { "X-User-Id": getOrCreateUserId() },
  });
  const body = await _handleResponse<{ deleted_samples: number }>(resp);
  return body.deleted_samples;
}

export async function submitFeedbackVideo(segmentId: string, video: Blob): Promise<void> {
  const form = new FormData();
  form.append("segment_id", segmentId);
  form.append("video", video, `${segmentId}.webm`);
  const resp = await fetch(apiUrl("/feedback/video"), {
    method: "POST",
    headers: videoHeaders(),
    body: form,
  });
  await _handleResponse(resp);
}

export interface FeedbackStats {
  pending_count: number;
  total_count: number;
  last_retrain_at: string | null;
  last_attempt_at?: string | null;
  feedback_version: string | null;
  model: string | null;
}

type FeedbackStrings = {
  feedbackErrorDuplicate: string;
  feedbackErrorRateLimit: string;
  feedbackErrorValidation: string;
  feedbackErrorConsent: string;
  correctionError: string;
  teachError: string;
};

export function feedbackErrorMessage(status: number, t: FeedbackStrings, kind: "correction" | "teach"): string {
  switch (status) {
    case 409:
      return t.feedbackErrorDuplicate;
    case 429:
      return t.feedbackErrorRateLimit;
    case 400:
    case 422:
      return t.feedbackErrorValidation;
    case 403:
      return t.feedbackErrorConsent;
    default:
      return kind === "teach" ? t.teachError : t.correctionError;
  }
}

export interface FeedbackSubmissionResult {
  segment_id: string;
  kind: string;
  status: string;
  message: string;
}

export interface CaptureQualityPayload {
  fps?: number;
  lighting_ok?: boolean;
  hand_present?: boolean;
  warning?: string | null;
  landmark_quality?: number | null;
  feature_schema?: "raw_mediapipe_543x4" | "raw_mediapipe_543x3";
  camera_facing?: "user" | "environment";
}

export interface CorrectionParams {
  frames: number[][][];
  predictedText: string;
  correctedText: string;
  model?: string;
  score?: number;
  captureQuality?: CaptureQualityPayload | null;
}

export interface TeachParams {
  frames: number[][][];
  labelText: string;
  captureQuality?: CaptureQualityPayload | null;
}

export async function submitCorrection(params: CorrectionParams): Promise<FeedbackSubmissionResult> {
  const { frames, predictedText, correctedText, model, score, captureQuality } = params;
  const resp = await fetch(apiUrl("/feedback/correction"), {
    method: "POST",
    headers: feedbackHeaders(),
    body: JSON.stringify({
      frames,
      predicted_text: predictedText,
      corrected_text: correctedText,
      model: model ?? null,
      score: score ?? null,
      capture_quality: captureQuality ?? null,
    }),
  });
  return _handleResponse<FeedbackSubmissionResult>(resp);
}

export async function submitTeach(params: TeachParams): Promise<FeedbackSubmissionResult> {
  const { frames, labelText, captureQuality } = params;
  const resp = await fetch(apiUrl("/feedback/teach"), {
    method: "POST",
    headers: feedbackHeaders(),
    body: JSON.stringify({
      frames,
      label_text: labelText,
      capture_quality: captureQuality ?? null,
    }),
  });
  return _handleResponse<FeedbackSubmissionResult>(resp);
}

export async function getFeedbackStats(): Promise<FeedbackStats> {
  const resp = await fetch(apiUrl("/feedback/stats"));
  return _handleResponse<FeedbackStats>(resp);
}
