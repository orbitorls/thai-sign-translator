/**
 * API client for the Thai Sign Language Translator backend.
 * All requests target relative URLs so they work via the Vite dev proxy
 * and directly when served by FastAPI in production.
 */

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
  const resp = await fetch("/models");
  return _handleResponse<ModelsResponse>(resp);
}

export interface TranslateParams {
  frames: number[][][];
  model?: string;
  featureSchema?: string;
  maxLen?: number;
}

export interface SupportedPhrasesResult {
  phrases: string[];
  total: number;
  note: string;
}

export async function getSupportedPhrases(): Promise<SupportedPhrasesResult> {
  const resp = await fetch("/supported-phrases");
  return _handleResponse<SupportedPhrasesResult>(resp);
}

export async function translate(params: TranslateParams): Promise<TranslateResult> {
  const { frames, model, featureSchema = "raw_mediapipe_543x3", maxLen = 128 } = params;
  const resp = await fetch("/translate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      frames,
      feature_schema: featureSchema,
      model: model ?? null,
      max_len: maxLen,
    }),
  });
  return _handleResponse<TranslateResult>(resp);
}
