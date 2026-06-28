/**
 * useTranslate — wraps the POST /translate API call with a simple state machine.
 */
import { useState, useCallback } from "react";
import { translate, TranslateResult, ApiError } from "../api/client";

export type TranslateStatus = "idle" | "loading" | "success" | "error";
export type ErrorKind = "api" | "network" | null;

export interface TranslateState {
  status: TranslateStatus;
  result: TranslateResult | null;
  error: string | null;
  /** Error HTTP status code, if known (503 = model unavailable, 400 = bad input) */
  errorStatus: number | null;
  errorKind: ErrorKind;
  lastLatencyMs: number | null;
  run: (frames: number[][][], model?: string) => Promise<void>;
  reset: () => void;
}

export function useTranslate(): TranslateState {
  const [status, setStatus] = useState<TranslateStatus>("idle");
  const [result, setResult] = useState<TranslateResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [errorStatus, setErrorStatus] = useState<number | null>(null);
  const [errorKind, setErrorKind] = useState<ErrorKind>(null);
  const [lastLatencyMs, setLastLatencyMs] = useState<number | null>(null);

  const run = useCallback(async (frames: number[][][], model?: string) => {
    setStatus("loading");
    setError(null);
    setErrorStatus(null);
    setErrorKind(null);
    const t0 = Date.now();
    try {
      const res = await translate({ frames, model });
      setLastLatencyMs(Date.now() - t0);
      setResult(res);
      setStatus("success");
    } catch (e) {
      if (e instanceof ApiError) {
        setError(e.detail);
        setErrorStatus(e.status);
        setErrorKind("api");
      } else {
        // fetch rejects with TypeError on network failure
        setError("เชื่อมต่อไม่ได้");
        setErrorStatus(null);
        setErrorKind("network");
      }
      setStatus("error");
    }
  }, []);

  const reset = useCallback(() => {
    setStatus("idle");
    setResult(null);
    setError(null);
    setErrorStatus(null);
    setErrorKind(null);
  }, []);

  return { status, result, error, errorStatus, errorKind, lastLatencyMs, run, reset };
}
