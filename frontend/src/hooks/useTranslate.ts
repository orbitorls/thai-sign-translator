/**
 * useTranslate — wraps the POST /translate API call with a simple state machine.
 */
import { useState, useCallback, useEffect, useRef } from "react";
import { translate, translateRealtime, closeRealtimeSocket, TranslateResult, ApiError } from "../api/client";
import { createMockTranslateResult, MOCKUP_MODE } from "../mockup";

export type TranslateStatus = "idle" | "loading" | "success" | "error";

export interface TranslateRunOptions {
  model?: string;
  featureSchema?: string;
}

export function inferFeatureSchema(frames: number[][][]): string {
  if (
    frames.length > 0 &&
    frames[0].length > 0 &&
    frames[0][0].length >= 4
  ) {
    return "raw_mediapipe_543x4";
  }
  return "raw_mediapipe_543x3";
}

export interface TranslateState {
  status: TranslateStatus;
  result: TranslateResult | null;
  error: string | null;
  /** Error HTTP status code, if known (503 = model unavailable, 400 = bad input) */
  errorStatus: number | null;
  pending: boolean;
  run: (frames: number[][][], options?: TranslateRunOptions) => Promise<void>;
  cancel: (options?: { keepResult?: boolean }) => void;
  reset: () => void;
}

export function useTranslate(): TranslateState {
  const [status, setStatus] = useState<TranslateStatus>("idle");
  const [result, setResult] = useState<TranslateResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [errorStatus, setErrorStatus] = useState<number | null>(null);
  const [pending, setPending] = useState(false);
  const requestIdRef = useRef(0);
  const abortRef = useRef<AbortController | null>(null);
  const resultRef = useRef<TranslateResult | null>(null);

  const run = useCallback(async (frames: number[][][], options?: TranslateRunOptions) => {
    const requestId = requestIdRef.current + 1;
    requestIdRef.current = requestId;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    const model = options?.model;
    const featureSchema = options?.featureSchema ?? inferFeatureSchema(frames);

    setPending(true);
    if (!resultRef.current) {
      setStatus("loading");
    }
    setError(null);
    setErrorStatus(null);
    // Keep previous result visible during loading — avoids blank flash.
    try {
      let res: TranslateResult;
      if (MOCKUP_MODE) {
        res = await new Promise<TranslateResult>((resolve, reject) => {
          const timer = window.setTimeout(() => {
            resolve(createMockTranslateResult(frames, model));
          }, 120);
          controller.signal.addEventListener(
            "abort",
            () => {
              window.clearTimeout(timer);
              reject(new DOMException("Mock translate aborted", "AbortError"));
            },
            { once: true },
          );
        });
      } else {
        try {
          res = await translateRealtime({
            frames,
            model,
            featureSchema,
            signal: controller.signal,
          });
        } catch (e) {
          if (e instanceof ApiError || controller.signal.aborted) throw e;
          res = await translate({
            frames,
            model,
            featureSchema,
            signal: controller.signal,
          });
        }
      }
      if (requestIdRef.current !== requestId) return;
      resultRef.current = res;
      setResult(res);
      setStatus("success");
    } catch (e) {
      if (controller.signal.aborted || requestIdRef.current !== requestId) return;
      if (e instanceof ApiError) {
        setError(e.detail);
        setErrorStatus(e.status);
      } else {
        setError("เกิดข้อผิดพลาดที่ไม่คาดคิด");
        setErrorStatus(null);
      }
      setStatus("error");
    } finally {
      if (requestIdRef.current === requestId) {
        abortRef.current = null;
        setPending(false);
      }
    }
  }, []);

  const cancel = useCallback((options?: { keepResult?: boolean }) => {
    requestIdRef.current += 1;
    abortRef.current?.abort();
    abortRef.current = null;
    closeRealtimeSocket();
    setPending(false);
    if (!options?.keepResult) {
      resultRef.current = null;
      setResult(null);
      setStatus("idle");
    }
  }, []);

  const reset = useCallback(() => {
    cancel();
    setStatus("idle");
    resultRef.current = null;
    setResult(null);
    setError(null);
    setErrorStatus(null);
  }, [cancel]);

  useEffect(() => () => {
    requestIdRef.current += 1;
    abortRef.current?.abort();
    abortRef.current = null;
  }, []);

  return { status, result, error, errorStatus, pending, run, cancel, reset };
}
