import { useCallback, useState } from "react";
import { predictSign, PredictResult, ApiError } from "../api/client";

export type PredictStatus = "idle" | "loading" | "success" | "error";

export function usePredict() {
  const [status, setStatus] = useState<PredictStatus>("idle");
  const [result, setResult] = useState<PredictResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async (frames: number[][][]) => {
    setStatus("loading");
    setError(null);
    try {
      const res = await predictSign(frames);
      setResult(res);
      setStatus("success");
    } catch (e) {
      setError(e instanceof ApiError ? e.detail : "เชื่อมต่อไม่ได้");
      setStatus("error");
    }
  }, []);

  const reset = useCallback(() => {
    setStatus("idle");
    setResult(null);
    setError(null);
  }, []);

  return { status, result, error, run, reset };
}
