import { useCallback, useRef, useState } from "react";

export type FeedbackKind = "success" | "error";

export interface Feedback {
  flash: FeedbackKind | null;
  signal: (kind: FeedbackKind) => void;
}

const VIBRATE: Record<FeedbackKind, number | number[]> = {
  success: 30,
  error: [60, 40, 60],
};

/** Non-audio feedback for Deaf users: a brief visual flash + haptic buzz. */
export function useFeedback(): Feedback {
  const [flash, setFlash] = useState<FeedbackKind | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout>>();

  const signal = useCallback((kind: FeedbackKind) => {
    if (typeof navigator !== "undefined" && "vibrate" in navigator) {
      try { navigator.vibrate(VIBRATE[kind]); } catch { /* ignore */ }
    }
    setFlash(kind);
    clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => setFlash(null), 600);
  }, []);

  return { flash, signal };
}
