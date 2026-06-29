import { useCallback, useState } from "react";
import { trainCustomSign, ApiError } from "../api/client";

export type TeachStatus = "idle" | "saving" | "saved" | "error";

export function useTeach() {
  const [clips, setClips] = useState<number[][][][]>([]);
  const [status, setStatus] = useState<TeachStatus>("idle");
  const [error, setError] = useState<string | null>(null);

  const addClip = useCallback((frames: number[][][]) => {
    setClips((c) => [...c, frames]);
    setStatus("idle");
  }, []);

  const removeClip = useCallback((i: number) => {
    setClips((c) => c.filter((_, idx) => idx !== i));
  }, []);

  const clear = useCallback(() => {
    setClips([]);
    setStatus("idle");
    setError(null);
  }, []);

  const submit = useCallback(
    async (name: string): Promise<boolean> => {
      if (!name.trim() || clips.length === 0) return false;
      setStatus("saving");
      setError(null);
      try {
        await trainCustomSign(name.trim(), clips);
        setStatus("saved");
        setClips([]);
        return true;
      } catch (e) {
        setError(e instanceof ApiError ? e.detail : "บันทึกไม่สำเร็จ");
        setStatus("error");
        return false;
      }
    },
    [clips]
  );

  return { clips, status, error, addClip, removeClip, clear, submit };
}
