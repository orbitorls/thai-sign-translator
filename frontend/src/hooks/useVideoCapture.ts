import { useCallback, useRef, useState } from "react";

export interface UseVideoCapture {
  recording: boolean;
  blob: Blob | null;
  error: string | null;
  start: (stream: MediaStream) => void;
  stop: () => Promise<Blob | null>;
  clear: () => void;
}

export function useVideoCapture(): UseVideoCapture {
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const [recording, setRecording] = useState(false);
  const [blob, setBlob] = useState<Blob | null>(null);
  const [error, setError] = useState<string | null>(null);

  const start = useCallback((stream: MediaStream) => {
    chunksRef.current = [];
    setBlob(null);
    setError(null);
    try {
      const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp8")
        ? "video/webm;codecs=vp8"
        : "video/webm";
      const recorder = new MediaRecorder(stream, { mimeType });
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };
      recorder.start(250);
      recorderRef.current = recorder;
      setRecording(true);
    } catch (err) {
      setError(`Recording unavailable: ${(err as Error)?.message ?? "unknown"}`);
      setRecording(false);
    }
  }, []);

  const stop = useCallback(async (): Promise<Blob | null> => {
    const recorder = recorderRef.current;
    if (!recorder || recorder.state === "inactive") {
      setRecording(false);
      return blob;
    }
    return new Promise((resolve) => {
      recorder.onstop = () => {
        const nextBlob = new Blob(chunksRef.current, { type: recorder.mimeType || "video/webm" });
        chunksRef.current = [];
        recorderRef.current = null;
        setBlob(nextBlob);
        setRecording(false);
        resolve(nextBlob);
      };
      recorder.stop();
    });
  }, [blob]);

  const clear = useCallback(() => {
    setBlob(null);
    setError(null);
    chunksRef.current = [];
  }, []);

  return { recording, blob, error, start, stop, clear };
}
