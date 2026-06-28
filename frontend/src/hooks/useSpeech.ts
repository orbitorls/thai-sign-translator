import { useCallback, useEffect, useRef, useState } from "react";

export interface SpeechController {
  speak: (text: string, lang?: string) => void;
  cancel: () => void;
  speaking: boolean;
  supported: boolean;
}

/** Thin wrapper over the Web Speech API (speechSynthesis). */
export function useSpeech(): SpeechController {
  const supported = typeof window !== "undefined" && "speechSynthesis" in window;
  const [speaking, setSpeaking] = useState(false);
  const voicesRef = useRef<SpeechSynthesisVoice[]>([]);

  useEffect(() => {
    if (!supported) return;
    const load = () => {
      voicesRef.current = window.speechSynthesis.getVoices();
    };
    load();
    window.speechSynthesis.addEventListener("voiceschanged", load);
    return () => window.speechSynthesis.removeEventListener("voiceschanged", load);
  }, [supported]);

  const speak = useCallback(
    (text: string, lang = "th-TH") => {
      if (!supported || !text) return;
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.lang = lang;
      const base = lang.split("-")[0];
      const voice =
        voicesRef.current.find((v) => v.lang === lang) ||
        voicesRef.current.find((v) => v.lang.startsWith(base));
      if (voice) u.voice = voice;
      u.onstart = () => setSpeaking(true);
      u.onend = () => setSpeaking(false);
      u.onerror = () => setSpeaking(false);
      window.speechSynthesis.speak(u);
    },
    [supported]
  );

  const cancel = useCallback(() => {
    if (!supported) return;
    window.speechSynthesis.cancel();
    setSpeaking(false);
  }, [supported]);

  return { speak, cancel, speaking, supported };
}
