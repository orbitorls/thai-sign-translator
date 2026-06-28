import { useCallback, useEffect, useState } from "react";

export type CameraFacing = "user" | "environment";

export interface Settings {
  voiceURI: string;
  animationSpeed: number;
  highContrast: boolean;
  autoScroll: boolean;
  fontSize: "small" | "medium" | "large";
  cameraFacing: CameraFacing;
  diagnosticsEnabled: boolean;
  /* Device selectors persisted across reloads. Empty string = use the
     first enumerated device. */
  cameraDeviceId: string;
  micDeviceId: string;
  speakerDeviceId: string;
  /* Accessibility / output toggles surfaced on the Settings page. */
  autoStopOnPause: boolean;
  speakAloud: boolean;
}

const DEFAULT_SETTINGS: Settings = {
  voiceURI: "",
  animationSpeed: 1.0,
  highContrast: false,
  autoScroll: true,
  fontSize: "medium",
  cameraFacing: "user",
  diagnosticsEnabled: true,
  cameraDeviceId: "",
  micDeviceId: "",
  speakerDeviceId: "",
  autoStopOnPause: false,
  speakAloud: false,
};

const STORAGE_KEY = "signbridge:settings";

function loadSettings(): Settings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_SETTINGS;
    return { ...DEFAULT_SETTINGS, ...JSON.parse(raw) };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

function saveSettings(s: Settings) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(s));
  } catch {
    // ignore
  }
}

export interface UseLocalSettings {
  settings: Settings;
  update: <K extends keyof Settings>(key: K, value: Settings[K]) => void;
  reset: () => void;
}

export function useLocalSettings(): UseLocalSettings {
  const [settings, setSettings] = useState<Settings>(loadSettings);

  useEffect(() => {
    saveSettings(settings);
  }, [settings]);

  // Sync high-contrast class on <html>
  useEffect(() => {
    const html = document.documentElement;
    if (settings.highContrast) {
      html.classList.add("contrast-high");
    } else {
      html.classList.remove("contrast-high");
    }
  }, [settings.highContrast]);

  // Sync font-size CSS variable
  useEffect(() => {
    const sizes: Record<string, string> = {
      small: "14px",
      medium: "18px",
      large: "22px",
    };
    document.documentElement.style.setProperty(
      "--transcript-font-size",
      sizes[settings.fontSize]
    );
  }, [settings.fontSize]);

  const update = useCallback(
    <K extends keyof Settings>(key: K, value: Settings[K]) => {
      setSettings((prev) => ({ ...prev, [key]: value }));
    },
    []
  );

  const reset = useCallback(() => {
    setSettings(DEFAULT_SETTINGS);
  }, []);

  return { settings, update, reset };
}
