import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  ReactNode,
} from "react";

export type Lang = "th" | "en";
export type CameraFacing = "user" | "environment";
export type FontSize = "small" | "medium" | "large";

export interface Settings {
  lang: Lang;
  showLandmarks: boolean;
  voiceURI: string;
  animationSpeed: number;
  highContrast: boolean;
  autoScroll: boolean;
  fontSize: FontSize;
  cameraFacing: CameraFacing;
  diagnosticsEnabled: boolean;
  cameraDeviceId: string;
  micDeviceId: string;
  speakerDeviceId: string;
  autoStopOnPause: boolean;
  speakAloud: boolean;
}

const DEFAULT_SETTINGS: Settings = {
  lang: "th",
  showLandmarks: false,
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

const STORAGE_KEY = "tsl.settings.v2";

function loadSettings(): Settings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_SETTINGS;
    const parsed = JSON.parse(raw) as Partial<Settings>;
    return { ...DEFAULT_SETTINGS, ...parsed, lang: parsed.lang === "en" ? "en" : "th" };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

interface SettingsContextValue {
  settings: Settings;
  update: <K extends keyof Settings>(key: K, value: Settings[K]) => void;
  reset: () => void;
  setLang: (lang: Lang) => void;
  setShowLandmarks: (v: boolean) => void;
  lang: Lang;
  showLandmarks: boolean;
}

const SettingsContext = createContext<SettingsContextValue>({
  settings: DEFAULT_SETTINGS,
  update: () => {},
  reset: () => {},
  setLang: () => {},
  setShowLandmarks: () => {},
  lang: "th",
  showLandmarks: false,
});

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<Settings>(loadSettings);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    } catch {
      /* ignore */
    }
  }, [settings]);

  useEffect(() => {
    const html = document.documentElement;
    if (settings.highContrast) html.classList.add("contrast-high");
    else html.classList.remove("contrast-high");
  }, [settings.highContrast]);

  useEffect(() => {
    const sizes: Record<FontSize, string> = {
      small: "14px",
      medium: "18px",
      large: "22px",
    };
    document.documentElement.style.setProperty(
      "--transcript-font-size",
      sizes[settings.fontSize]
    );
  }, [settings.fontSize]);

  const update = useCallback(<K extends keyof Settings>(key: K, value: Settings[K]) => {
    setSettings((prev) => ({ ...prev, [key]: value }));
  }, []);

  const reset = useCallback(() => setSettings(DEFAULT_SETTINGS), []);

  const setLang = useCallback((lang: Lang) => update("lang", lang), [update]);
  const setShowLandmarks = useCallback(
    (showLandmarks: boolean) => update("showLandmarks", showLandmarks),
    [update]
  );

  return (
    <SettingsContext.Provider
      value={{
        settings,
        update,
        reset,
        setLang,
        setShowLandmarks,
        lang: settings.lang,
        showLandmarks: settings.showLandmarks,
      }}
    >
      {children}
    </SettingsContext.Provider>
  );
}

export function useSettings(): SettingsContextValue {
  return useContext(SettingsContext);
}
