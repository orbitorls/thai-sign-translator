import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from "react";

export type Lang = "th" | "en";

export interface Settings {
  lang: Lang;
  showLandmarks: boolean;
}

const DEFAULT_SETTINGS: Settings = { lang: "th", showLandmarks: false };
const STORAGE_KEY = "tsl.settings.v1";

function loadSettings(): Settings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_SETTINGS;
    const parsed = JSON.parse(raw);
    return {
      lang: parsed.lang === "en" ? "en" : "th",
      showLandmarks: Boolean(parsed.showLandmarks),
    };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

interface SettingsContextValue extends Settings {
  setLang: (lang: Lang) => void;
  setShowLandmarks: (v: boolean) => void;
}

const SettingsContext = createContext<SettingsContextValue>({
  ...DEFAULT_SETTINGS,
  setLang: () => {},
  setShowLandmarks: () => {},
});

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<Settings>(loadSettings);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    } catch {
      /* ignore quota/availability errors */
    }
  }, [settings]);

  const setLang = useCallback((lang: Lang) => setSettings((s) => ({ ...s, lang })), []);
  const setShowLandmarks = useCallback(
    (showLandmarks: boolean) => setSettings((s) => ({ ...s, showLandmarks })),
    []
  );

  return (
    <SettingsContext.Provider value={{ ...settings, setLang, setShowLandmarks }}>
      {children}
    </SettingsContext.Provider>
  );
}

export function useSettings(): SettingsContextValue {
  return useContext(SettingsContext);
}
