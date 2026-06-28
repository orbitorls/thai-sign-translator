import { useSettings } from "../hooks/SettingsProvider";
import { th } from "./th";
import { en } from "./en";

export type Dict = typeof th;
export type Lang = "th" | "en";

export const dictionaries: Record<Lang, Dict> = { th, en: en as unknown as Dict };

/** Conductor-style hook — returns active dictionary. */
export function useI18n(): Dict {
  const { lang } = useSettings();
  return dictionaries[lang];
}

/** Stash-style hook — returns { lang, t, setLang, toggleLang }. */
export function useT() {
  const { lang, setLang, settings, update } = useSettings();
  const t = dictionaries[lang];
  const toggleLang = () => setLang(lang === "th" ? "en" : "th");
  return { lang, t, setLang, toggleLang, settings, update };
}
