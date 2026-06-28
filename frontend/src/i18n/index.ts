import { useSettings, Lang } from "../hooks/SettingsProvider";
import { th } from "./th";
import { en } from "./en";

export type Dict = typeof th;

export const dictionaries: Record<Lang, Dict> = { th, en };

/** Returns the active dictionary based on the user's language setting. */
export function useI18n(): Dict {
  const { lang } = useSettings();
  return dictionaries[lang];
}
