import { useSettings, type CameraFacing, type FontSize, type Settings } from "./SettingsProvider";

export type { CameraFacing, FontSize, Settings };

export interface UseLocalSettings {
  settings: Settings;
  update: <K extends keyof Settings>(key: K, value: Settings[K]) => void;
  reset: () => void;
}

/** Back-compat alias for stash components — reads unified SettingsProvider. */
export function useLocalSettings(): UseLocalSettings {
  const { settings, update, reset } = useSettings();
  return { settings, update, reset };
}
