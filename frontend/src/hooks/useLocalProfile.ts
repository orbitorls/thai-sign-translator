import { useCallback, useEffect, useState } from "react";

export interface Profile {
  name: string;
  email: string;
  /* Optional fields surfaced on the Settings profile card. */
  avatarColor?: string;
  bio?: string;
  createdAt?: string;
  updatedAt?: string;
}

const DEFAULT_PROFILE: Profile = {
  name: "ผู้ใช้ตัวอย่าง",
  email: "demo@conductor.app",
};

const STORAGE_KEY = "signbridge:profile";

function loadProfile(): Profile {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_PROFILE;
    const parsed = JSON.parse(raw) as Partial<Profile>;
    return {
      name: typeof parsed.name === "string" ? parsed.name : DEFAULT_PROFILE.name,
      email: typeof parsed.email === "string" ? parsed.email : DEFAULT_PROFILE.email,
      avatarColor: typeof parsed.avatarColor === "string" ? parsed.avatarColor : undefined,
      bio: typeof parsed.bio === "string" ? parsed.bio : undefined,
      createdAt: typeof parsed.createdAt === "string" ? parsed.createdAt : undefined,
      updatedAt: typeof parsed.updatedAt === "string" ? parsed.updatedAt : undefined,
    };
  } catch {
    return DEFAULT_PROFILE;
  }
}

function saveProfile(p: Profile) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(p));
  } catch {
    // ignore
  }
}

export interface UseLocalProfile {
  profile: Profile;
  update: <K extends keyof Profile>(key: K, value: Profile[K]) => void;
  reset: () => void;
}

export function useLocalProfile(): UseLocalProfile {
  const [profile, setProfile] = useState<Profile>(loadProfile);

  useEffect(() => {
    saveProfile(profile);
  }, [profile]);

  const update = useCallback(
    <K extends keyof Profile>(key: K, value: Profile[K]) => {
      setProfile((prev) => ({ ...prev, [key]: value }));
    },
    [],
  );

  const reset = useCallback(() => {
    setProfile(DEFAULT_PROFILE);
  }, []);

  return { profile, update, reset };
}
