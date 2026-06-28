import { useCallback, useEffect, useState } from "react";

export interface HistoryEntry {
  id: string;
  timeISO: string;
  sentence: string;
  score: number;
  model: string;
  chips: string[];
}

const STORAGE_KEY = "signbridge:history";

function loadHistory(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    return JSON.parse(raw) as HistoryEntry[];
  } catch {
    return [];
  }
}

function saveHistory(entries: HistoryEntry[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
  } catch {
    // ignore
  }
}

export interface UseLocalHistory {
  entries: HistoryEntry[];
  addEntry: (entry: Omit<HistoryEntry, "id" | "timeISO">) => void;
  removeEntry: (id: string) => void;
  clear: () => void;
}

export function useLocalHistory(): UseLocalHistory {
  const [entries, setEntries] = useState<HistoryEntry[]>(loadHistory);

  useEffect(() => {
    saveHistory(entries);
  }, [entries]);

  const addEntry = useCallback((entry: Omit<HistoryEntry, "id" | "timeISO">) => {
    const newEntry: HistoryEntry = {
      ...entry,
      id: crypto.randomUUID(),
      timeISO: new Date().toISOString(),
    };
    setEntries((prev) => [newEntry, ...prev]);
  }, []);

  const removeEntry = useCallback((id: string) => {
    setEntries((prev) => prev.filter((e) => e.id !== id));
  }, []);

  const clear = useCallback(() => {
    setEntries([]);
  }, []);

  return { entries, addEntry, removeEntry, clear };
}
