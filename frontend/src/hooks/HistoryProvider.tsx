import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from "react";

export interface HistoryItem {
  id: string;
  sentence: string;
  score: number;
  model: string;
  ts: number;
}

const STORAGE_KEY = "tsl.history.v1";
const CAP = 100;
const DEDUP_WINDOW_MS = 30_000;

function load(): HistoryItem[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as HistoryItem[]) : [];
  } catch {
    return [];
  }
}

function genId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

interface HistoryContextValue {
  items: HistoryItem[];
  add: (entry: { sentence: string; score: number; model: string }) => void;
  clear: () => void;
}

const HistoryContext = createContext<HistoryContextValue>({
  items: [],
  add: () => {},
  clear: () => {},
});

export function HistoryProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<HistoryItem[]>(load);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
    } catch {
      /* ignore */
    }
  }, [items]);

  const add = useCallback((entry: { sentence: string; score: number; model: string }) => {
    setItems((prev) => {
      const last = prev[0];
      // Dedup: skip identical sentence within the dedup window (auto-loop fires repeatedly).
      if (last && last.sentence === entry.sentence && Date.now() - last.ts < DEDUP_WINDOW_MS) {
        return prev;
      }
      const next: HistoryItem = { id: genId(), ts: Date.now(), ...entry };
      return [next, ...prev].slice(0, CAP);
    });
  }, []);

  const clear = useCallback(() => setItems([]), []);

  return <HistoryContext.Provider value={{ items, add, clear }}>{children}</HistoryContext.Provider>;
}

export function useHistory(): HistoryContextValue {
  return useContext(HistoryContext);
}
