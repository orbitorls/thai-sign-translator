import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  ReactNode,
} from "react";

export interface HistoryItem {
  id: string;
  sentence: string;
  score: number;
  model: string;
  ts: number;
}

export interface HistoryEntry {
  id: string;
  timeISO: string;
  sentence: string;
  score: number;
  model: string;
  chips: string[];
}

const STORAGE_KEY = "tsl.history.v2";
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

function toEntry(item: HistoryItem): HistoryEntry {
  return {
    id: item.id,
    timeISO: new Date(item.ts).toISOString(),
    sentence: item.sentence,
    score: item.score,
    model: item.model,
    chips: [],
  };
}

interface HistoryContextValue {
  items: HistoryItem[];
  entries: HistoryEntry[];
  add: (entry: { sentence: string; score: number; model: string }) => void;
  addEntry: (entry: Omit<HistoryEntry, "id" | "timeISO">) => void;
  removeEntry: (id: string) => void;
  clear: () => void;
}

const HistoryContext = createContext<HistoryContextValue>({
  items: [],
  entries: [],
  add: () => {},
  addEntry: () => {},
  removeEntry: () => {},
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
      if (last && last.sentence === entry.sentence && Date.now() - last.ts < DEDUP_WINDOW_MS) {
        return prev;
      }
      const next: HistoryItem = { id: genId(), ts: Date.now(), ...entry };
      return [next, ...prev].slice(0, CAP);
    });
  }, []);

  const addEntry = useCallback((entry: Omit<HistoryEntry, "id" | "timeISO">) => {
    add({ sentence: entry.sentence, score: entry.score, model: entry.model });
  }, [add]);

  const removeEntry = useCallback((id: string) => {
    setItems((prev) => prev.filter((it) => it.id !== id));
  }, []);

  const clear = useCallback(() => setItems([]), []);

  const entries = items.map(toEntry);

  return (
    <HistoryContext.Provider value={{ items, entries, add, addEntry, removeEntry, clear }}>
      {children}
    </HistoryContext.Provider>
  );
}

export function useHistory(): HistoryContextValue {
  return useContext(HistoryContext);
}

/** Back-compat alias for stash components. */
export function useLocalHistory() {
  const { entries, addEntry, removeEntry, clear } = useHistory();
  return { entries, addEntry, removeEntry, clear };
}
