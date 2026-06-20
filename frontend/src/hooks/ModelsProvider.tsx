/**
 * ModelsProvider — provides the list of available models and the user-selected model
 * throughout the app via React context.
 */
import React, { createContext, useContext, useEffect, useState, ReactNode } from "react";
import { getModels, ModelInfo } from "../api/client";

interface ModelsContextValue {
  models: ModelInfo[];
  selectedModelId: string | null;
  defaultModelId: string | null;
  loading: boolean;
  error: string | null;
  setSelectedModelId: (id: string) => void;
}

const ModelsContext = createContext<ModelsContextValue>({
  models: [],
  selectedModelId: null,
  defaultModelId: null,
  loading: true,
  error: null,
  setSelectedModelId: () => {},
});

export function ModelsProvider({ children }: { children: ReactNode }) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [defaultModelId, setDefaultModelId] = useState<string | null>(null);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getModels()
      .then((resp) => {
        setModels(resp.models);
        setDefaultModelId(resp.default);
        setSelectedModelId(resp.default);
      })
      .catch((e) => {
        setError(e instanceof Error ? e.message : "โหลดรายชื่อโมเดลล้มเหลว");
      })
      .finally(() => setLoading(false));
  }, []);

  return (
    <ModelsContext.Provider
      value={{ models, selectedModelId, defaultModelId, loading, error, setSelectedModelId }}
    >
      {children}
    </ModelsContext.Provider>
  );
}

export function useModels(): ModelsContextValue {
  return useContext(ModelsContext);
}
