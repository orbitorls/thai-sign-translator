/**
 * ModelsProvider — provides the list of available models and the user-selected model
 * throughout the app via React context.
 */
import React, { createContext, useContext, useEffect, useState, ReactNode } from "react";
import { getModels, ModelInfo } from "../api/client";
import { MOCK_MODELS, MOCKUP_MODE } from "../mockup";

interface ModelsContextValue {
  models: ModelInfo[];
  selectedModelId: string | null;
  defaultModelId: string | null;
  loading: boolean;
  error: string | null;
  setSelectedModelId: (id: string | null) => void;
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
    if (MOCKUP_MODE) {
      setModels(MOCK_MODELS);
      setDefaultModelId(MOCK_MODELS[0]?.id ?? null);
      setSelectedModelId(MOCK_MODELS[0]?.id ?? null);
      setLoading(false);
      return;
    }

    getModels()
      .then((resp) => {
        setModels(resp.models);
        setDefaultModelId(resp.default);
        const available = resp.models.filter((model) => model.available);
        const defaultModel = available.find((model) => model.id === resp.default);
        setSelectedModelId(defaultModel?.id ?? available[0]?.id ?? null);
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
