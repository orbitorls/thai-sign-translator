import React from "react";
import { useModels } from "../hooks/ModelsProvider";
import { useI18n } from "../i18n";

interface ModelPickerProps {
  className?: string;
}

export function ModelPicker({ className }: ModelPickerProps) {
  const th = useI18n();
  const { models, selectedModelId, loading, error, setSelectedModelId } = useModels();

  if (loading) {
    return (
      <span
        className={className ?? "glass-chip model-picker-select"}
        aria-label={th.modelLoading}
        aria-busy="true"
        style={{ opacity: 0.6, cursor: "default", userSelect: "none" }}
      >
        {th.modelLoading}
      </span>
    );
  }

  if (error || models.length === 0) {
    return (
      <span
        className={className ?? "glass-chip model-picker-select"}
        aria-label={th.modelLoadError}
        style={{ cursor: "default", userSelect: "none" }}
      >
        <span className="glass-text-warn">{th.modelLoadError}</span>
      </span>
    );
  }

  return (
    <select
      className={className ?? "glass-chip model-picker-select"}
      value={selectedModelId ?? ""}
      onChange={(e) => setSelectedModelId(e.target.value || null)}
      aria-label={th.modelLabel}
    >
      {models.map((m) => (
        <option key={m.id} value={m.id} disabled={!m.available}>
          {m.label_th}
          {!m.available ? ` (${th.modelUnavailable})` : ""}
        </option>
      ))}
    </select>
  );
}
