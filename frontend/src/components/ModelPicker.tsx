import React from "react";
import { ModelInfo } from "../api/client";
import { th } from "../i18n/th";

interface ModelPickerProps {
  models: ModelInfo[];
  selectedId: string | null;
  onChange: (id: string) => void;
  disabled?: boolean;
}

export function ModelPicker({ models, selectedId, onChange, disabled }: ModelPickerProps) {
  if (models.length === 0) return null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-2)" }}>
      <label
        htmlFor="model-picker"
        style={{
          fontSize: "var(--font-size-sm)",
          fontWeight: 600,
          color: "var(--color-text)",
        }}
      >
        {th.modelLabel}
      </label>
      <div
        style={{
          display: "grid",
          gap: "var(--space-2)",
          padding: "var(--space-2)",
          border: "1px solid var(--color-border)",
          borderRadius: "var(--radius-lg)",
          background: "var(--color-surface-soft)",
        }}
      >
        {models.map((m) => {
          const isSelected = m.id === selectedId;
          const isAvailable = m.available;
          return (
            <button
              key={m.id}
              id={m.id === models[0]?.id ? "model-picker" : undefined}
              onClick={() => isAvailable && onChange(m.id)}
              disabled={!isAvailable || disabled}
              aria-pressed={isSelected}
              title={!isAvailable ? th.modelUnavailable : undefined}
              style={{
                padding: "var(--space-3) var(--space-4)",
                borderRadius: "var(--radius-md)",
                border: `1px solid ${isSelected ? "var(--color-primary)" : "transparent"}`,
                background: isSelected ? "var(--color-primary)" : "var(--color-surface)",
                color: isSelected ? "#fff" : isAvailable ? "var(--color-text)" : "var(--color-text-placeholder)",
                fontFamily: "var(--font-family)",
                fontSize: "var(--font-size-sm)",
                fontWeight: isSelected ? 600 : 400,
                cursor: isAvailable && !disabled ? "pointer" : "not-allowed",
                opacity: isAvailable ? 1 : 0.45,
                transition: "var(--transition)",
                textAlign: "left",
                minHeight: "44px",
                boxShadow: isSelected ? "var(--shadow-sm)" : "none",
              }}
            >
              {m.label_th}
              {!isAvailable && (
                <span style={{ marginLeft: "var(--space-2)", fontSize: "var(--font-size-xs)" }}>
                  ({th.modelUnavailable})
                </span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
