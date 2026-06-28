import React, { useEffect, useRef, useState } from "react";
import { useModels } from "../hooks/ModelsProvider";
import { useI18n } from "../i18n";

interface ModelPickerProps {
  className?: string;
}

const ChevronIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"
    strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"
    className="model-picker-chevron">
    <path d="M6 9l6 6 6-6" />
  </svg>
);

const CheckIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"
    strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"
    className="model-picker-check">
    <path d="M20 6L9 17l-5-5" />
  </svg>
);

export function ModelPicker({ className }: ModelPickerProps) {
  const th = useI18n();
  const { models, selectedModelId, loading, error, setSelectedModelId } = useModels();
  const [open, setOpen] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    const closeKey = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", closeKey);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", closeKey);
    };
  }, [open]);

  const chipClass = className ?? "glass-chip model-picker-select";

  if (loading) {
    return (
      <span className={chipClass} aria-label={th.modelLoading}
        aria-busy="true" aria-disabled="true"
        style={{ opacity: 0.6, cursor: "default", userSelect: "none" }}>
        {th.modelLoading}
      </span>
    );
  }

  if (error || models.length === 0) {
    return (
      <span className={chipClass} aria-label={th.modelLoadError}
        aria-disabled="true" style={{ cursor: "default", userSelect: "none" }}>
        <span className="glass-text-warn" aria-hidden="true">{th.modelLoadError}</span>
      </span>
    );
  }

  const selected = models.find((m) => m.id === selectedModelId) ?? models[0];

  return (
    <div ref={wrapperRef} className="model-picker-wrap model-picker-select">
      <button
        type="button"
        className={`glass-chip model-picker-btn${open ? " open" : ""}`}
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={th.modelLabel}
      >
        <span className="model-picker-label">{selected?.label_th}</span>
        <ChevronIcon />
      </button>

      {open && (
        <div className="model-picker-panel" role="listbox" aria-label={th.modelLabel}>
          {models.map((m) => (
            <button
              key={m.id}
              type="button"
              role="option"
              aria-selected={m.id === selectedModelId}
              disabled={!m.available}
              className={`model-picker-option${m.id === selectedModelId ? " selected" : ""}${!m.available ? " unavailable" : ""}`}
              onClick={() => { setSelectedModelId(m.id); setOpen(false); }}
            >
              <span className="model-picker-option-check">
                {m.id === selectedModelId && <CheckIcon />}
              </span>
              <span className="model-picker-option-label">{m.label_th}</span>
              {!m.available && (
                <span className="model-picker-unavail">({th.modelUnavailable})</span>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
