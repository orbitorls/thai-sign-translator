import React from "react";
import { th } from "../i18n/th";

interface RecordButtonProps {
  recording: boolean;
  disabled?: boolean;
  frameCount?: number;
  onClick: () => void;
}

export function RecordButton({ recording, disabled, frameCount, onClick }: RecordButtonProps) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "var(--space-3)", width: "100%", justifyContent: "center" }}>
      <button
        onClick={onClick}
        disabled={disabled}
        aria-pressed={recording}
        aria-label={recording ? th.recordStop : th.recordStart}
        style={{
          minWidth: 168,
          minHeight: 52,
          borderRadius: "var(--radius-full)",
          border: `1px solid ${recording ? "var(--color-recording)" : "var(--color-primary)"}`,
          background: recording ? "var(--color-recording)" : "var(--color-primary)",
          color: "#fff",
          fontSize: "var(--font-size-base)",
          fontFamily: "var(--font-family)",
          fontWeight: 700,
          cursor: disabled ? "not-allowed" : "pointer",
          opacity: disabled ? 0.55 : 1,
          transition: "var(--transition)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          gap: "var(--space-2)",
          boxShadow: recording ? "0 0 0 8px rgba(220,38,38,0.18)" : "var(--shadow-md)",
        }}
      >
        <span
          style={{
            width: recording ? 22 : 18,
            height: recording ? 22 : 18,
            borderRadius: recording ? "4px" : "50%",
            background: "#fff",
            display: "block",
            transition: "var(--transition)",
          }}
        />
        <span>{recording ? th.recordStop : th.recordStart}</span>
      </button>
      <span
        style={{
          fontSize: "var(--font-size-sm)",
          fontWeight: 600,
          color: recording ? "var(--color-recording)" : "var(--color-text-muted)",
          minWidth: 76,
        }}
      >
        {recording && frameCount !== undefined ? th.frames(frameCount) : ""}
      </span>
    </div>
  );
}
