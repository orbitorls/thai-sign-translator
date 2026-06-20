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
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "var(--space-3)" }}>
      <button
        onClick={onClick}
        disabled={disabled}
        aria-pressed={recording}
        aria-label={recording ? th.recordStop : th.recordStart}
        style={{
          width: 80,
          height: 80,
          borderRadius: "50%",
          border: `3px solid ${recording ? "var(--color-recording)" : "var(--color-primary)"}`,
          background: recording ? "var(--color-recording)" : "var(--color-primary)",
          color: "#fff",
          fontSize: "var(--font-size-xs)",
          fontFamily: "var(--font-family)",
          fontWeight: 700,
          cursor: disabled ? "not-allowed" : "pointer",
          opacity: disabled ? 0.55 : 1,
          transition: "var(--transition)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexDirection: "column",
          gap: 4,
          boxShadow: recording ? "0 0 0 8px rgba(220,38,38,0.18)" : "var(--shadow-md)",
        }}
      >
        {/* Stop square / Record circle icon */}
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
      </button>
      <span
        style={{
          fontSize: "var(--font-size-sm)",
          fontWeight: 600,
          color: recording ? "var(--color-recording)" : "var(--color-text-muted)",
        }}
      >
        {recording
          ? `${th.recording}${frameCount !== undefined ? ` (${th.frames(frameCount)})` : ""}`
          : th.recordStart}
      </span>
    </div>
  );
}
