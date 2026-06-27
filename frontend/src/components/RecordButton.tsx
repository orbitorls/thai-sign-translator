import React from "react";
import { th } from "../i18n/th";

interface RecordButtonProps {
  recording: boolean;
  disabled?: boolean;
  frameCount?: number;
  onClick: () => void;
  variant?: "large";
}

export function RecordButton({ recording, disabled, frameCount, onClick, variant }: RecordButtonProps) {
  const isLarge = variant === "large";
  const size = isLarge ? 80 : 64;

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "var(--space-2)" }}>
      {/* Frame counter */}
      {recording && frameCount !== undefined && (
        <span
          style={{
            color: "#fff",
            fontSize: "var(--font-size-sm)",
            fontWeight: 700,
            textShadow: "0 1px 6px rgba(0,0,0,0.7)",
            letterSpacing: "0.04em",
          }}
        >
          {th.frames(frameCount)}
        </span>
      )}

      <button
        onClick={onClick}
        disabled={disabled}
        aria-pressed={recording}
        aria-label={recording ? th.recordStop : th.recordStart}
        style={{
          width: size,
          height: size,
          borderRadius: "50%",
          border: `${isLarge ? 4 : 3}px solid ${recording ? "rgba(220,38,38,0.9)" : "rgba(255,255,255,0.85)"}`,
          background: recording
            ? "rgba(220, 38, 38, 0.85)"
            : "rgba(255, 255, 255, 0.18)",
          backdropFilter: "blur(16px) saturate(160%)",
          WebkitBackdropFilter: "blur(16px) saturate(160%)",
          cursor: disabled ? "not-allowed" : "pointer",
          opacity: disabled ? 0.5 : 1,
          transition: "all 200ms ease",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          boxShadow: recording
            ? "0 0 0 10px rgba(220,38,38,0.22), 0 0 0 20px rgba(220,38,38,0.08), 0 4px 20px rgba(220,38,38,0.4)"
            : "0 0 0 6px rgba(255,255,255,0.12), 0 4px 24px rgba(0,0,0,0.3)",
        }}
      >
        {/* Inner shape: circle → rounded-square */}
        <span
          style={{
            width: recording ? (isLarge ? 28 : 22) : (isLarge ? 36 : 28),
            height: recording ? (isLarge ? 28 : 22) : (isLarge ? 36 : 28),
            borderRadius: recording ? "7px" : "50%",
            background: recording ? "#fff" : "rgba(255,255,255,0.9)",
            display: "block",
            transition: "all 200ms ease",
          }}
        />
      </button>

      {/* Label below button */}
      <span
        style={{
          color: recording ? "rgba(255,100,100,0.9)" : "rgba(255,255,255,0.7)",
          fontSize: "var(--font-size-xs)",
          fontWeight: 600,
          textShadow: "0 1px 4px rgba(0,0,0,0.6)",
          letterSpacing: "0.02em",
        }}
      >
        {recording ? th.recordStop : th.recordStart}
      </span>
    </div>
  );
}
