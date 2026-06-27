import React from "react";
import { TranslateResult } from "../api/client";
import { th } from "../i18n/th";

interface ResultCardProps {
  status: "idle" | "loading" | "success" | "error";
  result: TranslateResult | null;
  error: string | null;
  errorStatus: number | null;
  variant?: "glass";
}

export function ResultCard({ status, result, error, errorStatus, variant }: ResultCardProps) {
  const glass = variant === "glass";
  const pct = result ? Math.round(result.score * 100) : null;

  const textColor = glass ? "#fff" : "var(--color-text)";
  const mutedColor = glass ? "rgba(255,255,255,0.6)" : "var(--color-text-muted)";
  const placeholderColor = glass ? "rgba(255,255,255,0.28)" : "var(--color-text-placeholder)";
  const trackColor = glass ? "rgba(255,255,255,0.15)" : "var(--color-border)";

  return (
    <div
      aria-live="polite"
      aria-atomic="true"
      className={glass ? undefined : "message-bubble"}
      style={{
        position: "relative",
        minHeight: glass ? 72 : 120,
        display: "flex",
        flexDirection: "column",
        gap: "var(--space-3)",
      }}
    >
      {/* Loading bar at top edge of panel — visible during any loading */}
      {status === "loading" && (
        <div
          aria-hidden="true"
          style={{
            position: "absolute",
            top: 0,
            left: "-var(--space-5)",
            right: "-var(--space-5)",
            height: 2,
            overflow: "hidden",
            borderRadius: "var(--radius-lg) var(--radius-lg) 0 0",
          }}
        >
          <div style={{ animation: "scanning-bar 1.4s ease-in-out infinite" }} className="scanning-shimmer" />
        </div>
      )}

      {status === "loading" && result && (
        /* Previous result shown dimmed while new one loads */
        <>
          <p
            style={{
              fontSize: "var(--font-size-3xl)",
              fontWeight: 700,
              color: textColor,
              lineHeight: 1.4,
              wordBreak: "break-word",
              opacity: 0.4,
              transition: "opacity 300ms ease",
            }}
          >
            {result.sentence || th.resultPlaceholder}
          </p>
          {pct !== null && (
            <div style={{ opacity: 0.3, display: "flex", flexDirection: "column", gap: "var(--space-2)" }}>
              <span style={{ fontSize: "var(--font-size-sm)", color: mutedColor }}>{th.confidence(pct)}</span>
              <ConfidenceBar pct={pct} trackColor={trackColor} />
            </div>
          )}
        </>
      )}

      {status === "loading" && !result && (
        <div style={{ display: "flex", alignItems: "center", gap: "var(--space-3)", color: mutedColor }}>
          <Spinner glass={glass} />
          <span style={{ fontSize: "var(--font-size-lg)", color: mutedColor }}>{th.translating}</span>
        </div>
      )}

      {status === "error" && (
        <div style={{ color: glass ? "#fca5a5" : "var(--color-danger)", fontSize: "var(--font-size-base)" }}>
          {errorStatus === 503 ? th.errorModelUnavailable : error ?? th.errorGeneric}
        </div>
      )}

      {status === "success" && result && (
        <>
          <p
            style={{
              fontSize: "var(--font-size-3xl)",
              fontWeight: 700,
              color: textColor,
              lineHeight: 1.4,
              wordBreak: "break-word",
            }}
          >
            {result.sentence || th.resultPlaceholder}
          </p>
          {pct !== null && (
            <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-2)" }}>
              <span style={{ fontSize: "var(--font-size-sm)", color: mutedColor }}>
                {th.confidence(pct)}
              </span>
              <ConfidenceBar pct={pct} trackColor={trackColor} />
            </div>
          )}
        </>
      )}

      {(status === "idle") && (
        <p style={{ fontSize: "var(--font-size-3xl)", color: placeholderColor, fontWeight: 300 }}>
          {th.resultPlaceholder}
        </p>
      )}
    </div>
  );
}

function ConfidenceBar({ pct, trackColor }: { pct: number; trackColor: string }) {
  return (
    <div
      style={{
        height: 6,
        borderRadius: "var(--radius-full)",
        background: trackColor,
        overflow: "hidden",
      }}
    >
      <div
        style={{
          height: "100%",
          width: `${pct}%`,
          background: pct >= 70 ? "#22c55e" : pct >= 40 ? "#f59e0b" : "#ef4444",
          borderRadius: "var(--radius-full)",
          transition: "width 0.4s ease",
        }}
      />
    </div>
  );
}

function Spinner({ glass }: { glass?: boolean }) {
  return (
    <div
      style={{
        width: 20,
        height: 20,
        border: `3px solid ${glass ? "rgba(255,255,255,0.2)" : "var(--color-border)"}`,
        borderTopColor: glass ? "#fff" : "var(--color-primary)",
        borderRadius: "50%",
        flexShrink: 0,
        animation: "spin 0.8s linear infinite",
      }}
    />
  );
}
