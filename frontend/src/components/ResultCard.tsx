import React from "react";
import { TranslateResult } from "../api/client";
import { th } from "../i18n/th";

interface ResultCardProps {
  status: "idle" | "loading" | "success" | "error";
  result: TranslateResult | null;
  error: string | null;
  errorStatus: number | null;
}

export function ResultCard({ status, result, error, errorStatus }: ResultCardProps) {
  const pct = result ? Math.round(result.score * 100) : null;

  return (
    <div
      aria-live="polite"
      aria-atomic="true"
      style={{
        background: "var(--color-surface)",
        borderRadius: "var(--radius-lg)",
        padding: "var(--space-6)",
        boxShadow: "var(--shadow-sm)",
        border: "1px solid var(--color-border)",
        minHeight: 120,
        display: "flex",
        flexDirection: "column",
        gap: "var(--space-3)",
      }}
    >
      {status === "loading" && (
        <div style={{ display: "flex", alignItems: "center", gap: "var(--space-3)", color: "var(--color-text-muted)" }}>
          <Spinner />
          <span style={{ fontSize: "var(--font-size-lg)" }}>{th.translating}</span>
        </div>
      )}

      {status === "error" && (
        <div style={{ color: "var(--color-danger)", fontSize: "var(--font-size-base)" }}>
          {errorStatus === 503 ? th.errorModelUnavailable : error ?? th.errorGeneric}
        </div>
      )}

      {status === "success" && result && (
        <>
          <p
            style={{
              fontSize: "var(--font-size-3xl)",
              fontWeight: 700,
              color: "var(--color-text)",
              lineHeight: 1.4,
            }}
          >
            {result.sentence || th.resultPlaceholder}
          </p>
          {pct !== null && (
            <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-2)" }}>
              <span style={{ fontSize: "var(--font-size-sm)", color: "var(--color-text-muted)" }}>
                {th.confidence(pct)}
              </span>
              <div
                style={{
                  height: 6,
                  borderRadius: "var(--radius-full)",
                  background: "var(--color-border)",
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    height: "100%",
                    width: `${pct}%`,
                    background:
                      pct >= 70
                        ? "var(--color-success)"
                        : pct >= 40
                        ? "var(--color-warn)"
                        : "var(--color-danger)",
                    borderRadius: "var(--radius-full)",
                    transition: "width 0.4s ease",
                  }}
                />
              </div>
            </div>
          )}
        </>
      )}

      {status === "idle" && (
        <p
          style={{
            fontSize: "var(--font-size-3xl)",
            color: "var(--color-text-placeholder)",
            fontWeight: 300,
          }}
        >
          {th.resultPlaceholder}
        </p>
      )}
    </div>
  );
}

function Spinner() {
  return (
    <div
      style={{
        width: 20,
        height: 20,
        border: "3px solid var(--color-border)",
        borderTopColor: "var(--color-primary)",
        borderRadius: "50%",
        flexShrink: 0,
        animation: "spin 0.8s linear infinite",
      }}
    >
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
