import React, { useEffect, useState } from "react";
import { getSupportedPhrases, SupportedPhrasesResult } from "../api/client";
import { th } from "../i18n/th";

/**
 * SupportedPhrases — collapsible panel showing which phrases the model knows.
 *
 * The closed-vocab TSL-51 model only recognises the ~252 sentences it was
 * trained on. Showing this list prevents users from signing arbitrary
 * phrases and concluding the model is broken.
 */
export function SupportedPhrases() {
  const [data, setData] = useState<SupportedPhrasesResult | null>(null);
  const [error, setError] = useState(false);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    getSupportedPhrases()
      .then(setData)
      .catch(() => setError(true));
  }, []);

  const count = data?.total ?? 0;

  return (
    <section
      style={{
        background: "var(--color-primary-light)",
        border: "1px solid var(--color-primary)",
        borderRadius: "var(--radius-md)",
        overflow: "hidden",
      }}
    >
      {/* Header / toggle */}
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        style={{
          width: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "var(--space-3) var(--space-4)",
          background: "none",
          border: "none",
          cursor: "pointer",
          color: "var(--color-primary)",
          fontFamily: "var(--font-family)",
          fontSize: "var(--font-size-sm)",
          fontWeight: 600,
          textAlign: "left",
        }}
        aria-expanded={open}
      >
        <span>
          {th.supportedPhrasesTitle}
          {count > 0 && (
            <span
              style={{
                marginLeft: "var(--space-2)",
                background: "var(--color-primary)",
                color: "#fff",
                borderRadius: "var(--radius-full)",
                padding: "0 var(--space-2)",
                fontSize: "var(--font-size-xs)",
                fontWeight: 700,
              }}
            >
              {th.supportedPhrasesCount(count)}
            </span>
          )}
        </span>
        <span aria-hidden="true" style={{ fontSize: "var(--font-size-base)" }}>
          {open ? "▲" : "▼"}
        </span>
      </button>

      {/* Scope note — always visible */}
      <p
        style={{
          padding: "0 var(--space-4) var(--space-2)",
          fontSize: "var(--font-size-xs)",
          color: "var(--color-text-muted)",
        }}
      >
        {th.supportedPhrasesScope}
      </p>

      {/* Collapsed body */}
      {open && (
        <div
          style={{
            padding: "var(--space-2) var(--space-4) var(--space-4)",
            borderTop: "1px solid var(--color-primary)",
          }}
        >
          {error && (
            <p style={{ color: "var(--color-danger)", fontSize: "var(--font-size-sm)" }}>
              {th.supportedPhrasesUnavailable}
            </p>
          )}
          {!error && data && data.phrases.length === 0 && (
            <p style={{ color: "var(--color-text-muted)", fontSize: "var(--font-size-sm)" }}>
              {data.note || th.supportedPhrasesEmpty}
            </p>
          )}
          {!error && data && data.phrases.length > 0 && (
            <ul
              style={{
                listStyle: "none",
                display: "flex",
                flexWrap: "wrap",
                gap: "var(--space-2)",
                maxHeight: 200,
                overflowY: "auto",
              }}
            >
              {data.phrases.map((phrase) => (
                <li
                  key={phrase}
                  style={{
                    background: "var(--color-surface)",
                    border: "1px solid var(--color-border)",
                    borderRadius: "var(--radius-full)",
                    padding: "var(--space-1) var(--space-3)",
                    fontSize: "var(--font-size-sm)",
                    color: "var(--color-text)",
                    whiteSpace: "nowrap",
                  }}
                >
                  {phrase}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </section>
  );
}
