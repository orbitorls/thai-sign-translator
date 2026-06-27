import React, { useEffect, useState } from "react";
import { getSupportedPhrases, SupportedPhrasesResult } from "../api/client";
import { th } from "../i18n/th";

interface SupportedPhrasesProps {
  glass?: boolean;
}

export function SupportedPhrases({ glass }: SupportedPhrasesProps) {
  const [data, setData] = useState<SupportedPhrasesResult | null>(null);
  const [error, setError] = useState(false);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    getSupportedPhrases()
      .then(setData)
      .catch(() => setError(true));
  }, []);

  const count = data?.total ?? 0;

  // Glass variant: flat list, no nested toggle (parent panel handles open/close)
  if (glass) {
    return (
      <section>
        <p style={{ color: "rgba(255,255,255,0.55)", fontSize: "var(--font-size-xs)", marginBottom: "var(--space-3)" }}>
          {th.supportedPhrasesScope}
          {count > 0 && (
            <span style={{ marginLeft: "var(--space-2)", color: "rgba(255,255,255,0.75)", fontWeight: 600 }}>
              · {th.supportedPhrasesCount(count)}
            </span>
          )}
        </p>
        {error && (
          <p style={{ color: "#fca5a5", fontSize: "var(--font-size-sm)" }}>
            {th.supportedPhrasesUnavailable}
          </p>
        )}
        {!error && data && data.phrases.length === 0 && (
          <p style={{ color: "rgba(255,255,255,0.5)", fontSize: "var(--font-size-sm)" }}>
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
              maxHeight: "45dvh",
              overflowY: "auto",
            }}
          >
            {data.phrases.map((phrase) => (
              <li
                key={phrase}
                style={{
                  background: "rgba(255,255,255,0.12)",
                  border: "1px solid rgba(255,255,255,0.2)",
                  borderRadius: "var(--radius-full)",
                  padding: "var(--space-1) var(--space-3)",
                  fontSize: "var(--font-size-sm)",
                  color: "#fff",
                  whiteSpace: "nowrap",
                }}
              >
                {phrase}
              </li>
            ))}
          </ul>
        )}
      </section>
    );
  }

  // Default variant: collapsible card (used in legacy desktop layout if ever restored)
  return (
    <section
      style={{
        background: "var(--color-primary-light)",
        border: "1px solid var(--color-primary)",
        borderRadius: "var(--radius-md)",
        overflow: "hidden",
      }}
    >
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

      <p style={{ padding: "0 var(--space-4) var(--space-2)", fontSize: "var(--font-size-xs)", color: "var(--color-text-muted)" }}>
        {th.supportedPhrasesScope}
      </p>

      {open && (
        <div style={{ padding: "var(--space-2) var(--space-4) var(--space-4)", borderTop: "1px solid var(--color-primary)" }}>
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
