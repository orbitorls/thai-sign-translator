import React, { useState } from "react";
import { useI18n } from "../i18n";

export function TextToSignPanel() {
  const th = useI18n();
  const [text, setText] = useState("");
  const [open, setOpen] = useState(false);

  if (!open) {
    return (
      <button type="button" className="glass-chip" onClick={() => setOpen(true)}>
        {th.textToSign}
      </button>
    );
  }

  return (
    <div
      style={{
        padding: "var(--space-3)",
        borderRadius: "var(--radius-md)",
        background: "rgba(255,255,255,0.08)",
        border: "1px solid rgba(255,255,255,0.15)",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "var(--space-2)" }}>
        <span style={{ fontWeight: 700, fontSize: "var(--font-size-sm)" }}>{th.textToSign}</span>
        <button type="button" className="glass-chip" onClick={() => setOpen(false)} aria-label={th.close}>
          ✕
        </button>
      </div>
      <input
        type="text"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder={th.textToSignPlaceholder}
        className="glass-input"
        style={{
          width: "100%",
          padding: "var(--space-3)",
          borderRadius: "var(--radius-sm)",
          border: "1px solid rgba(255,255,255,0.2)",
          background: "rgba(0,0,0,0.25)",
          color: "#fff",
          fontFamily: "var(--font-family)",
        }}
      />
      <p style={{ marginTop: "var(--space-2)", fontSize: "var(--font-size-xs)", color: "rgba(255,255,255,0.55)" }}>
        {th.textToSignHint}
      </p>
    </div>
  );
}
