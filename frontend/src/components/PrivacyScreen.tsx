import React from "react";
import { useT } from "../i18n";
import { useConsent } from "../hooks/useConsent";
import { Toggle } from "./settings/Toggle";
import type { ConsentScope } from "../privacy/consentStorage";

interface PrivacyScreenProps {
  onBack?: () => void;
}

export function PrivacyScreen({ onBack }: PrivacyScreenProps) {
  const { t } = useT();
  const { consent, setScope, withdrawAndDelete, hasScope } = useConsent();
  const [withdrawing, setWithdrawing] = React.useState(false);
  const [withdrawMsg, setWithdrawMsg] = React.useState("");

  const scopeRows: { scope: ConsentScope; title: string; desc: string; required?: boolean }[] = [
    { scope: "service", title: t.consentServiceTitle, desc: t.consentServiceDesc, required: true },
    { scope: "model_improvement", title: t.consentModelTitle, desc: t.consentModelDesc },
    { scope: "video_research", title: t.consentVideoTitle, desc: t.consentVideoDesc },
    { scope: "academic_publication", title: t.consentAcademicTitle, desc: t.consentAcademicDesc },
  ];

  async function handleDelete() {
    if (!window.confirm(t.privacyWithdrawConfirm)) return;
    setWithdrawing(true);
    setWithdrawMsg("");
    try {
      const deleted = await withdrawAndDelete();
      setWithdrawMsg(t.privacyWithdrawSuccess(deleted));
    } catch {
      setWithdrawMsg(t.privacyWithdrawError);
    } finally {
      setWithdrawing(false);
    }
  }

  const lastUpdated = consent.updatedAt ? new Date(consent.updatedAt).toLocaleString() : "—";

  return (
    <div className="screen-sheet">
      <div className="sheet-head">
        <h2 className="sheet-title">{t.privacyPolicy}</h2>
        {onBack && (
          <button type="button" className="glass-chip" onClick={onBack}>
            {t.close}
          </button>
        )}
      </div>
      <p style={{ color: "rgba(255,255,255,0.65)", marginBottom: "var(--space-4)" }}>{t.privacyPageIntro}</p>

      <div className="settings-list" style={{ marginBottom: "var(--space-5)" }}>
        <p style={{ fontSize: "var(--font-size-sm)", color: "rgba(255,255,255,0.5)" }}>
          {t.privacyLastUpdated}: {lastUpdated}
        </p>
        {scopeRows.map((row) => (
          <div key={row.scope} className="settings-row" style={{ alignItems: "flex-start" }}>
            <div style={{ flex: 1, minWidth: 0 }}>
              <span className="settings-label">
                {row.title}
                {row.required && <span style={{ color: "#fca5a5", marginLeft: 4 }}>({t.consentRequired})</span>}
              </span>
              <p style={{ fontSize: "var(--font-size-sm)", color: "rgba(255,255,255,0.55)", marginTop: 4 }}>
                {row.desc}
              </p>
            </div>
            <Toggle
              id={`privacy-${row.scope}`}
              checked={hasScope(row.scope)}
              onChange={(v) => void setScope(row.scope, v)}
            />
          </div>
        ))}
      </div>

      <p style={{ fontSize: "var(--font-size-sm)", color: "rgba(255,255,255,0.55)", marginBottom: "var(--space-3)" }}>
        {t.privacyWithdrawDisclaimer}
      </p>
      <button type="button" className="glass-action-btn" onClick={() => void handleDelete()} disabled={withdrawing}>
        {withdrawing ? t.privacyWithdrawing : t.privacyWithdrawButton}
      </button>
      {withdrawMsg && (
        <p role="status" style={{ marginTop: "var(--space-3)", fontSize: "var(--font-size-sm)" }}>
          {withdrawMsg}
        </p>
      )}

      <article style={{ marginTop: "var(--space-6)", display: "flex", flexDirection: "column", gap: "var(--space-4)" }}>
        {[
          [t.privacySectionCollection, t.privacyCollectionBody],
          [t.privacySectionConsent, t.privacyConsentBody],
          [t.privacySectionWithdraw, t.privacyWithdrawBody],
          [t.privacySectionRetention, t.privacyRetentionBody],
        ].map(([title, body]) => (
          <section key={title}>
            <h3 style={{ fontWeight: 700, marginBottom: "var(--space-2)" }}>{title}</h3>
            <p style={{ fontSize: "var(--font-size-sm)", color: "rgba(255,255,255,0.65)", whiteSpace: "pre-line" }}>
              {body}
            </p>
          </section>
        ))}
      </article>
    </div>
  );
}
