import React, { useCallback, useRef, useState } from "react";
import { useT } from "../../i18n";
import { Toggle } from "../settings/Toggle";
import type { ConsentScope } from "../../privacy/consentStorage";
import { useFocusTrap } from "../../hooks/useFocusTrap";

interface ConsentModalProps {
  open: boolean;
  closing?: boolean;
  onClose: () => void;
  onAccept: (scopes: Partial<Record<ConsentScope, boolean>>) => Promise<void>;
  onOpenPrivacy?: () => void;
}

const OPTIONAL_SCOPES: ConsentScope[] = [
  "model_improvement",
  "video_research",
  "academic_publication",
];

export function ConsentModal({ open, closing, onClose, onAccept, onOpenPrivacy }: ConsentModalProps) {
  const { t } = useT();
  const [service, setService] = useState(false);
  const [modelImprovement, setModelImprovement] = useState(false);
  const [videoResearch, setVideoResearch] = useState(false);
  const [academic, setAcademic] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const dialogRef = useRef<HTMLDivElement>(null);

  // Block Esc when service consent is not yet granted (mandatory gate).
  // Once service is accepted the user may press Esc to dismiss.
  const stableClose = useCallback(() => onClose(), [onClose]);
  const escapeHandler = service ? stableClose : undefined;
  useFocusTrap(dialogRef, open, escapeHandler, "first");

  if (!open) return null;

  async function handleAccept() {
    if (!service) {
      setError(t.consentServiceRequired);
      return;
    }
    setSubmitting(true);
    setError("");
    try {
      await onAccept({
        service,
        model_improvement: modelImprovement,
        video_research: videoResearch,
        academic_publication: academic,
      });
      onClose();
    } catch {
      setError(t.consentSyncError);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div
      className={`glass-modal-backdrop${closing ? " is-closing" : ""}`}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 60,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "var(--space-4)",
        background: "rgba(0,0,0,0.4)",
        backdropFilter: "blur(4px)",
      }}
      role="dialog"
      aria-modal="true"
      aria-labelledby="consent-modal-title"
      aria-describedby="consent-modal-intro"
    >
      <div
        ref={dialogRef}
        tabIndex={-1}
        className="glass-card-strong glass-modal-card"
        style={{
          borderRadius: "var(--radius-lg)",
          maxWidth: 512,
          width: "100%",
          maxHeight: "90dvh",
          overflowY: "auto",
          padding: "var(--space-6)",
          outline: "none",
        }}
      >
        <h2
          id="consent-modal-title"
          style={{
            fontSize: "var(--font-size-xl)",
            fontWeight: 800,
            color: "#121c2a",
            marginBottom: "var(--space-2)",
          }}
        >
          {t.consentModalTitle}
        </h2>
        <p
          id="consent-modal-intro"
          style={{
            fontSize: "var(--font-size-sm)",
            color: "#526071",
            marginBottom: "var(--space-4)",
          }}
        >
          {t.consentModalIntro}
        </p>
        <p
          style={{
            fontSize: "var(--font-size-sm)",
            color: "#526071",
            marginBottom: "var(--space-4)",
          }}
        >
          {t.consentMinorNotice}
        </p>

        <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-4)", marginBottom: "var(--space-6)" }}>
          <ConsentRow
            id="consent-service"
            title={t.consentServiceTitle}
            desc={t.consentServiceDesc}
            required
            checked={service}
            onChange={setService}
          />
          {OPTIONAL_SCOPES.map((scope) => {
            const config = scopeConfig(scope, t);
            const checked =
              scope === "model_improvement"
                ? modelImprovement
                : scope === "video_research"
                  ? videoResearch
                  : academic;
            const onChange =
              scope === "model_improvement"
                ? setModelImprovement
                : scope === "video_research"
                  ? setVideoResearch
                  : setAcademic;
            return (
              <ConsentRow
                key={scope}
                id={`consent-${scope}`}
                title={config.title}
                desc={config.desc}
                checked={checked}
                onChange={onChange}
              />
            );
          })}
        </div>

        {error && (
          <p
            role="alert"
            style={{
              fontSize: "var(--font-size-sm)",
              color: "var(--color-danger)",
              marginBottom: "var(--space-3)",
            }}
          >
            {error}
          </p>
        )}

        <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-3)" }}>
          <button
            type="button"
            onClick={() => void handleAccept()}
            disabled={submitting}
            className="glass-button-primary"
            style={{
              width: "100%",
              minHeight: 44,
              padding: "var(--space-3) var(--space-6)",
              borderRadius: "var(--radius-md)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: "var(--space-2)",
            }}
          >
            {submitting ? t.consentSaving : t.consentAccept}
          </button>
          <button
            type="button"
            onClick={() => {
              onClose();
              onOpenPrivacy?.();
            }}
            className="glass-button"
            style={{
              width: "100%",
              minHeight: 44,
              padding: "var(--space-3) var(--space-6)",
              borderRadius: "var(--radius-md)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: "var(--space-2)",
            }}
          >
            {t.privacyPolicy}
          </button>
        </div>
      </div>
    </div>
  );
}

function scopeConfig(scope: ConsentScope, t: ReturnType<typeof useT>["t"]) {
  switch (scope) {
    case "model_improvement":
      return { title: t.consentModelTitle, desc: t.consentModelDesc };
    case "video_research":
      return { title: t.consentVideoTitle, desc: t.consentVideoDesc };
    case "academic_publication":
      return { title: t.consentAcademicTitle, desc: t.consentAcademicDesc };
    case "service":
      return { title: t.consentServiceTitle, desc: t.consentServiceDesc };
    default: {
      const _exhaustive: never = scope;
      return { title: String(_exhaustive), desc: "" };
    }
  }
}

interface ConsentRowProps {
  id: string;
  title: string;
  desc: string;
  required?: boolean;
  checked: boolean;
  onChange: (value: boolean) => void;
}

function ConsentRow({ id, title, desc, required, checked, onChange }: ConsentRowProps) {
  const { t } = useT();
  return (
    <div
      className="glass-panel"
      style={{
        borderRadius: "var(--radius-md)",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "flex-start",
        gap: "var(--space-4)",
        padding: "var(--space-4)",
      }}
    >
      <div style={{ minWidth: 0 }}>
        <h3
          style={{
            fontSize: "var(--font-size-base)",
            fontWeight: 700,
            color: "#121c2a",
          }}
        >
          {title}
          {required && (
            <span
              style={{
                marginLeft: "var(--space-2)",
                fontSize: "var(--font-size-sm)",
                color: "var(--color-danger)",
              }}
            >
              ({t.consentRequired})
            </span>
          )}
        </h3>
        <p
          style={{
            fontSize: "var(--font-size-sm)",
            color: "#526071",
            marginTop: "var(--space-1)",
          }}
        >
          {desc}
        </p>
      </div>
      <Toggle id={id} checked={checked} onChange={onChange} />
    </div>
  );
}
