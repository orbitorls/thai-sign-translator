import React, { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { useT } from "../../i18n";
import { Toggle } from "../settings/Toggle";
import type { ConsentScope } from "../../privacy/consentStorage";

interface ConsentModalProps {
  open: boolean;
  onClose: () => void;
  onAccept: (scopes: Partial<Record<ConsentScope, boolean>>) => Promise<void>;
}

const OPTIONAL_SCOPES: ConsentScope[] = [
  "model_improvement",
  "video_research",
  "academic_publication",
];

const FOCUSABLE_SELECTOR =
  'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])';

export function ConsentModal({ open, onClose, onAccept }: ConsentModalProps) {
  const { t } = useT();
  const [service, setService] = useState(false);
  const [modelImprovement, setModelImprovement] = useState(false);
  const [videoResearch, setVideoResearch] = useState(false);
  const [academic, setAcademic] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const dialogRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const dialog = dialogRef.current;
    if (!dialog) return;

    // Focus the dialog container so the modal is announced and Tab starts inside.
    dialog.focus();

    function handleKey(e: KeyboardEvent) {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
        return;
      }
      if (e.key === "Tab" && dialog) {
        const els = Array.from(
          dialog.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR),
        ).filter((el) => !el.hasAttribute("disabled"));
        if (els.length === 0) return;
        const firstEl = els[0];
        const lastEl = els[els.length - 1];
        if (e.shiftKey && document.activeElement === firstEl) {
          e.preventDefault();
          lastEl.focus();
        } else if (!e.shiftKey && document.activeElement === lastEl) {
          e.preventDefault();
          firstEl.focus();
        }
      }
    }
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [open, onClose]);

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
      className="fixed inset-0 z-[60] flex items-center justify-center p-4 bg-black/40 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-labelledby="consent-modal-title"
    >
      <div
        ref={dialogRef}
        tabIndex={-1}
        className="glass-card-strong rounded-2xl max-w-lg w-full max-h-[90vh] overflow-y-auto p-6 outline-none"
      >
        <h2 id="consent-modal-title" className="font-headline-md text-headline-md text-on-surface mb-2">
          {t.consentModalTitle}
        </h2>
        <p className="font-body-md text-body-md text-on-surface-variant mb-4">{t.consentModalIntro}</p>
        <p className="font-label-md text-label-md text-on-surface-variant mb-4">{t.consentMinorNotice}</p>

        <div className="flex flex-col gap-4 mb-6">
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
          <p className="font-label-md text-error mb-3" role="alert">
            {error}
          </p>
        )}

        <div className="flex flex-col sm:flex-row gap-3">
          <button
            type="button"
            onClick={() => void handleAccept()}
            disabled={submitting}
            className="glass-button-primary flex-1 min-h-[44px] px-6 py-3 rounded-xl gap-2 flex items-center justify-center transition-all duration-200 ease-out outline-none hover:brightness-110 hover:scale-105 focus-visible:ring-2 focus-visible:ring-primary/40 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
          >
            {submitting ? t.consentSaving : t.consentAccept}
          </button>
          <Link
            to="/privacy"
            className="glass-button flex-1 min-h-[44px] px-6 py-3 text-center rounded-xl gap-2 flex items-center justify-center text-on-surface transition-all duration-200 ease-out outline-none hover:brightness-110 hover:scale-105 focus-visible:ring-2 focus-visible:ring-primary/40 active:scale-95"
            onClick={onClose}
          >
            {t.privacyPolicy}
          </Link>
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
    <div className="glass-panel rounded-xl flex justify-between items-start gap-4 p-4">
      <div className="min-w-0">
        <h3 className="font-label-lg text-label-lg text-on-surface">
          {title}
          {required && (
            <span className="ml-2 text-error font-label-md">({t.consentRequired})</span>
          )}
        </h3>
        <p className="font-body-md text-body-md text-on-surface-variant text-sm mt-1">{desc}</p>
      </div>
      <Toggle id={id} checked={checked} onChange={onChange} />
    </div>
  );
}
