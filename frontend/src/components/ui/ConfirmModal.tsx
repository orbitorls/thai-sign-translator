import React, { useRef } from "react";
import { useT } from "../../i18n";
import { useFocusTrap } from "../../hooks/useFocusTrap";

interface ConfirmModalProps {
  open: boolean;
  title: string;          // i18n string passed by caller
  message: string;        // i18n string passed by caller
  confirmLabel?: string;  // default: t.confirm
  cancelLabel?: string;   // default: t.cancel
  onConfirm: () => void;
  onCancel: () => void;
  danger?: boolean;       // if true, confirm button label carries glass-text-danger treatment
}

/**
 * Glass-styled, focus-trapped confirmation dialog.
 * Uses useFocusTrap — Escape key triggers onCancel.
 * Cancel button is focused first (safer for destructive confirms).
 */
export function ConfirmModal({
  open,
  title,
  message,
  confirmLabel,
  cancelLabel,
  onConfirm,
  onCancel,
  danger = false,
}: ConfirmModalProps) {
  const { t } = useT();
  const dialogRef = useRef<HTMLDivElement>(null);

  // initialFocus "first" will land on the cancel button (it is rendered first in DOM)
  useFocusTrap(dialogRef, open, onCancel, "first");

  if (!open) return null;

  const resolvedConfirmLabel = confirmLabel ?? t.confirm;
  const resolvedCancelLabel = cancelLabel ?? t.cancel;

  return (
    <div
      className="fixed inset-0 z-[70] flex items-center justify-center p-4 bg-black/40 backdrop-blur-sm"
    >
      <div
        ref={dialogRef}
        tabIndex={-1}
        role="dialog"
        aria-modal="true"
        aria-labelledby="confirm-modal-title"
        className="glass-card-strong rounded-2xl max-w-sm w-full p-6 outline-none"
      >
        <h2
          id="confirm-modal-title"
          className="font-headline-md text-headline-md text-on-surface mb-2"
        >
          {title}
        </h2>
        <p className="font-body-md text-body-md text-on-surface-variant mb-6">
          {message}
        </p>

        <div className="flex flex-col sm:flex-row gap-3">
          {/* Cancel first in DOM — receives focus first via initialFocus:"first" */}
          <button
            type="button"
            onClick={onCancel}
            className="glass-button flex-1 min-h-[44px] px-6 py-3 rounded-xl flex items-center justify-center text-on-surface transition-all duration-200 ease-out outline-none hover:brightness-110 hover:scale-105 focus-visible:ring-2 focus-visible:ring-primary/40 active:scale-95"
          >
            {resolvedCancelLabel}
          </button>

          <button
            type="button"
            onClick={onConfirm}
            className="glass-button-primary flex-1 min-h-[44px] px-6 py-3 rounded-xl flex items-center justify-center transition-all duration-200 ease-out outline-none hover:brightness-110 hover:scale-105 focus-visible:ring-2 focus-visible:ring-primary/40 active:scale-95"
          >
            <span className={danger ? "glass-text-danger" : undefined}>
              {resolvedConfirmLabel}
            </span>
          </button>
        </div>
      </div>
    </div>
  );
}
