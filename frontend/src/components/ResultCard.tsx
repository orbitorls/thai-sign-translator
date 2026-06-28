import React, { useEffect, useState } from "react";
import {
  TranslateResult,
  submitCorrection,
  ApiError,
  feedbackErrorMessage,
  type CaptureQualityPayload,
} from "../api/client";
import { useI18n } from "../i18n";
import { useSpeech } from "../hooks/useSpeech";
import { useConsent } from "../hooks/useConsent";

interface ResultCardProps {
  status: "idle" | "loading" | "success" | "error";
  result: TranslateResult | null;
  error: string | null;
  errorStatus: number | null;
  variant?: "glass";
  lastFrames?: number[][][] | null;
  captureQuality?: CaptureQualityPayload | null;
  onRetry?: () => void;
}

type CorrectionState = "idle" | "editing" | "submitting" | "saved" | "error";

// Change 6: Warning code → i18n mapping
const WARNING_MAP: Record<string, keyof ReturnType<typeof useI18n>> = {
  low_light:         "warningLowLight",
  no_hands:          "warningNoHands",
  low_quality:       "warningLowQuality",
  low_confidence:    "warningLowConfidence",
  partial_occlusion: "warningPartialOcclusion",
};

function mapWarning(code: string, th: ReturnType<typeof useI18n>): string {
  const key = WARNING_MAP[code];
  return key ? (th[key] as string) : code;
}

export function ResultCard({
  status,
  result,
  error,
  errorStatus,
  variant,
  lastFrames,
  captureQuality,
  onRetry,
}: ResultCardProps) {
  const th = useI18n();
  const { speak, speaking, supported } = useSpeech();
  const { hasScope } = useConsent();
  const glass = variant === "glass";
  const pct = result ? Math.round(result.score * 100) : null;

  const [correctionState, setCorrectionState] = useState<CorrectionState>("idle");
  const [editedText, setEditedText] = useState("");
  const [correctionErrorMsg, setCorrectionErrorMsg] = useState("");

  const textColor = glass ? "#fff" : "var(--color-text)";
  const mutedColor = glass ? "rgba(255,255,255,0.6)" : "var(--color-text-muted)";
  const placeholderColor = glass ? "rgba(255,255,255,0.28)" : "var(--color-text-placeholder)";
  const trackColor = glass ? "rgba(255,255,255,0.15)" : "var(--color-border)";

  const showSpeaker = supported && Boolean(result?.sentence);
  const canCorrect = hasScope("model_improvement") && lastFrames && lastFrames.length > 0;

  useEffect(() => {
    setCorrectionState("idle");
    setEditedText("");
    setCorrectionErrorMsg("");
  }, [result?.sentence, result?.model]);

  async function handleSubmitCorrection() {
    if (!result || !lastFrames) return;
    const text = editedText.trim();
    if (!text) return;
    setCorrectionState("submitting");
    setCorrectionErrorMsg("");
    try {
      await submitCorrection({
        frames: lastFrames,
        predictedText: result.sentence,
        correctedText: text,
        model: result.model,
        captureQuality: captureQuality ?? undefined,
      });
      setCorrectionState("saved");
    } catch (err) {
      setCorrectionState("error");
      const code = err instanceof ApiError ? err.status : 0;
      setCorrectionErrorMsg(feedbackErrorMessage(code, th, "correction"));
    }
  }

  return (
    // Change 1a: Remove aria-live/aria-atomic from outer div
    <div
      style={{
        position: "relative",
        minHeight: glass ? 72 : 120,
        display: "flex",
        flexDirection: "column",
        gap: "var(--space-3)",
      }}
    >
      {showSpeaker && (
        <button
          type="button"
          className={`result-speaker-btn${speaking ? " playing" : ""}`}
          onClick={() => speak(result!.sentence)}
          aria-label={th.ariaSpeak}
        >
          <svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="2">
            <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
            <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
          </svg>
        </button>
      )}

      {status === "loading" && !result && (
        <div style={{ display: "flex", alignItems: "center", gap: "var(--space-3)", color: mutedColor }}>
          <Spinner glass={glass} />
          <span>{th.translating}</span>
        </div>
      )}

      {/* Change 5a: Replace hardcoded hex with token class */}
      {status === "error" && (
        <div className={glass ? "glass-text-danger" : ""} style={{ color: glass ? undefined : "var(--color-danger)" }}>
          {errorStatus === 503 ? th.errorModelUnavailable : error ?? th.errorGeneric}
          {onRetry && (
            <button type="button" className="glass-chip" style={{ marginLeft: "var(--space-2)" }} onClick={onRetry}>
              {th.retry}
            </button>
          )}
        </div>
      )}

      {(status === "success" || (status === "loading" && result)) && result && (
        <>
          {/* Change 1b: aria-live scoped to sentence only */}
          <p
            key={result.sentence}
            aria-live="polite"
            aria-atomic="true"
            className="result-sentence"
            style={{ fontSize: "var(--font-size-3xl)", fontWeight: 700, color: textColor, lineHeight: 1.4, wordBreak: "break-word" }}
          >
            {result.sentence || th.resultPlaceholder}
          </p>
          {/* Change 7: Updating dot when live re-translation is in progress */}
          {status === "loading" && result && (
            <span
              aria-live="off"
              aria-hidden="true"
              style={{
                display: "inline-block",
                width: 8,
                height: 8,
                borderRadius: "50%",
                background: "rgba(255,255,255,0.6)",
                animation: "pulse 1.2s ease-in-out infinite",
                marginTop: "var(--space-1)",
              }}
            />
          )}
          {/* Change 1c: aria-hidden on confidence section */}
          {pct !== null && (
            <div aria-hidden="true" style={{ display: "flex", flexDirection: "column", gap: "var(--space-2)" }}>
              <span style={{ fontSize: "var(--font-size-sm)", color: mutedColor }}>{th.confidence(pct)}</span>
              <ConfidenceBar pct={pct} trackColor={trackColor} />
            </div>
          )}
          {/* Change 5b + 6: Replace hex with token class and use i18n mapping */}
          {result.warning && (
            <p className="glass-text-warn" style={{ fontSize: "var(--font-size-sm)" }}>
              {mapWarning(result.warning, th)}
            </p>
          )}
        </>
      )}

      {status === "idle" && (
        <p style={{ fontSize: "var(--font-size-3xl)", color: placeholderColor, fontWeight: 300 }}>
          {th.resultPlaceholder}
        </p>
      )}

      {status === "success" && result && canCorrect && correctionState === "idle" && (
        <button type="button" className="glass-chip" onClick={() => { setEditedText(result.sentence); setCorrectionState("editing"); }}>
          {th.editTranslation}
        </button>
      )}

      {correctionState === "editing" && (
        <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-2)" }}>
          {/* Change 2: Add aria-label and aria-describedby to correction input */}
          <input
            type="text"
            id="correction-input"
            aria-label={th.editTranslation}
            aria-describedby="correction-error"
            value={editedText}
            onChange={(e) => setEditedText(e.target.value)}
            style={{
              padding: "var(--space-2)",
              borderRadius: "var(--radius-sm)",
              border: "1px solid rgba(255,255,255,0.25)",
              background: "rgba(0,0,0,0.3)",
              color: "#fff",
              fontFamily: "var(--font-family)",
            }}
          />
          <div style={{ display: "flex", gap: "var(--space-2)" }}>
            <button type="button" className="glass-action-btn" onClick={() => void handleSubmitCorrection()}>
              {th.submitCorrection}
            </button>
            <button type="button" className="glass-chip" onClick={() => setCorrectionState("idle")}>
              {th.cancel}
            </button>
          </div>
        </div>
      )}

      {correctionState === "submitting" && <p style={{ fontSize: "var(--font-size-sm)", color: mutedColor }}>{th.correctionSubmitting}</p>}
      {/* Change 3: Add role="status" and glass-text-success class to saved confirmation */}
      {correctionState === "saved" && (
        <p role="status" style={{ fontSize: "var(--font-size-sm)" }} className="glass-text-success">
          {th.correctionSaved}
        </p>
      )}
      {/* Change 4: Add id, role="alert", and glass-text-danger class to error message */}
      {correctionState === "error" && (
        <p id="correction-error" role="alert" style={{ fontSize: "var(--font-size-sm)" }} className="glass-text-danger">
          {correctionErrorMsg}
        </p>
      )}
      {!hasScope("model_improvement") && status === "success" && (
        <p style={{ fontSize: "var(--font-size-xs)", color: mutedColor }}>{th.correctionNeedsOptIn}</p>
      )}
    </div>
  );
}

function ConfidenceBar({ pct, trackColor }: { pct: number; trackColor: string }) {
  return (
    <div style={{ height: 6, borderRadius: "var(--radius-full)", background: trackColor, overflow: "hidden" }}>
      <div
        style={{
          height: "100%",
          width: `${pct}%`,
          // Change 5c: Replace hardcoded hex colors with CSS token variables
          background:
            pct >= 70 ? "var(--glass-success)"
            : pct >= 40 ? "var(--glass-warn)"
            : "var(--glass-danger)",
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
        animation: "spin 0.8s linear infinite",
      }}
    />
  );
}
