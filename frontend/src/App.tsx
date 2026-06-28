import React, { useEffect, useRef, useState } from "react";
import { TranslateResult } from "./api/client";
import { ModelsProvider, useModels } from "./hooks/ModelsProvider";
import { SettingsProvider } from "./hooks/SettingsProvider";
import { useHolisticCapture } from "./hooks/useHolisticCapture";
import { useTranslate } from "./hooks/useTranslate";
import { CameraView } from "./components/CameraView";
import { ResultCard } from "./components/ResultCard";
import { SupportedPhrases } from "./components/SupportedPhrases";
import { th } from "./i18n/th";

const CONFIDENCE_FLOOR = 0.3;
const HANDS_GONE_DEBOUNCE_MS = 800;
const MIN_HAND_FRAMES = 6;
const MIN_TOTAL_FRAMES = 8;

function Translator() {
  const { models, selectedModelId, loading: modelsLoading, error: modelsError, setSelectedModelId } = useModels();
  const [showLandmarks, setShowLandmarks] = useState(false);
  const capture = useHolisticCapture({ overlayEnabled: showLandmarks });
  const translator = useTranslate();
  const selectedModel = models.find((m) => m.id === selectedModelId);
  const [phrasesOpen, setPhrasesOpen] = useState(false);
  const [displayedResult, setDisplayedResult] = useState<TranslateResult | null>(null);

  // Stable refs for interval/timeout callbacks — always point at current values.
  const captureRef = useRef(capture);
  captureRef.current = capture;
  const translatorRef = useRef(translator);
  translatorRef.current = translator;
  const selectedModelIdRef = useRef(selectedModelId);
  selectedModelIdRef.current = selectedModelId;
  const debounceRef = useRef<ReturnType<typeof setTimeout>>();

  // Auto-start collection as soon as camera is ready.
  useEffect(() => {
    if (!capture.ready) return;
    captureRef.current.start();
  }, [capture.ready]);

  // Presence-based segmentation: translate when hands disappear for 0.8 s.
  useEffect(() => {
    if (!capture.ready) return;

    if (capture.handsPresent) {
      // Hands appeared — cancel pending translate trigger and ensure we're collecting.
      clearTimeout(debounceRef.current);
      if (!captureRef.current.recording) {
        captureRef.current.start();
      }
    } else {
      // Hands gone — schedule translate after debounce.
      clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => {
        const trans = translatorRef.current;
        if (trans.status === "loading") return;
        const { frames, handFrameCount } = captureRef.current.stop();
        captureRef.current.start(); // restart immediately for next sign
        if (handFrameCount >= MIN_HAND_FRAMES && frames.length >= MIN_TOTAL_FRAMES) {
          trans.run(frames, selectedModelIdRef.current ?? undefined);
        }
      }, HANDS_GONE_DEBOUNCE_MS);
    }

    return () => clearTimeout(debounceRef.current);
  }, [capture.handsPresent, capture.ready]);

  // Update displayed result only when confidence clears the floor.
  useEffect(() => {
    if (translator.status === "success" && translator.result) {
      if (translator.result.score >= CONFIDENCE_FLOOR) {
        setDisplayedResult(translator.result);
      }
    }
  }, [translator.status, translator.result]);

  // Auto-reset error after 3 s so the loop can continue.
  useEffect(() => {
    if (translator.status !== "error") return;
    const id = setTimeout(() => translatorRef.current.reset(), 3000);
    return () => clearTimeout(id);
  }, [translator.status]);

  const hasCameraError = Boolean(capture.cameraError);

  // Derive a status for the result card that uses displayedResult instead of
  // translator.result so low-confidence windows don't replace the displayed text.
  const resultStatus = translator.status === "success"
    ? (displayedResult && displayedResult === translator.result ? "success" : "loading")
    : translator.status;

  return (
    <main className="app-immersive">
      {/* Full-screen camera background */}
      <CameraView videoRef={capture.videoRef} overlayRef={capture.overlayRef} />

      {/* ── Top glass bar ── */}
      <div className="glass-top-bar">
        <div className="brand-glass">
          <div className="brand-mark-glass">TS</div>
          <span className="brand-name-glass">{th.appTitle}</span>
        </div>

        <div className="top-controls-right">
          <span className="live-chip">
            <span
              className={[
                "live-dot",
                !capture.ready ? "offline" : "",
                capture.ready && capture.handsPresent ? "hands" : "",
              ]
                .filter(Boolean)
                .join(" ")}
            />
            {capture.ready ? "กล้อง Live" : "กำลังเปิด..."}
          </span>

          {!modelsLoading && selectedModel && (
            <span className="glass-chip" style={{ cursor: "default" }}>
              {selectedModel.label_th}
            </span>
          )}

          <button
            className={`glass-chip${showLandmarks ? " active" : ""}`}
            onClick={() => setShowLandmarks((v) => !v)}
            aria-pressed={showLandmarks}
            title="แสดง/ซ่อนเส้นโครงร่าง MediaPipe Holistic"
          >
            ✦ เส้นโครงร่าง
          </button>

          <button
            className="glass-chip"
            onClick={() => setPhrasesOpen((v) => !v)}
            aria-expanded={phrasesOpen}
          >
            {th.supportedPhrasesTitle}
          </button>
        </div>
      </div>

      {/* ── Camera permission error ── */}
      {hasCameraError && (
        <div className="glass-camera-error">
          <p style={{ color: "#fca5a5", fontWeight: 700, fontSize: "var(--font-size-base)" }}>
            {th.cameraError}
          </p>
          <p style={{ color: "rgba(255,255,255,0.65)", fontSize: "var(--font-size-sm)" }}>
            {th.cameraErrorHint}
          </p>
          <button className="glass-action-btn" onClick={() => captureRef.current.start()}>
            {th.cameraRetry}
          </button>
        </div>
      )}

      {/* ── Always-visible live translation panel ── */}
      <div className="result-glass-panel live">
        <LiveStatusRow
          cameraReady={capture.ready}
          translating={translator.status === "loading"}
          hasResult={Boolean(displayedResult)}
          hasError={translator.status === "error"}
          modelsError={modelsError}
        />
        <ResultCard
          status={resultStatus}
          result={displayedResult}
          error={translator.error}
          errorStatus={translator.errorStatus}
          variant="glass"
        />
      </div>

      {/* ── Supported phrases glass panel (slides up) ── */}
      <div
        className={`result-glass-panel${phrasesOpen ? " open" : ""}`}
        style={{ zIndex: 30 }}
        aria-hidden={!phrasesOpen}
      >
        <div className="glass-panel-handle" />
        <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: "var(--space-3)" }}>
          <button className="glass-chip" onClick={() => setPhrasesOpen(false)} aria-label="ปิด">
            ✕ ปิด
          </button>
        </div>
        <SupportedPhrases glass />
      </div>
    </main>
  );
}

interface LiveStatusRowProps {
  cameraReady: boolean;
  translating: boolean;
  hasResult: boolean;
  hasError: boolean;
  modelsError: string | null;
}

function LiveStatusRow({ cameraReady, translating, hasResult, hasError, modelsError }: LiveStatusRowProps) {
  let label = "";
  let color = "rgba(255,255,255,0.5)";

  if (!cameraReady) {
    label = th.cameraInit;
  } else if (modelsError) {
    label = th.modelLoadError;
    color = "#fcd34d";
  } else if (hasError) {
    label = "";
  } else if (translating) {
    label = th.translating;
  } else if (!hasResult) {
    label = "แสดงภาษามือต่อกล้อง";
  }

  if (!label) return null;

  return (
    <p
      style={{
        fontSize: "var(--font-size-sm)",
        color,
        marginBottom: "var(--space-2)",
        fontWeight: 500,
      }}
    >
      {label}
    </p>
  );
}

export default function App() {
  return (
    <ModelsProvider>
      <SettingsProvider>
        <Translator />
      </SettingsProvider>
    </ModelsProvider>
  );
}
