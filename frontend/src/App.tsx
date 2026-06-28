import React, { useEffect, useRef, useState } from "react";
import { TranslateResult, type CaptureQualityPayload } from "./api/client";
import { ModelsProvider, useModels } from "./hooks/ModelsProvider";
import { SettingsProvider, useSettings } from "./hooks/SettingsProvider";
import { HistoryProvider, useHistory } from "./hooks/HistoryProvider";
import { useHolisticCapture, type QualityWarning } from "./hooks/useHolisticCapture";
import { useTranslate } from "./hooks/useTranslate";
import { useConsent } from "./hooks/useConsent";
import { useI18n } from "./i18n";
import { CameraView } from "./components/CameraView";
import { ResultCard } from "./components/ResultCard";
import { SupportedPhrases } from "./components/SupportedPhrases";
import { BottomNav, Screen } from "./components/BottomNav";
import { HistoryScreen } from "./components/HistoryScreen";
import { SettingsScreen } from "./components/SettingsScreen";
import { TeachScreen } from "./components/TeachScreen";
import { DictionaryScreen } from "./components/DictionaryScreen";
import { ModelPicker } from "./components/ModelPicker";
import { TextToSignPanel } from "./components/TextToSignPanel";
import { ConsentModal } from "./components/privacy/ConsentModal";
import { MOCKUP_MODE } from "./mockup";

const LIVE_INTERVAL_MS = 900;
const LIVE_WINDOW_MS = 2000;
const MIN_LIVE_FRAMES = 12;
const CALIBRATION_MS = 2000;
const HARD_BLOCK_WARNING_MS = 900;
const MIN_VISIBLE_SCORE = 0.55;
const MIN_VISIBLE_LANDMARK_QUALITY = 0.45;

function qualityWarningMessage(
  warning: QualityWarning,
  th: ReturnType<typeof useI18n>,
  diagnosticsEnabled: boolean
): string {
  switch (warning) {
    case "no_hands":   return th.qualityNoHands;   // always shown
    case "low_light":  return th.qualityLowLight;  // always shown
    case "low_fps":    return diagnosticsEnabled ? th.qualityLowFps : "";
    case "motion_blur": return diagnosticsEnabled ? th.qualityMotionBlur : "";
    default:           return "";
  }
}

function AppShell() {
  const th = useI18n();
  const { settings, showLandmarks } = useSettings();
  const { models, selectedModelId, loading: modelsLoading, error: modelsError } = useModels();
  const history = useHistory();
  const { needsConsentModal, hasScope, completeModal } = useConsent();
  const capture = useHolisticCapture({
    overlayEnabled: showLandmarks,
    facingMode: settings.cameraFacing,
  });
  const translator = useTranslate();
  const selectedModel = models.find((m) => m.id === selectedModelId);

  const [screen, setScreen] = useState<Screen>("camera");
  const [phrasesOpen, setPhrasesOpen] = useState(false);
  const phrasesCloseRef = useRef<HTMLButtonElement>(null);
  const [showConsent, setShowConsent] = useState(needsConsentModal);
  const [displayedResult, setDisplayedResult] = useState<TranslateResult | null>(null);
  const [lastFrames, setLastFrames] = useState<number[][][] | null>(null);
  const [lastCaptureQuality, setLastCaptureQuality] = useState<CaptureQualityPayload | null>(null);
  const [warningWindow, setWarningWindow] = useState<{
    warning: Exclude<QualityWarning, null>;
    startedAt: number;
  } | null>(null);
  const [hardBlocked, setHardBlocked] = useState(false);

  const pendingRef = useRef(false);
  const lastSavedRef = useRef("");
  const captureRef = useRef(capture);
  const translatorRef = useRef(translator);
  const selectedModelIdRef = useRef(selectedModelId);
  const historyRef = useRef(history);
  const captureQualityRef = useRef(capture.quality);
  const frameCountRef = useRef(capture.frameCount);
  const featureSchemaRef = useRef(capture.featureSchema);
  const cameraFacingRef = useRef(settings.cameraFacing);

  captureRef.current = capture;
  translatorRef.current = translator;
  selectedModelIdRef.current = selectedModelId;
  historyRef.current = history;
  captureQualityRef.current = capture.quality;
  frameCountRef.current = capture.frameCount;
  featureSchemaRef.current = capture.featureSchema;
  cameraFacingRef.current = settings.cameraFacing;

  const effectiveHardBlocked = MOCKUP_MODE ? false : hardBlocked;

  useEffect(() => {
    pendingRef.current = translator.pending;
  }, [translator.pending]);

  useEffect(() => {
    setShowConsent(needsConsentModal);
  }, [needsConsentModal]);

  // Auto-start capture on camera tab when ready.
  useEffect(() => {
    if (!capture.ready || screen !== "camera") return;
    if (!hasScope("service")) return;
    if (!capture.recording) captureRef.current.start();
  }, [capture.ready, screen, hasScope, capture.recording]);

  // Stop recording when leaving camera tab.
  useEffect(() => {
    if (screen === "camera") return;
    if (captureRef.current.recording) {
      captureRef.current.stop();
      translatorRef.current.cancel({ keepResult: true });
    }
  }, [screen]);

  useEffect(() => {
    const warning = capture.quality.lastWarning;
    const isCoreWarning = warning === "no_hands" || warning === "low_light";
    if (!warning || (!settings.diagnosticsEnabled && !isCoreWarning)) {
      setWarningWindow(null);
      return;
    }
    setWarningWindow((prev) =>
      prev?.warning === warning ? prev : { warning, startedAt: window.performance.now() }
    );
  }, [capture.quality.lastWarning, settings.diagnosticsEnabled]);

  useEffect(() => {
    if (!warningWindow) {
      setHardBlocked(false);
      return;
    }
    const elapsed = window.performance.now() - warningWindow.startedAt;
    if (elapsed >= HARD_BLOCK_WARNING_MS) {
      setHardBlocked(true);
      return;
    }
    setHardBlocked(false);
    const timer = window.setTimeout(() => setHardBlocked(true), HARD_BLOCK_WARNING_MS - elapsed);
    return () => window.clearTimeout(timer);
  }, [warningWindow]);

  // Live interval translate on camera tab.
  useEffect(() => {
    if (screen !== "camera" || !capture.recording || !capture.ready) return;
    if (!hasScope("service")) return;

    const tick = () => {
      if (pendingRef.current || effectiveHardBlocked) return;
      if (!MOCKUP_MODE && frameCountRef.current < MIN_LIVE_FRAMES) return;
      const frames = captureRef.current.getRecentFrames(LIVE_WINDOW_MS);
      if (frames.length < MIN_LIVE_FRAMES && !MOCKUP_MODE) return;

      const quality = captureQualityRef.current;
      setLastFrames(frames.map((f) => f.map((lm) => [...lm])));
      setLastCaptureQuality({
        fps: quality.fps,
        lighting_ok: quality.lightingOK,
        hand_present: quality.handPresent,
        warning: quality.lastWarning,
        feature_schema: featureSchemaRef.current,
        camera_facing: cameraFacingRef.current,
      });

      void translatorRef.current.run(frames, {
        model: selectedModelIdRef.current ?? undefined,
        featureSchema: featureSchemaRef.current,
      });
    };

    const startAt = window.performance.now();
    const gatedTick = () => {
      if (window.performance.now() - startAt < CALIBRATION_MS) return;
      tick();
    };
    gatedTick();
    const id = window.setInterval(gatedTick, LIVE_INTERVAL_MS);
    return () => window.clearInterval(id);
  }, [screen, capture.recording, capture.ready, effectiveHardBlocked, hasScope]);

  useEffect(() => {
    if (translator.status === "success" && translator.result) {
      const scoreOk = translator.result.score >= MIN_VISIBLE_SCORE;
      const qualityOk = (translator.result.landmark_quality ?? 1) >= MIN_VISIBLE_LANDMARK_QUALITY;
      const hasWarning = Boolean(translator.result.warning);
      if (scoreOk && qualityOk && !hasWarning) {
        setDisplayedResult(translator.result);
        const saveKey = `${translator.result.model}\n${translator.result.sentence}`;
        if (translator.result.sentence && saveKey !== lastSavedRef.current) {
          lastSavedRef.current = saveKey;
          historyRef.current.add({
            sentence: translator.result.sentence,
            score: translator.result.score,
            model: translator.result.model,
          });
        }
      }
    }
  }, [translator.status, translator.result]);

  useEffect(() => {
    if (!phrasesOpen) return;
    phrasesCloseRef.current?.focus();
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setPhrasesOpen(false);
    };
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [phrasesOpen]);

  const visibleResult = displayedResult ?? translator.result;
  const resultStatus =
    translator.pending && visibleResult
      ? "loading"
      : visibleResult
        ? "success"
        : translator.status;

  const qualityMsg =
    capture.quality.lastWarning
      ? qualityWarningMessage(capture.quality.lastWarning, th, settings.diagnosticsEnabled)
      : "";

  const hasCameraError = Boolean(capture.cameraError);

  return (
    <main className="app-immersive">
      <CameraView videoRef={capture.videoRef} overlayRef={capture.overlayRef} />

      {screen === "camera" && (
        <>
          <div className="glass-top-bar">
            <div className="brand-glass">
              <div className="brand-mark-glass">{th.brandShort}</div>
              <span className="brand-name-glass">{th.appTitle}</span>
            </div>
            <div className="top-controls-right">
              <span className="live-chip" role="status">
                <span
                  className={["live-dot", !capture.ready ? "offline" : "", capture.ready && capture.handsPresent ? "hands" : ""].filter(Boolean).join(" ")}
                  aria-hidden="true"
                />
                {!capture.ready
                  ? th.cameraOpening
                  : capture.handsPresent
                    ? `🤲 ${th.cameraLive}`
                    : th.cameraLive}
              </span>
              <ModelPicker />
              <button className="glass-chip" onClick={() => setPhrasesOpen((v) => !v)} aria-expanded={phrasesOpen}>
                {th.supportedPhrasesTitle}
              </button>
            </div>
          </div>

          {hasCameraError && (
            <div className="glass-camera-error" role="alert">
              <p className="glass-text-danger" style={{ fontWeight: 700 }}>{th.cameraError}</p>
              <p style={{ color: "rgba(255,255,255,0.65)", fontSize: "var(--font-size-sm)" }}>{th.cameraErrorHint}</p>
              <button className="glass-action-btn" onClick={() => captureRef.current.start()}>
                {th.cameraRetry}
              </button>
            </div>
          )}
          {!hasCameraError && !hasScope("service") && !showConsent && (
            <div className="glass-camera-error" role="status">
              <p style={{ color: "#fff", fontWeight: 700, textAlign: "center" }}>{th.cameraConsentRequired}</p>
              <button className="glass-action-btn" onClick={() => setShowConsent(true)}>
                {th.openConsentSettings}
              </button>
            </div>
          )}
        </>
      )}

      {screen === "camera" && (
        <div className="result-glass-panel live">
          <LiveStatusRow
            cameraReady={capture.ready}
            translating={translator.pending}
            qualityMsg={qualityMsg}
            recording={capture.recording}
            modelsError={modelsError}
            hardBlocked={effectiveHardBlocked}
          />
          <ResultCard
            status={resultStatus}
            result={visibleResult}
            error={translator.error}
            errorStatus={translator.errorStatus}
            variant="glass"
            lastFrames={lastFrames}
            captureQuality={lastCaptureQuality}
            onRetry={() => translator.reset()}
          />
          <div style={{ marginTop: "var(--space-3)" }}>
            <TextToSignPanel />
          </div>
        </div>
      )}

      {screen === "history" && (
        <div className="screen-overlay">
          <HistoryScreen />
        </div>
      )}
      {screen === "settings" && (
        <div className="screen-overlay">
          <SettingsScreen />
        </div>
      )}
      {screen === "teach" && (
        <div className="screen-overlay">
          <TeachScreen capture={capture} />
        </div>
      )}
      {screen === "dictionary" && (
        <div className="screen-overlay">
          <DictionaryScreen />
        </div>
      )}

      {screen === "camera" && phrasesOpen && (
        <div
          className="result-glass-panel open"
          style={{ zIndex: 30 }}
          role="dialog"
          aria-modal="true"
          aria-label={th.supportedPhrasesTitle}
        >
          <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: "var(--space-3)" }}>
            <button
              ref={phrasesCloseRef}
              className="glass-chip"
              onClick={() => setPhrasesOpen(false)}
              aria-label={th.close}
            >
              ✕
            </button>
          </div>
          <SupportedPhrases glass />
        </div>
      )}

      <BottomNav
        active={screen}
        onChange={(s) => {
          if (s === "camera" && !hasScope("service")) {
            setShowConsent(true);
            return;
          }
          setScreen(s);
          setPhrasesOpen(false);
        }}
      />

      <ConsentModal
        open={showConsent}
        onClose={() => setShowConsent(false)}
        onAccept={completeModal}
        onOpenPrivacy={() => {
          setShowConsent(false);
          setScreen("settings");
        }}
      />
    </main>
  );
}

interface LiveStatusRowProps {
  cameraReady: boolean;
  translating: boolean;
  qualityMsg: string;
  recording: boolean;
  modelsError: string | null;
  hardBlocked: boolean;
}

function LiveStatusRow({ cameraReady, translating, qualityMsg, recording, modelsError, hardBlocked }: LiveStatusRowProps) {
  const th = useI18n();
  let label = "";
  let color = "rgba(255,255,255,0.5)";

  if (!cameraReady) { label = th.cameraInit; }
  else if (modelsError) {
    label = th.modelLoadError;
    color = "var(--glass-warn)";
  } else if (qualityMsg) {
    label = hardBlocked
      ? `${qualityMsg} — ${th.translationPaused}`
      : qualityMsg;
    color = hardBlocked ? "var(--glass-danger)" : "var(--glass-warn)";
  } else if (translating) { label = th.translating; }
  else if (recording) { label = th.realtimeTranslating; }
  else { label = th.showSignHint; }

  if (!label) return null;
  return (
    <p style={{ fontSize: "var(--font-size-sm)", color, marginBottom: "var(--space-2)", fontWeight: 500 }}>{label}</p>
  );
}

export default function App() {
  return (
    <ModelsProvider>
      <SettingsProvider>
        <HistoryProvider>
          <AppShell />
        </HistoryProvider>
      </SettingsProvider>
    </ModelsProvider>
  );
}
