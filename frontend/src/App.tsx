import React from "react";
import { ModelsProvider, useModels } from "./hooks/ModelsProvider";
import { useHolisticCapture } from "./hooks/useHolisticCapture";
import { useTranslate } from "./hooks/useTranslate";
import { CameraView } from "./components/CameraView";
import { RecordButton } from "./components/RecordButton";
import { ModelPicker } from "./components/ModelPicker";
import { ResultCard } from "./components/ResultCard";
import { StatusBar } from "./components/StatusBar";
import { SupportedPhrases } from "./components/SupportedPhrases";
import { th } from "./i18n/th";

function Translator() {
  const { models, selectedModelId, loading: modelsLoading, error: modelsError, setSelectedModelId } = useModels();
  const capture = useHolisticCapture();
  const translator = useTranslate();

  function handleToggleRecord() {
    if (capture.recording) {
      const frames = capture.stop();
      if (frames.length === 0) {
        // No frames captured — show hint but don't call API
        translator.reset();
        return;
      }
      translator.run(frames, selectedModelId ?? undefined);
    } else {
      translator.reset();
      capture.start();
    }
  }

  const isDisabled = !capture.ready || modelsLoading;

  // Derive status bar message
  let statusMsg = "";
  let statusType: "info" | "warn" | "error" | "success" = "info";
  const hasCameraError = Boolean(capture.cameraError);
  if (hasCameraError) {
    statusMsg = `${th.cameraError} — ${th.cameraErrorHint}`;
    statusType = "error";
  } else if (!capture.ready) {
    statusMsg = th.cameraInit;
  } else if (modelsError) {
    statusMsg = th.modelLoadError;
    statusType = "warn";
  } else if (capture.recording && capture.frameCount === 0) {
    statusMsg = th.recording;
  }

  // After stop with no frames
  const showNoFrames = !capture.recording && translator.status === "idle" && capture.frameCount === 0 && capture.ready;

  return (
    <main
      style={{
        maxWidth: "var(--max-width)",
        margin: "0 auto",
        padding: "var(--space-6) var(--space-4)",
        display: "flex",
        flexDirection: "column",
        gap: "var(--space-6)",
        minHeight: "100dvh",
      }}
    >
      {/* Header */}
      <header style={{ textAlign: "center" }}>
        <h1
          style={{
            fontSize: "var(--font-size-2xl)",
            fontWeight: 700,
            color: "var(--color-primary)",
          }}
        >
          {th.appTitle}
        </h1>
        <p style={{ color: "var(--color-text-muted)", fontSize: "var(--font-size-sm)", marginTop: "var(--space-1)" }}>
          {th.appSubtitle}
        </p>
      </header>

      {/* Camera */}
      <CameraView videoRef={capture.videoRef} recording={capture.recording} />

      {/* Model picker */}
      {modelsLoading ? (
        <p style={{ textAlign: "center", color: "var(--color-text-muted)", fontSize: "var(--font-size-sm)" }}>
          {th.modelLoading}
        </p>
      ) : (
        <ModelPicker
          models={models}
          selectedId={selectedModelId}
          onChange={setSelectedModelId}
          disabled={capture.recording || translator.status === "loading"}
        />
      )}

      {/* Record button */}
      <div style={{ display: "flex", justifyContent: "center" }}>
        <RecordButton
          recording={capture.recording}
          disabled={isDisabled}
          frameCount={capture.recording ? capture.frameCount : undefined}
          onClick={handleToggleRecord}
        />
      </div>

      {/* Status bar + camera retry */}
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "var(--space-2)" }}>
        <StatusBar
          message={statusMsg || (showNoFrames ? th.noFrames : "")}
          type={statusType}
        />
        {hasCameraError && (
          <button
            type="button"
            onClick={() => capture.start()}
            style={{
              padding: "var(--space-2) var(--space-4)",
              borderRadius: "var(--radius-full)",
              border: "1px solid var(--color-primary)",
              background: "var(--color-primary-light)",
              color: "var(--color-primary)",
              fontSize: "var(--font-size-sm)",
              fontWeight: 600,
              cursor: "pointer",
              fontFamily: "var(--font-family)",
              transition: "background var(--transition)",
            }}
          >
            {th.cameraRetry}
          </button>
        )}
      </div>

      {/* Result */}
      <ResultCard
        status={translator.status}
        result={translator.result}
        error={translator.error}
        errorStatus={translator.errorStatus}
      />

      {/* Supported phrases — lets users know what they can sign */}
      <SupportedPhrases />
    </main>
  );
}

export default function App() {
  return (
    <ModelsProvider>
      <Translator />
    </ModelsProvider>
  );
}
