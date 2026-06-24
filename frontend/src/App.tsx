import React from "react";
import { ModelsProvider, useModels } from "./hooks/ModelsProvider";
import { useHolisticCapture } from "./hooks/useHolisticCapture";
import { useTranslate } from "./hooks/useTranslate";
import { CameraView } from "./components/CameraView";
import { RecordButton } from "./components/RecordButton";
import { ModelPicker } from "./components/ModelPicker";
import { ResultCard } from "./components/ResultCard";
import { StatusBar } from "./components/StatusBar";
import { th } from "./i18n/th";

function Translator() {
  const { models, selectedModelId, loading: modelsLoading, error: modelsError, setSelectedModelId } = useModels();
  const capture = useHolisticCapture();
  const translator = useTranslate();
  const selectedModel = models.find((model) => model.id === selectedModelId);

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
  if (capture.cameraError) {
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
    <main className="app-shell">
      <header className="app-header">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true">TS</div>
          <div style={{ minWidth: 0 }}>
            <h1
              style={{
                fontSize: "var(--font-size-xl)",
                fontWeight: 700,
                color: "var(--color-text)",
                lineHeight: 1.25,
              }}
            >
              {th.appTitle}
            </h1>
            <p style={{ color: "var(--color-text-muted)", fontSize: "var(--font-size-sm)" }}>
              {selectedModel?.label_th ?? th.appSubtitle}
            </p>
          </div>
        </div>
        <span
          style={{
            color: capture.ready ? "var(--color-success)" : "var(--color-text-muted)",
            fontSize: "var(--font-size-sm)",
            fontWeight: 600,
            whiteSpace: "nowrap",
          }}
        >
          {capture.ready ? th.cameraReady : th.cameraInit}
        </span>
      </header>

      <div className="chat-layout">
        <aside className="side-panel" aria-label={th.cameraPanelLabel}>
          <CameraView videoRef={capture.videoRef} recording={capture.recording} />
          {modelsLoading ? (
            <p style={{ color: "var(--color-text-muted)", fontSize: "var(--font-size-sm)" }}>
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
        </aside>

        <section className="chat-main" aria-label={th.chatPanelLabel}>
          <div className="chat-thread">
            <div className="message-row">
              <div className="message-bubble">
                <p style={{ color: "var(--color-text-muted)", fontSize: "var(--font-size-sm)", marginBottom: "var(--space-2)" }}>
                  {th.assistantLabel}
                </p>
                <p style={{ fontSize: "var(--font-size-lg)", fontWeight: 600, color: "var(--color-text)" }}>
                  {th.appSubtitle}
                </p>
              </div>
            </div>

            <div className="message-row user">
              <div className="message-bubble">
                <p style={{ color: "var(--color-text-muted)", fontSize: "var(--font-size-sm)", marginBottom: "var(--space-2)" }}>
                  {th.cameraPanelLabel}
                </p>
                <p style={{ fontSize: "var(--font-size-base)", fontWeight: 600 }}>
                  {capture.recording
                    ? `${th.recording} ${th.frames(capture.frameCount)}`
                    : selectedModel?.label_th ?? th.modelLabel}
                </p>
              </div>
            </div>

            <div className="message-row">
              <ResultCard
                status={translator.status}
                result={translator.result}
                error={translator.error}
                errorStatus={translator.errorStatus}
              />
            </div>
          </div>

          <div className="composer">
            <StatusBar
              message={statusMsg || (showNoFrames ? th.noFrames : "")}
              type={statusType}
            />
            <RecordButton
              recording={capture.recording}
              disabled={isDisabled}
              frameCount={capture.recording ? capture.frameCount : undefined}
              onClick={handleToggleRecord}
            />
          </div>
        </section>
      </div>
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
