// frontend/src/components/TeachScreen.tsx
import React, { useCallback, useEffect, useState } from "react";
import { useI18n } from "../i18n";
import { useTeach } from "../hooks/useTeach";
import { usePredict } from "../hooks/usePredict";
import { getSigns, deleteSign } from "../api/client";
import { RecordButton } from "./RecordButton";
import type { HolisticCaptureState } from "../hooks/useHolisticCapture";

const MIN_TOTAL_FRAMES = 8;

interface TeachScreenProps {
  capture: HolisticCaptureState;
  active: boolean;
}

export function TeachScreen({ capture, active }: TeachScreenProps) {
  const th = useI18n();
  const teach = useTeach();
  const predict = usePredict();
  const [mode, setMode] = useState<"teach" | "recognize">("teach");
  const [name, setName] = useState("");
  const [signs, setSigns] = useState<string[]>([]);
  const [frameCount, setFrameCount] = useState(0);

  const refreshSigns = useCallback(async () => {
    try {
      const { signs } = await getSigns();
      setSigns(signs);
    } catch {
      /* leave list as-is on error */
    }
  }, []);

  useEffect(() => {
    if (active) refreshSigns();
  }, [active, refreshSigns]);

  // Poll the live frame counter while recording.
  useEffect(() => {
    if (!capture.recording) return;
    const id = setInterval(() => setFrameCount((c) => c + 1), 100);
    return () => clearInterval(id);
  }, [capture.recording]);

  // The shared capture may arrive already "recording" (the camera tab leaves it
  // on). Discard that stale buffer on mount, and stop on unmount so the camera
  // tab can't inherit a buffer we started. start()/stop() touch only stable
  // refs/setters, so this is safe.
  useEffect(() => {
    if (capture.recording) capture.stop();
    return () => { capture.stop(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const onRecordToggle = useCallback(() => {
    if (capture.recording) {
      const { frames } = capture.stop();
      setFrameCount(0);
      if (frames.length < MIN_TOTAL_FRAMES) return; // too short, drop silently
      if (mode === "teach") teach.addClip(frames);
      else predict.run(frames);
    } else {
      if (mode === "recognize" && signs.length === 0) return;
      predict.reset();
      setFrameCount(0);
      capture.start();
    }
  }, [capture, mode, signs, teach, predict]);

  const onSave = useCallback(async () => {
    const ok = await teach.submit(name);
    if (ok) {
      setName("");
      refreshSigns();
    }
  }, [teach, name, refreshSigns]);

  const onDelete = useCallback(
    async (sign: string) => {
      try {
        await deleteSign(sign);
        refreshSigns();
      } catch {
        /* ignore */
      }
    },
    [refreshSigns]
  );

  const canSave = name.trim().length > 0 && teach.clips.length > 0 && teach.status !== "saving";

  return (
    <div className="teach-screen">
      <div className="teach-mode-switch">
        <button
          className={`glass-chip${mode === "teach" ? " on" : ""}`}
          aria-pressed={mode === "teach"}
          onClick={() => { setMode("teach"); predict.reset(); }}
        >
          {th.teachModeTab}
        </button>
        <button
          className={`glass-chip${mode === "recognize" ? " on" : ""}`}
          aria-pressed={mode === "recognize"}
          onClick={() => { setMode("recognize"); }}
        >
          {th.recognizeModeTab}
        </button>
      </div>

      <p className="teach-note">{th.untrainedEncoderNote}</p>

      {mode === "teach" && <p className="teach-hint">{th.teachIntro}</p>}

      <div className="teach-record-row">
        <RecordButton
          recording={capture.recording}
          frameCount={frameCount}
          onClick={onRecordToggle}
          variant="large"
        />
        <p className="teach-hint">
          {mode === "teach" ? th.recordClipHint : th.recognizeHint}
        </p>
      </div>

      {mode === "teach" ? (
        <>
          {teach.clips.length > 0 && (
            <div className="teach-clip-chips">
              {teach.clips.map((_, i) => (
                <span key={i} className="glass-chip">
                  {th.clipChip(i + 1)}
                  <button
                    className="teach-clip-del"
                    onClick={() => teach.removeClip(i)}
                    aria-label={`${th.actionDelete} ${th.clipChip(i + 1)}`}
                  >
                    ✕
                  </button>
                </span>
              ))}
            </div>
          )}

          <label className="teach-field">
            <span>{th.meaningLabel}</span>
            <input
              type="text"
              value={name}
              placeholder={th.meaningPlaceholder}
              onChange={(e) => setName(e.target.value)}
            />
          </label>

          <button className="glass-action-btn" disabled={!canSave} onClick={onSave}>
            {teach.status === "saving" ? th.savingSign : th.saveSign}
          </button>
          <div aria-live="polite">
            {teach.status === "saved" && <p className="teach-ok">{th.signSaved}</p>}
            {teach.status === "error" && <p className="teach-err">{teach.error ?? th.saveSignError}</p>}
          </div>
        </>
      ) : (
        <div className="teach-recognize-result" aria-live="polite">
          {signs.length === 0 && <p className="teach-hint">{th.recognizeNoSigns}</p>}
          {predict.status === "loading" && <p className="teach-hint">…</p>}
          {predict.status === "success" && predict.result && (
            <p className="teach-recognized">
              {th.recognizeResult} <strong>{predict.result.word}</strong>
            </p>
          )}
          {predict.status === "error" && <p className="teach-err">{predict.error}</p>}
        </div>
      )}

      <div className="teach-signs">
        <h3>{th.taughtSignsTitle}</h3>
        {signs.length === 0 ? (
          <p className="teach-hint">{th.noTaughtSigns}</p>
        ) : (
          <ul>
            {signs.map((s) => (
              <li key={s}>
                <span>{s}</span>
                <button
                  className="teach-clip-del"
                  onClick={() => onDelete(s)}
                  aria-label={th.deleteSignAria(s)}
                >
                  ✕
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
