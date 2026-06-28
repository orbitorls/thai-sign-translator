import React, { useEffect, useState } from "react";
import { useT } from "../i18n";
import { useSettings } from "../hooks/SettingsProvider";
import { useConsent } from "../hooks/useConsent";
import { useVideoCapture } from "../hooks/useVideoCapture";
import {
  getFeedbackStats,
  submitTeach,
  submitFeedbackVideo,
  ApiError,
  feedbackErrorMessage,
  type CaptureQualityPayload,
  type FeedbackStats,
} from "../api/client";
import { CameraView } from "./CameraView";
import type { HolisticCaptureState } from "../hooks/useHolisticCapture";
import { MOCKUP_MODE } from "../mockup";

const MIN_FRAMES = 12;
const CAPTURE_WINDOW_MS = 2000;
const TEACH_CATEGORIES = ["greetings", "numbers", "polite", "basic"] as const;

interface TeachScreenProps {
  capture: HolisticCaptureState;
}

export function TeachScreen({ capture }: TeachScreenProps) {
  const { t } = useT();
  const { settings, update } = useSettings();
  const { hasScope } = useConsent();
  const videoCapture = useVideoCapture();
  const [labelText, setLabelText] = useState("");
  const [category, setCategory] = useState<(typeof TEACH_CATEGORIES)[number]>("greetings");
  const [hasFrames, setHasFrames] = useState(false);
  const [stats, setStats] = useState<FeedbackStats | null>(null);
  const [submitState, setSubmitState] = useState<"idle" | "submitting" | "saved" | "error">("idle");
  const [statusMsg, setStatusMsg] = useState("");

  useEffect(() => {
    if (MOCKUP_MODE) {
      setStats({
        pending_count: 3,
        total_count: 24,
        last_retrain_at: null,
        last_attempt_at: null,
        feedback_version: "demo",
        model: "conductor_core",
      });
      return;
    }
    void getFeedbackStats()
      .then(setStats)
      .catch(() => setStats(null));
  }, [submitState]);

  async function handleSubmit() {
    if (!hasScope("model_improvement")) {
      setStatusMsg(t.teachNeedsOptIn);
      return;
    }
    const frames = capture.getRecentFrames(CAPTURE_WINDOW_MS);
    if (frames.length < MIN_FRAMES) {
      setStatusMsg(t.recordingHint);
      return;
    }
    const text = labelText.trim();
    if (!text) return;

    setSubmitState("submitting");
    setStatusMsg("");
    try {
      const captureQuality: CaptureQualityPayload = {
        fps: capture.quality.fps,
        lighting_ok: capture.quality.lightingOK,
        hand_present: capture.quality.handPresent,
        warning: capture.quality.lastWarning,
        feature_schema: capture.featureSchema,
        camera_facing: settings.cameraFacing,
      };
      const result = await submitTeach({
        frames: frames.map((frame) => frame.map((lm) => [...lm])),
        labelText: text,
        captureQuality,
      });
      if (hasScope("video_research")) {
        const stream = capture.videoRef.current?.srcObject;
        let blob = videoCapture.blob;
        if (videoCapture.recording && stream instanceof MediaStream) {
          blob = await videoCapture.stop();
        }
        if (blob && blob.size > 0) {
          await submitFeedbackVideo(result.segment_id, blob);
        }
      }
      videoCapture.clear();
      setSubmitState("saved");
      setStatusMsg(t.teachSaved);
      setLabelText("");
      setHasFrames(false);
    } catch (err) {
      setSubmitState("error");
      const status = err instanceof ApiError ? err.status : 0;
      setStatusMsg(feedbackErrorMessage(status, t, "teach"));
    }
  }

  function handleToggleRecord() {
    if (capture.recording) {
      void videoCapture.stop();
      capture.stop();
      setHasFrames(true);
    } else {
      setSubmitState("idle");
      setStatusMsg("");
      setHasFrames(false);
      videoCapture.clear();
      capture.start();
      if (hasScope("video_research")) {
        const stream = capture.videoRef.current?.srcObject;
        if (stream instanceof MediaStream) videoCapture.start(stream);
      }
    }
  }

  return (
    <div className="screen-sheet">
      <h2 className="sheet-title" style={{ marginBottom: "var(--space-2)" }}>
        {t.teachTitle}
      </h2>
      <p style={{ color: "rgba(255,255,255,0.6)", marginBottom: "var(--space-4)", fontSize: "var(--font-size-sm)" }}>
        {t.teachDesc}
      </p>

      {stats && (
        <p style={{ fontSize: "var(--font-size-sm)", color: "rgba(255,255,255,0.55)", marginBottom: "var(--space-3)" }}>
          {t.teachStatsPending(stats.pending_count)} · {t.teachStatsTotal(stats.total_count)}
        </p>
      )}

      <div
        style={{
          borderRadius: "var(--radius-md)",
          overflow: "hidden",
          marginBottom: "var(--space-4)",
          aspectRatio: "4/3",
          maxHeight: 200,
          position: "relative",
        }}
      >
        <CameraView videoRef={capture.videoRef} overlayRef={capture.overlayRef} />
      </div>

      <div style={{ display: "flex", gap: "var(--space-2)", marginBottom: "var(--space-4)", flexWrap: "wrap" }}>
        <button type="button" className="glass-action-btn" onClick={handleToggleRecord}>
          {capture.recording ? t.recordStop : t.recordStart}
        </button>
        <button
          type="button"
          className="glass-chip"
          onClick={() => update("cameraFacing", settings.cameraFacing === "user" ? "environment" : "user")}
        >
          {t.switchCamera}
        </button>
      </div>

      <label style={{ display: "block", marginBottom: "var(--space-2)", fontWeight: 600 }}>{t.teachLabelTitle}</label>
      <input
        type="text"
        value={labelText}
        onChange={(e) => setLabelText(e.target.value)}
        placeholder={t.teachLabelPlaceholder}
        style={{
          width: "100%",
          padding: "var(--space-3)",
          marginBottom: "var(--space-3)",
          borderRadius: "var(--radius-md)",
          border: "1px solid rgba(255,255,255,0.2)",
          background: "rgba(0,0,0,0.25)",
          color: "#fff",
          fontFamily: "var(--font-family)",
        }}
      />

      <label style={{ display: "block", marginBottom: "var(--space-2)", fontWeight: 600 }}>{t.teachCategoryLabel}</label>
      <select
        value={category}
        onChange={(e) => setCategory(e.target.value as (typeof TEACH_CATEGORIES)[number])}
        className="glass-chip select-full"
        style={{ marginBottom: "var(--space-4)" }}
      >
        {TEACH_CATEGORIES.map((c) => (
          <option key={c} value={c}>
            {c}
          </option>
        ))}
      </select>

      <button
        type="button"
        className="glass-action-btn"
        onClick={() => void handleSubmit()}
        disabled={submitState === "submitting" || (!hasFrames && !capture.recording)}
      >
        {submitState === "submitting" ? t.teachSubmitting : t.teachSubmit}
      </button>

      {statusMsg && (
        <p
          role="status"
          style={{
            marginTop: "var(--space-3)",
            fontSize: "var(--font-size-sm)",
            color: submitState === "error" ? "#fca5a5" : "rgba(255,255,255,0.75)",
          }}
        >
          {statusMsg}
        </p>
      )}
    </div>
  );
}
