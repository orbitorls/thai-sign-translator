import React from "react";

interface DevMetricsOverlayProps {
  fps: number;
  latencyMs: number | null;
  confidence: number | null; // 0..1
}

export function DevMetricsOverlay({ fps, latencyMs, confidence }: DevMetricsOverlayProps) {
  return (
    <div className="dev-metrics" aria-hidden="true">
      <span>FPS {fps}</span>
      <span>· {latencyMs == null ? "–" : `${latencyMs} ms`}</span>
      <span>· {confidence == null ? "–" : `${Math.round(confidence * 100)}%`}</span>
    </div>
  );
}
