import React from "react";

interface CameraViewProps {
  videoRef: React.RefObject<HTMLVideoElement>;
  recording: boolean;
}

export function CameraView({ videoRef, recording }: CameraViewProps) {
  return (
    <div
      style={{
        position: "relative",
        borderRadius: "var(--radius-lg)",
        overflow: "hidden",
        background: "#000",
        boxShadow: "var(--shadow-md)",
        width: "100%",
        maxWidth: "480px",
        aspectRatio: "4/3",
        margin: "0 auto",
      }}
    >
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{
          width: "100%",
          height: "100%",
          objectFit: "cover",
          transform: "scaleX(-1)", // mirror
          display: "block",
        }}
      />
      {recording && (
        <div
          style={{
            position: "absolute",
            top: "var(--space-3)",
            left: "var(--space-3)",
            display: "flex",
            alignItems: "center",
            gap: "var(--space-2)",
            background: "rgba(0,0,0,0.6)",
            borderRadius: "var(--radius-full)",
            padding: "var(--space-1) var(--space-3)",
          }}
        >
          <span
            style={{
              width: 10,
              height: 10,
              borderRadius: "50%",
              background: "var(--color-recording)",
              animation: "pulse 1s infinite",
              display: "inline-block",
            }}
          />
          <span style={{ color: "#fff", fontSize: "var(--font-size-sm)", fontWeight: 600 }}>
            REC
          </span>
        </div>
      )}
      <style>{`@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }`}</style>
    </div>
  );
}
