import React from "react";

interface CameraViewProps {
  videoRef: React.RefObject<HTMLVideoElement>;
}

export function CameraView({ videoRef }: CameraViewProps) {
  return (
    <div style={{ position: "absolute", inset: 0, background: "#000", overflow: "hidden" }}>
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{
          width: "100%",
          height: "100%",
          objectFit: "cover",
          transform: "scaleX(-1)",
          display: "block",
        }}
      />

      {/* AR-style corner-bracket guide */}
      <svg
        aria-hidden="true"
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
      >
        <path d="M 10 22 L 10 10 L 22 10" fill="none" stroke="rgba(255,255,255,0.45)" strokeWidth="0.9" strokeLinecap="round" />
        <path d="M 78 10 L 90 10 L 90 22" fill="none" stroke="rgba(255,255,255,0.45)" strokeWidth="0.9" strokeLinecap="round" />
        <path d="M 10 78 L 10 90 L 22 90" fill="none" stroke="rgba(255,255,255,0.45)" strokeWidth="0.9" strokeLinecap="round" />
        <path d="M 78 90 L 90 90 L 90 78" fill="none" stroke="rgba(255,255,255,0.45)" strokeWidth="0.9" strokeLinecap="round" />
      </svg>
    </div>
  );
}
