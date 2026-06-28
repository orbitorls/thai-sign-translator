/**
 * useHolisticCapture — MediaPipe Holistic capture hook (Tasks API).
 *
 * Uses @mediapipe/tasks-vision HolisticLandmarker. Frame order:
 *   face(468) → leftHand(21) → pose(33) → rightHand(21) = 543 landmarks.
 */
import { useEffect, useRef, useState } from "react";
import {
  FilesetResolver,
  HolisticLandmarker,
  DrawingUtils,
  type HolisticLandmarkerResult,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";
import type { CameraFacing } from "./useLocalSettings";
import { useI18n } from "../i18n";

const WASM_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task";

const N_FACE = 468;
const N_HAND = 21;
const N_POSE = 33;
const EXPECTED_FRAME_LEN = N_FACE + N_HAND + N_POSE + N_HAND;
const MAX_BUFFER_MS = 6000;
const FPS_LOW_THRESHOLD = 15;
const NO_HANDS_DURATION_MS = 2000;

type Landmark = [number, number, number];
type Frame = Landmark[];

interface TimestampedFrame {
  ts: number;
  frame: Frame;
}

export type FeatureSchemaId = "raw_mediapipe_543x3" | "raw_mediapipe_543x4";
export type QualityWarning = "low_light" | "no_hands" | "motion_blur" | "low_fps" | null;

export interface CaptureQuality {
  fps: number;
  handPresent: boolean;
  lightingOK: boolean;
  lastWarning: QualityWarning;
}

export interface StopResult {
  frames: Frame[];
  handFrameCount: number;
}

function zeros(n: number): Landmark[] {
  return Array.from({ length: n }, () => [0.0, 0.0, 0.0] as Landmark);
}

function fillBlock(block: Landmark[], landmarks: NormalizedLandmark[] | null): void {
  if (!landmarks) return;
  for (let i = 0; i < block.length && i < landmarks.length; i++) {
    const lm = landmarks[i];
    block[i] = [lm.x, lm.y, lm.z ?? 0.0];
  }
}

function first(arr: NormalizedLandmark[][] | undefined): NormalizedLandmark[] | null {
  return arr && arr.length ? arr[0] : null;
}

interface Picked {
  face: NormalizedLandmark[] | null;
  left: NormalizedLandmark[] | null;
  pose: NormalizedLandmark[] | null;
  right: NormalizedLandmark[] | null;
}

function pick(result: HolisticLandmarkerResult): Picked {
  return {
    face: first(result.faceLandmarks),
    left: first(result.leftHandLandmarks),
    pose: first(result.poseLandmarks),
    right: first(result.rightHandLandmarks),
  };
}

function assembleFrame(p: Picked): Frame {
  const face = zeros(N_FACE);
  const left = zeros(N_HAND);
  const pose = zeros(N_POSE);
  const right = zeros(N_HAND);
  fillBlock(face, p.face);
  fillBlock(left, p.left);
  fillBlock(pose, p.pose);
  fillBlock(right, p.right);
  return [...face, ...left, ...pose, ...right];
}

function meanLuminance(video: HTMLVideoElement, canvas: HTMLCanvasElement): number {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx || video.videoWidth === 0) return 255;
  const w = 32;
  const h = 24;
  canvas.width = w;
  canvas.height = h;
  ctx.drawImage(video, 0, 0, w, h);
  const data = ctx.getImageData(0, 0, w, h).data;
  let sum = 0;
  for (let i = 0; i < data.length; i += 4) {
    sum += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }
  return sum / (data.length / 4);
}

const INITIAL_QUALITY: CaptureQuality = {
  fps: 0,
  handPresent: false,
  lightingOK: true,
  lastWarning: null,
};

export interface HolisticCaptureState {
  videoRef: React.RefObject<HTMLVideoElement>;
  overlayRef: React.RefObject<HTMLCanvasElement>;
  ready: boolean;
  recording: boolean;
  handsPresent: boolean;
  frameCount: number;
  cameraError: string | null;
  quality: CaptureQuality;
  featureSchema: FeatureSchemaId;
  start: () => void;
  stop: () => StopResult;
  getRecentFrames: (windowMs?: number) => Frame[];
}

export interface HolisticCaptureOptions {
  overlayEnabled?: boolean;
  facingMode?: CameraFacing;
}

function drawOverlay(
  canvas: HTMLCanvasElement | null,
  video: HTMLVideoElement | null,
  p: Picked,
  enabled: boolean
): void {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const w = video?.videoWidth || canvas.width || 480;
  const h = video?.videoHeight || canvas.height || 360;
  if (canvas.width !== w) canvas.width = w;
  if (canvas.height !== h) canvas.height = h;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!enabled) return;

  const du = new DrawingUtils(ctx);
  if (p.pose) {
    du.drawConnectors(p.pose, HolisticLandmarker.POSE_CONNECTIONS, {
      color: "rgba(255,255,255,0.55)",
      lineWidth: 2,
    });
  }
  if (p.face) {
    du.drawConnectors(p.face, HolisticLandmarker.FACE_LANDMARKS_TESSELATION, {
      color: "rgba(37,99,235,0.30)",
      lineWidth: 0.5,
    });
  }
  for (const hand of [p.left, p.right]) {
    if (!hand) continue;
    du.drawConnectors(hand, HolisticLandmarker.HAND_CONNECTIONS, {
      color: "rgba(22,163,74,0.9)",
      lineWidth: 3,
    });
    du.drawLandmarks(hand, { color: "#ffffff", radius: 2.5, lineWidth: 1 });
  }
}

const MEDIAPIPE_TIMEOUT_MS = 20_000;

export function useHolisticCapture(
  options: HolisticCaptureOptions = {}
): HolisticCaptureState {
  const t = useI18n();
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const [ready, setReady] = useState(false);
  const [recording, setRecording] = useState(false);
  const [handsPresent, setHandsPresent] = useState(false);
  const [frameCount, setFrameCount] = useState(0);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [quality, setQuality] = useState<CaptureQuality>(INITIAL_QUALITY);

  const recordingRef = useRef(false);
  const frameBufferRef = useRef<TimestampedFrame[]>([]);
  const handFrameCountRef = useRef(0);
  const handsPresentRef = useRef(false);
  const overlayEnabledRef = useRef(Boolean(options.overlayEnabled));
  const facingModeRef = useRef(options.facingMode ?? "user");
  const fpsTimestampsRef = useRef<number[]>([]);
  const noHandsSinceRef = useRef<number | null>(null);
  const sampleCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const lastLuminanceCheckRef = useRef(0);

  overlayEnabledRef.current = Boolean(options.overlayEnabled);
  facingModeRef.current = options.facingMode ?? "user";

  useEffect(() => {
    let cancelled = false;
    let rafId = 0;
    let landmarker: HolisticLandmarker | null = null;
    let stream: MediaStream | null = null;
    let lastTs = -1;

    async function init() {
      const loadTimeoutId = window.setTimeout(() => {
        if (!cancelled) setCameraError(t.mediaPipeSlowNetwork);
      }, MEDIAPIPE_TIMEOUT_MS);

      try {
        const vision = await FilesetResolver.forVisionTasks(WASM_BASE);
        if (cancelled) { clearTimeout(loadTimeoutId); return; }
        landmarker = await HolisticLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_URL },
          runningMode: "VIDEO",
        });
        clearTimeout(loadTimeoutId);
        if (cancelled) return;

        const video = videoRef.current;
        if (!video) return;
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: 480,
            height: 360,
            facingMode: facingModeRef.current,
          },
          audio: false,
        });
        if (cancelled) {
          stream.getTracks().forEach((s) => s.stop());
          return;
        }
        video.srcObject = stream;
        await video.play();
        setReady(true);
        loop();
      } catch (err) {
        clearTimeout(loadTimeoutId);
        if (!cancelled) {
          const errName = (err as Error)?.name ?? "";
          const errMsg =
            errName === "NotAllowedError"  ? t.cameraDenied :
            errName === "NotFoundError"    ? t.cameraNotFound :
            errName === "NotReadableError" ? t.cameraBusy :
            t.cameraError;
          setCameraError(errMsg);
        }
      }
    }

    function updateQuality(hasHands: boolean, now: number) {
      fpsTimestampsRef.current.push(now);
      fpsTimestampsRef.current = fpsTimestampsRef.current.filter((t) => now - t < 1000);
      const fps = fpsTimestampsRef.current.length;

      if (!hasHands) {
        if (noHandsSinceRef.current === null) noHandsSinceRef.current = now;
      } else {
        noHandsSinceRef.current = null;
      }

      let lastWarning: QualityWarning = null;
      if (noHandsSinceRef.current !== null && now - noHandsSinceRef.current > NO_HANDS_DURATION_MS) {
        lastWarning = "no_hands";
      } else if (fps > 0 && fps < FPS_LOW_THRESHOLD) {
        lastWarning = "low_fps";
      }

      const video = videoRef.current;
      if (video && now - lastLuminanceCheckRef.current > 500) {
        lastLuminanceCheckRef.current = now;
        if (!sampleCanvasRef.current) sampleCanvasRef.current = document.createElement("canvas");
        const lum = meanLuminance(video, sampleCanvasRef.current);
        if (lum < 50) lastWarning = "low_light";
        setQuality({ fps, handPresent: hasHands, lightingOK: lum >= 50, lastWarning });
      } else {
        setQuality((prev) => ({ ...prev, fps, handPresent: hasHands, lastWarning }));
      }
    }

    function loop() {
      if (cancelled) return;
      rafId = requestAnimationFrame(loop);

      const video = videoRef.current;
      if (!landmarker || !video || video.readyState < 2) return;

      let ts = performance.now();
      if (ts <= lastTs) ts = lastTs + 1;
      lastTs = ts;

      let result: HolisticLandmarkerResult;
      try {
        result = landmarker.detectForVideo(video, ts);
      } catch {
        return;
      }

      const p = pick(result);
      drawOverlay(overlayRef.current, video, p, overlayEnabledRef.current);

      const hasHands = Boolean(p.left || p.right);
      if (hasHands !== handsPresentRef.current) {
        handsPresentRef.current = hasHands;
        setHandsPresent(hasHands);
      }
      updateQuality(hasHands, ts);

      const frame = assembleFrame(p);
      if (frame.length !== EXPECTED_FRAME_LEN) return;

      const cutoff = ts - MAX_BUFFER_MS;
      frameBufferRef.current.push({ ts, frame });
      frameBufferRef.current = frameBufferRef.current.filter((f) => f.ts >= cutoff);

      if (recordingRef.current) {
        if (hasHands) handFrameCountRef.current += 1;
        setFrameCount((c) => c + 1);
      }
    }

    init();

    return () => {
      cancelled = true;
      cancelAnimationFrame(rafId);
      if (stream) stream.getTracks().forEach((t) => t.stop());
      if (landmarker) landmarker.close();
    };
  }, [options.facingMode]);

  function start() {
    handFrameCountRef.current = 0;
    recordingRef.current = true;
    setRecording(true);
    setFrameCount(0);
  }

  function stop(): StopResult {
    recordingRef.current = false;
    setRecording(false);
    const recent = getRecentFrames();
    return {
      frames: recent,
      handFrameCount: handFrameCountRef.current,
    };
  }

  function getRecentFrames(windowMs = 2000): Frame[] {
    const now = performance.now();
    const cutoff = now - windowMs;
    return frameBufferRef.current.filter((f) => f.ts >= cutoff).map((f) => f.frame);
  }

  return {
    videoRef,
    overlayRef,
    ready,
    recording,
    handsPresent,
    frameCount,
    cameraError,
    quality,
    featureSchema: "raw_mediapipe_543x3",
    start,
    stop,
    getRecentFrames,
  };
}
