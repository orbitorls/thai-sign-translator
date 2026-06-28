/**
 * useHolisticCapture — MediaPipe Holistic capture hook (Tasks API).
 *
 * Uses @mediapipe/tasks-vision HolisticLandmarker (the maintained replacement
 * for the EOL @mediapipe/holistic Solutions build, whose WASM aborts on
 * Chrome 130+). The WASM runtime and the .task model are loaded from the CDN;
 * only the JS wrapper is bundled.
 *
 * Frame assembly order (unchanged, for backend compatibility):
 *   face(468) → leftHand(21) → pose(33) → rightHand(21) = 543 landmarks.
 * Each landmark is [x, y, z]. The Tasks face model returns 478 points (468 mesh
 * + 10 iris); we keep only the first 468 to match the legacy Solutions output
 * the backend model was trained on.
 *
 * handsPresent: true when the current frame has at least one hand detected.
 * State update is edge-triggered (only fires when the value transitions), so
 * it does NOT trigger a React re-render every frame — only on appear/disappear.
 *
 * stop() returns both frames and the handFrameCount for that window so callers
 * can gate translation on real hand activity without a stale-closure issue.
 */
import { useEffect, useRef, useState } from "react";
import {
  FilesetResolver,
  HolisticLandmarker,
  DrawingUtils,
  type HolisticLandmarkerResult,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";

const WASM_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task";

const N_FACE = 468;
const N_HAND = 21;
const N_POSE = 33;
const EXPECTED_FRAME_LEN = N_FACE + N_HAND + N_POSE + N_HAND;

type Landmark = [number, number, number];
type Frame = Landmark[];

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

/** Tasks results are arrays-of-arrays (one entry per detected instance). */
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
  fillBlock(face, p.face); // 478 → first 468 kept by fillBlock's length cap
  fillBlock(left, p.left);
  fillBlock(pose, p.pose);
  fillBlock(right, p.right);
  return [...face, ...left, ...pose, ...right];
}

export interface HolisticCaptureState {
  videoRef: React.RefObject<HTMLVideoElement>;
  /** Canvas overlay for the optional landmark skeleton; sits above the video. */
  overlayRef: React.RefObject<HTMLCanvasElement>;
  ready: boolean;
  recording: boolean;
  handsPresent: boolean;
  cameraError: string | null;
  start: () => void;
  /** Returns frames and hand-frame-count for the window just stopped. */
  stop: () => StopResult;
  pause: () => void;
  resume: () => void;
}

export interface HolisticCaptureOptions {
  /** When true, draw the Holistic skeleton onto the overlay canvas. */
  overlayEnabled?: boolean;
}

/**
 * Draw the Holistic connections/landmarks onto the overlay canvas.
 * Always clears first so disabling the overlay leaves no stale frame.
 */
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

export function useHolisticCapture(
  options: HolisticCaptureOptions = {}
): HolisticCaptureState {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const [ready, setReady] = useState(false);
  const [recording, setRecording] = useState(false);
  const [handsPresent, setHandsPresent] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);

  const recordingRef = useRef(false);
  const frameBufferRef = useRef<Frame[]>([]);
  const handFrameCountRef = useRef(0);
  // Edge-triggered: only call setHandsPresent when the boolean actually flips.
  const handsPresentRef = useRef(false);
  // Ref-gated so toggling the overlay never re-inits MediaPipe.
  const overlayEnabledRef = useRef(Boolean(options.overlayEnabled));
  overlayEnabledRef.current = Boolean(options.overlayEnabled);
  const pausedRef = useRef(false);

  useEffect(() => {
    let cancelled = false;
    let rafId = 0;
    let landmarker: HolisticLandmarker | null = null;
    let stream: MediaStream | null = null;
    let lastTs = -1;

    async function init() {
      try {
        const vision = await FilesetResolver.forVisionTasks(WASM_BASE);
        if (cancelled) return;
        landmarker = await HolisticLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_URL },
          runningMode: "VIDEO",
        });
        if (cancelled) return;

        const video = videoRef.current;
        if (!video) return;
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 480, height: 360 },
          audio: false,
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        video.srcObject = stream;
        await video.play();
        setReady(true);
        loop();
      } catch (err) {
        if (!cancelled) {
          setCameraError(
            (err as Error)?.message ?? "ไม่สามารถเข้าถึงกล้องได้"
          );
        }
      }
    }

    function loop() {
      if (cancelled) return;
      rafId = requestAnimationFrame(loop);

      const video = videoRef.current;
      if (!landmarker || !video || video.readyState < 2) return;
      if (pausedRef.current) return; // paused: keep a light rAF tick, skip detection

      // detectForVideo requires strictly increasing timestamps (ms).
      let ts = performance.now();
      if (ts <= lastTs) ts = lastTs + 1;
      lastTs = ts;

      let result: HolisticLandmarkerResult;
      try {
        result = landmarker.detectForVideo(video, ts);
      } catch {
        return; // transient frame error — skip this frame
      }

      const p = pick(result);
      drawOverlay(overlayRef.current, video, p, overlayEnabledRef.current);

      const hasHands = Boolean(p.left || p.right);
      // Edge-triggered state update — avoids a re-render every frame.
      if (hasHands !== handsPresentRef.current) {
        handsPresentRef.current = hasHands;
        setHandsPresent(hasHands);
      }

      if (!recordingRef.current) return;

      const frame = assembleFrame(p);
      if (frame.length === EXPECTED_FRAME_LEN) {
        frameBufferRef.current.push(frame);
        if (hasHands) handFrameCountRef.current += 1;
      }
    }

    init();

    return () => {
      cancelled = true;
      cancelAnimationFrame(rafId);
      if (stream) stream.getTracks().forEach((t) => t.stop());
      if (landmarker) landmarker.close();
    };
  }, []);

  function start() {
    frameBufferRef.current = [];
    handFrameCountRef.current = 0;
    recordingRef.current = true;
    setRecording(true);
  }

  function stop(): StopResult {
    recordingRef.current = false;
    setRecording(false);
    return {
      frames: [...frameBufferRef.current],
      handFrameCount: handFrameCountRef.current,
    };
  }

  function pause() {
    pausedRef.current = true;
    videoRef.current?.pause();
  }

  function resume() {
    pausedRef.current = false;
    videoRef.current?.play().catch(() => {});
  }

  return { videoRef, overlayRef, ready, recording, handsPresent, cameraError, start, stop, pause, resume };
}
