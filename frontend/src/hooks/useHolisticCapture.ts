/**
 * useHolisticCapture — MediaPipe Holistic capture hook.
 *
 * Reads window.Holistic and window.Camera from CDN globals (do not bundle
 * @mediapipe/holistic — wasm paths are fragile in bundlers).
 *
 * Frame assembly order: face(468) → leftHand(21) → pose(33) → rightHand(21) = 543 landmarks.
 * Each landmark is [x, y, z].
 *
 * handsPresent: true when the current frame has at least one hand detected.
 * State update is edge-triggered (only fires when the value transitions), so
 * it does NOT trigger a React re-render every 30fps — only on appear/disappear.
 *
 * stop() returns both frames and the handFrameCount for that window so callers
 * can gate translation on real hand activity without a stale-closure issue.
 */
import { useEffect, useRef, useState } from "react";

declare const Holistic: any;
declare const Camera: any;

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

function fillBlock(block: Landmark[], landmarks: any[] | undefined | null): void {
  if (!landmarks) return;
  for (let i = 0; i < block.length && i < landmarks.length; i++) {
    const lm = landmarks[i];
    block[i] = [lm.x, lm.y, lm.z ?? 0.0];
  }
}

function assembleFrame(results: any): Frame {
  const face = zeros(N_FACE);
  const left = zeros(N_HAND);
  const pose = zeros(N_POSE);
  const right = zeros(N_HAND);
  fillBlock(face, results.faceLandmarks);
  fillBlock(left, results.leftHandLandmarks);
  fillBlock(pose, results.poseLandmarks);
  fillBlock(right, results.rightHandLandmarks);
  return [...face, ...left, ...pose, ...right];
}

export interface HolisticCaptureState {
  videoRef: React.RefObject<HTMLVideoElement>;
  ready: boolean;
  recording: boolean;
  handsPresent: boolean;
  cameraError: string | null;
  start: () => void;
  /** Returns frames and hand-frame-count for the window just stopped. */
  stop: () => StopResult;
}

export function useHolisticCapture(): HolisticCaptureState {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [ready, setReady] = useState(false);
  const [recording, setRecording] = useState(false);
  const [handsPresent, setHandsPresent] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);

  const recordingRef = useRef(false);
  const frameBufferRef = useRef<Frame[]>([]);
  const handFrameCountRef = useRef(0);
  // Edge-triggered: only call setHandsPresent when the boolean actually flips.
  const handsPresentRef = useRef(false);

  useEffect(() => {
    let pollId: ReturnType<typeof setTimeout>;

    function init() {
      if (typeof Holistic === "undefined" || typeof Camera === "undefined") {
        pollId = setTimeout(init, 200);
        return;
      }

      const holistic = new Holistic({
        locateFile: (f: string) =>
          `https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.5.1675471629/${f}`,
      });
      holistic.setOptions({ modelComplexity: 1, refineFaceLandmarks: false });

      holistic.onResults((results: any) => {
        const hasHands = Boolean(
          results.leftHandLandmarks || results.rightHandLandmarks
        );
        // Edge-triggered state update — avoids 30 re-renders/sec.
        if (hasHands !== handsPresentRef.current) {
          handsPresentRef.current = hasHands;
          setHandsPresent(hasHands);
        }

        if (!recordingRef.current) return;

        const frame = assembleFrame(results);
        if (frame.length === EXPECTED_FRAME_LEN) {
          frameBufferRef.current.push(frame);
          if (hasHands) {
            handFrameCountRef.current += 1;
          }
        }
      });

      if (!videoRef.current) return;

      const camera = new Camera(videoRef.current, {
        onFrame: async () => {
          if (holistic && videoRef.current) {
            await holistic.send({ image: videoRef.current });
          }
        },
        width: 480,
        height: 360,
      });

      camera.start()
        .then(() => setReady(true))
        .catch((err: Error) => {
          setCameraError(err.message ?? "ไม่สามารถเข้าถึงกล้องได้");
        });
    }

    init();
    return () => { clearTimeout(pollId); };
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

  return { videoRef, ready, recording, handsPresent, cameraError, start, stop };
}
