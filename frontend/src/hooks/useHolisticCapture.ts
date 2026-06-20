/**
 * useHolisticCapture — MediaPipe Holistic capture hook.
 *
 * Reads window.Holistic and window.Camera from CDN globals (do not bundle
 * @mediapipe/holistic — wasm paths are fragile in bundlers).
 *
 * Frame assembly order: face(468) → leftHand(21) → pose(33) → rightHand(21) = 543 landmarks.
 * Each landmark is [x, y, z].
 */
import { useEffect, useRef, useState } from "react";

// CDN global types (loaded from index.html <script> tags)
declare const Holistic: any;
declare const Camera: any;

const N_FACE = 468;
const N_HAND = 21;
const N_POSE = 33;

type Landmark = [number, number, number];
type Frame = Landmark[];

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
  frameCount: number;
  cameraError: string | null;
  start: () => void;
  stop: () => Frame[];
}

export function useHolisticCapture(): HolisticCaptureState {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [ready, setReady] = useState(false);
  const [recording, setRecording] = useState(false);
  const [frameCount, setFrameCount] = useState(0);
  const [cameraError, setCameraError] = useState<string | null>(null);

  const recordingRef = useRef(false);
  const frameBufferRef = useRef<Frame[]>([]);
  const cameraRef = useRef<any>(null);
  const holisticRef = useRef<any>(null);

  useEffect(() => {
    // Poll until CDN globals are available (they load asynchronously)
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
        if (!recordingRef.current) return;
        const frame = assembleFrame(results);
        if (frame.length === N_FACE + N_HAND + N_POSE + N_HAND) {
          frameBufferRef.current.push(frame);
          setFrameCount(frameBufferRef.current.length);
        }
      });
      holisticRef.current = holistic;

      if (!videoRef.current) return;

      const camera = new Camera(videoRef.current, {
        onFrame: async () => {
          if (holisticRef.current && videoRef.current) {
            await holisticRef.current.send({ image: videoRef.current });
          }
        },
        width: 480,
        height: 360,
      });

      camera.start().then(() => {
        setReady(true);
      }).catch((err: Error) => {
        setCameraError(err.message ?? "ไม่สามารถเข้าถึงกล้องได้");
      });

      cameraRef.current = camera;
    }

    init();

    return () => {
      clearTimeout(pollId);
      // Note: MediaPipe Camera has no clean stop API on CDN version — leave running
    };
  }, []); // run once on mount

  function start() {
    frameBufferRef.current = [];
    setFrameCount(0);
    recordingRef.current = true;
    setRecording(true);
  }

  function stop(): Frame[] {
    recordingRef.current = false;
    setRecording(false);
    return [...frameBufferRef.current];
  }

  return { videoRef, ready, recording, frameCount, cameraError, start, stop };
}
