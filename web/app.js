const N_FACE = 468, N_HAND = 21, N_POSE = 33, N_LANDMARKS = 543;
const video = document.getElementById("video");
const recordBtn = document.getElementById("recordBtn");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const topkEl = document.getElementById("topk");
let recording = false;
let frameBuffer = [];

function zeros(n) { const out = new Array(n); for (let i = 0; i < n; i++) out[i] = [0.0, 0.0, 0.0]; return out; }
function fillBlock(block, landmarks) { if (!landmarks) return; for (let i = 0; i < block.length && i < landmarks.length; i++) { const lm = landmarks[i]; block[i] = [lm.x, lm.y, lm.z || 0.0]; } }
function assembleFrame(results) { const face = zeros(N_FACE), left = zeros(N_HAND), pose = zeros(N_POSE), right = zeros(N_HAND); fillBlock(face, results.faceLandmarks); fillBlock(left, results.leftHandLandmarks); fillBlock(pose, results.poseLandmarks); fillBlock(right, results.rightHandLandmarks); return face.concat(left, pose, right); }

const holistic = new Holistic({ locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${f}` });
holistic.setOptions({ modelComplexity: 1, refineFaceLandmarks: false });
holistic.onResults((results) => { if (!recording) return; const frame = assembleFrame(results); if (frame.length === N_LANDMARKS) frameBuffer.push(frame); });

const camera = new Camera(video, { onFrame: async () => { await holistic.send({ image: video }); }, width: 480, height: 360 });
camera.start();

async function sendForPrediction() {
  if (frameBuffer.length === 0) {
    statusEl.textContent = "No frames captured.";
    return;
  }
  statusEl.textContent = "Translating sentence...";
  topkEl.textContent = "";
  const resp = await fetch("/translate-sentence", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      frames: frameBuffer,
      feature_schema: "raw_mediapipe_543x3"
    })
  });
  const data = await resp.json();
  if (!resp.ok) {
    resultEl.textContent = "Sentence model unavailable";
    statusEl.textContent = data.detail || "Translation failed.";
    return;
  }
  resultEl.textContent = data.sentence || "-";
  statusEl.textContent = "";
}

recordBtn.addEventListener("click", () => { recording = !recording; if (recording) { frameBuffer = []; recordBtn.textContent = "Stop recording"; statusEl.textContent = "Recording…"; statusEl.className = "recording"; } else { recordBtn.textContent = "Start recording"; statusEl.className = ""; sendForPrediction(); } });
