const N_FACE = 468, N_HAND = 21, N_POSE = 33, N_LANDMARKS = 543;
const video = document.getElementById("video");
const recordBtn = document.getElementById("recordBtn");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const confidenceEl = document.getElementById("confidence");
const modelSelect = document.getElementById("modelSelect");
const cameraState = document.getElementById("cameraState");
const captureSummary = document.getElementById("captureSummary");
const cameraBox = document.getElementById("cameraBox");
let recording = false;
let frameBuffer = [];
let selectedModel = null;

function zeros(n) { const out = new Array(n); for (let i = 0; i < n; i++) out[i] = [0.0, 0.0, 0.0]; return out; }
function fillBlock(block, landmarks) { if (!landmarks) return; for (let i = 0; i < block.length && i < landmarks.length; i++) { const lm = landmarks[i]; block[i] = [lm.x, lm.y, lm.z || 0.0]; } }
function assembleFrame(results) { const face = zeros(N_FACE), left = zeros(N_HAND), pose = zeros(N_POSE), right = zeros(N_HAND); fillBlock(face, results.faceLandmarks); fillBlock(left, results.leftHandLandmarks); fillBlock(pose, results.poseLandmarks); fillBlock(right, results.rightHandLandmarks); return face.concat(left, pose, right); }

const holistic = new Holistic({ locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${f}` });
holistic.setOptions({ modelComplexity: 1, refineFaceLandmarks: false });
holistic.onResults((results) => { if (!recording) return; const frame = assembleFrame(results); if (frame.length === N_LANDMARKS) frameBuffer.push(frame); });

const camera = new Camera(video, { onFrame: async () => { await holistic.send({ image: video }); }, width: 480, height: 360 });
camera.start().then(() => { cameraState.textContent = "กล้องพร้อม"; }).catch(() => {
  cameraState.textContent = "ไม่สามารถเข้าถึงกล้องได้";
  statusEl.textContent = "กรุณาอนุญาตการใช้กล้องในเบราว์เซอร์";
});

async function loadModels() {
  try {
    const resp = await fetch("/models");
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "โหลดรายชื่อโมเดลล้มเหลว");
    modelSelect.innerHTML = "";
    data.models.forEach((model) => {
      const option = document.createElement("option");
      option.value = model.id;
      option.disabled = !model.available;
      option.textContent = `${model.label_th}${model.available ? "" : " (ไม่พร้อมใช้งาน)"}`;
      modelSelect.appendChild(option);
    });
    selectedModel = data.default;
    modelSelect.value = data.default;
    modelSelect.disabled = false;
    captureSummary.textContent = modelSelect.selectedOptions[0]?.textContent || "เลือกโมเดลแล้วเริ่มบันทึก";
  } catch (err) {
    modelSelect.innerHTML = "<option>โหลดโมเดลล้มเหลว</option>";
    statusEl.textContent = err.message || "โหลดรายชื่อโมเดลล้มเหลว";
  }
}

modelSelect.addEventListener("change", () => {
  selectedModel = modelSelect.value;
  captureSummary.textContent = modelSelect.selectedOptions[0]?.textContent || "เลือกโมเดลแล้วเริ่มบันทึก";
});

async function sendForPrediction() {
  if (frameBuffer.length === 0) {
    statusEl.textContent = "ไม่พบการเคลื่อนไหว ลองเริ่มบันทึกใหม่";
    return;
  }
  statusEl.textContent = "กำลังแปล...";
  confidenceEl.textContent = "";
  const resp = await fetch("/translate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      frames: frameBuffer,
      feature_schema: "raw_mediapipe_543x3",
      model: selectedModel,
      max_len: 128
    })
  });
  const data = await resp.json();
  if (!resp.ok) {
    resultEl.textContent = resp.status === 503 ? "โมเดลนี้ยังไม่พร้อมใช้งาน" : "เกิดข้อผิดพลาด กรุณาลองใหม่";
    statusEl.textContent = data.detail || "แปลไม่สำเร็จ";
    return;
  }
  const pct = Math.round((data.score || 0) * 100);
  resultEl.textContent = data.sentence || "-";
  confidenceEl.textContent = `ความมั่นใจ ${pct}%`;
  statusEl.textContent = "";
}

recordBtn.addEventListener("click", () => {
  recording = !recording;
  if (recording) {
    frameBuffer = [];
    recordBtn.textContent = "หยุดบันทึก";
    recordBtn.classList.add("recording");
    cameraBox.classList.add("recording");
    statusEl.textContent = "กำลังบันทึก...";
    captureSummary.textContent = "กำลังบันทึก... 0 เฟรม";
  } else {
    recordBtn.textContent = "เริ่มบันทึก";
    recordBtn.classList.remove("recording");
    cameraBox.classList.remove("recording");
    captureSummary.textContent = `บันทึกแล้ว ${frameBuffer.length} เฟรม`;
    sendForPrediction();
  }
});

setInterval(() => {
  if (recording) captureSummary.textContent = `กำลังบันทึก... ${frameBuffer.length} เฟรม`;
}, 300);

loadModels();
