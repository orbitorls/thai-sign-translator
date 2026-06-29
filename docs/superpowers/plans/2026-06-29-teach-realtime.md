# แท็บ "สอน" — Realtime Few-Shot Teaching — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 4th bottom-nav tab "สอน" where the user records a few clips of a sign, types its meaning, saves it to the prototype recognizer, and immediately tests recognition (closed teach→recognize loop).

**Architecture:** Frontend-heavy. The backend `/train-custom-sign`, `/predict`, `/signs` endpoints already exist; we add encoder-seed determinism + a `DELETE /signs/{name}` endpoint. The frontend reuses the single shared `useHolisticCapture` instance (already feeding a full-screen `CameraView` mounted across tabs) for manual clip recording and prediction, behind a new `TeachScreen` overlay.

**Tech Stack:** FastAPI + PyTorch (backend), React 18 + TypeScript + Vite (frontend), MediaPipe Tasks Vision (capture).

## Global Constraints

- Backend Python deps unchanged; `transformers>=5,<6` already pinned; do NOT add torchao.
- Frontend has **no JS test runner** — verify frontend tasks with `npm run build` (tsc typecheck) + manual browser test. Backend uses `pytest`.
- All API requests use relative URLs (Vite proxy forwards `/signs`, `/predict`, `/train-custom-sign` to `127.0.0.1:8000`; `/signs` already proxied — verify the proxy list includes it).
- Feature schema is raw 543×3 landmark frames; server normalizes. No client-side transform.
- Run backend with `PYTHONPATH=src`; run pytest from repo root.
- Thai UI copy lives in `frontend/src/i18n/th.ts`; English mirror in `en.ts` must get the same keys (build fails otherwise — `Dict = typeof th`).

---

### Task 1: Backend — deterministic encoder fallback

Seed the random encoder init so a saved `prototypes.pt` survives a server restart (embeddings reproducible).

**Files:**
- Modify: `src/tsl/api/app.py` (`_build_encoder`, ~line 90)
- Test: `tests/api/test_encoder_seed.py`

**Interfaces:**
- Produces: `_build_encoder() -> LandmarkEncoder` — same signature; now deterministic when no weights file exists.

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_encoder_seed.py
import numpy as np
import torch
import tsl.api.app as appmod


def test_build_encoder_is_deterministic_without_weights(monkeypatch):
    # Force the no-weights branch.
    monkeypatch.setattr(appmod.config, "ENCODER_WEIGHTS_PATH", None, raising=False)
    enc_a = appmod._build_encoder()
    enc_b = appmod._build_encoder()

    # Same random seed → identical parameters → identical embeddings.
    seq = np.zeros((4, len(appmod.SELECTED_LANDMARKS) * 3), dtype=np.float32)
    x = torch.from_numpy(seq).unsqueeze(0)
    lengths = torch.tensor([4], dtype=torch.long)
    with torch.no_grad():
        emb_a = enc_a(x, lengths)
        emb_b = enc_b(x, lengths)
    assert torch.allclose(emb_a, emb_b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/api/test_encoder_seed.py -v`
Expected: FAIL — `emb_a` and `emb_b` differ (independent random inits).

- [ ] **Step 3: Implement the seed**

In `src/tsl/api/app.py`, change `_build_encoder` so the no-weights branch seeds torch first:

```python
def _build_encoder() -> LandmarkEncoder:
    """Load the trained encoder weights from disk if available; else a
    deterministic random init (fixed seed) so a saved prototype store stays
    valid across restarts."""
    weights_path = getattr(config, "ENCODER_WEIGHTS_PATH", None)
    if weights_path and os.path.exists(weights_path):
        enc = LandmarkEncoder(input_dim=len(SELECTED_LANDMARKS) * 3)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        enc.load_state_dict(state)
    else:
        # No trained encoder shipped — seed so the random weights are
        # reproducible (prototypes.pt embeddings remain valid after restart).
        gen_state = torch.random.get_rng_state()
        torch.manual_seed(20260629)
        try:
            enc = LandmarkEncoder(input_dim=len(SELECTED_LANDMARKS) * 3)
        finally:
            torch.random.set_rng_state(gen_state)
    enc.eval()
    return enc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/api/test_encoder_seed.py -v`
Expected: PASS.

- [ ] **Step 5: Run the existing API tests to confirm no regression**

Run: `PYTHONPATH=src pytest tests/api -v`
Expected: PASS (predict/signs/translate suites still green).

- [ ] **Step 6: Commit**

```bash
git add src/tsl/api/app.py tests/api/test_encoder_seed.py
git commit -m "feat(api): deterministic encoder fallback so taught signs survive restart"
```

---

### Task 2: Backend — `DELETE /signs/{name}` endpoint

**Files:**
- Modify: `src/tsl/api/schemas.py` (add `DeleteSignResponse`)
- Modify: `src/tsl/api/app.py` (add endpoint near `list_signs`, ~line 236)
- Test: `tests/api/test_app_signs.py` (extend)

**Interfaces:**
- Consumes: `get_store()` dependency; `store.remove_sign(name)`, `store.names()`, `_persist_store(store)` (all existing).
- Produces: `DELETE /signs/{name}` → 200 `{name: str, total_signs: int}`; 404 `{detail}` when the sign is absent.

- [ ] **Step 1: Write the failing test**

Add to `tests/api/test_app_signs.py` — extend `StubStore` with `remove_sign`, then two tests:

```python
class StubStore:
    def __init__(self):
        self._signs: dict[str, list] = {"existing": []}

    def add_sign(self, name, clips):
        self._signs[name] = clips

    def remove_sign(self, name):
        self._signs.pop(name, None)

    def names(self):
        return list(self._signs.keys())


def test_delete_sign_removes_and_returns_total():
    stub = StubStore()
    app.dependency_overrides[get_store] = lambda: stub
    try:
        client = TestClient(app)
        resp = client.delete("/signs/existing")
        assert resp.status_code == 200
        assert resp.json() == {"name": "existing", "total_signs": 0}
        assert "existing" not in stub.names()
    finally:
        app.dependency_overrides.clear()


def test_delete_unknown_sign_returns_404():
    stub = StubStore()
    app.dependency_overrides[get_store] = lambda: stub
    try:
        client = TestClient(app)
        resp = client.delete("/signs/nope")
        assert resp.status_code == 404
    finally:
        app.dependency_overrides.clear()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/api/test_app_signs.py -v`
Expected: FAIL — 405/404 routing (endpoint not defined).

- [ ] **Step 3: Add the response schema**

In `src/tsl/api/schemas.py`, after `TrainSignResponse`:

```python
class DeleteSignResponse(BaseModel):
    name: str
    total_signs: int
```

- [ ] **Step 4: Add the endpoint**

In `src/tsl/api/app.py`, import the schema (add `DeleteSignResponse` to the existing schemas import) and add after `list_signs`:

```python
@app.delete("/signs/{name}", response_model=DeleteSignResponse)
def delete_sign(name: str, store: PrototypeStore = Depends(get_store)) -> DeleteSignResponse:
    if name not in store.names():
        raise HTTPException(status_code=404, detail=f"unknown sign {name!r}")
    store.remove_sign(name)
    _persist_store(store)
    return DeleteSignResponse(name=name, total_signs=len(store.names()))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/api/test_app_signs.py -v`
Expected: PASS (all four tests).

- [ ] **Step 6: Commit**

```bash
git add src/tsl/api/app.py src/tsl/api/schemas.py tests/api/test_app_signs.py
git commit -m "feat(api): DELETE /signs/{name} to remove a taught sign"
```

---

### Task 3: Frontend — API client functions

**Files:**
- Modify: `frontend/src/api/client.ts`
- Modify: `frontend/vite.config.ts` (ensure `/train-custom-sign` and `/predict` are proxied)

**Interfaces:**
- Produces:
  - `interface PredictResult { word: string; score: number; topk: { word: string; score: number }[] }`
  - `trainCustomSign(name: string, clips: number[][][][]): Promise<{ name: string; num_clips: number; total_signs: number }>`
  - `predictSign(frames: number[][][]): Promise<PredictResult>`
  - `getSigns(): Promise<{ signs: string[] }>`
  - `deleteSign(name: string): Promise<{ name: string; total_signs: number }>`

- [ ] **Step 1: Confirm the proxy covers the new paths**

Read `frontend/vite.config.ts`. If `proxy` keys do not already include `/train-custom-sign` and `/predict`, add them mirroring the existing `/translate` entry (same target `http://127.0.0.1:8000`). `/signs` is already listed.

- [ ] **Step 2: Add the client functions**

Append to `frontend/src/api/client.ts`:

```typescript
export interface PredictResult {
  word: string;
  score: number;
  topk: { word: string; score: number }[];
}

export async function trainCustomSign(
  name: string,
  clips: number[][][][]
): Promise<{ name: string; num_clips: number; total_signs: number }> {
  const resp = await fetch("/train-custom-sign", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, clips }),
  });
  return _handleResponse(resp);
}

export async function predictSign(frames: number[][][]): Promise<PredictResult> {
  const resp = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ frames }),
  });
  return _handleResponse<PredictResult>(resp);
}

export async function getSigns(): Promise<{ signs: string[] }> {
  const resp = await fetch("/signs");
  return _handleResponse<{ signs: string[] }>(resp);
}

export async function deleteSign(
  name: string
): Promise<{ name: string; total_signs: number }> {
  const resp = await fetch(`/signs/${encodeURIComponent(name)}`, { method: "DELETE" });
  return _handleResponse(resp);
}
```

- [ ] **Step 3: Typecheck**

Run: `cd frontend && npm run build`
Expected: build succeeds (no TS errors).

- [ ] **Step 4: Commit**

```bash
git add frontend/src/api/client.ts frontend/vite.config.ts
git commit -m "feat(ui): API client for teach/predict/signs endpoints"
```

---

### Task 4: Frontend — `useTeach` and `usePredict` hooks

**Files:**
- Create: `frontend/src/hooks/useTeach.ts`
- Create: `frontend/src/hooks/usePredict.ts`

**Interfaces:**
- Consumes: `trainCustomSign`, `predictSign`, `PredictResult`, `ApiError` from `../api/client`.
- Produces:
  - `useTeach(): { clips: number[][][][]; status: "idle"|"saving"|"saved"|"error"; error: string|null; addClip(frames: number[][][]): void; removeClip(i: number): void; clear(): void; submit(name: string): Promise<boolean> }`
  - `usePredict(): { status: "idle"|"loading"|"success"|"error"; result: PredictResult|null; error: string|null; run(frames: number[][][]): Promise<void>; reset(): void }`

- [ ] **Step 1: Write `useTeach`**

```typescript
// frontend/src/hooks/useTeach.ts
import { useCallback, useState } from "react";
import { trainCustomSign, ApiError } from "../api/client";

export type TeachStatus = "idle" | "saving" | "saved" | "error";

export function useTeach() {
  const [clips, setClips] = useState<number[][][][]>([]);
  const [status, setStatus] = useState<TeachStatus>("idle");
  const [error, setError] = useState<string | null>(null);

  const addClip = useCallback((frames: number[][][]) => {
    setClips((c) => [...c, frames]);
    setStatus("idle");
  }, []);

  const removeClip = useCallback((i: number) => {
    setClips((c) => c.filter((_, idx) => idx !== i));
  }, []);

  const clear = useCallback(() => {
    setClips([]);
    setStatus("idle");
    setError(null);
  }, []);

  const submit = useCallback(
    async (name: string): Promise<boolean> => {
      if (!name.trim() || clips.length === 0) return false;
      setStatus("saving");
      setError(null);
      try {
        await trainCustomSign(name.trim(), clips);
        setStatus("saved");
        setClips([]);
        return true;
      } catch (e) {
        setError(e instanceof ApiError ? e.detail : "บันทึกไม่สำเร็จ");
        setStatus("error");
        return false;
      }
    },
    [clips]
  );

  return { clips, status, error, addClip, removeClip, clear, submit };
}
```

- [ ] **Step 2: Write `usePredict`**

```typescript
// frontend/src/hooks/usePredict.ts
import { useCallback, useState } from "react";
import { predictSign, PredictResult, ApiError } from "../api/client";

export type PredictStatus = "idle" | "loading" | "success" | "error";

export function usePredict() {
  const [status, setStatus] = useState<PredictStatus>("idle");
  const [result, setResult] = useState<PredictResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async (frames: number[][][]) => {
    setStatus("loading");
    setError(null);
    try {
      const res = await predictSign(frames);
      setResult(res);
      setStatus("success");
    } catch (e) {
      setError(e instanceof ApiError ? e.detail : "เชื่อมต่อไม่ได้");
      setStatus("error");
    }
  }, []);

  const reset = useCallback(() => {
    setStatus("idle");
    setResult(null);
    setError(null);
  }, []);

  return { status, result, error, run, reset };
}
```

- [ ] **Step 3: Typecheck**

Run: `cd frontend && npm run build`
Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/hooks/useTeach.ts frontend/src/hooks/usePredict.ts
git commit -m "feat(ui): useTeach and usePredict hooks"
```

---

### Task 5: Frontend — i18n strings + 4th nav tab

**Files:**
- Modify: `frontend/src/i18n/th.ts`
- Modify: `frontend/src/i18n/en.ts`
- Modify: `frontend/src/components/BottomNav.tsx`

**Interfaces:**
- Produces: `Screen` type now `"camera" | "history" | "settings" | "teach"`; `BottomNav` shows a 4th "สอน" item.
- New i18n keys (both dictionaries): `navTeach, teachModeTab, recognizeModeTab, teachIntro, recordClipHint, clipChip(n), meaningLabel, meaningPlaceholder, saveSign, savingSign, signSaved, saveSignError, taughtSignsTitle, noTaughtSigns, recognizeHint, recognizeResult, recognizeNoSigns, untrainedEncoderNote, deleteSignAria(name)`.

- [ ] **Step 1: Add Thai strings**

Add to the object in `frontend/src/i18n/th.ts` (before the closing `};`):

```typescript
  // Teach tab
  navTeach: "สอน",
  teachModeTab: "สอนท่า",
  recognizeModeTab: "จำท่า",
  teachIntro: "อัดคลิปท่าเดิมซ้ำ 3–5 คลิป แล้วกรอกความหมาย",
  recordClipHint: "กดเพื่ออัดคลิป แล้วกดอีกครั้งเพื่อหยุด",
  clipChip: (n: number) => `คลิป ${n}`,
  meaningLabel: "ความหมาย",
  meaningPlaceholder: "เช่น สวัสดี",
  saveSign: "บันทึกท่า",
  savingSign: "กำลังบันทึก...",
  signSaved: "บันทึกท่าแล้ว",
  saveSignError: "บันทึกไม่สำเร็จ",
  taughtSignsTitle: "ท่าที่สอนไว้",
  noTaughtSigns: "ยังไม่มีท่าที่สอนไว้",
  recognizeHint: "ทำท่าหน้ากล้องเพื่อให้ระบบทาย",
  recognizeResult: "ระบบทายว่า",
  recognizeNoSigns: "ยังไม่มีท่าที่สอนไว้ ไปที่แท็บ \"สอนท่า\" ก่อน",
  untrainedEncoderNote: "โหมดทดลอง: ใช้ตัวเข้ารหัสแบบยังไม่ฝึก เหมาะกับท่าที่ต่างกันชัดเจน",
  deleteSignAria: (name: string) => `ลบท่า ${name}`,
```

- [ ] **Step 2: Add matching English strings**

Add the same keys to `frontend/src/i18n/en.ts` with English copy (e.g. `navTeach: "Teach"`, `teachModeTab: "Teach"`, `recognizeModeTab: "Recognize"`, `clipChip: (n) => \`Clip ${n}\``, `deleteSignAria: (name) => \`Delete sign ${name}\``, etc.). Every key from Step 1 must be present or the build fails.

- [ ] **Step 3: Extend `Screen` and add the nav item**

In `frontend/src/components/BottomNav.tsx`:
- Change the type: `export type Screen = "camera" | "history" | "settings" | "teach";`
- Add a teach icon component:

```tsx
const TeachIcon = () => (
  <svg viewBox="0 0 24 24" {...stroke}>
    <path d="M12 3 2 8l10 5 10-5-10-5z" />
    <path d="M6 10.5V16c0 1 2.7 2.5 6 2.5s6-1.5 6-2.5v-5.5" />
  </svg>
);
```

- Add `{ key: "teach", label: th.navTeach, icon: <TeachIcon /> }` to the `items` array (after `camera`, so order is กล้อง / สอน / ประวัติ / ตั้งค่า).

- [ ] **Step 4: Typecheck**

Run: `cd frontend && npm run build`
Expected: build succeeds; no missing-key errors between `th` and `en`.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/i18n/th.ts frontend/src/i18n/en.ts frontend/src/components/BottomNav.tsx
git commit -m "feat(ui): add สอน tab to bottom nav + i18n strings"
```

---

### Task 6: Frontend — `TeachScreen` component + App wiring

**Files:**
- Create: `frontend/src/components/TeachScreen.tsx`
- Modify: `frontend/src/App.tsx`

**Interfaces:**
- Consumes: `useTeach`, `usePredict`, `getSigns`, `deleteSign` (client), `RecordButton`, `HolisticCaptureState` (type from `../hooks/useHolisticCapture`), `useI18n`.
- `TeachScreen` props: `{ capture: HolisticCaptureState; active: boolean }` (`active` = teach screen is visible, used to gate the signs fetch).

- [ ] **Step 1: Write `TeachScreen`**

```tsx
// frontend/src/components/TeachScreen.tsx
import React, { useCallback, useEffect, useState } from "react";
import { useI18n } from "../i18n";
import { useTeach } from "../hooks/useTeach";
import { usePredict } from "../hooks/usePredict";
import { getSigns, deleteSign } from "../api/client";
import { RecordButton } from "./RecordButton";
import type { HolisticCaptureState } from "../hooks/useHolisticCapture";

const MIN_TOTAL_FRAMES = 8;

interface TeachScreenProps {
  capture: HolisticCaptureState;
  active: boolean;
}

export function TeachScreen({ capture, active }: TeachScreenProps) {
  const th = useI18n();
  const teach = useTeach();
  const predict = usePredict();
  const [mode, setMode] = useState<"teach" | "recognize">("teach");
  const [name, setName] = useState("");
  const [signs, setSigns] = useState<string[]>([]);
  const [frameCount, setFrameCount] = useState(0);

  const refreshSigns = useCallback(async () => {
    try {
      const { signs } = await getSigns();
      setSigns(signs);
    } catch {
      /* leave list as-is on error */
    }
  }, []);

  useEffect(() => {
    if (active) refreshSigns();
  }, [active, refreshSigns]);

  // Poll the live frame counter while recording.
  useEffect(() => {
    if (!capture.recording) return;
    const id = setInterval(() => setFrameCount((c) => c + 1), 100);
    return () => clearInterval(id);
  }, [capture.recording]);

  const onRecordToggle = useCallback(() => {
    if (capture.recording) {
      const { frames } = capture.stop();
      setFrameCount(0);
      if (frames.length < MIN_TOTAL_FRAMES) return; // too short, drop silently
      if (mode === "teach") teach.addClip(frames);
      else predict.run(frames);
    } else {
      predict.reset();
      setFrameCount(0);
      capture.start();
    }
  }, [capture, mode, teach, predict]);

  const onSave = useCallback(async () => {
    const ok = await teach.submit(name);
    if (ok) {
      setName("");
      refreshSigns();
    }
  }, [teach, name, refreshSigns]);

  const onDelete = useCallback(
    async (sign: string) => {
      try {
        await deleteSign(sign);
        refreshSigns();
      } catch {
        /* ignore */
      }
    },
    [refreshSigns]
  );

  const canSave = name.trim().length > 0 && teach.clips.length > 0 && teach.status !== "saving";

  return (
    <div className="teach-screen">
      <div className="teach-mode-switch">
        <button
          className={`glass-chip${mode === "teach" ? " on" : ""}`}
          onClick={() => { setMode("teach"); predict.reset(); }}
        >
          {th.teachModeTab}
        </button>
        <button
          className={`glass-chip${mode === "recognize" ? " on" : ""}`}
          onClick={() => { setMode("recognize"); }}
        >
          {th.recognizeModeTab}
        </button>
      </div>

      <p className="teach-note">{th.untrainedEncoderNote}</p>

      <div className="teach-record-row">
        <RecordButton
          recording={capture.recording}
          frameCount={frameCount}
          onClick={onRecordToggle}
          variant="large"
        />
        <p className="teach-hint">
          {mode === "teach" ? th.recordClipHint : th.recognizeHint}
        </p>
      </div>

      {mode === "teach" ? (
        <>
          {teach.clips.length > 0 && (
            <div className="teach-clip-chips">
              {teach.clips.map((_, i) => (
                <span key={i} className="glass-chip">
                  {th.clipChip(i + 1)}
                  <button
                    className="teach-clip-del"
                    onClick={() => teach.removeClip(i)}
                    aria-label={`${th.actionDelete} ${th.clipChip(i + 1)}`}
                  >
                    ✕
                  </button>
                </span>
              ))}
            </div>
          )}

          <label className="teach-field">
            <span>{th.meaningLabel}</span>
            <input
              type="text"
              value={name}
              placeholder={th.meaningPlaceholder}
              onChange={(e) => setName(e.target.value)}
            />
          </label>

          <button className="glass-action-btn" disabled={!canSave} onClick={onSave}>
            {teach.status === "saving" ? th.savingSign : th.saveSign}
          </button>
          {teach.status === "saved" && <p className="teach-ok">{th.signSaved}</p>}
          {teach.status === "error" && <p className="teach-err">{teach.error ?? th.saveSignError}</p>}
        </>
      ) : (
        <div className="teach-recognize-result" aria-live="polite">
          {signs.length === 0 && <p className="teach-hint">{th.recognizeNoSigns}</p>}
          {predict.status === "loading" && <p className="teach-hint">…</p>}
          {predict.status === "success" && predict.result && (
            <p className="teach-recognized">
              {th.recognizeResult} <strong>{predict.result.word}</strong>
            </p>
          )}
          {predict.status === "error" && <p className="teach-err">{predict.error}</p>}
        </div>
      )}

      <div className="teach-signs">
        <h3>{th.taughtSignsTitle}</h3>
        {signs.length === 0 ? (
          <p className="teach-hint">{th.noTaughtSigns}</p>
        ) : (
          <ul>
            {signs.map((s) => (
              <li key={s}>
                <span>{s}</span>
                <button
                  className="teach-clip-del"
                  onClick={() => onDelete(s)}
                  aria-label={th.deleteSignAria(s)}
                >
                  ✕
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Wire into `App.tsx`**

Three edits in `frontend/src/App.tsx`:

1. Add the import near the other component imports:
```tsx
import { TeachScreen } from "./components/TeachScreen";
```

2. Extend the capture resume condition so capture runs on the teach screen too. Change the `apply` body in the visibilitychange effect (~line 52):
```tsx
      const shouldRun = !document.hidden && (screen === "camera" || screen === "teach");
```

3. Render the overlay alongside the history/settings overlays (~after line 229):
```tsx
      {screen === "teach" && (
        <div className="screen-overlay">
          <TeachScreen capture={capture} active={screen === "teach"} />
        </div>
      )}
```

- [ ] **Step 3: Add CSS for the teach screen**

Append to `frontend/src/styles/tokens.css` (confirmed home of `.screen-overlay`, `.glass-chip`, `.glass-action-btn`). Add:

```css
.teach-screen { display: flex; flex-direction: column; gap: var(--space-4); padding: var(--space-5); overflow-y: auto; height: 100%; color: #fff; }
.teach-mode-switch { display: flex; gap: var(--space-2); }
.glass-chip.on { background: rgba(255,255,255,0.28); font-weight: 700; }
.teach-note { font-size: var(--font-size-xs); color: rgba(255,255,255,0.6); }
.teach-record-row { display: flex; flex-direction: column; align-items: center; gap: var(--space-2); }
.teach-hint { font-size: var(--font-size-sm); color: rgba(255,255,255,0.7); text-align: center; }
.teach-clip-chips { display: flex; flex-wrap: wrap; gap: var(--space-2); }
.teach-clip-del { background: none; border: none; color: inherit; cursor: pointer; margin-left: 6px; opacity: 0.7; }
.teach-field { display: flex; flex-direction: column; gap: var(--space-2); }
.teach-field input { padding: var(--space-3); border-radius: var(--radius-md); border: 1px solid rgba(255,255,255,0.25); background: rgba(0,0,0,0.25); color: #fff; font-size: var(--font-size-base); }
.teach-ok { color: #22c55e; }
.teach-err { color: #fca5a5; }
.teach-recognized { font-size: var(--font-size-xl); }
.teach-signs ul { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: var(--space-2); }
.teach-signs li { display: flex; justify-content: space-between; align-items: center; padding: var(--space-2) var(--space-3); background: rgba(255,255,255,0.08); border-radius: var(--radius-md); }
```

- [ ] **Step 4: Typecheck the build**

Run: `cd frontend && npm run build`
Expected: build succeeds.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/TeachScreen.tsx frontend/src/App.tsx frontend/src/styles/tokens.css
git commit -m "feat(ui): TeachScreen with teach/recognize modes wired into app"
```

---

### Task 7: End-to-end manual verification

**Files:** none (verification only).

- [ ] **Step 1: Start both servers**

```bash
PYTHONPATH=src python -m uvicorn tsl.api.app:app --host 127.0.0.1 --port 8000 &
cd frontend && npm run dev
```

- [ ] **Step 2: Teach a sign**

Open http://localhost:5173 → tap **สอน** tab → ensure **สอนท่า** mode → record 3 clips of a distinct gesture (each ≥ ~1s) → type a meaning (e.g. "สวัสดี") → tap **บันทึกท่า**. Expect "บันทึกท่าแล้ว" and the sign appears under "ท่าที่สอนไว้".

- [ ] **Step 3: Recognize it**

Switch to **จำท่า** mode → perform the same gesture → tap stop. Expect "ระบบทายว่า <meaning>".

- [ ] **Step 4: Verify persistence across restart**

Stop the backend, restart it (same command), reload the page, go to **สอน** → the taught sign still listed; recognize it again → still works (confirms encoder-seed determinism).

- [ ] **Step 5: Delete a sign**

Tap ✕ next to a sign under "ท่าที่สอนไว้" → it disappears and `GET /signs` no longer lists it.

- [ ] **Step 6: Final commit (if any verification fix was needed)**

```bash
git add -A && git commit -m "fix(ui): teach feature verification adjustments"
```

---

## Self-Review

**Spec coverage:**
- §3 encoder seed → Task 1. §4 backend DELETE → Task 2; client funcs → Task 3; hooks → Task 4; nav+i18n → Task 5; TeachScreen+App wiring (incl. capture resume on teach screen) → Task 6. §5 data flow exercised in Task 7. §6 validation (name required, ≥1 clip, MIN_TOTAL_FRAMES, ApiError) → Tasks 4 & 6. §8 testing → Tasks 1, 2 (pytest) + Task 7 (manual). All sections mapped.

**Placeholder scan:** No TBD/“handle edge cases”/“similar to” — every code step shows full code. CSS target pinned to `frontend/src/styles/tokens.css` (verified). Vite proxy confirmed to lack `/train-custom-sign` + `/predict` — Task 3 Step 1 adds them.

**Type consistency:** `PredictResult` defined in Task 3, consumed by `usePredict` (Task 4) and `TeachScreen` (Task 6). `useTeach` API (`clips/status/error/addClip/removeClip/clear/submit`) defined in Task 4, consumed in Task 6. `Screen` union extended in Task 5, used in Task 6. `HolisticCaptureState` imported as a type in Task 6 — it is exported from `useHolisticCapture.ts` (verified). `capture.recording/start/stop` match the hook's returned shape.
