# Design: แท็บ "สอน" — Realtime Few-Shot Sign Teaching

**Date:** 2026-06-29
**Branch:** feature/ux-tier1-polish
**Status:** Approved (user: "โอเค สร้างเลย")

## 1. Goal

Add a 4th bottom-nav tab **"สอน"** that lets a user teach the recognizer a new sign in
realtime by recording a few clips and typing its meaning, then immediately test
recognition (closed teach→recognize loop). This is **few-shot word recognition** on the
prototype track — entirely separate from the PoseT5 sentence translator on the camera tab.

## 2. Decisions (gathered in brainstorming)

- Closed loop: teach → recognize immediately via `/predict`.
- New 4th bottom-nav tab "สอน" (current nav: กล้อง / ประวัติ / ตั้งค่า).
- 3–5 clips per sign (flexible: min 1, recommended 3–5).

## 3. Key constraint — random-but-seeded encoder

The prototype recognizer needs `checkpoints/encoder.pt` (a `LandmarkEncoder`). **That file
is not in the repo / not on GitHub LFS** (only PoseT5's `pose_encoder.pt`, a different
architecture). So `_build_encoder()` falls back to a **randomly-initialised** encoder.

Implications:
- Within one server process the encoder is fixed (built once, cached), so teach→recognize
  in the same session is self-consistent and works for visually distinct gestures.
- **Currently the random init is non-deterministic**, so a saved `prototypes.pt` becomes
  invalid after a server restart (new random weights → embeddings no longer match).

**Fix (in scope):** seed the random init in `_build_encoder()` when no weights file exists,
so embeddings are reproducible across restarts and taught signs persist. This is a small,
high-value backend change. Recognition quality remains modest (untrained encoder) — that is
acceptable for a demo/few-shot feature and will be surfaced in the UI copy.

## 4. Architecture

Single shared `useHolisticCapture` instance already lives in `AppShell` and feeds a
full-screen `CameraView` that stays mounted across tabs. The Teach tab reuses that same
capture (manual `start()`/`stop()`), so no second camera/MediaPipe instance is created.

### Backend (`src/tsl/api/`)
1. **Seed encoder fallback** — in `app.py` `_build_encoder()`, when `ENCODER_WEIGHTS_PATH`
   is absent, set a fixed torch manual seed before constructing `LandmarkEncoder` so the
   random weights are deterministic across restarts.
2. **`DELETE /signs/{name}`** — new endpoint: `store.remove_sign(name)` + `_persist_store`,
   returns `{name, total_signs}`. 404 if the sign does not exist.
3. Reused as-is: `POST /train-custom-sign`, `POST /predict`, `GET /signs`.

### Frontend (`frontend/src/`)
1. **`api/client.ts`** — add:
   - `trainCustomSign(name, clips)` → POST `/train-custom-sign`
   - `predictSign(frames)` → POST `/predict` → `{word, score, topk}`
   - `getSigns()` → GET `/signs` → `{signs: string[]}`
   - `deleteSign(name)` → DELETE `/signs/{name}`
2. **`hooks/useTeach.ts`** — state machine for: clip buffer (array of frame-arrays),
   `addClip(frames)`, `removeClip(i)`, `submit(name)` (calls `trainCustomSign`), status
   (idle/saving/saved/error). Reuses `ApiError`/network pattern from `useTranslate`.
3. **`hooks/usePredict.ts`** — wraps `predictSign`, same state-machine shape as `useTranslate`.
4. **`components/TeachScreen.tsx`** — the "สอน" overlay. Receives the shared `capture` via
   props. Two sub-modes:
   - **สอน (Teach):** record-clip button (tap start → tap stop, reuses `RecordButton` +
     `capture.start()/stop()`), records into the clip buffer; chips "คลิป 1..n" with frame
     counts + delete-per-clip; text input for *ความหมาย*; "บันทึกท่า" button enabled when
     name non-empty AND ≥1 clip. On success: toast + clears buffer + refreshes sign list.
   - **จำท่า (Recognize):** record a gesture → `predictSign` → show recognized word + score
     (reuse `ResultCard`).
   - **รายการท่าที่สอนไว้:** from `getSigns()`, each with a delete button (`deleteSign`).
5. **`components/BottomNav.tsx`** — add 4th item "สอน" with a teach icon; extend
   `Screen` type to `"camera" | "history" | "settings" | "teach"`.
6. **`App.tsx`** — render `<TeachScreen capture={capture} />` in a `screen-overlay` when
   `screen === "teach"`; extend the capture resume condition (line ~52) so capture also runs
   on the teach screen: `shouldRun = !document.hidden && (screen === "camera" || screen === "teach")`.
   The camera-tab auto-segmentation effect stays gated on `screen === "camera"`, so it does
   not interfere with the teach tab's manual recording.
7. **`i18n` (`th.ts`)** — new strings: `navTeach`, `teachMode`, `recognizeMode`, `teachHint`,
   `clipLabel`, `meaningPlaceholder`, `saveSign`, `signSaved`, `taughtSigns`, `noSigns`,
   `recognizeHint`, `untrainedEncoderNote`, etc.

## 5. Data flow

- **Teach:** camera → `capture.start()` → tap stop → `capture.stop()` returns 543×3 raw
  frames → push to clip buffer → on save POST `/train-custom-sign {name, clips}` → server
  `normalize_sequence` + `store.add_sign` → `prototypes.pt` updated.
- **Recognize:** camera → record → 543×3 frames → POST `/predict {frames}` →
  `{word, score, topk}` → display.
- Feature schema is the same raw 543×3 the camera tab already produces; server normalizes.
  No client-side transform needed.

## 6. Validation & errors

- Name required (trim non-empty); ≥1 clip; warn if a clip has < `MIN_TOTAL_FRAMES` (8) frames.
- Reuse `ApiError` handling (api vs network) from `useTranslate`.
- Recognize with zero taught signs → friendly "ยังไม่มีท่าที่สอนไว้" message.

## 7. Out of scope (YAGNI)

No multi-user, no cloud sync, no per-user stores, no encoder training. Prototypes remain a
single server-side global store. Does not touch the PoseT5 sentence path at all.

## 8. Testing

- **Backend (pytest):** `DELETE /signs/{name}` (success + 404); encoder-seed determinism
  (two `_build_encoder()` builds with no weights → identical embedding for the same input).
  Existing `train_custom_sign` / `predict` tests already cover add + recognize.
- **Frontend:** component smoke test for `TeachScreen` (renders, buttons gate correctly);
  manual browser test — teach a distinct gesture with 3 clips, switch to จำท่า, confirm it
  is recognized; restart backend, confirm taught sign still recognized (seed determinism).

## 9. Files touched

Backend: `src/tsl/api/app.py` (seed + DELETE), `src/tsl/api/schemas.py` (DeleteSign response),
`tests/` (new test file).
Frontend: `api/client.ts`, `hooks/useTeach.ts` (new), `hooks/usePredict.ts` (new),
`components/TeachScreen.tsx` (new), `components/BottomNav.tsx`, `App.tsx`, `i18n` strings,
CSS additions for the teach panel.
