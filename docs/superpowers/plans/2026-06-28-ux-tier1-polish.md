# UX Tier 1 Polish — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** เพิ่ม 6 ฟีเจอร์ quick-win ต่อยอด P1 (Developer Mode+เมตริก, history copy/ลบ/แชร์, haptic+แฟลช, pause ตอนพักหลัง, PWA ติดตั้งได้, offline/error state) — frontend-only

**Architecture:** ต่อยอดบน hooks/components ของ P1; เพิ่ม 2 hook/component ใหม่ (`useFeedback`, `DevMetricsOverlay`) + manifest; ขยาย `useHolisticCapture` (fps, pause/resume), `useTranslate` (latency, errorKind), `SettingsProvider` (showLandmarks→devMode), `HistoryProvider` (remove). ใช้ web API มาตรฐานล้วน

**Tech Stack:** React 18 + TS + Vite; web APIs: `navigator.vibrate`, `navigator.clipboard`, `navigator.share`, `navigator.onLine`, `visibilitychange`, Web App Manifest. ไม่เพิ่ม dependency

**อ้างอิง spec:** `docs/superpowers/specs/2026-06-28-ux-tier1-polish-design.md`

## Global Constraints

- **frontend (`frontend/`) เท่านั้น** — ไม่แตะ backend/ML, **ไม่เพิ่ม npm dependency**
- ทำบน branch `feature/ux-tier1-polish` (แตกจาก P1 แล้ว) — **ไม่แตะ main, ไม่ force-push** (repo มี collab)
- ไม่มี test framework → เกณฑ์ผ่านแต่ละ task = `cd frontend && npm run build` ผ่าน + ตรวจเบราว์เซอร์ (ห้ามเพิ่ม test runner)
- คง glass design + พฤติกรรมลูปแปลของ P1; เคารพ `prefers-reduced-motion` กับ animation ใหม่
- ทุกความสามารถใหม่ guard ด้วย feature-detection (vibrate/share/clipboard อาจไม่มีในบางเบราว์เซอร์ → ซ่อน/ข้ามอย่างนุ่มนวล)
- i18n: ทุกคีย์ใหม่ต้องมีครบทั้ง `th.ts` และ `en.ts` (en typed `: typeof th` → build บังคับ)
- localStorage settings key คงเดิม `tsl.settings.v1`

---

## File map

**สร้างใหม่:**
- `frontend/src/hooks/useFeedback.ts`
- `frontend/src/components/DevMetricsOverlay.tsx`
- `frontend/public/manifest.webmanifest`, `frontend/public/icon-192.png`, `frontend/public/icon-512.png`

**แก้ไข:**
- `frontend/src/hooks/useTranslate.ts` — `lastLatencyMs`, `errorKind`
- `frontend/src/hooks/useHolisticCapture.ts` — `fps`, `pause()`, `resume()`
- `frontend/src/hooks/SettingsProvider.tsx` — `showLandmarks` → `devMode` (+legacy migration)
- `frontend/src/hooks/HistoryProvider.tsx` — `remove(id)`
- `frontend/src/components/HistoryScreen.tsx` — ปุ่ม copy/delete/share
- `frontend/src/components/SettingsScreen.tsx` — toggle "Developer Mode"
- `frontend/src/App.tsx` — offline state, feedback, pause logic, DevMetricsOverlay, devMode wiring
- `frontend/src/i18n/th.ts` + `frontend/src/i18n/en.ts` — คีย์ใหม่
- `frontend/src/styles/tokens.css` — สไตล์ flash/pulse, dev-metrics, history actions, offline chip
- `frontend/index.html` — manifest/theme-color/apple-touch-icon

---

## Task 1 (F): Network/offline + error-kind + latency in useTranslate

**Files:**
- Modify: `frontend/src/hooks/useTranslate.ts`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/i18n/th.ts`, `frontend/src/i18n/en.ts`
- Modify: `frontend/src/styles/tokens.css`

**Interfaces:**
- Produces: `useTranslate()` now also returns `lastLatencyMs: number | null` and `errorKind: "api" | "network" | null`

- [ ] **Step 1: extend `useTranslate.ts`**

Add two state fields and set them in `run`. Replace the state block + `run` + `reset` + return with:

```ts
export type TranslateStatus = "idle" | "loading" | "success" | "error";
export type ErrorKind = "api" | "network" | null;

export interface TranslateState {
  status: TranslateStatus;
  result: TranslateResult | null;
  error: string | null;
  errorStatus: number | null;
  errorKind: ErrorKind;
  lastLatencyMs: number | null;
  run: (frames: number[][][], model?: string) => Promise<void>;
  reset: () => void;
}

export function useTranslate(): TranslateState {
  const [status, setStatus] = useState<TranslateStatus>("idle");
  const [result, setResult] = useState<TranslateResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [errorStatus, setErrorStatus] = useState<number | null>(null);
  const [errorKind, setErrorKind] = useState<ErrorKind>(null);
  const [lastLatencyMs, setLastLatencyMs] = useState<number | null>(null);

  const run = useCallback(async (frames: number[][][], model?: string) => {
    setStatus("loading");
    setError(null);
    setErrorStatus(null);
    setErrorKind(null);
    const t0 = Date.now();
    try {
      const res = await translate({ frames, model });
      setLastLatencyMs(Date.now() - t0);
      setResult(res);
      setStatus("success");
    } catch (e) {
      if (e instanceof ApiError) {
        setError(e.detail);
        setErrorStatus(e.status);
        setErrorKind("api");
      } else {
        // fetch rejects with TypeError on network failure
        setError("เชื่อมต่อไม่ได้");
        setErrorStatus(null);
        setErrorKind("network");
      }
      setStatus("error");
    }
  }, []);

  const reset = useCallback(() => {
    setStatus("idle");
    setResult(null);
    setError(null);
    setErrorStatus(null);
    setErrorKind(null);
  }, []);

  return { status, result, error, errorStatus, errorKind, lastLatencyMs, run, reset };
}
```

- [ ] **Step 2: add i18n keys** — append to `th.ts` (before `}`):

```ts
  offline: "ออฟไลน์",
  offlineHint: "ไม่มีการเชื่อมต่ออินเทอร์เน็ต",
  retry: "ลองใหม่",
```
and to `en.ts`:
```ts
  offline: "Offline",
  offlineHint: "No internet connection",
  retry: "Retry",
```

- [ ] **Step 3: App — online state + offline chip + network error UI**

In `AppShell` (App.tsx), add online tracking near the other state:
```tsx
  const [online, setOnline] = useState(typeof navigator === "undefined" ? true : navigator.onLine);
  useEffect(() => {
    const up = () => setOnline(true);
    const down = () => setOnline(false);
    window.addEventListener("online", up);
    window.addEventListener("offline", down);
    return () => { window.removeEventListener("online", up); window.removeEventListener("offline", down); };
  }, []);
```
In the top bar `top-controls-right`, add (before the phrases button):
```tsx
{!online && <span className="glass-chip offline-chip">⚠ {th.offline}</span>}
```
In `LiveStatusRow`, treat offline/network specially — change its props to also receive `online` and `errorKind`, and add a branch: if `!online || errorKind === "network"` → show `th.offlineHint`, color `#fca5a5`. (Pass `online={online}` and `errorKind={translator.errorKind}` from AppShell; add them to `LiveStatusRowProps` and the function signature, and read `const th = useI18n()` already present.)

Add the retry affordance inside the live result panel, after `<ResultCard .../>`, only when network error:
```tsx
{translator.status === "error" && translator.errorKind === "network" && (
  <button className="glass-action-btn" onClick={() => translatorRef.current.reset()}>{th.retry}</button>
)}
```

- [ ] **Step 4: CSS** — append to `tokens.css`:
```css
.offline-chip { background: rgba(220,38,38,0.35); border-color: rgba(255,255,255,0.4); }
```

- [ ] **Step 5: Build & commit**
```bash
cd frontend && npm run build
```
Expected: pass.
```bash
git add frontend/src/hooks/useTranslate.ts frontend/src/App.tsx frontend/src/i18n/th.ts frontend/src/i18n/en.ts frontend/src/styles/tokens.css
git commit -m "feat(net): offline/network error state + retry; expose translate latency"
```

---

## Task 2 (C): Haptic + visual flash feedback

**Files:**
- Create: `frontend/src/hooks/useFeedback.ts`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles/tokens.css`

**Interfaces:**
- Produces: `useFeedback(): { flash: "success" | "error" | null; signal: (kind: "success" | "error") => void }`

- [ ] **Step 1: create `useFeedback.ts`**
```ts
import { useCallback, useRef, useState } from "react";

export type FeedbackKind = "success" | "error";

export interface Feedback {
  flash: FeedbackKind | null;
  signal: (kind: FeedbackKind) => void;
}

const VIBRATE: Record<FeedbackKind, number | number[]> = {
  success: 30,
  error: [60, 40, 60],
};

/** Non-audio feedback for Deaf users: a brief visual flash + haptic buzz. */
export function useFeedback(): Feedback {
  const [flash, setFlash] = useState<FeedbackKind | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout>>();

  const signal = useCallback((kind: FeedbackKind) => {
    if (typeof navigator !== "undefined" && "vibrate" in navigator) {
      try { navigator.vibrate(VIBRATE[kind]); } catch { /* ignore */ }
    }
    setFlash(kind);
    clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => setFlash(null), 600);
  }, []);

  return { flash, signal };
}
```

- [ ] **Step 2: App wiring** — in `AppShell`:
```tsx
  const feedback = useFeedback();
  const feedbackRef = useRef(feedback); feedbackRef.current = feedback;
```
In the displayed-result effect, after `setDisplayedResult(...)` + history add, call `feedbackRef.current.signal("success")`. Add a new effect for errors:
```tsx
  useEffect(() => {
    if (translator.status === "error") feedbackRef.current.signal("error");
  }, [translator.status]);
```
Render the flash overlay as the first child of `<main className="app-immersive">`:
```tsx
{feedback.flash && <div className={`feedback-flash ${feedback.flash}`} aria-hidden="true" />}
```

- [ ] **Step 3: CSS** — append to `tokens.css`:
```css
@keyframes feedback-fade { 0% { opacity: 0.55; } 100% { opacity: 0; } }
.feedback-flash {
  position: absolute; inset: 0; z-index: 45; pointer-events: none;
  animation: feedback-fade 600ms ease-out forwards;
}
.feedback-flash.success { box-shadow: inset 0 0 0 4px rgba(34,197,94,0.9); }
.feedback-flash.error   { box-shadow: inset 0 0 0 4px rgba(239,68,68,0.9); }
@media (prefers-reduced-motion: reduce) { .feedback-flash { animation-duration: 1ms; } }
```

- [ ] **Step 4: Build & commit**
```bash
cd frontend && npm run build
git add frontend/src/hooks/useFeedback.ts frontend/src/App.tsx frontend/src/styles/tokens.css
git commit -m "feat(a11y): haptic + visual flash feedback on translate success/error"
```

---

## Task 3 (B): History copy / delete / share

**Files:**
- Modify: `frontend/src/hooks/HistoryProvider.tsx`
- Modify: `frontend/src/components/HistoryScreen.tsx`
- Modify: `frontend/src/i18n/th.ts`, `frontend/src/i18n/en.ts`
- Modify: `frontend/src/styles/tokens.css`

**Interfaces:**
- Consumes: `useI18n()`
- Produces: `useHistory()` now also returns `remove: (id: string) => void`

- [ ] **Step 1: HistoryProvider — add `remove`**

In the context value interface add `remove: (id: string) => void;`, default `remove: () => {}`. In the provider:
```tsx
  const remove = useCallback((id: string) => {
    setItems((prev) => prev.filter((it) => it.id !== id));
  }, []);
```
Add `remove` to the provider `value={{ items, add, clear, remove }}`.

- [ ] **Step 2: i18n keys** — `th.ts`:
```ts
  actionCopy: "คัดลอก",
  actionDelete: "ลบ",
  actionShare: "แชร์",
  copied: "คัดลอกแล้ว",
```
`en.ts`:
```ts
  actionCopy: "Copy",
  actionDelete: "Delete",
  actionShare: "Share",
  copied: "Copied",
```

- [ ] **Step 3: HistoryScreen — action buttons per row**

Pull `remove` from `useHistory()`. The existing row is a `<button className="history-row" onClick={() => speak(it.sentence)}>` — keep tap-to-speak on the word area, but move the actions into a sibling group so they don't trigger speak. Restructure each `<li>` to:
```tsx
<li key={it.id} className="history-row">
  <button type="button" className="history-word-btn" onClick={() => speak(it.sentence)}>
    <div className="history-word">
      <b>{it.sentence}</b>
      <small>{formatRelative(it.ts, th, lang)}</small>
    </div>
  </button>
  <div className="history-actions">
    <button type="button" aria-label={th.actionCopy} title={th.actionCopy}
      onClick={() => { if (navigator.clipboard) navigator.clipboard.writeText(it.sentence); }}>
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
    </button>
    {typeof navigator !== "undefined" && "share" in navigator && (
      <button type="button" aria-label={th.actionShare} title={th.actionShare}
        onClick={() => { navigator.share({ text: it.sentence }).catch(() => {}); }}>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.6" y1="13.5" x2="15.4" y2="17.5"/><line x1="15.4" y1="6.5" x2="8.6" y2="10.5"/></svg>
      </button>
    )}
    <button type="button" aria-label={th.actionDelete} title={th.actionDelete}
      onClick={() => remove(it.id)}>
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
    </button>
  </div>
</li>
```
(The `formatRelative(it.ts, th, lang)` helper and the `speak`, `lang`, `th` bindings already exist in this file from P1.)

- [ ] **Step 4: CSS** — append to `tokens.css`:
```css
.history-row { /* now a flex container holding word-button + actions */
  align-items: stretch; gap: var(--space-2);
}
.history-word-btn { flex: 1; min-width: 0; display: flex; align-items: center; text-align: left;
  background: none; border: none; color: #fff; font-family: var(--font-family); cursor: pointer; padding: 0; }
.history-actions { display: flex; align-items: center; gap: var(--space-1); flex-shrink: 0; }
.history-actions button { width: 38px; height: 38px; border-radius: var(--radius-md);
  display: grid; place-items: center; background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.14); color: rgba(255,255,255,0.85); cursor: pointer; }
.history-actions button:hover { background: rgba(255,255,255,0.16); }
.history-actions svg { width: 18px; height: 18px; }
```
> Note: the P1 `.history-row` rule already sets `display:flex; padding; background; border`. Keep those; this append only adds `align-items/gap` overrides and the new child classes. (CSS later rule wins.)

- [ ] **Step 5: Build & commit**
```bash
cd frontend && npm run build
git add frontend/src/hooks/HistoryProvider.tsx frontend/src/components/HistoryScreen.tsx frontend/src/i18n/th.ts frontend/src/i18n/en.ts frontend/src/styles/tokens.css
git commit -m "feat(history): per-item copy, delete, share"
```

---

## Task 4 (D): Pause MediaPipe on background / off-camera

**Files:**
- Modify: `frontend/src/hooks/useHolisticCapture.ts`
- Modify: `frontend/src/App.tsx`

**Interfaces:**
- Produces: `useHolisticCapture()` now also returns `pause: () => void` and `resume: () => void`

- [ ] **Step 1: useHolisticCapture — add pause/resume**

Add a hook-scope ref above the effect (near `overlayEnabledRef`):
```ts
  const pausedRef = useRef(false);
```
In `loop()`, after computing `rafId = requestAnimationFrame(loop);` and getting `video`, add an early skip when paused — insert right after the `if (!landmarker || !video || video.readyState < 2) return;` line:
```ts
      if (pausedRef.current) return; // paused: keep a light rAF tick, skip detection
```
Add `pause`/`resume` functions (next to `start`/`stop`):
```ts
  function pause() {
    pausedRef.current = true;
    videoRef.current?.pause();
  }
  function resume() {
    pausedRef.current = false;
    videoRef.current?.play().catch(() => {});
  }
```
Add them to the returned object: `return { videoRef, overlayRef, ready, recording, handsPresent, cameraError, start, stop, pause, resume };` and to the `HolisticCaptureState` interface:
```ts
  pause: () => void;
  resume: () => void;
```

- [ ] **Step 2: App — drive pause/resume**

In `AppShell`, add an effect (after the capture refs):
```tsx
  useEffect(() => {
    const apply = () => {
      const shouldRun = !document.hidden && screen === "camera";
      if (shouldRun) captureRef.current.resume();
      else captureRef.current.pause();
    };
    apply();
    document.addEventListener("visibilitychange", apply);
    return () => document.removeEventListener("visibilitychange", apply);
  }, [screen, capture.ready]);
```
(`capture.ready` in deps so the first resume runs once the camera initialised.)

- [ ] **Step 3: Build & manual check**
```bash
cd frontend && npm run build
```
Manual (`npm run dev`): switch to History/Settings tab → camera video freezes (paused), CPU drops; back to camera → resumes instantly (no re-init / no permission re-prompt). Hide the tab (switch app) → paused; return → resumes.

- [ ] **Step 4: Commit**
```bash
git add frontend/src/hooks/useHolisticCapture.ts frontend/src/App.tsx
git commit -m "feat(perf): pause MediaPipe when backgrounded or off camera tab"
```

---

## Task 5 (A): Developer Mode + metrics (fps / latency / confidence)

**Files:**
- Modify: `frontend/src/hooks/useHolisticCapture.ts` (add `fps`)
- Modify: `frontend/src/hooks/SettingsProvider.tsx` (`showLandmarks` → `devMode` + migration)
- Modify: `frontend/src/components/SettingsScreen.tsx` (Developer Mode toggle)
- Create: `frontend/src/components/DevMetricsOverlay.tsx`
- Modify: `frontend/src/App.tsx` (wire devMode + overlay)
- Modify: `frontend/src/i18n/th.ts`, `frontend/src/i18n/en.ts` (`settingsDevMode`)
- Modify: `frontend/src/styles/tokens.css` (overlay styles)

**Interfaces:**
- Consumes: `useTranslate().lastLatencyMs` (Task 1), `useHolisticCapture()` (Task 4)
- Produces: `useSettings()` exposes `devMode: boolean`, `setDevMode(v)`; `useHolisticCapture()` exposes `fps: number`; `DevMetricsOverlay({ fps, latencyMs, confidence })`

- [ ] **Step 1: useHolisticCapture — fps (only computed when overlay/dev on)**

Add state + refs at hook scope:
```ts
  const [fps, setFps] = useState(0);
  const fpsCountRef = useRef(0);
  const fpsStartRef = useRef(0);
```
In `loop()`, right after a successful `result = landmarker.detectForVideo(...)` and the `pick`/draw, when overlay (dev) is enabled, tally fps. Insert after `drawOverlay(...)`:
```ts
      if (overlayEnabledRef.current) {
        fpsCountRef.current += 1;
        if (fpsStartRef.current === 0) fpsStartRef.current = ts;
        const elapsed = ts - fpsStartRef.current;
        if (elapsed >= 1000) {
          setFps(Math.round((fpsCountRef.current * 1000) / elapsed));
          fpsCountRef.current = 0;
          fpsStartRef.current = ts;
        }
      } else if (fpsStartRef.current !== 0) {
        fpsStartRef.current = 0; fpsCountRef.current = 0; // reset when dev off
      }
```
Add `fps` to the return object and to `HolisticCaptureState` (`fps: number;`).

- [ ] **Step 2: SettingsProvider — rename showLandmarks → devMode (+migration)**

Replace the `Settings` interface, defaults, loader, context value, and setter:
```ts
export interface Settings {
  lang: Lang;
  devMode: boolean;
}
const DEFAULT_SETTINGS: Settings = { lang: "th", devMode: false };
```
In `loadSettings`, map legacy `showLandmarks`:
```ts
    return {
      lang: parsed.lang === "en" ? "en" : "th",
      devMode: Boolean(parsed.devMode ?? parsed.showLandmarks),
    };
```
Context value + setter: replace `showLandmarks`/`setShowLandmarks` with `devMode`/`setDevMode`:
```ts
interface SettingsContextValue extends Settings {
  setLang: (lang: Lang) => void;
  setDevMode: (v: boolean) => void;
}
```
default context `{ ...DEFAULT_SETTINGS, setLang: () => {}, setDevMode: () => {} }`; setter:
```ts
  const setDevMode = useCallback((devMode: boolean) => setSettings((s) => ({ ...s, devMode })), []);
```
provider value `{ ...settings, setLang, setDevMode }`.

- [ ] **Step 3: i18n** — `th.ts`: `settingsDevMode: "โหมดนักพัฒนา",`  ·  `en.ts`: `settingsDevMode: "Developer Mode",`

- [ ] **Step 4: SettingsScreen — Developer Mode toggle**

Replace the landmarks row: pull `devMode, setDevMode` from `useSettings()` (instead of `showLandmarks/setShowLandmarks`), change label to `th.settingsDevMode`, and toggle `setDevMode(!devMode)` with `aria-checked={devMode}`. (Keep the same `.toggle` markup; swap the icon if desired — keep existing.)

- [ ] **Step 5: create `DevMetricsOverlay.tsx`**
```tsx
import React from "react";

interface DevMetricsOverlayProps {
  fps: number;
  latencyMs: number | null;
  confidence: number | null; // 0..1
}

export function DevMetricsOverlay({ fps, latencyMs, confidence }: DevMetricsOverlayProps) {
  return (
    <div className="dev-metrics" aria-hidden="true">
      <span>FPS {fps}</span>
      <span>· {latencyMs == null ? "–" : `${latencyMs} ms`}</span>
      <span>· {confidence == null ? "–" : `${Math.round(confidence * 100)}%`}</span>
    </div>
  );
}
```

- [ ] **Step 6: App — wire devMode + overlay**

In `AppShell`: change `const { showLandmarks } = useSettings();` → `const { devMode } = useSettings();`; change `useHolisticCapture({ overlayEnabled: showLandmarks })` → `useHolisticCapture({ overlayEnabled: devMode })`. Import `DevMetricsOverlay`. Render it when `devMode && screen === "camera"`, after the top bar:
```tsx
{devMode && screen === "camera" && (
  <DevMetricsOverlay fps={capture.fps} latencyMs={translator.lastLatencyMs} confidence={displayedResult?.score ?? null} />
)}
```

- [ ] **Step 7: CSS** — append to `tokens.css`:
```css
.dev-metrics {
  position: absolute; top: calc(env(safe-area-inset-top, 0px) + 64px); left: var(--space-4); z-index: 12;
  display: flex; gap: var(--space-2); padding: var(--space-1) var(--space-3);
  border-radius: var(--radius-full); background: rgba(0,0,0,0.55);
  border: 1px solid var(--glass-border); color: #4ade80;
  font-size: var(--font-size-xs); font-weight: 700; font-variant-numeric: tabular-nums;
  backdrop-filter: var(--glass-blur);
}
```

- [ ] **Step 8: Build & commit**
```bash
cd frontend && npm run build
git add frontend/src/hooks/useHolisticCapture.ts frontend/src/hooks/SettingsProvider.tsx frontend/src/components/SettingsScreen.tsx frontend/src/components/DevMetricsOverlay.tsx frontend/src/App.tsx frontend/src/i18n/th.ts frontend/src/i18n/en.ts frontend/src/styles/tokens.css
git commit -m "feat(dev): Developer Mode toggle with fps/latency/confidence metrics"
```

---

## Task 6 (E): PWA — installable

**Files:**
- Create: `frontend/public/manifest.webmanifest`, `frontend/public/icon-192.png`, `frontend/public/icon-512.png`
- Modify: `frontend/index.html`

- [ ] **Step 1: create icon assets (one-time, via Chrome headless already on this machine)**

Write a temporary SVG and render two PNGs into `frontend/public/`. Run from repo root (Git Bash):
```bash
mkdir -p frontend/public
cat > /tmp/icon.svg <<'SVG'
<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512">
  <rect width="512" height="512" rx="112" fill="#0b1020"/>
  <text x="50%" y="54%" text-anchor="middle" dominant-baseline="middle"
        font-family="'Noto Sans Thai',system-ui,sans-serif" font-size="190" font-weight="800" fill="#ffffff">วท</text>
</svg>
SVG
CHROME="/c/Program Files/Google/Chrome/Application/chrome.exe"
"$CHROME" --headless=new --disable-gpu --force-device-scale-factor=1 --window-size=512,512 --default-background-color=00000000 --screenshot="frontend/public/icon-512.png" "file:///tmp/icon.svg"
"$CHROME" --headless=new --disable-gpu --force-device-scale-factor=1 --window-size=192,192 --default-background-color=00000000 --screenshot="frontend/public/icon-192.png" "file:///tmp/icon.svg"
ls -la frontend/public/icon-*.png
```
Expected: two non-empty PNGs. (If the headless render fails, fall back to any 192/512 PNG placeholder committed under `frontend/public/` — installability just needs the two sizes to exist.)

- [ ] **Step 2: create `frontend/public/manifest.webmanifest`**
```json
{
  "name": "วาทยากร",
  "short_name": "วาทยากร",
  "description": "แปลภาษามือไทยแบบเรียลไทม์",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#0b1020",
  "theme_color": "#0b1020",
  "icons": [
    { "src": "/icon-192.png", "sizes": "192x192", "type": "image/png" },
    { "src": "/icon-512.png", "sizes": "512x512", "type": "image/png" },
    { "src": "/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable" }
  ]
}
```

- [ ] **Step 3: link from `index.html`** — inside `<head>`, after the viewport meta, add:
```html
  <link rel="manifest" href="/manifest.webmanifest" />
  <meta name="theme-color" content="#0b1020" />
  <link rel="apple-touch-icon" href="/icon-192.png" />
```
Also update `<title>` to `วาทยากร | Conductor`.

- [ ] **Step 4: Build & verify**
```bash
cd frontend && npm run build
ls -la dist/manifest.webmanifest dist/icon-192.png dist/icon-512.png
```
Expected: build passes and the manifest + icons are copied into `dist/` (Vite copies `public/` verbatim). Manual: in Chrome devtools → Application → Manifest shows name/icons with no errors; install icon appears.

- [ ] **Step 5: Commit**
```bash
git add frontend/public/manifest.webmanifest frontend/public/icon-192.png frontend/public/icon-512.png frontend/index.html
git commit -m "feat(pwa): installable web app manifest + icons + theme-color"
```

---

## Self-Review (เทียบ spec)

**1. Spec coverage:**
- §3 Developer Mode + metrics → Task 5 ✓ (fps in useHolisticCapture, latency in useTranslate(Task 1), confidence from displayedResult, merged toggle + migration)
- §4 history copy/delete/share → Task 3 ✓ (no search, per decision)
- §5 haptic + flash → Task 2 ✓ (vibrate guarded, reduced-motion)
- §6 pause on background → Task 4 ✓ (pause/resume, no teardown)
- §7 PWA installable (no SW) → Task 6 ✓
- §8 network/offline + errorKind + retry → Task 1 ✓
- §9 file list → matches ✓ ; §11 out-of-scope (search/SW/STT) not built ✓

**2. Placeholder scan:** ไม่มี TBD/TODO; ทุก step มีโค้ด/คำสั่งจริง (icon gen มี fallback ระบุไว้) ✓

**3. Type consistency:** `devMode`/`setDevMode` ใช้ตรงกันใน SettingsProvider/SettingsScreen/App; `fps` (number) จาก useHolisticCapture → DevMetricsOverlay; `lastLatencyMs: number|null` + `errorKind` จาก useTranslate ใช้ใน App/DevMetricsOverlay; `remove(id)` จาก HistoryProvider ใช้ใน HistoryScreen; `useFeedback().signal/flash` ใช้ใน App ✓

> หมายเหตุข้าม-task: Task 5 เปลี่ยน `showLandmarks`→`devMode`. ก่อน Task 5 โค้ด P1 ยังอ้าง `showLandmarks` อยู่ — Task 5 แก้ทุกจุดที่อ้าง (SettingsProvider, SettingsScreen, App) ในคราวเดียว จึง build ผ่านหลัง Task 5. Tasks 1–4 ไม่แตะ `showLandmarks` จึง build ผ่านระหว่างทาง
