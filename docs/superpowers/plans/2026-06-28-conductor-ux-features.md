# Conductor (วาทยากร) UX/UI v2 — Implementation Plan (P1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** เพิ่มความใช้งานได้จริงให้แอปแปลภาษามือ "วาทยากร" — floating glass bottom nav (กล้อง/ประวัติ/ตั้งค่า), ปุ่มลำโพง TTS บนการ์ดผลแปล, ประวัติ (localStorage), และตั้งค่าสลับภาษา UI ไทย/EN

**Architecture:** เพิ่ม React Context providers (Settings, History) ครอบแอป + hooks (`useSpeech`, `useI18n`) แล้ว refactor `App.tsx` เป็น shell ที่คง `CameraView`+`useHolisticCapture` mount ไว้ตลอด และสลับ overlay ของหน้า ประวัติ/ตั้งค่า ทับกล้อง โดยลูปแปลอัตโนมัติเดิมทำงานเฉพาะแท็บกล้อง

**Tech Stack:** React 18 + TypeScript + Vite, `@mediapipe/tasks-vision`, Web Speech API (`speechSynthesis`), localStorage. ไม่มี dependency ใหม่

**อ้างอิง spec:** `docs/superpowers/specs/2026-06-28-conductor-ux-features-design.md`

## Global Constraints

- ขอบเขต = **P1 เท่านั้น** — ไม่ทำ favorites, autoSpeak, automated tests (เลื่อน P2)
- ไม่เพิ่ม npm dependency ใหม่ และ **ไม่แตะ backend/ML** (`frontend/` เท่านั้น)
- ไม่มี test framework ใน P1 → **เกณฑ์ผ่านแต่ละ task = `cd frontend && npm run build` ผ่าน (tsc + vite) + ตรวจในเบราว์เซอร์ตามที่ระบุ**
- ทำงานบนสาขา `feature/conductor-ux-v2` (สร้างแล้ว) — commit ถี่ ทีละ task
- **ห้าม unmount `CameraView` / `useHolisticCapture` ตอนสลับแท็บ** (re-init MediaPipe แพง)
- ลูปแปลอัตโนมัติ (presence-based) ต้อง **หยุด trigger เมื่อ `screen !== "camera"`**
- ประวัติ: key `tsl.history.v1`, cap 100, dedup (คำเดิมซ้ำภายใน 30 วิ ไม่บันทึก); บันทึกเฉพาะผลที่ `score >= CONFIDENCE_FLOOR (0.3)`
- การตั้งค่า: key `tsl.settings.v1` (`lang`, `showLandmarks`)
- ผลแปลภาษามือยังเป็นภาษาไทย (เอาต์พุตโมเดล) — i18n แปลเฉพาะ UI chrome
- แบรนด์ = "วาทยากร"
- TTS เริ่มต้นภาษา `th-TH` ผ่าน Web Speech API; ถ้าเบราว์เซอร์ไม่รองรับให้ซ่อนปุ่มลำโพง (graceful)
- ทุกคำสั่ง `npm` รันใน `frontend/` (เช่น `cd frontend && npm run build`)

---

## File map (ภาพรวมไฟล์)

**สร้างใหม่:**
- `frontend/src/hooks/SettingsProvider.tsx` — context: lang + showLandmarks + persist
- `frontend/src/hooks/HistoryProvider.tsx` — context: list + add(dedup/cap) + clear + persist
- `frontend/src/hooks/useSpeech.ts` — Web Speech API wrapper
- `frontend/src/i18n/en.ts` — พจนานุกรมอังกฤษ (ครบทุก key ของ th)
- `frontend/src/i18n/index.ts` — รวม dictionary + `useI18n()`
- `frontend/src/components/BottomNav.tsx` — floating glass nav 3 แท็บ
- `frontend/src/components/HistoryScreen.tsx` — หน้าประวัติ
- `frontend/src/components/SettingsScreen.tsx` — หน้าตั้งค่า

**แก้ไข:**
- `frontend/src/i18n/th.ts` — เพิ่ม key ใหม่ + เปลี่ยน `appTitle` เป็น "วาทยากร"
- `frontend/src/components/ResultCard.tsx` — เพิ่มปุ่มลำโพง + ใช้ `useI18n`
- `frontend/src/components/SupportedPhrases.tsx` — ใช้ `useI18n`
- `frontend/src/App.tsx` — restructure เป็น shell + providers + nav + overlays
- `frontend/src/styles/tokens.css` — เพิ่มสไตล์ nav/ลำโพง/overlay/ประวัติ/ตั้งค่า + แก้ overflow

> หมายเหตุ: `RecordButton.tsx`, `ModelPicker.tsx`, `StatusBar.tsx` เป็น dead code (ไม่ถูก `App` ใช้) และยัง `import { th }` ได้ปกติ เพราะ `th.ts` ยังอยู่ — **ไม่ต้องแก้** ในรอบนี้

---

## Task 1: Settings provider (lang + landmarks + persist)

**Files:**
- Create: `frontend/src/hooks/SettingsProvider.tsx`
- Modify: `frontend/src/App.tsx` (ครอบ provider ชั่วคราว)

**Interfaces:**
- Produces: `type Lang = "th" | "en"`; `SettingsProvider`; `useSettings(): { lang: Lang; showLandmarks: boolean; setLang(l: Lang): void; setShowLandmarks(v: boolean): void }`

- [ ] **Step 1: สร้าง `frontend/src/hooks/SettingsProvider.tsx`**

```tsx
import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from "react";

export type Lang = "th" | "en";

export interface Settings {
  lang: Lang;
  showLandmarks: boolean;
}

const DEFAULT_SETTINGS: Settings = { lang: "th", showLandmarks: false };
const STORAGE_KEY = "tsl.settings.v1";

function loadSettings(): Settings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_SETTINGS;
    const parsed = JSON.parse(raw);
    return {
      lang: parsed.lang === "en" ? "en" : "th",
      showLandmarks: Boolean(parsed.showLandmarks),
    };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

interface SettingsContextValue extends Settings {
  setLang: (lang: Lang) => void;
  setShowLandmarks: (v: boolean) => void;
}

const SettingsContext = createContext<SettingsContextValue>({
  ...DEFAULT_SETTINGS,
  setLang: () => {},
  setShowLandmarks: () => {},
});

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<Settings>(loadSettings);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    } catch {
      /* ignore quota/availability errors */
    }
  }, [settings]);

  const setLang = useCallback((lang: Lang) => setSettings((s) => ({ ...s, lang })), []);
  const setShowLandmarks = useCallback(
    (showLandmarks: boolean) => setSettings((s) => ({ ...s, showLandmarks })),
    []
  );

  return (
    <SettingsContext.Provider value={{ ...settings, setLang, setShowLandmarks }}>
      {children}
    </SettingsContext.Provider>
  );
}

export function useSettings(): SettingsContextValue {
  return useContext(SettingsContext);
}
```

- [ ] **Step 2: ครอบ `SettingsProvider` ใน `App.tsx`**

แก้เฉพาะ default export ท้ายไฟล์ `frontend/src/App.tsx`:

```tsx
import { SettingsProvider } from "./hooks/SettingsProvider";
// ...
export default function App() {
  return (
    <ModelsProvider>
      <SettingsProvider>
        <Translator />
      </SettingsProvider>
    </ModelsProvider>
  );
}
```

- [ ] **Step 3: Build**

Run: `cd frontend && npm run build`
Expected: ผ่าน ไม่มี error TypeScript

- [ ] **Step 4: Commit**

```bash
git add frontend/src/hooks/SettingsProvider.tsx frontend/src/App.tsx
git commit -m "feat(settings): add SettingsProvider (lang + landmarks, localStorage)"
```

---

## Task 2: i18n multi-language (en.ts + index + useI18n)

**Files:**
- Modify: `frontend/src/i18n/th.ts`
- Create: `frontend/src/i18n/en.ts`
- Create: `frontend/src/i18n/index.ts`

**Interfaces:**
- Consumes: `useSettings()` (Task 1)
- Produces: `type Dict = typeof th`; `useI18n(): Dict`; พจนานุกรมมี key ใหม่: `brandShort, navCamera, navHistory, navSettings, cameraLive, cameraOpening, showSignHint, ariaSpeak, speaking, historyTitle, historyEmpty, settingsTitle, settingsLanguage, settingsLandmarks, settingsClearHistory, confirmClear`

- [ ] **Step 1: เพิ่ม key ใหม่ + เปลี่ยน appTitle ใน `frontend/src/i18n/th.ts`**

เปลี่ยนบรรทัด `appTitle`:

```ts
  appTitle: "วาทยากร",
```

แล้วเพิ่ม key เหล่านี้ก่อน `} as const;` (ต่อท้ายรายการเดิม):

```ts
  // Brand / nav
  brandShort: "วท",
  navCamera: "กล้อง",
  navHistory: "ประวัติ",
  navSettings: "ตั้งค่า",

  // Camera live chip / hint
  cameraLive: "กล้อง Live",
  cameraOpening: "กำลังเปิด...",
  showSignHint: "แสดงภาษามือต่อกล้อง",

  // Speaker / TTS
  ariaSpeak: "ฟังเสียงอ่าน",
  speaking: "กำลังพูด…",

  // History
  historyTitle: "ประวัติ",
  historyEmpty: "ยังไม่มีประวัติ",

  // Settings
  settingsTitle: "ตั้งค่า",
  settingsLanguage: "ภาษา",
  settingsLandmarks: "เส้นโครงร่าง",
  settingsClearHistory: "ล้างประวัติ",
  confirmClear: "ล้างประวัติทั้งหมด?",
```

- [ ] **Step 2: สร้าง `frontend/src/i18n/en.ts` (ครบทุก key, typed `: typeof th`)**

```ts
import { th } from "./th";

/** English UI strings. Typed against `th` so missing keys are a compile error. */
export const en: typeof th = {
  appTitle: "Conductor",
  appSubtitle: "Record a sign and get the translation",
  assistantLabel: "Result",
  cameraPanelLabel: "Camera",
  chatPanelLabel: "Conversation",

  cameraInit: "Preparing camera...",
  cameraReady: "Camera ready",
  cameraError: "Cannot access the camera",
  cameraErrorHint: "Please allow camera access in your browser",
  cameraRetry: "Try again",

  recordStart: "Start recording",
  recordStop: "Stop recording",
  recording: "Recording...",
  frames: (n: number) => `${n} frames`,

  translating: "Translating...",
  resultPlaceholder: "—",
  confidence: (pct: number) => `Confidence ${pct}%`,
  noFrames: "No motion detected. Try recording again",

  modelLabel: "Model",
  modelUnavailable: "Unavailable",
  modelLoading: "Loading models...",
  modelLoadError: "Failed to load model list",

  errorModelUnavailable: "This model is not available yet",
  errorGeneric: "Something went wrong. Please try again",

  supportedPhrasesTitle: "Supported phrases",
  supportedPhrasesScope: "The current model only knows phrases from the TSL-51 dataset",
  supportedPhrasesUnavailable: "Could not load the phrase list",
  supportedPhrasesEmpty: "No supported phrases yet",
  supportedPhrasesCount: (n: number) => `${n} phrases`,
  supportedPhrasesShow: "Show supported phrases",
  supportedPhrasesHide: "Hide",

  // Brand / nav
  brandShort: "CD",
  navCamera: "Camera",
  navHistory: "History",
  navSettings: "Settings",

  // Camera live chip / hint
  cameraLive: "Live",
  cameraOpening: "Opening...",
  showSignHint: "Show a sign to the camera",

  // Speaker / TTS
  ariaSpeak: "Listen",
  speaking: "Speaking…",

  // History
  historyTitle: "History",
  historyEmpty: "No history yet",

  // Settings
  settingsTitle: "Settings",
  settingsLanguage: "Language",
  settingsLandmarks: "Landmarks",
  settingsClearHistory: "Clear history",
  confirmClear: "Clear all history?",
};
```

- [ ] **Step 3: สร้าง `frontend/src/i18n/index.ts`**

```ts
import { useSettings, Lang } from "../hooks/SettingsProvider";
import { th } from "./th";
import { en } from "./en";

export type Dict = typeof th;

export const dictionaries: Record<Lang, Dict> = { th, en };

/** Returns the active dictionary based on the user's language setting. */
export function useI18n(): Dict {
  const { lang } = useSettings();
  return dictionaries[lang];
}
```

- [ ] **Step 4: Build (ตรวจ key ครบ — ถ้า en ขาด key จะ error)**

Run: `cd frontend && npm run build`
Expected: ผ่าน — ถ้า `en.ts` ขาด key ใด TypeScript จะ error `Property 'X' is missing`

- [ ] **Step 5: Commit**

```bash
git add frontend/src/i18n/th.ts frontend/src/i18n/en.ts frontend/src/i18n/index.ts
git commit -m "feat(i18n): add English dictionary + useI18n hook"
```

---

## Task 3: Refactor live components to useI18n + brand rename

**Files:**
- Modify: `frontend/src/components/SupportedPhrases.tsx`
- Modify: `frontend/src/App.tsx`
- (ResultCard เปลี่ยนใน Task 5)

**Interfaces:**
- Consumes: `useI18n()` (Task 2)

- [ ] **Step 1: `SupportedPhrases.tsx` — ใช้ useI18n**

ใน `frontend/src/components/SupportedPhrases.tsx`:
- ลบ `import { th } from "../i18n/th";`
- เพิ่ม `import { useI18n } from "../i18n";`
- ในตัวฟังก์ชัน `SupportedPhrases` เพิ่มบรรทัดแรก: `const th = useI18n();`

(ที่เหลือใช้ `th.x` ได้เหมือนเดิมเพราะตั้งชื่อตัวแปรเป็น `th`)

- [ ] **Step 2: `App.tsx` — ใช้ useI18n ใน `Translator` และ `LiveStatusRow`**

ใน `frontend/src/App.tsx`:
- ลบ `import { th } from "./i18n/th";` → เพิ่ม `import { useI18n } from "./i18n";`
- ในฟังก์ชัน `Translator()` เพิ่มบรรทัดแรก: `const th = useI18n();`
- ในฟังก์ชัน `LiveStatusRow(...)` เพิ่มบรรทัดแรก: `const th = useI18n();`
- เปลี่ยน brand mark จากค่าคงที่ `"TS"` เป็น `{th.brandShort}`:

```tsx
<div className="brand-mark-glass">{th.brandShort}</div>
```

- เปลี่ยน live chip ให้ใช้ i18n:

```tsx
{capture.ready ? th.cameraLive : th.cameraOpening}
```

- ใน `LiveStatusRow` เปลี่ยนข้อความ hardcoded `"แสดงภาษามือต่อกล้อง"` เป็น `th.showSignHint`

- [ ] **Step 3: Build**

Run: `cd frontend && npm run build`
Expected: ผ่าน

- [ ] **Step 4: ตรวจในเบราว์เซอร์**

Run: `cd frontend && npm run dev` → เปิด URL ที่ Vite แจ้ง
Expected: แบรนด์ขึ้น "วาทยากร" และ chip กล้องขึ้น "กล้อง Live"

- [ ] **Step 5: Commit**

```bash
git add frontend/src/App.tsx frontend/src/components/SupportedPhrases.tsx
git commit -m "refactor(i18n): use useI18n in App + SupportedPhrases; brand = วาทยากร"
```

---

## Task 4: useSpeech (Web Speech API)

**Files:**
- Create: `frontend/src/hooks/useSpeech.ts`

**Interfaces:**
- Produces: `useSpeech(): { speak(text: string, lang?: string): void; cancel(): void; speaking: boolean; supported: boolean }`

- [ ] **Step 1: สร้าง `frontend/src/hooks/useSpeech.ts`**

```ts
import { useCallback, useEffect, useRef, useState } from "react";

export interface SpeechController {
  speak: (text: string, lang?: string) => void;
  cancel: () => void;
  speaking: boolean;
  supported: boolean;
}

/** Thin wrapper over the Web Speech API (speechSynthesis). */
export function useSpeech(): SpeechController {
  const supported = typeof window !== "undefined" && "speechSynthesis" in window;
  const [speaking, setSpeaking] = useState(false);
  const voicesRef = useRef<SpeechSynthesisVoice[]>([]);

  useEffect(() => {
    if (!supported) return;
    const load = () => {
      voicesRef.current = window.speechSynthesis.getVoices();
    };
    load();
    window.speechSynthesis.addEventListener("voiceschanged", load);
    return () => window.speechSynthesis.removeEventListener("voiceschanged", load);
  }, [supported]);

  const speak = useCallback(
    (text: string, lang = "th-TH") => {
      if (!supported || !text) return;
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.lang = lang;
      const base = lang.split("-")[0];
      const voice =
        voicesRef.current.find((v) => v.lang === lang) ||
        voicesRef.current.find((v) => v.lang.startsWith(base));
      if (voice) u.voice = voice;
      u.onstart = () => setSpeaking(true);
      u.onend = () => setSpeaking(false);
      u.onerror = () => setSpeaking(false);
      window.speechSynthesis.speak(u);
    },
    [supported]
  );

  const cancel = useCallback(() => {
    if (!supported) return;
    window.speechSynthesis.cancel();
    setSpeaking(false);
  }, [supported]);

  return { speak, cancel, speaking, supported };
}
```

- [ ] **Step 2: Build**

Run: `cd frontend && npm run build`
Expected: ผ่าน

- [ ] **Step 3: Commit**

```bash
git add frontend/src/hooks/useSpeech.ts
git commit -m "feat(tts): add useSpeech hook (Web Speech API)"
```

---

## Task 5: Speaker button on ResultCard + overflow fix

**Files:**
- Modify: `frontend/src/components/ResultCard.tsx`
- Modify: `frontend/src/styles/tokens.css`

**Interfaces:**
- Consumes: `useSpeech()` (Task 4), `useI18n()` (Task 2)

- [ ] **Step 1: แก้ overflow ใน `tokens.css` + เพิ่มสไตล์ปุ่มลำโพง**

ใน `frontend/src/styles/tokens.css` หา block `.result-glass-panel.live { transform: none; }` แล้วเพิ่ม `overflow: visible;` และ `max-height: none;`:

```css
/* Live mode: always visible (no slide-in needed) */
.result-glass-panel.live {
  transform: none;
  overflow: visible;   /* allow the speaker button to float past the panel edge */
  max-height: none;
}
```

ต่อท้ายไฟล์ เพิ่มสไตล์ปุ่มลำโพง:

```css
/* ── Result speaker button (floats on the card's top-right corner) ── */
.result-speaker-btn {
  position: absolute;
  top: -28px;
  right: 16px;
  width: 56px;
  height: 56px;
  border-radius: 50%;
  display: grid;
  place-items: center;
  color: #fff;
  cursor: pointer;
  z-index: 2;
  background: linear-gradient(180deg, rgba(37, 99, 235, 0.96), rgba(29, 78, 216, 0.96));
  border: 2px solid rgba(255, 255, 255, 0.5);
  box-shadow: 0 10px 24px rgba(37, 99, 235, 0.5), 0 2px 6px rgba(0, 0, 0, 0.4);
  transition: transform var(--transition);
}
.result-speaker-btn:active { transform: scale(0.94); }
.result-speaker-btn svg { width: 26px; height: 26px; }
.result-speaker-btn.playing { animation: spk-ring 1.4s ease-out infinite; }
@keyframes spk-ring {
  0%   { box-shadow: 0 10px 24px rgba(37, 99, 235, 0.5), 0 0 0 0 rgba(74, 222, 128, 0.6); }
  100% { box-shadow: 0 10px 24px rgba(37, 99, 235, 0.5), 0 0 0 16px rgba(74, 222, 128, 0); }
}
```

- [ ] **Step 2: เพิ่มปุ่มลำโพงใน `ResultCard.tsx`**

แทนที่ทั้งไฟล์ `frontend/src/components/ResultCard.tsx` ด้วย:

```tsx
import React from "react";
import { TranslateResult } from "../api/client";
import { useI18n } from "../i18n";
import { useSpeech } from "../hooks/useSpeech";

interface ResultCardProps {
  status: "idle" | "loading" | "success" | "error";
  result: TranslateResult | null;
  error: string | null;
  errorStatus: number | null;
  variant?: "glass";
}

export function ResultCard({ status, result, error, errorStatus, variant }: ResultCardProps) {
  const th = useI18n();
  const { speak, speaking, supported } = useSpeech();
  const glass = variant === "glass";
  const pct = result ? Math.round(result.score * 100) : null;

  const textColor = glass ? "#fff" : "var(--color-text)";
  const mutedColor = glass ? "rgba(255,255,255,0.6)" : "var(--color-text-muted)";
  const placeholderColor = glass ? "rgba(255,255,255,0.28)" : "var(--color-text-placeholder)";
  const trackColor = glass ? "rgba(255,255,255,0.15)" : "var(--color-border)";

  // Show the speaker whenever a result sentence is present and TTS is supported.
  const showSpeaker = supported && Boolean(result?.sentence);

  return (
    <div
      aria-live="polite"
      aria-atomic="true"
      className={glass ? undefined : "message-bubble"}
      style={{
        position: "relative",
        minHeight: glass ? 72 : 120,
        display: "flex",
        flexDirection: "column",
        gap: "var(--space-3)",
      }}
    >
      {showSpeaker && (
        <button
          type="button"
          className={`result-speaker-btn${speaking ? " playing" : ""}`}
          onClick={() => speak(result!.sentence)}
          aria-label={th.ariaSpeak}
          title={th.ariaSpeak}
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
            <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
            <path d="M19.07 4.93a10 10 0 0 1 0 14.14" />
          </svg>
        </button>
      )}

      {status === "loading" && (
        <div
          aria-hidden="true"
          style={{
            position: "absolute",
            top: 0,
            left: "-var(--space-5)",
            right: "-var(--space-5)",
            height: 2,
            overflow: "hidden",
            borderRadius: "var(--radius-lg) var(--radius-lg) 0 0",
          }}
        >
          <div style={{ animation: "scanning-bar 1.4s ease-in-out infinite" }} className="scanning-shimmer" />
        </div>
      )}

      {status === "loading" && result && (
        <>
          <p
            style={{
              fontSize: "var(--font-size-3xl)",
              fontWeight: 700,
              color: textColor,
              lineHeight: 1.4,
              wordBreak: "break-word",
              opacity: 0.4,
              transition: "opacity 300ms ease",
            }}
          >
            {result.sentence || th.resultPlaceholder}
          </p>
          {pct !== null && (
            <div style={{ opacity: 0.3, display: "flex", flexDirection: "column", gap: "var(--space-2)" }}>
              <span style={{ fontSize: "var(--font-size-sm)", color: mutedColor }}>{th.confidence(pct)}</span>
              <ConfidenceBar pct={pct} trackColor={trackColor} />
            </div>
          )}
        </>
      )}

      {status === "loading" && !result && (
        <div style={{ display: "flex", alignItems: "center", gap: "var(--space-3)", color: mutedColor }}>
          <Spinner glass={glass} />
          <span style={{ fontSize: "var(--font-size-lg)", color: mutedColor }}>{th.translating}</span>
        </div>
      )}

      {status === "error" && (
        <div style={{ color: glass ? "#fca5a5" : "var(--color-danger)", fontSize: "var(--font-size-base)" }}>
          {errorStatus === 503 ? th.errorModelUnavailable : error ?? th.errorGeneric}
        </div>
      )}

      {status === "success" && result && (
        <>
          <p
            style={{
              fontSize: "var(--font-size-3xl)",
              fontWeight: 700,
              color: textColor,
              lineHeight: 1.4,
              wordBreak: "break-word",
            }}
          >
            {result.sentence || th.resultPlaceholder}
          </p>
          {pct !== null && (
            <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-2)" }}>
              <span style={{ fontSize: "var(--font-size-sm)", color: mutedColor }}>{th.confidence(pct)}</span>
              <ConfidenceBar pct={pct} trackColor={trackColor} />
            </div>
          )}
        </>
      )}

      {status === "idle" && (
        <p style={{ fontSize: "var(--font-size-3xl)", color: placeholderColor, fontWeight: 300 }}>
          {th.resultPlaceholder}
        </p>
      )}
    </div>
  );
}

function ConfidenceBar({ pct, trackColor }: { pct: number; trackColor: string }) {
  return (
    <div style={{ height: 6, borderRadius: "var(--radius-full)", background: trackColor, overflow: "hidden" }}>
      <div
        style={{
          height: "100%",
          width: `${pct}%`,
          background: pct >= 70 ? "#22c55e" : pct >= 40 ? "#f59e0b" : "#ef4444",
          borderRadius: "var(--radius-full)",
          transition: "width 0.4s ease",
        }}
      />
    </div>
  );
}

function Spinner({ glass }: { glass?: boolean }) {
  return (
    <div
      style={{
        width: 20,
        height: 20,
        border: `3px solid ${glass ? "rgba(255,255,255,0.2)" : "var(--color-border)"}`,
        borderTopColor: glass ? "#fff" : "var(--color-primary)",
        borderRadius: "50%",
        flexShrink: 0,
        animation: "spin 0.8s linear infinite",
      }}
    />
  );
}
```

- [ ] **Step 3: Build**

Run: `cd frontend && npm run build`
Expected: ผ่าน

- [ ] **Step 4: ตรวจในเบราว์เซอร์**

Run: `cd frontend && npm run dev`
Expected: เมื่อมีผลแปล ปุ่มลำโพงลอยมุมขวาบนของการ์ด **ไม่ถูกตัด**; กดแล้วมีเสียงอ่านคำ (ถ้าเครื่องมีเสียงไทย) และปุ่มเรืองแสงระหว่างพูด

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/ResultCard.tsx frontend/src/styles/tokens.css
git commit -m "feat(tts): add floating speaker button on result card + overflow fix"
```

---

## Task 6: History provider + record on translate success

**Files:**
- Create: `frontend/src/hooks/HistoryProvider.tsx`
- Modify: `frontend/src/App.tsx`

**Interfaces:**
- Produces: `interface HistoryItem { id: string; sentence: string; score: number; model: string; ts: number }`; `HistoryProvider`; `useHistory(): { items: HistoryItem[]; add(e: { sentence: string; score: number; model: string }): void; clear(): void }`

- [ ] **Step 1: สร้าง `frontend/src/hooks/HistoryProvider.tsx`**

```tsx
import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from "react";

export interface HistoryItem {
  id: string;
  sentence: string;
  score: number;
  model: string;
  ts: number;
}

const STORAGE_KEY = "tsl.history.v1";
const CAP = 100;
const DEDUP_WINDOW_MS = 30_000;

function load(): HistoryItem[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as HistoryItem[]) : [];
  } catch {
    return [];
  }
}

function genId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

interface HistoryContextValue {
  items: HistoryItem[];
  add: (entry: { sentence: string; score: number; model: string }) => void;
  clear: () => void;
}

const HistoryContext = createContext<HistoryContextValue>({
  items: [],
  add: () => {},
  clear: () => {},
});

export function HistoryProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<HistoryItem[]>(load);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
    } catch {
      /* ignore */
    }
  }, [items]);

  const add = useCallback((entry: { sentence: string; score: number; model: string }) => {
    setItems((prev) => {
      const last = prev[0];
      // Dedup: skip identical sentence within the dedup window (auto-loop fires repeatedly).
      if (last && last.sentence === entry.sentence && Date.now() - last.ts < DEDUP_WINDOW_MS) {
        return prev;
      }
      const next: HistoryItem = { id: genId(), ts: Date.now(), ...entry };
      return [next, ...prev].slice(0, CAP);
    });
  }, []);

  const clear = useCallback(() => setItems([]), []);

  return <HistoryContext.Provider value={{ items, add, clear }}>{children}</HistoryContext.Provider>;
}

export function useHistory(): HistoryContextValue {
  return useContext(HistoryContext);
}
```

- [ ] **Step 2: ครอบ `HistoryProvider` + บันทึกผลใน `App.tsx`**

ใน `frontend/src/App.tsx`:
- เพิ่ม import: `import { HistoryProvider, useHistory } from "./hooks/HistoryProvider";`
- ใน `Translator()` เพิ่ม: `const history = useHistory();` และ ref: `const historyRef = useRef(history); historyRef.current = history;`
- ใน effect ที่อัปเดต `displayedResult` (เงื่อนไข `translator.result.score >= CONFIDENCE_FLOOR`) เพิ่มการบันทึก:

```tsx
  useEffect(() => {
    if (translator.status === "success" && translator.result) {
      if (translator.result.score >= CONFIDENCE_FLOOR) {
        setDisplayedResult(translator.result);
        historyRef.current.add({
          sentence: translator.result.sentence,
          score: translator.result.score,
          model: translator.result.model,
        });
      }
    }
  }, [translator.status, translator.result]);
```

- แก้ default export ให้ครอบ provider:

```tsx
export default function App() {
  return (
    <ModelsProvider>
      <SettingsProvider>
        <HistoryProvider>
          <Translator />
        </HistoryProvider>
      </SettingsProvider>
    </ModelsProvider>
  );
}
```

- [ ] **Step 3: Build**

Run: `cd frontend && npm run build`
Expected: ผ่าน

- [ ] **Step 4: ตรวจในเบราว์เซอร์ (DevTools)**

Run: `cd frontend && npm run dev` → แปลภาษามือ 1-2 ครั้ง → เปิด DevTools › Application › Local Storage
Expected: มี key `tsl.history.v1` เก็บรายการ; คำเดิมซ้ำติด ๆ กันไม่เพิ่มซ้ำ

- [ ] **Step 5: Commit**

```bash
git add frontend/src/hooks/HistoryProvider.tsx frontend/src/App.tsx
git commit -m "feat(history): add HistoryProvider + record successful translations"
```

---

## Task 7: BottomNav (floating glass)

**Files:**
- Create: `frontend/src/components/BottomNav.tsx`
- Modify: `frontend/src/styles/tokens.css`

**Interfaces:**
- Consumes: `useI18n()` (Task 2)
- Produces: `type Screen = "camera" | "history" | "settings"`; `BottomNav({ active: Screen; onChange(s: Screen): void })`

- [ ] **Step 1: เพิ่มสไตล์ nav ต่อท้าย `tokens.css`**

```css
/* ── Floating glass bottom nav ── */
.bottom-nav-float {
  position: absolute;
  left: 50%;
  bottom: max(var(--space-5), env(safe-area-inset-bottom, 0px));
  transform: translateX(-50%);
  z-index: 40;
  display: flex;
  gap: var(--space-1);
  padding: var(--space-2);
  background: var(--glass-bg);
  backdrop-filter: var(--glass-blur);
  -webkit-backdrop-filter: var(--glass-blur);
  border: 1px solid var(--glass-border);
  border-top-color: var(--glass-border-bright);
  border-radius: var(--radius-lg);
  box-shadow: var(--glass-shadow);
}
.bottom-nav-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
  padding: var(--space-2) var(--space-4);
  border-radius: var(--radius-md);
  background: none;
  border: none;
  color: rgba(255, 255, 255, 0.6);
  font-family: var(--font-family);
  font-size: var(--font-size-xs);
  font-weight: 700;
  cursor: pointer;
  transition: background var(--transition), color var(--transition);
}
.bottom-nav-item.on { background: rgba(37, 99, 235, 0.55); color: #fff; }
.bottom-nav-ico svg { width: 22px; height: 22px; display: block; }
```

- [ ] **Step 2: สร้าง `frontend/src/components/BottomNav.tsx`**

```tsx
import React from "react";
import { useI18n } from "../i18n";

export type Screen = "camera" | "history" | "settings";

const stroke = {
  fill: "none" as const,
  stroke: "currentColor",
  strokeWidth: 2,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
};

const CameraIcon = () => (
  <svg viewBox="0 0 24 24" {...stroke}>
    <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
    <circle cx="12" cy="13" r="4" />
  </svg>
);
const HistoryIcon = () => (
  <svg viewBox="0 0 24 24" {...stroke}>
    <circle cx="12" cy="12" r="9" />
    <path d="M12 7v5l3 2" />
  </svg>
);
const SettingsIcon = () => (
  <svg viewBox="0 0 24 24" {...stroke}>
    <circle cx="12" cy="12" r="3" />
    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
  </svg>
);

interface BottomNavProps {
  active: Screen;
  onChange: (s: Screen) => void;
}

export function BottomNav({ active, onChange }: BottomNavProps) {
  const th = useI18n();
  const items: { key: Screen; label: string; icon: JSX.Element }[] = [
    { key: "camera", label: th.navCamera, icon: <CameraIcon /> },
    { key: "history", label: th.navHistory, icon: <HistoryIcon /> },
    { key: "settings", label: th.navSettings, icon: <SettingsIcon /> },
  ];
  return (
    <nav className="bottom-nav-float">
      {items.map((it) => (
        <button
          key={it.key}
          type="button"
          className={`bottom-nav-item${active === it.key ? " on" : ""}`}
          onClick={() => onChange(it.key)}
          aria-current={active === it.key ? "page" : undefined}
          aria-label={it.label}
        >
          <span className="bottom-nav-ico">{it.icon}</span>
          <span>{it.label}</span>
        </button>
      ))}
    </nav>
  );
}
```

- [ ] **Step 3: Build**

Run: `cd frontend && npm run build`
Expected: ผ่าน (ยังไม่ถูกใช้ — แค่คอมไพล์ได้)

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/BottomNav.tsx frontend/src/styles/tokens.css
git commit -m "feat(nav): add floating glass BottomNav component"
```

---

## Task 8: History + Settings screens + overlay/sheet styles

**Files:**
- Create: `frontend/src/components/HistoryScreen.tsx`
- Create: `frontend/src/components/SettingsScreen.tsx`
- Modify: `frontend/src/styles/tokens.css`

**Interfaces:**
- Consumes: `useHistory()` (Task 6), `useSettings()` (Task 1), `useSpeech()` (Task 4), `useI18n()` (Task 2)
- Produces: `HistoryScreen()`, `SettingsScreen()` (ไม่มี props)

- [ ] **Step 1: เพิ่มสไตล์ overlay/sheet/list/settings ต่อท้าย `tokens.css`**

```css
/* ── Full-screen glass overlay for History / Settings ── */
.screen-overlay {
  position: absolute;
  inset: 0;
  z-index: 25;
  background: rgba(8, 10, 22, 0.72);
  backdrop-filter: var(--glass-blur);
  -webkit-backdrop-filter: var(--glass-blur);
  overflow-y: auto;
  padding: calc(env(safe-area-inset-top, 0px) + var(--space-6)) var(--space-4) 120px;
}
.screen-sheet { max-width: 620px; margin: 0 auto; color: #fff; }
.sheet-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: var(--space-5); gap: var(--space-3); }
.sheet-title { font-size: var(--font-size-2xl); font-weight: 800; color: #fff; }

/* History list */
.history-list { list-style: none; display: flex; flex-direction: column; gap: var(--space-3); }
.history-row {
  display: flex; align-items: center; gap: var(--space-3);
  padding: var(--space-4); border-radius: var(--radius-md);
  background: rgba(255, 255, 255, 0.06); border: 1px solid rgba(255, 255, 255, 0.12);
  color: #fff; cursor: pointer; text-align: left; font-family: var(--font-family); width: 100%;
}
.history-row:hover { background: rgba(255, 255, 255, 0.1); }
.history-row svg { width: 22px; height: 22px; color: rgba(255, 255, 255, 0.55); flex-shrink: 0; }
.history-word { flex: 1; min-width: 0; }
.history-word b { display: block; font-size: var(--font-size-xl); font-weight: 800; word-break: break-word; }
.history-word small { font-size: var(--font-size-xs); color: rgba(255, 255, 255, 0.5); }

.empty-state {
  display: flex; flex-direction: column; align-items: center; gap: var(--space-3);
  padding: var(--space-12) var(--space-4); color: rgba(255, 255, 255, 0.5); text-align: center;
}
.empty-state svg { width: 48px; height: 48px; opacity: 0.5; }

/* Settings */
.settings-list { display: flex; flex-direction: column; gap: var(--space-3); }
.settings-row {
  display: flex; align-items: center; gap: var(--space-3);
  padding: var(--space-4); border-radius: var(--radius-md);
  background: rgba(255, 255, 255, 0.06); border: 1px solid rgba(255, 255, 255, 0.12); color: #fff;
}
.settings-row svg { width: 22px; height: 22px; color: #9fb6e6; flex-shrink: 0; }
.settings-label { flex: 1; font-size: var(--font-size-base); font-weight: 600; }
.settings-action { width: 100%; text-align: left; font-family: var(--font-family); cursor: pointer; }
.settings-action:hover { background: rgba(255, 255, 255, 0.1); }

/* Segmented language toggle */
.seg { display: flex; background: rgba(255, 255, 255, 0.1); border-radius: var(--radius-full); padding: 3px; }
.seg button {
  border: none; background: none; color: rgba(255, 255, 255, 0.6);
  font-family: var(--font-family); font-weight: 800; font-size: var(--font-size-sm);
  padding: 6px 14px; border-radius: var(--radius-full); cursor: pointer;
}
.seg button.on { background: var(--color-primary); color: #fff; }

/* Toggle switch */
.toggle {
  width: 46px; height: 26px; border-radius: var(--radius-full);
  background: rgba(255, 255, 255, 0.2); border: none; position: relative; cursor: pointer; flex-shrink: 0;
  transition: background var(--transition);
}
.toggle.on { background: var(--color-success); }
.toggle::after {
  content: ""; position: absolute; top: 3px; left: 3px; width: 20px; height: 20px;
  border-radius: 50%; background: #fff; transition: left var(--transition);
}
.toggle.on::after { left: 23px; }
```

- [ ] **Step 2: สร้าง `frontend/src/components/HistoryScreen.tsx`**

```tsx
import React from "react";
import { useHistory } from "../hooks/HistoryProvider";
import { useSpeech } from "../hooks/useSpeech";
import { useSettings } from "../hooks/SettingsProvider";
import { useI18n } from "../i18n";

const stroke = {
  fill: "none" as const,
  stroke: "currentColor",
  strokeWidth: 2,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
};

function formatTime(ts: number, lang: "th" | "en"): string {
  const locale = lang === "th" ? "th-TH" : "en-US";
  return new Date(ts).toLocaleString(locale, {
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function HistoryScreen() {
  const { items, clear } = useHistory();
  const { speak } = useSpeech();
  const { lang } = useSettings();
  const th = useI18n();

  return (
    <div className="screen-sheet">
      <div className="sheet-head">
        <h2 className="sheet-title">{th.historyTitle}</h2>
        {items.length > 0 && (
          <button
            type="button"
            className="glass-chip"
            onClick={() => {
              if (window.confirm(th.confirmClear)) clear();
            }}
          >
            ✕ {th.settingsClearHistory}
          </button>
        )}
      </div>

      {items.length === 0 ? (
        <div className="empty-state">
          <svg viewBox="0 0 24 24" {...stroke}>
            <circle cx="12" cy="12" r="9" />
            <path d="M12 7v5l3 2" />
          </svg>
          <p>{th.historyEmpty}</p>
        </div>
      ) : (
        <ul className="history-list">
          {items.map((it) => (
            <li key={it.id}>
              <button type="button" className="history-row" onClick={() => speak(it.sentence)}>
                <div className="history-word">
                  <b>{it.sentence}</b>
                  <small>{formatTime(it.ts, lang)}</small>
                </div>
                <svg viewBox="0 0 24 24" {...stroke}>
                  <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
                  <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
                  <path d="M19.07 4.93a10 10 0 0 1 0 14.14" />
                </svg>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
```

- [ ] **Step 3: สร้าง `frontend/src/components/SettingsScreen.tsx`**

```tsx
import React from "react";
import { useSettings } from "../hooks/SettingsProvider";
import { useHistory } from "../hooks/HistoryProvider";
import { useI18n } from "../i18n";

const stroke = {
  fill: "none" as const,
  stroke: "currentColor",
  strokeWidth: 2,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
};

export function SettingsScreen() {
  const { lang, setLang, showLandmarks, setShowLandmarks } = useSettings();
  const { clear } = useHistory();
  const th = useI18n();

  return (
    <div className="screen-sheet">
      <h2 className="sheet-title" style={{ marginBottom: "var(--space-5)" }}>
        {th.settingsTitle}
      </h2>
      <div className="settings-list">
        {/* Language */}
        <div className="settings-row">
          <svg viewBox="0 0 24 24" {...stroke}>
            <circle cx="12" cy="12" r="10" />
            <line x1="2" y1="12" x2="22" y2="12" />
            <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
          </svg>
          <span className="settings-label">{th.settingsLanguage}</span>
          <div className="seg">
            <button type="button" className={lang === "th" ? "on" : ""} onClick={() => setLang("th")}>
              ไทย
            </button>
            <button type="button" className={lang === "en" ? "on" : ""} onClick={() => setLang("en")}>
              EN
            </button>
          </div>
        </div>

        {/* Landmarks toggle */}
        <div className="settings-row">
          <svg viewBox="0 0 24 24" {...stroke}>
            <polygon points="12 2 15 9 22 9 16 14 18 21 12 17 6 21 8 14 2 9 9 9" />
          </svg>
          <span className="settings-label">{th.settingsLandmarks}</span>
          <button
            type="button"
            role="switch"
            aria-checked={showLandmarks}
            aria-label={th.settingsLandmarks}
            className={`toggle${showLandmarks ? " on" : ""}`}
            onClick={() => setShowLandmarks(!showLandmarks)}
          />
        </div>

        {/* Clear history */}
        <button
          type="button"
          className="settings-row settings-action"
          onClick={() => {
            if (window.confirm(th.confirmClear)) clear();
          }}
        >
          <svg viewBox="0 0 24 24" {...stroke}>
            <polyline points="3 6 5 6 21 6" />
            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
          </svg>
          <span className="settings-label">{th.settingsClearHistory}</span>
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Build**

Run: `cd frontend && npm run build`
Expected: ผ่าน (ยังไม่ถูกใช้)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/HistoryScreen.tsx frontend/src/components/SettingsScreen.tsx frontend/src/styles/tokens.css
git commit -m "feat(screens): add History + Settings screens and styles"
```

---

## Task 9: Shell integration — tab switching, overlays, pause translate off-camera

**Files:**
- Modify: `frontend/src/App.tsx`

**Interfaces:**
- Consumes: `BottomNav`/`Screen` (Task 7), `HistoryScreen` (Task 8), `SettingsScreen` (Task 8), `useSettings` (Task 1)

- [ ] **Step 1: แทนที่ทั้งไฟล์ `frontend/src/App.tsx`**

(รวมทุกอย่างของ Task 1/3/6 ที่แก้มาก่อนหน้า — เป็นเวอร์ชันสมบูรณ์)

```tsx
import React, { useEffect, useRef, useState } from "react";
import { TranslateResult } from "./api/client";
import { ModelsProvider, useModels } from "./hooks/ModelsProvider";
import { SettingsProvider, useSettings } from "./hooks/SettingsProvider";
import { HistoryProvider, useHistory } from "./hooks/HistoryProvider";
import { useHolisticCapture } from "./hooks/useHolisticCapture";
import { useTranslate } from "./hooks/useTranslate";
import { useI18n } from "./i18n";
import { CameraView } from "./components/CameraView";
import { ResultCard } from "./components/ResultCard";
import { SupportedPhrases } from "./components/SupportedPhrases";
import { BottomNav, Screen } from "./components/BottomNav";
import { HistoryScreen } from "./components/HistoryScreen";
import { SettingsScreen } from "./components/SettingsScreen";

const CONFIDENCE_FLOOR = 0.3;
const HANDS_GONE_DEBOUNCE_MS = 800;
const MIN_HAND_FRAMES = 6;
const MIN_TOTAL_FRAMES = 8;

function AppShell() {
  const th = useI18n();
  const { models, selectedModelId, loading: modelsLoading, error: modelsError } = useModels();
  const { showLandmarks } = useSettings();
  const history = useHistory();
  const capture = useHolisticCapture({ overlayEnabled: showLandmarks });
  const translator = useTranslate();
  const selectedModel = models.find((m) => m.id === selectedModelId);

  const [screen, setScreen] = useState<Screen>("camera");
  const [phrasesOpen, setPhrasesOpen] = useState(false);
  const [displayedResult, setDisplayedResult] = useState<TranslateResult | null>(null);

  const captureRef = useRef(capture);
  captureRef.current = capture;
  const translatorRef = useRef(translator);
  translatorRef.current = translator;
  const selectedModelIdRef = useRef(selectedModelId);
  selectedModelIdRef.current = selectedModelId;
  const historyRef = useRef(history);
  historyRef.current = history;
  const debounceRef = useRef<ReturnType<typeof setTimeout>>();

  // Auto-start capture once the camera is ready.
  useEffect(() => {
    if (!capture.ready) return;
    captureRef.current.start();
  }, [capture.ready]);

  // Presence-based segmentation — only while on the camera screen.
  useEffect(() => {
    if (!capture.ready) return;
    if (screen !== "camera") {
      clearTimeout(debounceRef.current);
      return;
    }

    if (capture.handsPresent) {
      clearTimeout(debounceRef.current);
      if (!captureRef.current.recording) captureRef.current.start();
    } else {
      clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => {
        const trans = translatorRef.current;
        if (trans.status === "loading") return;
        const { frames, handFrameCount } = captureRef.current.stop();
        captureRef.current.start();
        if (handFrameCount >= MIN_HAND_FRAMES && frames.length >= MIN_TOTAL_FRAMES) {
          trans.run(frames, selectedModelIdRef.current ?? undefined);
        }
      }, HANDS_GONE_DEBOUNCE_MS);
    }

    return () => clearTimeout(debounceRef.current);
  }, [capture.handsPresent, capture.ready, screen]);

  // Update displayed result + record to history when confidence clears the floor.
  useEffect(() => {
    if (translator.status === "success" && translator.result) {
      if (translator.result.score >= CONFIDENCE_FLOOR) {
        setDisplayedResult(translator.result);
        historyRef.current.add({
          sentence: translator.result.sentence,
          score: translator.result.score,
          model: translator.result.model,
        });
      }
    }
  }, [translator.status, translator.result]);

  // Auto-reset error after 3 s so the loop can continue.
  useEffect(() => {
    if (translator.status !== "error") return;
    const id = setTimeout(() => translatorRef.current.reset(), 3000);
    return () => clearTimeout(id);
  }, [translator.status]);

  const hasCameraError = Boolean(capture.cameraError);
  const resultStatus =
    translator.status === "success"
      ? displayedResult && displayedResult === translator.result
        ? "success"
        : "loading"
      : translator.status;

  return (
    <main className="app-immersive">
      {/* Full-screen camera background — stays mounted across tabs */}
      <CameraView videoRef={capture.videoRef} overlayRef={capture.overlayRef} />

      {/* Top glass bar (hidden under overlays on other tabs) */}
      <div className="glass-top-bar">
        <div className="brand-glass">
          <div className="brand-mark-glass">{th.brandShort}</div>
          <span className="brand-name-glass">{th.appTitle}</span>
        </div>

        <div className="top-controls-right">
          <span className="live-chip">
            <span
              className={[
                "live-dot",
                !capture.ready ? "offline" : "",
                capture.ready && capture.handsPresent ? "hands" : "",
              ]
                .filter(Boolean)
                .join(" ")}
            />
            {capture.ready ? th.cameraLive : th.cameraOpening}
          </span>

          {!modelsLoading && selectedModel && (
            <span className="glass-chip" style={{ cursor: "default" }}>
              {selectedModel.label_th}
            </span>
          )}

          <button className="glass-chip" onClick={() => setPhrasesOpen((v) => !v)} aria-expanded={phrasesOpen}>
            {th.supportedPhrasesTitle}
          </button>
        </div>
      </div>

      {/* Camera permission error */}
      {hasCameraError && (
        <div className="glass-camera-error">
          <p style={{ color: "#fca5a5", fontWeight: 700, fontSize: "var(--font-size-base)" }}>{th.cameraError}</p>
          <p style={{ color: "rgba(255,255,255,0.65)", fontSize: "var(--font-size-sm)" }}>{th.cameraErrorHint}</p>
          <button className="glass-action-btn" onClick={() => captureRef.current.start()}>
            {th.cameraRetry}
          </button>
        </div>
      )}

      {/* Live translation panel — only on the camera screen */}
      {screen === "camera" && (
        <div className="result-glass-panel live">
          <LiveStatusRow
            cameraReady={capture.ready}
            translating={translator.status === "loading"}
            hasResult={Boolean(displayedResult)}
            hasError={translator.status === "error"}
            modelsError={modelsError}
          />
          <ResultCard
            status={resultStatus}
            result={displayedResult}
            error={translator.error}
            errorStatus={translator.errorStatus}
            variant="glass"
          />
        </div>
      )}

      {/* History / Settings overlays (camera keeps running underneath) */}
      {screen === "history" && (
        <div className="screen-overlay">
          <HistoryScreen />
        </div>
      )}
      {screen === "settings" && (
        <div className="screen-overlay">
          <SettingsScreen />
        </div>
      )}

      {/* Supported phrases slide-up panel */}
      <div className={`result-glass-panel${phrasesOpen ? " open" : ""}`} style={{ zIndex: 30 }} aria-hidden={!phrasesOpen}>
        <div className="glass-panel-handle" />
        <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: "var(--space-3)" }}>
          <button className="glass-chip" onClick={() => setPhrasesOpen(false)} aria-label={th.navSettings}>
            ✕
          </button>
        </div>
        <SupportedPhrases glass />
      </div>

      {/* Floating bottom nav — always visible, above overlays */}
      <BottomNav active={screen} onChange={setScreen} />
    </main>
  );
}

interface LiveStatusRowProps {
  cameraReady: boolean;
  translating: boolean;
  hasResult: boolean;
  hasError: boolean;
  modelsError: string | null;
}

function LiveStatusRow({ cameraReady, translating, hasResult, hasError, modelsError }: LiveStatusRowProps) {
  const th = useI18n();
  let label = "";
  let color = "rgba(255,255,255,0.5)";

  if (!cameraReady) {
    label = th.cameraInit;
  } else if (modelsError) {
    label = th.modelLoadError;
    color = "#fcd34d";
  } else if (hasError) {
    label = "";
  } else if (translating) {
    label = th.translating;
  } else if (!hasResult) {
    label = th.showSignHint;
  }

  if (!label) return null;

  return (
    <p style={{ fontSize: "var(--font-size-sm)", color, marginBottom: "var(--space-2)", fontWeight: 500 }}>{label}</p>
  );
}

export default function App() {
  return (
    <ModelsProvider>
      <SettingsProvider>
        <HistoryProvider>
          <AppShell />
        </HistoryProvider>
      </SettingsProvider>
    </ModelsProvider>
  );
}
```

> หมายเหตุ: เวอร์ชันนี้ตัดปุ่ม toggle "เส้นโครงร่าง" ออกจาก top bar (ย้ายไปหน้าตั้งค่าแล้ว) และเอา local state `showLandmarks` ออก (อ่านจาก `useSettings`)

- [ ] **Step 2: Build**

Run: `cd frontend && npm run build`
Expected: ผ่าน

- [ ] **Step 3: ตรวจในเบราว์เซอร์ (ครบทุกเกณฑ์)**

Run: `cd frontend && npm run dev`
ตรวจ:
- แตะ 3 แท็บล่างสลับหน้าได้ (กล้อง/ประวัติ/ตั้งค่า) — กล้องไม่กระพริบ/ไม่ re-init ตอนสลับกลับ
- หน้ากล้อง: แปลได้ ปุ่มลำโพงทำงาน ไม่ถูกตัด
- สลับไปแท็บอื่นแล้วโบกมือหน้ากล้อง → **ไม่แปล/ไม่เพิ่มประวัติ**; กลับมาแท็บกล้องแล้วแปลได้อีก
- หน้าประวัติ: เห็นคำที่แปล, แตะการ์ดมีเสียงอ่าน, ล้างได้
- หน้าตั้งค่า: สลับ ไทย/EN แล้ว UI เปลี่ยนภาษาทันที; toggle เส้นโครงร่างมีผลกับกล้อง; รีโหลดแล้วค่าคงอยู่

- [ ] **Step 4: Commit**

```bash
git add frontend/src/App.tsx
git commit -m "feat(shell): tab nav + history/settings overlays; pause translate off-camera"
```

---

## Task 10: Final pass — build, baseline regression check

**Files:** (ไม่มีไฟล์ใหม่ — ตรวจรวม)

- [ ] **Step 1: Production build**

Run: `cd frontend && npm run build`
Expected: ผ่าน ไม่มี warning ที่เป็น error

- [ ] **Step 2: ตรวจ regression ของลูปแปลเดิม**

Run: `cd frontend && npm run preview` (เสิร์ฟ build จริง) หรือรันคู่กับ backend ตามปกติ
Expected: พฤติกรรมแปลอัตโนมัติ (โบกมือ→หยุด 0.8 วิ→แปล) เหมือนเดิม; ผลแปลที่ score ≥ 0.3 ขึ้นการ์ดและถูกบันทึกประวัติ

- [ ] **Step 3: ตรวจ EN เต็มรูปแบบ**

สลับเป็น EN ในตั้งค่า แล้วไล่ดูทั้ง 3 หน้า: ทุกข้อความ UI เป็นอังกฤษ (ยกเว้นผลแปลภาษามือซึ่งเป็นไทยตามโมเดล)

- [ ] **Step 4: Commit (ถ้ามีการปรับ pixel เล็กน้อยระหว่างตรวจ)**

```bash
git add -A
git commit -m "chore(ux): final polish pass for Conductor UX v2 (P1)"
```

---

## Self-Review (ผู้เขียนแผนตรวจเองเทียบ spec)

**1. Spec coverage:**
- §3 IA + floating nav → Task 7, 9 ✓
- §4 หน้ากล้อง + ปุ่มลำโพง + ย้าย landmark → Task 5, 9 ✓
- §5 ประวัติ (dedup/cap/clear/persist) → Task 6, 8 ✓
- §6 ตั้งค่า (lang/landmarks/clear) → Task 1, 8 ✓
- §7 i18n (en + index + useI18n + refactor) → Task 2, 3 ✓
- §8 TTS useSpeech → Task 4, 5 ✓
- §9 providers/ไฟล์ → ครบตาม file map ✓
- §10 overflow fix + camera lifecycle + dedup + บันทึกเฉพาะ ≥ floor → Task 5, 9, 6 ✓
- §11 P1 only (ไม่ทำ favorites/autoSpeak/tests) ✓

**2. Placeholder scan:** ไม่มี TBD/TODO; ทุก step มีโค้ดจริง/คำสั่งจริง ✓

**3. Type consistency:** `Lang` (SettingsProvider) ใช้ใน i18n/index, HistoryScreen formatTime; `Screen` (BottomNav) ใช้ใน App; `HistoryItem`/`add({sentence,score,model})` ตรงกันระหว่าง Task 6 และจุดเรียกใน App; `useSpeech().speak/speaking/supported` ตรงกับการใช้ใน ResultCard/HistoryScreen ✓
