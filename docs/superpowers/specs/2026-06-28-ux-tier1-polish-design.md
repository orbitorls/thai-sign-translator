# Design Spec — UX Tier 1 Polish (วาทยากร / Conductor)

วันที่: 2026-06-28
สถานะ: อนุมัติ scope แล้ว — รอ review spec ก่อนทำ plan
ขอบเขต: frontend (`frontend/`) เท่านั้น — ไม่แตะ backend/ML, ไม่เพิ่ม npm dependency
Base branch: `feature/ux-tier1-polish` (แตกจาก `feature/conductor-ux-v2` = P1)

---

## 1. ภาพรวมและเป้าหมาย

ชุดฟีเจอร์ "quick win" ต่อยอดจาก P1 เพื่อเพิ่ม accessibility, ความทนทาน, และคุณค่าเชิงสาธิต โดยทุกชิ้นเป็นฟีเจอร์อิสระ ทยอย ship/รีวิวแยกได้ และใช้ web API มาตรฐานล้วน (ไม่เพิ่ม dependency)

6 ฟีเจอร์: (A) Developer Mode + เมตริก, (B) History copy/ลบเดี่ยว/แชร์, (C) Haptic + แฟลชภาพ, (D) Pause กล้องตอนพักหลัง, (E) PWA ติดตั้งได้, (F) Network-loss/error state

**การตัดสินใจที่ล็อกแล้ว:** Developer Mode = toggle เดียวรวม skeleton+เมตริก (แทน toggle "เส้นโครงร่าง" เดิม) · History ไม่มีช่องค้นหา · PWA = ติดตั้งได้อย่างเดียว (ไม่มี service worker/offline cache)

---

## 2. หลักการ & ความปลอดภัย (collab)

- ทำบน `feature/ux-tier1-polish` เท่านั้น — **ไม่แตะ `main`, ไม่ force-push, ไม่ rewrite history ที่ share**
- frontend-only, ไม่แก้ backend, ไม่เพิ่ม dependency
- ไม่มี test framework (เลื่อน P2) → เกณฑ์ผ่านแต่ละงาน = `cd frontend && npm run build` ผ่าน + ตรวจเบราว์เซอร์
- คง glass design system เดิม; ไม่ทำลายลูปแปล/พฤติกรรม P1
- เคารพ `prefers-reduced-motion` กับทุก animation ใหม่ (แฟลช/พัลส์)

---

## 3. ฟีเจอร์ A — Developer Mode + เมตริก

รวม toggle "เส้นโครงร่าง" เดิมเป็น **Developer Mode** ตัวเดียว

**พฤติกรรม:** เปิด = วาด MediaPipe skeleton overlay (เหมือนเดิม) **+** overlay เมตริกมุมจอ (เช่น มุมบนซ้าย ใต้ top bar) แสดง:
- **FPS** ของลูป MediaPipe (เฟรม/วินาที)
- **Latency** การแปลล่าสุด (ms, เวลา round-trip ของ `/translate`)
- **Confidence** ของผลล่าสุด (%)

ปิด = ไม่วาดทั้ง skeleton และเมตริก (optimize)

**การเปลี่ยน interface:**
- `SettingsProvider`: เปลี่ยน field `showLandmarks` → **`devMode: boolean`** (+ `setDevMode`). ตอน load จาก localStorage: ถ้าไม่มี `devMode` แต่มี legacy `showLandmarks` ให้ใช้ค่านั้น (migration นุ่มนวล) key คงเดิม `tsl.settings.v1`
- `useHolisticCapture`: เพิ่มค่า return **`fps: number`** (คำนวณจากจำนวนเฟรมที่ประมวลผลต่อวินาที)
- `useTranslate`: เพิ่ม **`lastLatencyMs: number | null`** (จับเวลา `Date.now()` ก่อน/หลัง `translate()`)
- `App`: `overlayEnabled` = `devMode`; เมื่อ `devMode` แสดงคอมโพเนนต์ใหม่ `DevMetricsOverlay` (fps/latency/confidence)
- `SettingsScreen`: เปลี่ยน label/toggle เป็น "Developer Mode" (เพิ่มคีย์ i18n `settingsDevMode`)

**ไฟล์ใหม่:** `frontend/src/components/DevMetricsOverlay.tsx`

---

## 4. ฟีเจอร์ B — History: คัดลอก / ลบเดี่ยว / แชร์

- แต่ละการ์ดในหน้าประวัติเพิ่มปุ่มไอคอน: **คัดลอก** (`navigator.clipboard.writeText`), **ลบรายการนี้**, **แชร์** (`navigator.share({text})` → ถ้าไม่รองรับ fallback เป็นคัดลอก)
- การแตะตัวการ์ด (พื้นที่คำ) ยังคง = อ่านออกเสียง (เหมือน P1); ปุ่มไอคอนแยก action ชัดเจน (มี `stopPropagation`)
- `HistoryProvider`: เพิ่ม **`remove(id: string)`**
- i18n: เพิ่มคีย์ `actionCopy`, `actionDelete`, `actionShare`, `copied`
- คัดลอก/แชร์ guard ด้วย feature-detection; ปุ่มที่ไม่รองรับให้ซ่อน (เช่น share บนเดสก์ท็อปบางเบราว์เซอร์)

---

## 5. ฟีเจอร์ C — Haptic + แฟลชภาพ (feedback ที่ไม่พึ่งเสียง)

core accessibility: ผู้ใช้หูหนวกต้องรับรู้สถานะด้วย **ภาพ + การสั่น** ไม่ใช่เสียง

- เมื่อ **แปลสำเร็จ** (มี displayedResult ใหม่): พัลส์/แฟลชเขียวเบา ๆ ที่ขอบการ์ดผลแปล + `navigator.vibrate(30)`
- เมื่อ **error / เชื่อมต่อไม่ได้**: แฟลชแดง + `navigator.vibrate([60,40,60])`
- ไฟล์ใหม่ `frontend/src/hooks/useFeedback.ts` — คืน `{ flash: "success" | "error" | null, signal(kind) }`; `signal` สั่ง vibrate (guard `"vibrate" in navigator`) + ตั้ง `flash` ชั่วคราว (auto-clear ~600ms)
- `App` เรียก `signal("success")` เมื่อ displayedResult เปลี่ยน, `signal("error")` เมื่อ translator เข้า error/offline; เรนเดอร์ flash overlay (ขอบจอ/การ์ด) ตาม `flash`
- หมายเหตุ: `navigator.vibrate` ไม่ทำงานบน iOS Safari → ใช้ภาพเป็นหลัก, การสั่นเป็น progressive enhancement
- เคารพ `prefers-reduced-motion`: ลด/งดพัลส์

---

## 6. ฟีเจอร์ D — Pause กล้องตอนพักหลัง

ประหยัดแบต/ความร้อน (NFR ของ brief)

- `useHolisticCapture`: เพิ่ม **`pause()` / `resume()`** (หยุด/เริ่มลูปประมวลผล MediaPipe โดย **ไม่ teardown** กล้อง/โมเดล — resume ต้องเร็ว ไม่ re-init)
- `App`: หยุดประมวลผลเมื่อ `document.hidden` **หรือ** `screen !== "camera"`; resume เมื่อกลับมาที่แท็บกล้องและหน้าต่าง visible — ผูกกับ `visibilitychange` + state `screen`
- คง invariant ของ P1: ไม่ unmount `CameraView`/hook ตอนสลับแท็บ (แค่ pause ลูป)

---

## 7. ฟีเจอร์ E — PWA ติดตั้งได้ (installable เท่านั้น)

- เพิ่ม `frontend/public/manifest.webmanifest`: `name "วาทยากร"`, `short_name "วาทยากร"`, `display "standalone"`, `background_color`/`theme_color` (โทน glass เข้ม เช่น `#0b1020`), `start_url "/"`, `icons` 192 + 512 (+ maskable)
- สร้างไอคอน PNG 192/512 จากโลโก้ SVG (เรนเดอร์ด้วย Chrome headless ที่มีในเครื่อง — ขั้นตอนสร้าง asset ครั้งเดียว) วางใน `frontend/public/`
- `frontend/index.html`: เพิ่ม `<link rel="manifest">`, `<meta name="theme-color">`, `<link rel="apple-touch-icon">`
- **ไม่มี** service worker / offline cache (เลี่ยงปัญหา cache ค้างตอน dev/collab)

---

## 8. ฟีเจอร์ F — Network-loss / error state ชัดเจน

- `useTranslate`: แยกชนิด error — เพิ่ม **`errorKind: "api" | "network" | null`** (network = fetch โยน `TypeError`/ไม่มี response; api = `ApiError` มี status)
- `App`: ถ้า `errorKind === "network"` หรือ `!navigator.onLine` → แสดงสถานะ "เชื่อมต่อไม่ได้/ออฟไลน์" (chip/แถบ) + ปุ่มลองใหม่ แทนข้อความ error ทั่วไป; ฟัง event `online`/`offline`
- i18n: เพิ่มคีย์ `offline`, `offlineHint`, `retry`
- คง auto-reset error 3s เดิมไว้สำหรับ api error; network/offline แสดงค้างจนกลับมาออนไลน์

---

## 9. สรุปไฟล์

**สร้างใหม่:**
- `frontend/src/components/DevMetricsOverlay.tsx`
- `frontend/src/hooks/useFeedback.ts`
- `frontend/public/manifest.webmanifest` + `frontend/public/icon-192.png` + `icon-512.png` (+ maskable ถ้าทำ)

**แก้ไข:**
- `frontend/src/hooks/SettingsProvider.tsx` — `showLandmarks` → `devMode` (+ migration)
- `frontend/src/hooks/useHolisticCapture.ts` — `fps`, `pause()`, `resume()`
- `frontend/src/hooks/useTranslate.ts` — `lastLatencyMs`, `errorKind`
- `frontend/src/hooks/HistoryProvider.tsx` — `remove(id)`
- `frontend/src/components/HistoryScreen.tsx` — ปุ่ม copy/delete/share ต่อรายการ
- `frontend/src/components/SettingsScreen.tsx` — toggle "Developer Mode"
- `frontend/src/App.tsx` — DevMetricsOverlay, feedback signals + flash overlay, pause logic, offline state
- `frontend/src/i18n/th.ts` + `en.ts` — คีย์ใหม่ (settingsDevMode, actionCopy/Delete/Share, copied, offline, offlineHint, retry)
- `frontend/src/styles/tokens.css` — สไตล์ dev-metrics overlay, flash/pulse, history action buttons, offline chip
- `frontend/index.html` — manifest/theme-color/apple-touch-icon links

---

## 10. ลำดับการทำ (independent, ship แยกได้)

1. F (network/error state) — เล็ก, อิสระ
2. C (haptic + flash) — เล็ก
3. B (history actions) — เล็ก
4. D (pause on background) — แตะ useHolisticCapture
5. A (Developer Mode + metrics) — แตะ useHolisticCapture/useTranslate/Settings (ทำหลัง D เพราะแตะ hook เดียวกัน)
6. E (PWA) — แยกขาด, ทำเมื่อไรก็ได้

---

## 11. ขอบเขตที่ไม่ทำ (YAGNI / Out of scope)

- ค้นหาในประวัติ (ตัดออก)
- service worker / offline cache (PWA ติดตั้งได้อย่างเดียว)
- Two-way STT (Hearing→Deaf) = Tier 2 รอบถัดไป
- Teach-a-Sign, WebSocket streaming, Text-to-Sign, theme/high-contrast = รอบถัดไป
- automated tests = P2

---

## 12. เกณฑ์ความสำเร็จ

- Developer Mode toggle เดียวเปิด/ปิด skeleton + เมตริก (fps/latency/confidence) ได้ และ migrate ค่า showLandmarks เดิม
- History: คัดลอก/ลบเดี่ยว/แชร์ ได้ (แตะการ์ด = อ่านออกเสียงยังทำงาน)
- แปลสำเร็จ/ผิดพลาด มี feedback ภาพ (+สั่นบนอุปกรณ์ที่รองรับ)
- สลับแท็บ/พักแอป → MediaPipe หยุดประมวลผล, กลับมา resume เร็ว ไม่ re-init
- ติดตั้งเป็น PWA บนมือถือได้
- เน็ตหลุด/ออฟไลน์ มีสถานะชัด + ลองใหม่ได้
- ไม่มี regression กับ P1; `npm run build` ผ่าน
