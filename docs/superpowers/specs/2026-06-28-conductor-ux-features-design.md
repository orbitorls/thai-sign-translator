# Design Spec — วาทยากร (Conductor) UX/UI v2

วันที่: 2026-06-28
สถานะ: อนุมัติแล้ว — ขอบเขตงานปัจจุบัน **P1 เท่านั้น** (favorites ตัดออก), พร้อมทำ implementation plan
ขอบเขต: เฉพาะ frontend (`frontend/`) — ไม่แตะ backend/ML

---

## 1. ภาพรวมและเป้าหมาย

ยกระดับ UX/UI ของแอปแปลภาษามือไทย "วาทยากร / Conductor" ให้ผู้ใช้งานจริง (โดยเฉพาะผู้พิการทางหูที่อาจมีทักษะการอ่านน้อย) ใช้งานได้สะดวกขึ้น โดย:

1. ลดข้อความ ใช้ **ไอคอนเป็นหลัก + คำเดี่ยว** ปุ่มกดใหญ่
2. เพิ่มฟีเจอร์:
   - (a) **ประวัติการใช้งาน** (History)
   - (b) **การตั้งค่า** (Settings) — ภายในมี **สลับภาษา UI ไทย/EN** (ตามที่ผู้ใช้เลือก: เปลี่ยนภาษาหน้าจอ ไม่ใช่แปลผลลัพธ์เป็นภาษาอื่น)
   - (c) **อ่านออกเสียงผลแปล (TTS)** — ปุ่มลำโพงบนการ์ดผลแปล กดแล้วเสียง AI อ่านคำดังออกลำโพง ให้ทั้งผู้ใช้และคู่สนทนาที่ได้ยินรับรู้ (สื่อสารสองทาง)

ผลลัพธ์การแปล (sign → ข้อความ) ยังเป็น **ภาษาไทย** เหมือนเดิม

> ข้อสังเกต (จาก advisor): การสลับภาษา UI ไทย/EN ช่วยผู้ใช้ที่อ่านออก/คนได้ยิน มากกว่ากลุ่มหูหนวกอ่านน้อยที่เป็นผู้ใช้หลัก จึงไม่ให้งานนี้มาแย่งความสำคัญจากงาน icon-first ที่ตอบโจทย์ผู้ใช้หลักโดยตรง

---

## 2. หลักการออกแบบ (Design principles)

- **Icon-first / minimal text**: ทุกการนำทางและการกระทำหลักมีไอคอนชัด + คำกำกับสั้น (1 คำ)
- **คำเดี่ยวเด่น**: ผลแปลแสดงเป็นคำใหญ่ อ่านง่ายระยะไกล
- **Touch target ใหญ่**: ปุ่ม ≥ 44px เหมาะกับการกดเร็วและผู้ใช้ที่ควบคุมการเคลื่อนไหวจำกัด
- **คง glass design system เดิม** (`tokens.css`) — เพิ่ม token เท่าที่จำเป็น
- **ไม่ทำลายของที่ทำงานอยู่**: ลูปแปลอัตโนมัติ (presence-based segmentation) ใน `App.tsx` ต้องทำงานเหมือนเดิม

---

## 3. โครงสร้างหน้าจอ (IA) และการนำทาง

3 หน้าหลัก สลับด้วย **floating glass bottom nav** (แถบลอย โปร่งแก้ว เว้น margin รอบตัว มองเห็นตลอด):

| แท็บ | ไอคอน | หน้าที่ |
|------|-------|---------|
| กล้อง | camera | แปลภาษามือ (หน้าหลัก) |
| ประวัติ | clock | ดูคำที่เคยแปล |
| ตั้งค่า | gear | ตั้งค่า + สลับภาษา |

**กลไกการสลับหน้า (สำคัญ — โหลดแบริ่งของสถาปัตยกรรม):**

- `CameraView` + hook `useHolisticCapture` **mount อยู่ตลอด เป็นพื้นหลัง ห้าม unmount เมื่อสลับแท็บ** (re-init MediaPipe ช้าและทำลายความรู้สึก immersive)
- หน้า **ประวัติ/ตั้งค่า** เรนเดอร์เป็น **overlay (glass panel) ทับกล้อง** เมื่อแท็บนั้น active
- เมื่อ `screen !== 'camera'`: **หยุด trigger การแปลอัตโนมัติ** (ลูป segmentation early-return) เพื่อไม่ให้แปล/บันทึกประวัติขณะผู้ใช้ไม่ได้มองกล้อง — ตัวกล้องยังเปิดอยู่ (warm) แค่ไม่ trigger
- ไม่ต้องใช้ router library — ใช้ state `screen: 'camera' | 'history' | 'settings'` ใน shell component

---

## 4. หน้าจอ 1 — กล้อง (Translate)

โครงเดิม + เพิ่มปุ่มลำโพง:

- Top glass bar: แบรนด์ **"วาทยากร"** (เปลี่ยนจาก "แปลภาษามือไทย") + LIVE chip
  - **ย้ายปุ่มเส้นโครงร่าง (landmark)** ออกจาก top bar ไปไว้ในหน้า ตั้งค่า
  - **ปุ่ม "วลีที่รองรับ"** คงไว้เป็น chip เหมือนเดิมใน P1 (ไม่ย้าย เพื่อลดการเปลี่ยนแปลง) — พิจารณาย้ายไป ตั้งค่า ภายหลังได้
- การ์ดผลแปล (glass) ล่างจอ: คำใหญ่ + แถบความมั่นใจ (เหมือนเดิม)
- **ปุ่มลำโพง 🔊 (ใหม่):**
  - วาง `position:absolute; top:-28px; right:16px` เทียบกับ `.card` → ลอยกึ่งกลางบนขอบมุมขวาบน (ครึ่งบนลอยเหนือการ์ด ครึ่งล่างเกยการ์ด) — ดีไซน์ตามที่อนุมัติ (ตัวเลือก A)
  - กด → เรียก TTS อ่าน `result.sentence` ดังออกลำโพงเครื่อง
  - มี state "กำลังพูด" (ปุ่มเรืองแสง + pill "กำลังพูด…" + คลื่นเสียง)
  - ถ้าไม่มีผลแปล (idle) → ซ่อน/ปิดปุ่ม

---

## 5. หน้าจอ 2 — ประวัติ (History)

- รายการคำที่เคยแปล เป็น **การ์ด glass ใหญ่**: คำ (เด่น) + เวลาแบบสัมพัทธ์ ("2 นาทีที่แล้ว")
- การกระทำ:
  - แตะการ์ด → อ่านออกเสียงคำนั้นซ้ำ (ใช้ `useSpeech`)
  - ปุ่มล้างทั้งหมด (trash) — มีการยืนยันสั้น ๆ
- Empty state: ไอคอน + คำสั้น ("ยังไม่มีประวัติ")
- เก็บใน **localStorage** (ไม่มี backend)

**Data model:**
```ts
interface HistoryItem {
  id: string;        // crypto.randomUUID()
  sentence: string;
  score: number;
  model: string;
  ts: number;        // Date.now()
}
```
- localStorage key: `tsl.history.v1`
- **Cap 100 รายการล่าสุด** — เมื่อเกิน ตัดรายการเก่าสุดออก
- **Dedup**: ไม่บันทึกถ้า `sentence` ตรงกับรายการล่าสุด และห่างกัน < 30 วินาที (กันคำซ้ำจากลูปอัตโนมัติ)

---

## 6. หน้าจอ 3 — ตั้งค่า (Settings)

แถวการตั้งค่าแบบไอคอน + คำสั้น:

| รายการ | ไอคอน | ชนิด | ค่าเริ่มต้น |
|--------|-------|------|-------------|
| ภาษา / Language | globe | segmented toggle ไทย/EN | `th` |
| เส้นโครงร่าง | star/overlay | toggle | off |
| อ่านออกเสียงอัตโนมัติ | volume | toggle (P2) | off |
| ล้างประวัติ | trash | action | — |

**Data model:**
```ts
interface Settings {
  lang: 'th' | 'en';
  showLandmarks: boolean;
  autoSpeak: boolean; // P2: อ่านออกเสียงทันทีเมื่อมีผลแปลใหม่
}
```
- localStorage key: `tsl.settings.v1`

---

## 7. ระบบ i18n (สลับภาษา UI)

- ปัจจุบัน: มีเฉพาะ `frontend/src/i18n/th.ts` และทุก component `import { th }` ตรง ๆ
- เปลี่ยนเป็น:
  - เพิ่ม `frontend/src/i18n/en.ts` — `const en: typeof th = { ... }` (TypeScript บังคับ key ครบ → ถ้าขาด = compile error)
  - เพิ่ม `frontend/src/i18n/index.ts` — รวม dictionary + type `Dict = typeof th`
  - hook `useI18n()` อ่านภาษาจาก `SettingsProvider` แล้วคืน dictionary ที่ใช้งาน
  - **refactor ทุก component** ที่ใช้ `th` ให้เรียก `useI18n()` แทน (เป็นการแก้ mechanical กระทบหลายไฟล์ — วางแผนล่วงหน้า ไม่ใช่ค้นพบกลางทาง)
- ฟังก์ชันใน dictionary (เช่น `frames(n)`, `confidence(pct)`) ต้องมีครบทั้ง th/en (type ครอบให้แล้ว)
- ผลแปลภาษามือ (`result.sentence`) ไม่ถูกแปล — เป็นเอาต์พุตของโมเดล (ไทย)

---

## 8. TTS — `useSpeech`

- ใช้ **Web Speech API** (`window.speechSynthesis`) — ฟรี, ทำงานออฟไลน์บน Windows/Android, มีเสียงไทย
- hook `useSpeech()`:
  - `speak(text: string, lang = 'th-TH')` — เลือก voice ภาษาไทยถ้ามี, fallback voice ปริยาย
  - `speaking: boolean` — สำหรับ state "กำลังพูด"
  - `supported: boolean` — ถ้าเบราว์เซอร์ไม่รองรับ → ซ่อน/disable ปุ่มลำโพง (graceful degradation)
  - เคารพ `prefers-reduced-motion` เฉพาะส่วน animation (ไม่กระทบเสียง)
- หมายเหตุ: "เสียง AI" คุณภาพสูงขึ้น (cloud TTS) เป็น P2/อนาคต ไม่อยู่ในขอบเขตนี้

---

## 9. สถาปัตยกรรม / คอมโพเนนต์ / ไฟล์

**Providers (ครอบใน `App.tsx`):**
```
ModelsProvider
  └ SettingsProvider        (ใหม่: lang/overlay/autoSpeak + persist localStorage)
      └ HistoryProvider     (ใหม่: list + add/clear + persist localStorage)
          └ AppShell        (state screen + bottom nav + overlays)
```

**ไฟล์ใหม่:**
- `frontend/src/hooks/SettingsProvider.tsx` + `useSettings()`
- `frontend/src/hooks/HistoryProvider.tsx` + `useHistory()`
- `frontend/src/hooks/useSpeech.ts`
- `frontend/src/i18n/en.ts`, `frontend/src/i18n/index.ts`
- `frontend/src/components/BottomNav.tsx`
- `frontend/src/components/HistoryScreen.tsx`
- `frontend/src/components/SettingsScreen.tsx`

**ไฟล์ที่แก้:**
- `frontend/src/App.tsx` — แยกเป็น shell: คง `CameraView`+capture mount ตลอด, สลับ overlay, หยุด trigger แปลเมื่อออกจากแท็บกล้อง, บันทึกผลลง history เมื่อแปลสำเร็จ (≥ confidence floor)
- `frontend/src/components/ResultCard.tsx` — เพิ่มปุ่มลำโพง + state กำลังพูด, `overflow:visible`
- `frontend/src/styles/tokens.css` — เพิ่มสไตล์ `.bottom-nav-float`, ปุ่มลำโพง, การ์ดประวัติ, แถวตั้งค่า; **แก้ `.result-glass-panel.live` ให้ `overflow: visible`** (ดูข้อ 10)
- component อื่น ๆ ที่ใช้ `th` → ใช้ `useI18n()`

---

## 10. จุดที่ต้องระวัง (Implementation notes)

1. **overflow clip ปุ่มลำโพง**: `.result-glass-panel` ปัจจุบันตั้ง `overflow-y: auto` (สำหรับ panel เลื่อนของ "วลีที่รองรับ") ซึ่งจะ **clip ปุ่มลำโพงที่ยื่นออก** → ให้แผง **live** (`.result-glass-panel.live`) ใช้ `overflow: visible` แยกจากแผง phrases ที่ยังต้องเลื่อนได้
2. **camera lifecycle**: ห้าม unmount กล้องตอนสลับแท็บ (ข้อ 3)
3. **dedup ประวัติ**: ข้อ 5
4. **i18n เป็น mechanical change** กระทบหลายไฟล์: ข้อ 7
5. **บันทึกประวัติ**: บันทึกเฉพาะผลที่ผ่าน `CONFIDENCE_FLOOR` (เหมือนเงื่อนไข `displayedResult` ปัจจุบัน) เพื่อไม่ให้คำมั่ว ๆ ลงประวัติ

---

## 11. การแบ่งเฟส (Phasing)

> เหตุผล: อาจใกล้เดดไลน์ I-New Gen (30 มิ.ย.) และต้องไม่ทำให้แอปที่ทำงานอยู่พังก่อนส่ง — แบ่งเฟสให้มี increment ที่ปลอดภัยพร้อมใช้ก่อน (ถ้าเดดไลน์ไม่ใช่ตัวเร่ง ผู้ใช้ปรับลำดับได้)

**P1 — ฟีเจอร์ครบ เสี่ยงต่ำ (ส่งได้/ถ่ายภาพประกอบได้):**
- Floating bottom nav + สลับหน้า (กล้อง mount ตลอด)
- ปุ่มลำโพงบน ResultCard + `useSpeech` (กดเอง) + แก้ overflow
- ประวัติ: บันทึก + รายการ + ล้าง + persist + dedup
- ตั้งค่า: สลับภาษา ไทย/EN (i18n) + ย้าย toggle เส้นโครงร่างมาที่นี่ + ล้างประวัติ
- เปลี่ยนแบรนด์เป็น "วาทยากร"

**P2 — ขัดเกลา + ความทนทาน (เลื่อนไปภายหลัง — ไม่อยู่ในงานรอบนี้):**
- autoSpeak (อ่านออกเสียงอัตโนมัติเมื่อมีผลแปลใหม่), เวลาแบบสัมพัทธ์ที่ละเอียดขึ้น
- ชุดเทสต์ Vitest (ดูข้อ 13)
- ขัดเกลา empty state / reduced-motion

> **ขอบเขตงานรอบนี้ = P1 เท่านั้น** (ผู้ใช้ยืนยัน) — favorites ถูกตัดออกถาวร (ดูข้อ 12)

---

## 12. ขอบเขตที่ไม่ทำ (YAGNI / Out of scope)

- แปลผลลัพธ์เป็นภาษาอื่น (ผู้ใช้เลือกแค่สลับภาษา UI)
- cloud/AI TTS คุณภาพสูง (ใช้ browser TTS ก่อน)
- backend/บัญชีผู้ใช้/sync ข้ามอุปกรณ์ (ประวัติเก็บ localStorage เครื่องเดียว)
- โหมดขยายผลแปลเต็มจอ (ตัดออกตามที่ผู้ใช้ระบุ)
- **ดาวคำโปรด / ปักคำ (favorites)** — ตัดออกตามที่ผู้ใช้ระบุ (ประวัติเรียงตามเวลาอย่างเดียว ไม่มี field `favorite`)

---

## 13. การทดสอบ (scale ตามเวลา)

- หน่วย logic บริสุทธิ์ (คุ้มค่า ทำก่อน): history store (add/dedup/cap/clear/persist), settings store (persist/สลับภาษา), i18n completeness (ทุก key ของ th มีใน en — TS ครอบแล้ว + เทสต์ย้ำ), `useSpeech` (mock `speechSynthesis`)
- เครื่องมือ: เพิ่ม **Vitest** (frontend ยังไม่มี test setup) — เป็น P2 และเป็นสิ่งแรกที่ตัดได้ถ้าเวลาไม่พอ
- component/RTL test: เบา ๆ หรือทดสอบด้วยมือ

---

## 14. เกณฑ์ความสำเร็จ (Acceptance criteria)

- เข้าถึงทั้ง 3 แท็บผ่าน floating nav ได้ และ **กล้องไม่ re-init เมื่อสลับแท็บ**
- ลูปแปลทำงานเหมือนเดิมบนแท็บกล้อง และ **หยุด trigger เมื่อออกจากแท็บ**
- ปุ่มลำโพงอ่านคำด้วย TTS ได้ และ **ไม่ถูก clip**
- ประวัติบันทึกคำที่แปลสำเร็จ (deduped), คงอยู่หลังรีโหลด, ล้างได้
- สลับภาษา UI ไทย/EN ได้ครบทุกข้อความ และคงค่าหลังรีโหลด
- ไม่มี regression กับพฤติกรรมการแปลเดิม
