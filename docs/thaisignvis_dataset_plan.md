# Plan: สร้าง training set ภาษามือไทยระดับประโยคจาก ThaiSignVis

สถานะ: ร่าง (รออนุมัติก่อนลงมือ)
เป้าหมาย: เปลี่ยนวิดีโอ ThaiSignVis (Kaggle, Apache 2.0) ให้เป็น dataset
ระดับประโยคที่ป้อนเข้า `SignToTextTransformer` ได้โดยตรง โดยใช้ pipeline
และ schema ที่มีอยู่เดิมให้ได้มากที่สุด

---

## 1. ข้อเท็จจริงของแหล่งข้อมูล

- **ThaiSignVis** (`kaggle.com/datasets/thanawuttimpitak/thaisignvis`)
  - วิดีโอ MP4 ภาษามือไทย, ~165 GB, License **Apache 2.0** (redistribute ได้)
  - โครงสร้าง: `process_videos/` (มี annotation) + `no_labels_videos/`
  - ไฟล์ประกอบ: `metadata.csv`, `transcript_window_*.csv`, `detection_results.csv`
  - **ไม่มี landmark สำเร็จ** — ต้องสกัด MediaPipe เอง
  - `detection_results.csv` = ผล detection (ไม่ใช่ keypoint 162-dim) → ไม่ใช้เป็น feature

---

## 2. จุดตัดสินใจสำคัญ: schema ของ feature (ต้องเลือกก่อน)

repo มี landmark layout 2 แบบที่ **ไม่เข้ากัน**:

| Layout | ที่มา | มิติ | ใช้กับ |
|---|---|---|---|
| TSL-51 | `data/tsl51.py` | **162** (6 pose + 6 face + 21×2 hands = 54 pts ×3) | SLT model (`_INPUT_DIM=162`) |
| normalize | `features/normalize.py` | **312** (104 pts ×3) | encoder/ISLR pipeline |

**ทางเลือก:**

- **A) จับคู่ TSL-51 layout 162-dim เป๊ะ** → ThaiSignVis เข้ากันกับ checkpoint/finetune
  TSL-51 ได้ทันที **แต่** ต้องรู้ว่า TSL-51 เลือก 6 pose + 6 face ตัวไหน
  (ชุดข้อมูลภายนอก `Namonpas/thai-sign-language-tsl51` — ต้องตรวจ column spec จริง)
- **B) นิยาม 162-dim ของเราเอง (เลือก subset ชัดเจน, internally consistent)** →
  ใช้ ThaiSignVis เทรน standalone ได้เลย แต่ **ไม่ transfer** กับ TSL-51 checkpoint
- **C) ใช้ `normalize.py` (312-dim) ที่มีอยู่** → ต้องตั้ง `input_dim=312` แยก stage
  (ได้ normalization ฟรี: nose-center + inter-shoulder scale) ไม่ผูกกับ TSL-51

> **ตัดสินใจแล้ว: เลือก C (312-dim ผ่าน `normalize.py`).** เหตุผล:
> ได้ normalization (nose-center + inter-shoulder scale) ที่ทดสอบแล้วฟรี;
> ThaiSignVis เป็นชุดหลักจึงควรออกแบบรอบมัน ไม่บีบเข้า 162-dim เก่า;
> ไม่ผูกกับ spec 6 pose/6 face ของ TSL-51 ที่เรายังไม่เห็น;
> TSL-51 สกัดเป็น 312-dim ด้วย pipeline เดียวกันภายหลังได้ → schema เดียวกันหมด.
> ผล: `input_dim=312`, stage ใหม่แยกจาก 162-dim เดิม (checkpoint TSL-51 เก่าไม่ transfer ตรง — ยอมรับได้).

---

## 3. Pipeline ภาพรวม

```
process_videos/*.mp4 + transcript_window_*.csv + metadata.csv
  │
  ├─[1] parse manifest: จับคู่ (video, ช่วงเวลา start/end, ข้อความไทย) → segment list
  │
  ├─[2] สกัด landmark: cv2 อ่านเฟรมตามช่วง → MediaPipe Holistic (543,3)
  │        → เลือก subset ตาม schema ที่เลือก (ข้อ 2) → (T, D) float32
  │
  ├─[3] cache: เซฟ .npy ต่อ segment (สกัดครั้งเดียว)
  │
  └─[4] loader: อ่าน .npy + ข้อความ → SignTextExample → slt_collate → model
```

---

## 4. ไฟล์ที่จะสร้าง / แก้

1. **`scripts/extract_thaisignvis_landmarks.py`** (รันครั้งเดียว, ต้องมี `mediapipe`+`opencv`)
   - อ่าน `metadata.csv` + `transcript_window_*.csv`
   - ตัด segment ตาม timestamp, สกัด Holistic, เขียน `<seg_id>.npy`
   - เขียน `thaisignvis_manifest.csv` (cols: `segment_id, npy_path, text, video_id, start, end`)
   - flags: `--data-root`, `--out-dir`, `--limit`, `--fps`, `--schema {tsl51,norm312}`
   - idempotent: ข้าม segment ที่มี .npy แล้ว

2. **`src/tsl/data/thaisignvis.py`** (loader — แนวเดียวกับ `how2sign.py`)
   - `load_thaisignvis_manifest(data_root, split) -> list[SignTextExample]`
   - `load_thaisignvis_features(npy_path) -> np.ndarray`
   - reuse `SignTextExample`

3. **แก้ `src/tsl/train/train_slt.py`**
   - เพิ่ม choice `--stage thaisignvis` ใน `_parse_args`
   - เพิ่มกิ่งใน `_load_data` + `_resolve_input_dim` (162 หรือ 312 ตามข้อ 2)

4. **tests**
   - `tests/data/test_thaisignvis.py` — manifest parsing + feature shape (mock CSV/npy)
   - เพิ่ม case ใน `test_train_slt_stages.py`

---

## 5. การจัดการ transcript / segmentation

- ต้องดู column จริงของ `transcript_window_*.csv` ก่อน (คาดว่ามี start/end time + text)
- 1 transcript window = 1 ประโยค/ตัวอย่าง = 1 segment
- ถ้า window ยาวเกิน → cap ตาม `max_frames` (กัน memory)
- ทิ้ง segment ที่ text ว่าง หรือเฟรมที่มือหายทั้งหมด (NaN เกิน threshold)

---

## 6. กลยุทธ์ subset / cost (165 GB)

- เริ่มจาก subset: `--limit N` หรือเลือกบางโฟลเดอร์ใน `process_videos/`
- สกัด landmark แล้ว .npy เล็กกว่าวิดีโอมาก → เก็บเฉพาะ .npy, ลบ MP4 ได้
- รองรับ resume (idempotent) เพราะสกัดทั้งชุดใช้เวลานาน

---

## 7. ลำดับการทำ (proposed order)

1. ✅ schema = **C (312-dim, `normalize.py`)** — ตัดสินใจแล้ว
2. ตรวจ column จริงของ `metadata.csv` + `transcript_window_*.csv` (ดาวน์โหลด subset เล็ก) ← **ต้องทำก่อนเขียน loader**
3. เขียน `extract_thaisignvis_landmarks.py` + ทดสอบกับ 2-3 วิดีโอ
4. เขียน `thaisignvis.py` loader + tests
5. ต่อ `--stage thaisignvis` ใน `train_slt.py`
6. smoke train บน subset → ตรวจ loss ลด + decode ออกข้อความไทย
7. (ต่อยอด) เพิ่ม chrF eval ตามที่เคยเสนอ

---

## 8. ความเสี่ยง / ข้อควรระวัง

- **schema mismatch** (ข้อ 2) — เลือกผิดทำให้ feature เข้าโมเดลไม่ได้/transfer ไม่ได้
- **column ของ transcript ไม่ตรงคาด** — ต้องเห็นไฟล์จริงก่อน finalize loader
- **คุณภาพ transcript** — ถ้าเป็น auto-caption อาจ noisy ต้อง clean
- **เวลา/ดิสก์** — สกัด 165 GB ใช้เวลานาน + ต้องมี mediapipe ติดตั้งได้บน Windows
- **ความสอดคล้อง normalization** — ถ้าเลือก A ต้อง normalize แบบเดียวกับ TSL-51 ด้วย
  (ไม่ใช่แค่เลือก landmark ให้ตรง)
```
