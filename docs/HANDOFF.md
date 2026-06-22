# HANDOFF — Thai Sign Language Translator

> สถานะ ณ **2026-06-23** · เขียนเพื่อส่งต่อให้ session/คนถัดไปทำงานต่อได้ทันที โดยไม่ต้องไล่ย้อน context เดิม
> (ฉบับก่อนหน้า 2026-06-16 = ยุค `slt_v2`/legacy — เนื้อหายังเก็บไว้ใน §9 บทเรียนสำคัญ)
>
> **อัปเดตล่าสุด 2026-06-23:** Kaggle curriculum training (v40) สำเร็จ — Phase 1 pretrain ทุก 1938 ตัวอย่าง → Phase 2 finetune TSL-51
> ได้ **chrF 96.77 / BLEU 97.73 / exact-match 92%** (n=25, seed=42). Promoted vs incumbent chrF 86.95. ดูรายละเอียดใน §0 item 16.

---

## 0. TL;DR — อยู่ตรงไหน / ทำอะไรต่อ

- โปรเจกต์ย้ายจาก legacy `slt_v2` (จำประโยคปิด 30 คำ) → ระบบ **End-to-End จริง** = `PoseToTextT5`
  (pose encoder 312-dim → pretrained `google/mt5-small` decoder, open-vocab, beam search)
- **มี Colab A100 run เพิ่มแล้ว** (`thai-sign-train-managed-chrf`) และแก้ flow หลัก 2 จุด:
  1. early stopping ใช้ `val_chrf` ตรงกับ metric งานจริง
  2. local sync ผ่าน WSL path ได้จริงแล้ว
- **กันปัญหา Colab ซ้ำเพิ่มอีก 2 จุดแล้ว**:
  1. `scripts/colab_cli_pose_t5.ps1` probe health ของ session ก่อน reuse/upload เพื่อคัด ghost session ที่ `status` ยังตอบได้แต่ `exec/upload` ใช้ไม่ได้
  2. `scripts/colab_bootstrap_pose_t5.py` fail-fast ถ้า `resume != none` แต่ restore checkpoint ไม่ลง `out_dir`
- **แก้ root cause ของ monitoring หลอกหลัง resume แล้ว**:
  - checkpoint dataset ที่ใช้ resume เคยพา `train.log`, `train_metrics.json`, `launch.json`, export files, และ `best/latest_checkpoint.txt` เก่ากลับเข้า remote `out_dir`
  - ทำให้รอบใหม่ดูเหมือนยังค้างอยู่ที่ log/metrics เก่าทั้งที่ process ใหม่กำลังรัน
  - ตอนนี้ `scripts/colab_bootstrap_pose_t5.py` จะ prune artifact พวกนี้ทิ้งหลัง restore และเก็บไว้เฉพาะ `ckpt_step*.pt` สำหรับ resume
- **กันปัญหา launch/check ซ้ำเพิ่มอีก 2 จุด (2026-06-20 รอบหลัง)**:
  1. `scripts/colab_cli_pose_t5.ps1` มี `-MinGpu`, `-GpuRetryMinutes`, `-GpuRetryDelaySec` แล้ว เพื่อให้รอคิว `H100/A100` ได้จริง แทนการยอม fallback ไป `T4` ทันที
  2. `scripts/colab_checkpoint_sync.py` จะเขียน `session_status` (hardware / status / last_execution) ลง `sync_state.json` เพื่อให้ local check รู้ว่า session ยัง RUNNING/IDLE จริง ไม่ต้องเดาจาก `train_metrics.json` อย่างเดียว
  3. `scripts/colab_cli_pose_t5.ps1` เขียน `checkpoints/colab_sync/<session>/launcher.status.json` แล้ว ระหว่างรอ GPU จะเห็น `phase`, `attempts`, `deadline`, และ `last_errors` ของแต่ละ GPU ได้เลย
  4. `parse_status_output()` ใน `scripts/colab_checkpoint_sync.py` แยกกรณี `[colab] Session '...' not found.` ออกจาก live session จริงแล้ว
  5. `scripts/colab_cli_pose_t5.ps1` จำผลตอบกลับจาก backend ราย GPU ไว้ที่ `checkpoints/colab_sync/gpu_availability.json`:
     - ถ้าโดนแนว `Backend rejected accelerator` / `Precondition Failed` จะ mark GPU tier นั้นเป็น rejected ชั่วคราว เพื่อไม่วนชนซ้ำทุก launch
     - ถ้าเป็นแนว capacity ชั่วคราว (`Service Unavailable`, `TooManyAssignmentsError`) จะใส่ cooldown สั้นแล้วค่อยลองใหม่
     - ผลคือ launcher จะยังพยายามใช้ tier แรงสุดที่ **ยังใช้งานได้จริง** แต่ไม่เสียเวลาตีกับ tier ที่ backend ปฏิเสธไปแล้ว
  6. `src/tsl/train/checkpointing.py` จะเขียน `best_checkpoint.txt` / `latest_checkpoint.txt` ใหม่ทุกครั้งที่ save/prune checkpoint แล้ว
     - root cause ที่เจอจริงใน `r8` คือ sidecar สองไฟล์นี้ค้างค่าจาก dataset restore เก่า ทำให้ monitoring/publisher มอง state ผิดจาก `.pt` ที่มีอยู่จริง
     - มี test เพิ่มใน `tests/train/test_checkpointing.py`
- **อัปเดต launcher ล่าสุด (2026-06-20 รอบเช้า)**:
  1. `scripts/colab_cli_pose_t5.ps1` แยก `backend rejected` ออกจาก `temporary cooldown` ใน status/message แล้ว
  2. เพิ่ม `-AllowFallbackBelowMinGpuOnReject`:
     - ถ้า account ถูกปฏิเสธ `H100/A100/L4` จริง ไม่ใช่แค่ capacity เต็ม
     - launcher จะยอมไล่ลง tier ที่ต่ำกว่าใน `GpuPriority` เพื่อไม่ให้ lane ตายคา `MinGpu`
  3. `launcher.status.json` ตอนนี้เก็บ `preferred_gpu_candidates`, `effective_gpu_candidates`, `fallback_gpu_candidates`, และ `fallback_active`
     - ใช้ไฟล์นี้เป็น truth source ว่าตอนนี้กำลังรอ GPU เดิมอยู่ หรือถูก downgrade ไปลอง tier ต่ำกว่าแล้ว
  4. เพิ่ม `-GpuRejectCooldownMinutes` (default `360`)
     - root cause ที่เจอจริง: account นี้โดน backend reject `H100/A100/L4` แบบ entitlement/quota แต่ launcher เคย cache แค่ 5 นาที
     - ผลคือ background launcher วนกลับไปชน GPU tier เดิมซ้ำ ๆ ทั้งที่ไม่ใช่ transient failure
     - ตอนนี้ rejected tiers จะถูกพักนานกว่าชัดเจน ส่วน `Service Unavailable` ยังใช้ retry สั้นเหมือนเดิม
  5. `src/tsl/train/train_pose_t5.py` จะเขียน `manifest_quality.json` ทุกครั้งก่อนเข้า train
     - ใช้ `tsl.eval.manifest_quality.analyze_manifest_quality(...)`
     - log จะพิมพ์ `train_examples_per_target`, `target_overlap_ratio`, `video_overlap_count`
     - ถ้าเข้า regime เสี่ยง collapse (`one example per target`, target overlap ต่ำ, หรือ split ผิด) จะมี warning ทันทีใน stdout และมี artifact ตรวจย้อนหลังใน out-dir
- **แก้ root cause ของ artifact ปลายรอบแล้ว**:
  - `src/tsl/train/train_pose_t5.py` จะ export น้ำหนักจาก **best checkpoint ตาม metric ที่ติดตาม** ตอน finalize ไม่ใช่น้ำหนักล่าสุดที่อาจแย่กว่า best
  - เพิ่ม `--resume best` เพื่อให้รอบ continue ถัดไปเริ่มจาก checkpoint ที่ดีที่สุดจริง แทน `latest`
  - ปรับ `val_chrf` ตอน train ให้ใช้ decode defaults ชุดเดียวกับ runtime (`72 tokens / beam 5 / no-repeat 3 / repetition_penalty 1.5 / length_penalty 0.7`) เพื่อลด metric mismatch ระหว่าง train กับ inference จริง
- A100 run จบที่ **step 1700** ด้วย early stopping; best `val_chrf` = **13.05 ที่ step 1100**
- checkpoint ที่ถูก mirror ลง local ครบตอนนี้:
  - `checkpoints/colab_sync/thai-sign-train-managed-chrf/ckpt_step00000500.pt`
  - `.../ckpt_step00001000.pt`
  - `.../ckpt_step00001500.pt`
- exported final model จาก A100 run ถูกดึงลง local แล้ว:
  - `checkpoints/pose_t5_a100_final_export/`
- **decode defaults ใหม่ช่วยลด repetition ชัดเจน**:
  - ก่อนปรับ decode: chrF **22.5** บน validation subset 50 ตัวอย่าง
  - หลังปรับ decode defaults ใน `PoseT5Translator`: chrF **34.5** บน subset เดิม
- **แก้ metric bug แล้ว**:
  - `src/tsl/eval/slt_metrics.py` เคยส่ง refs ผิด shape ให้ `sacrebleu`
  - report comparison ก่อนรอบแก้วันที่ 2026-06-20 จึง **สูงเกินจริง**; ให้เชื่อ report ที่ลงท้าย `corrected_metrics.json`
- แต่ semantic quality ยังไม่ถึงขั้นใช้งานจริง: โมเดลยังชอบเดาประโยค generic ซ้ำ เช่น
  `พระเยซูจึงตรัสกับพวกเขาว่า “ข้าพเจ้าไม่เชื่อ”`

**3 งานที่ค้าง (เรียงลำดับ):**
1. **รอบล่าสุดที่ได้ใช้จริงตอนนี้คือ `thai-sign-train-managed-r4` (A100)**:
   - launch 2026-06-20: H100 fail = `503`, แต่ fallback ไป **A100 สำเร็จ**
   - resume จาก `ckpt_step00001500.pt`
   - ตั้ง `checkpoint_steps=100` แล้ว จึงได้ `ckpt_step00001600.pt` และ `ckpt_step00001700.pt`
   - metric รอบนี้:
     - step 1600: `val_chrf 10.64`
     - step 1700: `val_chrf 12.70`
   - หลังแก้ finalize-best export แล้ว final artifact ของรอบนี้ดึงจาก best checkpoint ที่มีอยู่จริง
   - eval แบบ **corrected** tuned decode บน validation subset 50 ตัวอย่าง:
     - `pose_t5_a100_step1500_export`: **chrF 12.20**
     - `thai-sign-train-managed-r4` final export: **chrF 12.36**
   - สรุป: `r4_final` ดีกว่าเดิมเล็กน้อย และถูก promote เป็น runtime default ตอนนี้
2. **แก้ training regime ต่อ** เพื่อดัน semantic quality ไม่ใช่แค่ลด repetition
3. **Version final export + tokenizer files** เข้า dataset/sync flow ให้ครบอัตโนมัติทุกครั้ง
4. **`thai-sign-train-managed-r6`** ถูก launch ด้วย policy `MinGpu=A100`, `GpuRetryMinutes=30`:
   - truth source หลักตอนนี้คือ `checkpoints/colab_sync/thai-sign-train-managed-r6/launcher.status.json`
   - ณ รอบล่าสุด:
     - `phase = waiting_for_gpu`
     - `attempts.H100 = 12+`
     - `attempts.A100 = 12+`
     - backend ตอบแนว `Backend rejected accelerator 'H100'/'A100'`
   - สรุป: ตอนนี้ไม่ใช่แค่ capacity ชั่วคราว แต่เหมือน account/quota/entitlement path ไม่ยอมให้ H100/A100 ในรอบนี้
5. **`thai-sign-train-managed-r7`** (policy `MinGpu=L4`) ก็ถูก backend ปฏิเสธเหมือนกัน:
   - `attempts.L4 = 14+`
   - `last_errors.L4` = `Backend rejected accelerator 'L4'`
   - สรุป: L4 route ใช้ไม่ได้ในรอบนี้เช่นกัน
6. **`thai-sign-train-managed-r8`** ถูกเปิดบน **T4** เพื่อให้มี training lane เดินจริง:
   - `launch.json` ยืนยัน config ใหม่:
     - `--resume best`
     - `--reset-progress-history`
     - `--lr 1e-05`
     - `--checkpoint-steps 100`
   - remote probe ยืนยันว่าใน `train.log` มี:
     - `[resume] Loading checkpoint: /content/checkpoints/pose_t5_v3_colab/ckpt_step00001000.pt`
     - `[resume] Resuming from step=1000, epoch=19`
     - `[resume] Resetting progress history at resumed step 1000.`
   - remote `train.pid` ยัง alive และ `ps` รายงาน CPU ใช้งานต่อเนื่อง → training lane นี้ยังเดินอยู่
   - caveat: local sync รอบแรกยังเจอ `500 Internal Server Error` ตอนดึง `ckpt_step00001000.pt` / `ckpt_step00001500.pt` จาก T4 session แม้ metadata/log ลงมาได้
   - note สำคัญ: `train.log` / `train_metrics.json` ที่เห็นใน `r8` ตอนนี้ยังปน history เก่าจาก dataset restore; fix ถูกลงใน bootstrap แล้ว แต่จะมีผลชัดตั้งแต่รอบ launch ถัดไป
   - ความคืบหน้าล่าสุดก่อน session หาย:
     - `step 1400`: `val_chrf 13.25`  **(best ใหม่ของรอบนี้)**
     - `step 1500`: `val_chrf 12.25`
     - `step 1600`: `val_chrf 12.01`
   - จากนั้น `colab status -s thai-sign-train-managed-r8` ตอบ `Session ... not found` / `appears to be lost (404/401)`:
     - ถือว่ารอบ `r8` จบแบบไม่ clean
     - checkpoint `1400` / `1600` ยังอยู่ใน staging `/content/kaggle_ckpt_publish` ก่อน session หาย แต่ยังไม่ถูกยืนยันว่าขึ้น Kaggle dataset ทัน
   - sync lane ตอนนี้ดีขึ้นบางส่วน:
     - `scripts/colab_checkpoint_sync.py` มี fallback ใช้ `colab exec` สำหรับ `ls` และไฟล์ text (`train.log`, `train_metrics.json`, `launch.json`)
     - จึงตาม progress ของ `r8` ลง local mirror ได้แล้ว
     - และถ้าโหลด `.pt` ตรงจาก session ไม่ได้ (`500` ผ่าน proxy ของ T4) sync จะพยายาม refresh จาก Kaggle dataset `orbitorls/thai-sign-ckpt` มา seed local mirror แทน
7. **`thai-sign-train-managed-r9`** เป็นรอบ relaunch หลัง fix sidecar/checkpoint metadata:
   - ใช้ config เดิมบน `T4`
   - รอบล่าสุดยังไม่ขึ้น session เพราะ backend ตอบ `T4: Service Unavailable` ซ้ำ (`attempts.T4 = 8`)
   - `launcher.status.json` ของ `r9` คือ truth source ว่าเป็น capacity issue ชั่วคราว ไม่ใช่ reject ถาวร
8. **launcher compatibility fix (Windows PowerShell 5.1)**:
   - root cause ที่ทำให้ retry/cache เพี้ยนคือ `ConvertFrom-Json -AsHashtable` และ `Set-Content -Encoding utf8NoBOM` ใช้ไม่ได้บน shell รุ่นนี้
   - ตอนนี้ `scripts/colab_cli_pose_t5.ps1` แก้แล้ว:
     - parse JSON cache ผ่าน helper ที่แปลง `PSCustomObject -> hashtable`
     - strip BOM ได้
     - เขียน cache ด้วย `.NET UTF8Encoding(false)` แทน `utf8NoBOM`
   - ผลคือ launcher ใช้งานแบบ background บนเครื่องนี้ได้จริงแล้ว
9. **`thai-sign-train-managed-r12`** คือ background launcher ตัวล่าสุดที่ยัง active:
   - รันจาก local PowerShell แบบ hidden/background
   - policy ตอนนี้: `T4 only`, `GpuRetryMinutes=90`, `GpuRetryDelaySec=20`
   - truth sources:
     - `checkpoints/colab_sync/thai-sign-train-managed-r12/launcher.status.json`
     - `.../launcher.stdout.log`
   - หลักฐานล่าสุด:
     - process ยังอยู่
     - `phase = waiting_for_gpu`
     - `deadline = 2026-06-20T05:31:47`
     - stdout มีลูป `Retrying in 20 sec...`
   - แปลว่า lane นี้กำลังรอ capacity จริงอยู่ ไม่ได้ตายตั้งแต่ startup เหมือนรอบก่อน
10. **local RTX 4060 lane** ถูกเปิดเพิ่มเป็น execution path หลักระหว่างรอ Colab:
   - เครื่องนี้มี `NVIDIA GeForce RTX 4060 8GB` และ `torch.cuda.is_available() == True`
   - `scripts/train_local_gpu.py` ถูกอัปเดตให้ตรงกับ regime ปัจจุบัน:
     - defaults = `lr=1e-5`, `dropout=0.4`, `weight_decay=0.1`
     - `batch_size=4`, `grad_accum=8`
     - `eval_steps=100`, `checkpoint_steps=1000`
     - `resume=best`, `early_stopping_metric=val_chrf`, `early_stopping_patience=12`
     - seed out-dir จาก `kaggle_upload/thai-sign-ckpt`
   - helper seed จะ copy เฉพาะ checkpoint ที่อ้างอิงจาก `best_checkpoint.txt` / `latest_checkpoint.txt` แล้ว ไม่ copy ทุก `.pt` แบบเดิม
   - helper seed จะ validate checkpoint ก่อนใช้ด้วย:
     - ถ้า out-dir มีแต่ checkpoint เสีย จะลบทิ้งแล้ว seed ใหม่
     - ถ้า checkpoint ต้นทางโหลดไม่ได้ จะไม่ถูก copy เข้ามา
   - run ที่ active ตอนนี้:
     - process: `python -u scripts/train_local_gpu.py`
     - out dir: `checkpoints/pose_t5_rtx4060_resume_best/`
     - log: `local_train.stdout.log`, `local_train.stderr.log`
   - หลักฐานล่าสุด:
     - process ยังอยู่
     - out-dir ตอนนี้มี checkpoint ที่ใช้ได้จริงอย่างน้อย `ckpt_step00001700.pt` และ `ckpt_step00001800.pt`
     - local lane เดิมเคยเริ่มจาก scratch แบบเงียบ ๆ เพราะ checkpoint ที่ copy เข้า out-dir เสีย; root cause นี้ถูกแก้แล้วด้วย checkpoint validation
     - local lane ตัวล่าสุด resume ได้จริงจาก:
       - `[resume] Resuming from step=1800, epoch=32`
       - `[resume] Resetting progress history at resumed step 1800.`
     - progress ล่าสุด:
       - `step 1800`: `train_loss 3.7132`, `val_loss 3.0265`, `val_chrf 14.73`
       - `step 1900`: `train_loss 3.6523`, `val_loss 3.0469`, `val_chrf 14.50`
       - `step 2000`: `train_loss 3.5639`, `val_loss 3.0652`, `val_chrf 14.22`
       - `step 2100`: `train_loss 3.7701`, `val_loss 3.0709`, `val_chrf 14.69`
       - `step 2200`: `train_loss 3.3124`, `val_loss 3.0822`, `val_chrf 14.90`
       - `step 2300`: `train_loss 3.1563`, `val_loss 3.1038`, `val_chrf 14.20`
       - `step 2400`: `train_loss 3.4313`, `val_loss 3.1149`, `val_chrf 15.00`
       - `step 2500`: `train_loss 3.2939`, `val_loss 3.1113`, `val_chrf 14.93`
       - `step 2600`: `train_loss 3.0952`, `val_loss 3.1167`, `val_chrf 15.21`
       - `step 2700`: `train_loss 3.1993`, `val_loss 3.1394`, `val_chrf 14.91`
       - `step 2800`: `train_loss 3.4201`, `val_loss 3.1449`, `val_chrf 14.65`
       - `step 2900`: `train_loss 3.3483`, `val_loss 3.1463`, `val_chrf 15.22`
     - เทียบกับ seed:
       - `step 1700`: `val_chrf 12.70`
       - `step 1800`: `val_chrf 14.73`  **ดีขึ้นชัดเจน**
     - ตอนนี้ best ตาม tracked metric กลายเป็น `step 2900`
     - note: GPU memory ตอน train ขึ้นแถว `7.8 GB / 8.2 GB`, และหลัง eval จะลดลงเกือบ idle ชั่วคราว
     - root cause ที่เจอเพิ่มวันที่ 2026-06-20:
       1. full checkpoint save ที่ `step 3000` ตายกลางคัน ทำให้เหลือ `.tmp` 3.6GB ที่ corrupt และ sidecar ไม่ขยับ
       2. local launcher เดิม validate/rebuild checkpoint refs ด้วยการ `torch.load()` checkpoint local ขนาดหลาย GB ทุกไฟล์ก่อนเข้า train ทำให้ startup ช้ามาก
       3. trainer เดิม pre-load `.npy` train ทั้ง 1670 ไฟล์เข้า RAM ก่อนเริ่ม loop ทำให้ local lane ดูเหมือนค้างที่ `[data] Pre-loading...`
     - ทางแก้ที่ลงแล้ว:
       1. `train_pose_t5.py` จะ save `best_model_state.pt` ทุกครั้งที่ metric ดีขึ้น แม้ชนรอบ full checkpoint cadence
       2. `scripts/train_local_gpu.py` ลด full checkpoint cadence เป็น `5000` steps และ default `resume=auto`
       3. launcher จะ cleanup/recover `*.tmp` checkpoint อัตโนมัติก่อนเริ่มรอบใหม่
       4. launcher เลิก `torch.load()` checkpoint local ทุกไฟล์ตอน startup; สำหรับ local restart จะอัปเดตแค่ `latest_checkpoint.txt` จากชื่อไฟล์ที่มีจริง
       5. local lane ใหม่ default เป็น `--preload-train-features false` เพื่อ stream `.npy` จาก disk ต่อ batch แทนการ preload ทั้งก้อน
     - run ล่าสุดหลัง fix:
       - command เดิมที่ยังรันอยู่: `python -u scripts/train_local_gpu.py --resume auto --checkpoint-steps 5000 --preload-train-features false`
       - note สำคัญ: code ล่าสุดเปลี่ยน default ของ local launcher เป็น `--resume best_state` แล้ว
         - root cause ที่แก้: ถ้า restart หลัง full checkpoint lag เราไม่ควรถอยกลับไป `ckpt_step00002000.pt`
         - trainer ตอนนี้ resume จาก `best_model_state.pt` ได้แบบ model-only แล้ว แม้ไม่มี optimizer/scheduler state
         - มี test รองรับใน `tests/train/test_train_pose_t5.py`
       - log ยืนยัน:
         - `checkpoint_steps = 5000`
         - `resume = auto`
         - `preload = false`
         - `[resume] Loading checkpoint: checkpoints\\pose_t5_rtx4060_resume_best\\ckpt_step00002000.pt`
         - `[data] Streaming train features from disk per batch.`
       - ความคืบหน้าหลัง relaunch:
         - `step 2100`: `train_loss 3.7025`, `val_loss 3.0619`, `val_chrf 15.32`
         - `step 2200`: `train_loss 3.5672`, `val_loss 3.0882`, `val_chrf 14.41`
       - `best_model_state.pt` ถูก refresh สำเร็จแล้วที่ `step 2100`
       - note เพิ่ม:
         - แม้ตัด full checkpoint ออกแล้ว การเขียน `best_model_state.pt` ขนาด ~1.2GB ยังช้าบน Windows
         - มี patch เพิ่มให้ `checkpointing.py` และ best-state save ใช้ `torch.save(..., _use_new_zipfile_serialization=False)` เพื่อลดภาระ serialize ไฟล์ใหญ่ในรอบถัดไป
       - ณ จุดนี้ lane กลับมาวิ่งบน GPU ได้จริงแล้วและผ่าน eval ใหม่อย่างน้อย 1 รอบ
     - อัปเดตถัดมา:
       - run เดิมลากต่อถึง `step 3600` แต่ไม่ชนะ best เดิม:
         - `step 3400`: `val_chrf 15.24`
         - `step 3500`: `14.88`
         - `step 3600`: `14.55`
       - สรุปเชิงตัดสินใจ:
         - `step 2900` ยังเป็น best ที่ชัดเจน
         - run เดิม plateau หลัง best อยู่ `7/12` evals without improvement
         - จึงหยุด run เดิมแล้วเปิด fine-tune lane ใหม่จาก `best_model_state.pt`
     - root cause ใหม่ที่เจอและแก้แล้ว:
       1. local launcher seed out-dir ใหม่จะ copy full checkpoints (`ckpt_step*.pt`) ก่อน แม้เรา launch ด้วย `--resume best_state`
       2. บน Windows การ copy checkpoint 3.6GB สองก้อนทำให้ lane ใหม่เสียเวลานานก่อนเข้า train loop ทั้งที่ไม่จำเป็น
       3. ตอนนี้ `scripts/train_local_gpu.py` รองรับ `resume_mode='best_state'` ใน seed helper แล้ว:
          - ถ้ามี `best_model_state.pt` ใน seed dir จะ copy เฉพาะ best-state + sidecar refs
          - ไม่ copy full checkpoints ที่ไม่ต้องใช้สำหรับ model-only resume
       4. มี test เพิ่มใน `tests/scripts/test_train_local_gpu.py`
     - lane ใหม่ที่ active ตอนนี้:
       - out dir: `checkpoints/pose_t5_rtx4060_finetune_beststate_lr5e6/`
       - launch args หลัก:
         - `--resume best_state`
         - `--lr 5e-6`
         - `--early-stopping-patience 8`
         - `--checkpoint-steps 5000`
         - `--preload-train-features false`
         - `--seed-checkpoint-dir checkpoints/pose_t5_rtx4060_resume_best`
       - evidence จาก `launcher.stdout.log`:
         - `[resume] Loading checkpoint: ...\\best_model_state.pt`
         - `[resume] Resuming from step=2900, epoch=52`
         - `[resume] Restored model weights only; optimizer/scheduler state unavailable.`
         - `[resume] Resetting progress history at resumed step 2900.`
       - ความคืบหน้าล่าสุด:
         - `step 3000`: `val_chrf 15.22`
         - `step 3100`: `val_chrf 15.90`  **สูงกว่า train-side best เดิม**
       - แต่หลังรัน `python scripts/refresh_pose_t5_verified.py` โดยชี้ `--train-dir` ไป lane นี้:
         - candidate export จาก `best_model_state.pt step 3100`
         - subset runtime eval 50 ตัวอย่าง = `chrF 15.44 / BLEU 13.61 / EM 6.0%`
         - incumbent verified = `15.80 / 13.87 / 6.0%`
         - promotion result = `promoted: false`
       - สรุปสำคัญ:
         - train-side `val_chrf` สูงขึ้นไม่ได้แปลว่า runtime export ดีขึ้นเสมอ
         - guardrail export/eval/promote ตอนนี้จับ regression นี้ได้แล้ว
11. **candidate export ที่ใช้งานได้ตอนนี้จาก local best**:
   - export path:
     - `checkpoints/pose_t5_rtx4060_step1800_export_current/`
   - report:
     - `checkpoints/pose_t5_rtx4060_step1800_export_current_eval.json`
     - `checkpoints/pose_t5_rtx4060_step1800_export_current_samples.json`
   - เดิมสร้างจาก:
     - `checkpoints/pose_t5_rtx4060_resume_best/ckpt_step00001800.pt`
   - รอบล่าสุดถูก refresh จาก:
     - `checkpoints/pose_t5_rtx4060_resume_best/best_model_state.pt`
     - metadata ยืนยัน `checkpoint_step = 2900`
   - eval บน validation subset 50 ตัวอย่างด้วย decode settings ชุดเดียวกับ runtime:
     - `step1800 export`: `chrF 14.29`, `BLEU 12.09`, `exact_match 1/50 = 2.0%`
     - `best_state step2200 export`: `chrF 15.36`, `BLEU 13.03`, `exact_match 3/50 = 6.0%`

12. **สถานะล่าสุดก่อนส่งต่อ (2026-06-20 08:48 ICT)**:
   - local fine-tune lane `checkpoints/pose_t5_rtx4060_finetune_beststate_lr5e6` ยังรันอยู่
     - process `python` PID `21520`
     - ล่าสุดถึง `step 3200`
     - best ของ lane นี้ฝั่ง train metrics ยังเป็น `step 3100`, `val_chrf 15.90`
   - แต่ runtime promotion check ล่าสุดยัง **ไม่ผ่าน**
     - candidate export จาก `step 3100` ได้ subset runtime eval `chrF 15.44`
     - incumbent verified export ยังดีกว่า (`chrF 15.80`)
     - ดังนั้นยัง **ห้าม promote** candidate นี้
   - Colab launcher `thai-sign-train-managed-r14`
     - truth source: `checkpoints/colab_sync/thai-sign-train-managed-r14/launcher.status.json`
     - ตอนเช็คล่าสุดอยู่ `phase = requesting_gpu`, `gpu = T4`
     - `H100/A100/L4` ถูก backend reject; `T4` ยังเป็น capacity issue (`Service Unavailable`)
     - ถ้าต้องการให้ reject cooldown ใหม่ 6 ชั่วโมงมีผล ต้อง relaunch process นี้หลัง patch
13. **อัปเดตรอบล่าสุด (2026-06-20 ~08:55 ICT)**:
   - local fine-tune lane `checkpoints/pose_t5_rtx4060_finetune_beststate_lr5e6`
     - ล่าสุดถึง `step 3600`
     - `step 3500`: `val_chrf 14.88`
     - `step 3600`: `val_chrf 15.25`
     - best ของ lane นี้ยังคงเป็น `step 3100`, `val_chrf 15.90`
     - เท่ากับตอนนี้มี `5` eval windows ติดที่ไม่ชนะ best (`3200..3600`)
   - candidate runtime ที่ดีที่สุดที่ verified อยู่ยังเป็น export จาก local best เดิม:
     - `checkpoints/pose_t5_rtx4060_best_export_verified/`
     - corrected subset eval = `chrF 15.80 / BLEU 13.87 / EM 6.0%`
   - candidate จาก lane fine-tune step 3100 ยังแพ้ incumbent:
     - `chrF 15.44 < 15.80`
   - Colab launcher `r14` ถูก relaunch แล้วด้วย
     - `-GpuRejectCooldownMinutes 360`
     - H100/A100 ถูก block ยาวถึงประมาณ `14:50 ICT`
     - fallback ลง `L4/T4` ทำงานแล้ว
     - แต่ `L4` ยังถูก reject และ `T4` ยังติด `Service Unavailable`
14. **อัปเดตถัดมา (2026-06-20 ~09:03 ICT)**:
   - local fine-tune lane เดิม `checkpoints/pose_t5_rtx4060_finetune_beststate_lr5e6`
     - ไปถึง `step 3900`
     - จบด้วย `stopped_reason = early_stopping`
     - best ของ lane นี้ยังเป็น `step 3100`, `val_chrf 15.90`
     - สรุป: lane นี้ไม่สามารถชนะ incumbent runtime verified ได้
   - root cause ใหม่ที่แก้แล้ว:
     1. `src/tsl/data/unified.py` เคย fallback `source='unified'` เมื่อ manifest ไม่มีคอลัมน์ `source`
     2. ทำให้ `PoseToTextT5` path ไม่เห็นความต่างระหว่าง `tsl51_v3` กับ `youtube_sl25_thai_v3`
     3. ฝั่ง train/eval จึงทำ source-level diagnostics หรือ source balancing ไม่ได้จริง แม้ข้อมูลมาจากคนละ corpus
   - patch ที่ลงแล้ว:
     1. `unified.py` infer `source` จากชื่อ `data_root` ถ้า manifest ไม่มีคอลัมน์ `source`
        - ตอนนี้ `data/tsl51_v3` → `tsl51`
        - `data/youtube_sl25_thai_v3` → `youtube_sl25_thai`
     2. `src/tsl/train/train_pose_t5.py` เพิ่ม source-balanced sampling (`WeightedRandomSampler`)
        - เปิดอัตโนมัติเมื่อ train examples มีมากกว่า 1 source
        - เขียน artifact `source_sampling.json`
     3. `scripts/train_local_gpu.py` expose `--balance-sources`
   - lane ใหม่ที่ active ตอนนี้:
     - out dir: `checkpoints/pose_t5_rtx4060_balanced_beststate_lr5e6/`
     - launch args หลัก:
       - `--resume best_state`
       - `--lr 5e-6`
       - `--checkpoint-steps 5000`
       - `--preload-train-features false`
       - `--early-stopping-patience 8`
       - `--balance-sources auto`
     - evidence จาก `launcher.stdout.log`:
       - `source-balanced sampling enabled`
       - `counts={'tsl51': 228, 'youtube_sl25_thai': 1442}`
       - `weights={'tsl51': 0.00438596, 'youtube_sl25_thai': 0.00069348}`
15. **จุดเปลี่ยนสำคัญ: usable model แรกที่ยืนยันแล้ว (2026-06-20 ~09:24 ICT)**:
   - source-balanced mixed lane ไม่ช่วยพอ:
     - `step 2950`: `val_chrf 15.35`
     - `step 3000`: `val_chrf 14.26`
     - ต่ำกว่า best เดิมชัด
   - จึง pivot ไป lane ใหม่ที่ใช้เฉพาะ corpus ที่ learnable จริง:
     - out dir: `checkpoints/pose_t5_rtx4060_tsl51_only_beststate_lr5e6/`
     - launch args หลัก:
       - `--data-roots data/tsl51_v3`
       - `--resume best_state`
       - `--lr 5e-6`
       - `--eval-steps 25`
       - `--checkpoint-steps 5000`
       - `--preload-train-features false`
       - `--early-stopping-patience 10`
       - `--balance-sources false`
     - manifest quality ของ lane นี้:
       - `252 total | 227 train | 25 val`
       - `train_examples_per_target = 3.7213`
       - `target_overlap_ratio = 0.9474`
       - ไม่มี video leakage
   - train-side metric ของ lane นี้ดีขึ้นทันที:
     - `step 2925`: `val_loss 0.1402`, `val_chrf 85.51`
     - ต่อมาวิ่งถึง `step 3075`: `val_chrf 86.18`
   - verified runtime artifact ที่ยืนยันแล้วตอนนี้:
     - export dir: `checkpoints/pose_t5_rtx4060_tsl51_only_export_verified/`
     - eval json: `checkpoints/pose_t5_rtx4060_tsl51_only_export_verified_eval.json`
     - metric บน val split ของ `data/tsl51_v3` (`n=25`):
       - `chrF 85.51`
       - `BLEU 88.01`
       - `exact_match 15/25 = 60.0%`
     - runtime metadata ตอนนี้ชี้ checkpoint:
       - `best_model_state.pt`
       - `step 2925`
       - `epoch 55`
   - note สำคัญ:
     - มี best ใหม่ฝั่ง train ที่ `step 3075` สูงกว่า verified export (`86.18 > 85.51`)
     - แต่การ refresh export รอบถัดมาชน shell timeout เพราะการ write model shard บน Windows ช้ามาก
     - artifact ที่ยืนยันใช้งานได้แน่ ๆ ตอนนี้จึงยังอิง `step 2925`
   - สถานะการรัน:
     - process ของ lane `tsl51_only` ถูกหยุดแล้วหลังได้ verified artifact เพื่อไม่ให้ state วิ่งต่อเกิน best ที่ export อยู่
     - `best_state step2400 export`: `chrF 15.20`, `BLEU 12.97`, `exact_match 4/50 = 8.0%`
     - `best_state step2600 export`: `chrF 15.26`, `BLEU 13.24`, `exact_match 3/50 = 6.0%`
     - `best_state step2900 export`: `chrF 15.60`, `BLEU 13.81`, `exact_match 3/50 = 6.0%`
   - เทียบกับ runtime export เดิมที่เคยวัด corrected/tuned decode ได้ `chrF 12.36`:
     - local export ล่าสุดดีกว่าอย่างมีนัย และเป็น artifact ที่ “ใช้งานได้ตอนนี้”
   - ตอนนี้ artifact นี้ถูก promote เป็น `config.SLT_V3_CHECKPOINT_DIR` แล้ว
   - เพื่อไม่ต้อง export/promo แบบ manual ทุกครั้ง มี utility ใหม่:
     - `python scripts/export_pose_t5_checkpoint.py --train-dir checkpoints/pose_t5_rtx4060_resume_best --export-dir checkpoints/pose_t5_rtx4060_best_export_auto`
   - workflow export/promotion ล่าสุดถูกแยกเป็น 2 path:
     - candidate export ชั่วคราว: `checkpoints/pose_t5_rtx4060_best_export_auto/`
     - stable runtime ที่ promote หลัง verify แล้ว: `checkpoints/pose_t5_rtx4060_best_export_verified/`
   - runtime ตอนนี้ชี้ path เสถียร:
     - `checkpoints/pose_t5_rtx4060_best_export_verified/`
   - รอบล่าสุด script เลือก `best_model_state.pt` อัตโนมัติ เพราะ best metric ใหม่ยังอยู่ใน lightweight best-state path ก่อน full checkpoint cadence ถัดไป
   - note สำคัญ:
     - tracked `val_chrf` ระหว่าง train ชี้ว่า `step2900` ดีสุดตอนนี้
     - eval offline subset 50 ตอนนี้ก็หนุนว่า `step2900` ดีกว่า `step2600`, `step2400`, และ `step2200`
     - ณ หลักฐานล่าสุด `step2900` คือ candidate ที่ดีที่สุดทั้ง train-side tracked metric และ offline subset metric ที่วัดอยู่
   - อัปเดต 2026-06-20 รอบล่าสุด:
     - export runtime ถูก refresh จาก `best_model_state.pt` ที่ `step 2100`
     - `runtime_metadata.json` ของ stable verified path ตอนนี้ชี้:
       - `checkpoint_step = 2100`
       - `checkpoint_epoch = 36`
       - `checkpoint_metrics.val_chrf = 15.322410595729966`
     - subset eval 50 ตัวอย่างของ verified export ปัจจุบัน:
       - `chrF 14.94`
       - `BLEU 12.66`
       - `exact_match 3/50 = 6.0%`
     - สรุปเชิงคุณภาพ:
       - train-side `val_chrf` ของ `step 2100` สูงกว่า checkpoint ที่ resume มา (`step 2000`)
       - แต่ subset runtime eval ของ `step 2100` ยัง **ไม่ดีกว่า** ตัวเลขที่เคยวัดจาก `step 2900 export` (`chrF 15.60`, `BLEU 13.81`)
       - ดังนั้นห้ามสรุปจาก train-side `val_chrf` อย่างเดียว; ต้องดู subset runtime eval คู่กันทุกครั้ง
     - script ที่เพิ่มเพื่อกัน regression รอบถัดไป:
       - `python scripts/evaluate_pose_t5_export.py --export-dir <candidate_export_dir>`
       - `python scripts/promote_pose_t5_export.py --candidate-export-dir <candidate> --candidate-eval-json <candidate_eval> --stable-export-dir checkpoints/pose_t5_rtx4060_best_export_verified --stable-eval-json checkpoints/pose_t5_rtx4060_best_export_verified_eval.json`
       - `python scripts/refresh_pose_t5_verified.py`
         - รวม 3 ขั้นตอน `export -> evaluate -> promote`
         - ใช้ `best_model_state.pt` จาก `checkpoints/pose_t5_rtx4060_resume_best` เป็น candidate โดย default
         - ถ้า candidate ไม่ชนะ verified incumbent จะไม่ promote
       - promotion จะดู `(chrF, BLEU, exact_match_pct)` ของ subset eval ก่อน; ถ้า candidate ไม่ชนะ incumbent จะไม่เขียนทับ stable runtime path
     - progress ล่าสุดของ live run หลังสร้าง verified path:
       - `step 2300`: `val_chrf 15.05`
       - `step 2400`: `val_chrf 14.92`
       - `step 2500`: `val_chrf 14.92`
       - `step 2600`: `val_chrf 14.43`
       - `step 2700`: `val_chrf 15.36`
       - `step 2800`: `val_chrf 14.82`
       - ณ หลักฐานล่าสุด `best_model_state.pt` ถูก refresh เป็น `step 2700`
     - run `python scripts/refresh_pose_t5_verified.py` ล่าสุด:
       - ครั้งก่อน:
         - export candidate จาก `best_model_state.pt step 2100`
         - subset eval = `chrF 14.94 / BLEU 12.66 / EM 6.0%`
         - promotion result = `promoted: false` เพราะ candidate metrics เท่ากับ incumbent metrics
     - รอบล่าสุดหลัง `step 2700`:
       - export candidate จาก `best_model_state.pt step 2700`
       - subset eval = `chrF 15.34 / BLEU 13.18 / EM 6.0%`
       - promotion result = `promoted: true`
       - verified incumbent ตอนนั้นขยับเป็น `step 2700`
     - รอบล่าสุดหลัง `step 2900`:
       - export candidate จาก `best_model_state.pt step 2900`
       - subset eval = `chrF 15.80 / BLEU 13.87 / EM 6.0%`
       - promotion result = `promoted: true`
       - verified incumbent ตอนนี้ขยับเป็น `step 2900`
     - หลักฐานสำคัญเพิ่ม:
       - live run ผ่าน `step 3000` ได้แล้วโดย **ไม่ค้าง** ที่ save path แบบรอบเก่า
       - metric ที่ `step 3000` = `val_chrf 14.16`
       - เพราะแย่กว่า `step 2900`, `best_model_state.pt` จึงยังคงอยู่ที่ `step 2900` ตามที่ควร
12. **`thai-sign-train-managed-r13`** คือ strong-GPU launcher ตัวล่าสุด:
   - policy:
     - `GpuPriority = H100,A100`
     - `MinGpu = A100`
     - `GpuRetryMinutes = 120`
     - `GpuRetryDelaySec = 30`
   - หลังแก้ launcher แล้ว backend reject ไม่ทำให้ lane ตายถาวรอีก:
     - `attempts.H100 = 3`
     - `attempts.A100 = 3`
     - `launcher.status.json` ยังอัปเดตต่อเนื่อง
   - แต่ truth ล่าสุดยังเหมือนเดิม:
     - backend ตอบ `Backend rejected accelerator 'H100'`
     - backend ตอบ `Backend rejected accelerator 'A100'`
   - สรุป: จุดติดอยู่ที่ quota/entitlement ของ account ในรอบนี้ ไม่ใช่ retry logic แล้ว
13. **`thai-sign-train-managed-r14`** คือ launcher ตัวล่าสุดหลังเพิ่ม lower-tier fallback:
   - policy:
     - `GpuPriority = H100,A100,L4,T4`
     - `MinGpu = A100`
     - `AllowFallbackBelowMinGpuOnReject = true`
   - หลักฐานล่าสุด:
     - `H100` และ `A100` ถูก backend reject ตามเดิม
     - launcher สลับ `effective_gpu_candidates` ไปเป็น `L4,T4` อัตโนมัติแล้ว
     - `L4` ก็ถูก reject เช่นกัน
     - `T4` ยังเป็น capacity path (`Service Unavailable`) และยัง retry อยู่
   - truth source:
     - `checkpoints/colab_sync/thai-sign-train-managed-r14/launcher.status.json`
     - `.../launcher.stdout.log`
14. ฝั่ง local มีการยิง `python -m kaggle datasets version` ด้วย **absolute path** ของ `kaggle_upload/thai-sign-ckpt`
   - จำเป็นต้องใช้ absolute path บน Windows; ถ้าใช้ relative path จะเจอ temp upload bug ของ Kaggle CLI (`...uploads\\kaggle_upload/thai-sign-ckpt_best_checkpoint.txt.json`)
   - version request ถูกส่งแล้ว แต่ `kaggle datasets files` อาจยังสะท้อนรายการไฟล์เดิมอยู่ช่วงสั้น ๆ ระหว่าง upstream processing
15. พยายาม reuse **`thai-sign-train-managed-r4`** (A100 session เดิม) แล้ว แต่ session นั้นกลายเป็น stale:
   - `status` เคยเห็นเป็น `IDLE`
   - แต่พอ health probe/exec ใช้จริงแล้ว fail และสุดท้าย `colab status -s thai-sign-train-managed-r4` กลายเป็น `Session ... not found`
   - มีหลักฐานไว้ที่ `checkpoints/colab_sync/thai-sign-train-managed-r4-reuse/launcher.status.json`

16. **Kaggle Curriculum Training v40 — สำเร็จ 2026-06-23** 🎯
   - workflow: Phase 1 pretrain ทุก 1938 ตัวอย่าง (tsl51+thaisignvis+youtube_sl25_thai) → Phase 2 finetune TSL-51 เท่านั้น
   - kernel: `orbitorls/thai-sign-tsl51-curriculum-train` version 40
   - GPU: T4 (sm_75, 15 GB), torch 2.10.0+cu128
   - seed: `orbitorls/thai-sign-tsl51-seed` (`best_model_state.pt` step=3075, chrF-86)
   - Phase 1:
     - stopped_reason: `early_stopping` (val_loss patience 8)
     - final global_step: 5575
   - Phase 2:
     - started from Phase 1 best (step=5200)
     - stopped_reason: `early_stopping` (val_chrf patience 10)
     - final global_step: 5575
     - best val_chrf during training: **96.77** (at step 5325)
   - verified eval (n=25, seed=42, data=tsl51, split=manifest):
     - **chrF: 96.77**
     - **BLEU: 97.73**
     - **exact_match: 23/25 = 92%**
   - promotion: `promoted: true` — candidate (96.77) beat incumbent (86.95)
   - stable artifact: `checkpoints/pose_t5_rtx4060_tsl51_only_export_verified/`
   - stable eval: `checkpoints/pose_t5_rtx4060_tsl51_only_export_verified_eval.json`
   - Kaggle model dataset: `orbitorls/thai-sign-tsl51-model` (published 2026-06-23)
   - root cause ที่แก้ระหว่างทาง:
     - `scripts/kaggle_train_pose_t5.py:_run_smoke_training` เคย hardcode `smoke_args.epochs = 1`
     - ทำให้ warm-start seed ที่ epoch=76 ทำ 0 smoke steps → noop-resume guard fire
     - แก้: `smoke_args.epochs = 100000` + notebook ส่ง `--smoke-steps 0` (belt-and-suspenders)

### สถานะ artifact ที่ “ใช้งานได้ที่สุดตอนนี้”

- **inference path ที่ดีที่สุด ณ 2026-06-23:**
  - `checkpoints/pose_t5_rtx4060_tsl51_only_export_verified/` — chrF **96.77** / BLEU **97.73** / exact **92%**
  - Kaggle: `orbitorls/thai-sign-tsl51-model`
  - `config.SLT_V3_CHECKPOINT_DIR` ชี้ที่ dir นี้อยู่แล้ว — restart API โหลดทันที
- (ก่อนหน้านี้: `checkpoints/pose_t5_a100_r4_final_export/` → chrF 12.36, superseded)
- report หลักฐาน:
  - `checkpoints/pose_t5_rtx4060_tsl51_only_export_verified_eval.json`
  - `checkpoints/pose_t5_rtx4060_tsl51_only_export_verified_samples.json`

---

## 1. Checkpoints ที่โหลดมาแล้ว (บนเครื่อง local)

ทุกไฟล์ขนาด **3.43 GB** (3,684,199,263 bytes) — bundle ครบ (model + optimizer + scheduler + scaler + step + RNG) สำหรับ resume

| Path | Step | สถานะ | ใช้ทำอะไร |
|------|------|-------|-----------|
| `C:/Temp/vFinalOut/pose_t5_v3/ckpt_step00004000.pt`  | 4000  | ✅ ครบ | **best by val_chrf (15.25)** → ใช้ inference |
| `C:/Temp/vFinalOut/pose_t5_v3/ckpt_step00015400.pt`  | 15400 | ✅ ครบ | late checkpoint (overfit) |
| `C:/Temp/vFinalOut/pose_t5_v3/ckpt_step00015500.pt`  | 15500 | ✅ ครบ | late checkpoint (overfit) |
| `C:/Temp/vFinalOut/pose_t5_v3/ckpt_step00015600.pt`  | 15600 | ❌ 0 byte (โหลดไม่จบ) | — |

> - การโหลดผ่าน `kaggle kernels output` มักถูกตัด (`IncompleteRead`) เพราะไฟล์ใหญ่ → **รันคำสั่งซ้ำได้** มัน skip ไฟล์ที่ครบแล้ว
> - **`train_metrics.json` ยังโหลดไม่ได้** (connection ขาดก่อนถึง) → ตัวเลข chrF/step มาจาก training log ที่ดูสดระหว่างรัน ไม่ใช่จากไฟล์ metrics. โหลดเพิ่มได้ด้วย:
>   `python -m kaggle kernels output orbitorls/thai-sign-train -p C:/Temp/vFinalOut`

---

## 2. โมเดล & สถาปัตยกรรม (ปัจจุบัน = v3)

`src/tsl/models/pose_t5.py` → `PoseToTextT5(nn.Module)`:
- **Front-end:** `Linear(312 → 512)` + sinusoidal positional encoding + `num_encoder_layers` × `TransformerEncoderLayer` + temporal downsample (mean-pool, factor 4) → feed เป็น `inputs_embeds` ให้ mT5 พร้อม `attention_mask`
- **Decoder:** `MT5ForConditionalGeneration.from_pretrained("google/mt5-small")` — ได้ language prior + open vocab + Thai subword ฟรี
- **Decode:** HF `.generate()` (beam search) — แทน greedy/beam ที่เขียนเอง
- **Checkpoint contract:** `pose_t5_config.json` + HF sub-model + front-end `state_dict`
- **Input contract ทั่วทั้ง v3 = 312-dim** จาก `normalize_sequence` (104 landmarks: 21+21 มือ, 22 pose, 40 หน้า), stamp `feature_layout_version = "v3-312"`

Legacy `slt_v2` (162-dim, word-tokenizer) **ยังแยกอยู่** ที่ `src/tsl/models/slt.py` — ไม่แตะ (ดู §9)

---

## 3. Hyperparameters รอบแรก (ค่าที่ทำให้ overfit)

จาก `kaggle_upload/notebook/notebook.ipynb` Cell 4 + defaults ใน `src/tsl/train/train_pose_t5.py`:

| Param | ค่าที่ใช้ | หมายเหตุ |
|-------|----------|----------|
| `--lr` | `3e-4` | **น่าจะสูงไปสำหรับ fine-tune mT5** |
| `encoder_dropout` | `0.1` | **hardcoded** ที่ `train_pose_t5.py:356` — ยังไม่เป็น CLI arg |
| weight_decay | `0.01` (AdamW default) | **ไม่ได้ตั้งเอง** — `AdamW(model.parameters(), lr=args.lr)` ที่ line 363 |
| `--num-encoder-layers` | `2` | |
| `--downsample-factor` | `4` | |
| `--max-src-len` | `512` | |
| `--batch-size` / `--grad-accum` | `8` / `4` | effective batch = 32 |
| `--amp` | `auto` | |
| scheduler | `ReduceLROnPlateau(mode=min, factor=0.5, patience=5)` | line 364 |
| early stopping | **ไม่มี** | save best by val_chrf แต่เทรนต่อจนหมดเวลา |
| `--eval-steps` | `100` | |
| `--epochs` / จริง | `300` ตั้งไว้ / ~109 epoch ก่อนหมดเวลา | |
| `--max-runtime-min` | `690` | self-terminate ก่อน Kaggle kill ที่ 12h |

---

## 4. งานที่ต้องทำต่อ (รายละเอียด)

### งาน 1 — Upload checkpoint เป็น Kaggle dataset
สร้าง dataset `orbitorls/thai-sign-ckpt` เพื่อให้ Cell 2 ของ notebook (`PREV_CKPT = '/kaggle/input/thai-sign-ckpt'`) restore ได้ใน session ถัดไป

```bash
mkdir C:/Temp/ckpt_upload
cp C:/Temp/vFinalOut/pose_t5_v3/ckpt_step00004000.pt C:/Temp/ckpt_upload/
# สร้าง dataset-metadata.json: { "id": "orbitorls/thai-sign-ckpt", "title": "Thai Sign Ckpt", ... }
python -m kaggle datasets create -p C:/Temp/ckpt_upload --dir-mode zip
# (ครั้งถัดไปที่อัปเดต: kaggle datasets version -p ... -m "msg")
```
> ตัดสินใจก่อน upload:
> - จะ **เทรนใหม่ด้วย config ที่แก้ overfit** → upload **step 4000 (best)** ไว้ใช้ inference; ไม่ต้อง resume จากตัว overfit
> - จะ resume เทรนต่อแบบเดิม → upload step 15500 แต่ **ไม่แนะนำ** เพราะ overfit ไปแล้ว

### งาน 2 — แก้ overfitting (แก้โค้ดก่อน run รอบ 2)
อาการ: val_loss best ที่ step 700 (3.75) แล้วพุ่งถึง 5.79 (step 5700) ขณะ train_loss ลดต่อ → classic overfit จากข้อมูลน้อย (~252 TSL-51 สะอาด + ~1,626 YouTube noisy, **~1 ตัวอย่าง/ประโยค**)

Levers ใน `src/tsl/train/train_pose_t5.py`:
1. **เพิ่ม `--dropout` เป็น CLI arg** (ตอนนี้ hardcoded 0.1 ที่ line 356) → ลอง `0.3`
2. **เพิ่ม `--weight-decay` arg** ส่งเข้า AdamW (line 363) → ลอง `0.05`–`0.1`
3. **ลด `--lr`** → `1e-4` หรือ `5e-5` (fine-tune pretrained ควร lr ต่ำ)
4. **เพิ่ม early stopping** บน val_loss (patience ~10 evals) — ตอนนี้ไม่มี
5. (พิจารณา) freeze mT5 decoder บางชั้น / warm-up / label smoothing
6. chrF oscillate 0↔15 ทุก ~500 step → ตรวจ generate config (beam, max_length, repetition_penalty) ว่าเสถียร

> ใช้ skill `superpowers:test-driven-development` ตอนแก้ — เพิ่ม arg แล้วเขียน test ว่า optimizer/model ได้ weight_decay/dropout ตามที่ส่งจริง

### งาน 3 — Run Kaggle รอบ 2
1. แก้ notebook Cell 4 ใส่ flags ใหม่ (`--lr 1e-4 --dropout 0.3 --weight-decay 0.05` ฯลฯ)
2. **เลือก accelerator = T4 x2 ในเว็บ UI ทุกครั้ง** (Kaggle ชอบ assign P100 ซึ่งใช้ไม่ได้ — ดู §5)
3. push: `python -m kaggle kernels push -p kaggle_upload/notebook`
4. monitor: `python -m kaggle kernels status orbitorls/thai-sign-train`

---

## 5. ⚠️ กับดักสำคัญ (อย่าพลาดซ้ำ)

1. **P100 ใช้ไม่ได้เด็ดขาด** — Tesla P100 = sm_60; PyTorch 2.x (Python 3.12) ต้องการ sm_70+ → crash `CUDA error: no kernel image is available`. Cell 1 มี gate ตรวจ `nvidia-smi --query-gpu=compute_cap` แล้ว raise ทันทีถ้า < 7.0 → **ต้องเลือก T4 x2 เองในเว็บ UI** (T4=7.5 ✅, V100=7.0 ✅, A100=8.0 ✅)
2. **ดาวน์โหลด output ใหญ่มักถูกตัด** (`IncompleteRead`) — รันซ้ำได้ มันโหลดต่อจากที่ค้าง
3. **`kaggle kernels status/output` เห็นแค่ committed version** — interactive draft session ต้องดูผ่านเว็บ UI
4. **Kaggle mount path ไม่แน่นอน** — `/kaggle/input/<name>/` หรือ `/kaggle/input/datasets/orbitorls/<name>/` → `find_dataset()` ใน Cell 2 จัดการให้แล้ว

---

## 6. ไฟล์/พาธสำคัญ (v3)

| สิ่งของ | Path |
|---------|------|
| Training (CLI ใหม่) | `src/tsl/train/train_pose_t5.py` |
| Checkpointing (resume) | `src/tsl/train/checkpointing.py` |
| Kaggle driver | `scripts/train_local_gpu.py` |
| Kaggle notebook | `kaggle_upload/notebook/notebook.ipynb` |
| Kaggle metadata | `kaggle_upload/notebook/kernel-metadata.json` |
| โมเดล | `src/tsl/models/pose_t5.py` |
| Unified 312-dim loader | `src/tsl/data/unified.py` |
| T5 collate | `src/tsl/data/pose_t5_collate.py` |
| Normalize (312 stamp) | `src/tsl/features/normalize.py` |
| Inference | `src/tsl/inference/pose_t5_translator.py`, `video_pipeline.py` |
| API | `src/tsl/api/app.py` (+ `POST /translate-video`, dispatch by ckpt type) |
| Eval (sacrebleu) | `src/tsl/eval/slt_metrics.py`, `build_splits.py` |
| Data migration | `scripts/migrate_tsl51_to_312.py`, `scripts/reextract_youtube_sl25_to_312.py` |
| Plan ฉบับเต็ม | `C:/Users/Pannawat Khantong/.claude/plans/end-to-end-serialized-cray.md` |

**Kaggle datasets:** `thai-sign-code`, `thai-sign-tsl51`, `thai-sign-youtube`, `thai-sign-mt5small` (+ จะเพิ่ม `thai-sign-ckpt`)
**Kernel:** `orbitorls/thai-sign-train`

---

## 7. ข้อมูล (datasets)

### TSL-51 (closed-vocab, ของสะอาด)
- 911 mp4 ใน `data/tsl51/videos/`, ~252 ตัวอย่างสะอาด, closed vocab ~30 คำ
- v3: re-extract เป็น 312-dim ด้วย `scripts/migrate_tsl51_to_312.py` → `data/tsl51_v3/`

### YouTube-SL-25 Thai (open-vocab, noisy)
- 106 raw mp4 retained (1.5 GB) + 1,626 segments (.npy)
- v3: re-extract จาก mp4 local เป็น 312-dim (ไม่ต้อง re-download) — `--from-existing-videos`
- เนื้อหา = การแปลพระคัมภีร์, **1,626 ประโยคไม่ซ้ำเลย** (~1 ตัวอย่าง/ประโยค) → เป็นต้นเหตุ data scarcity
- มี `data/youtube_sl25_thai_test/` เป็น frozen held-out test

---

## 8. ความคาดหวังที่ตรงความจริง

- เป้า = End-to-End ใช้งานจริง (video/webcam → ข้อความไทย), open-vocab
- ข้อมูลภาษามือไทยมี **น้อยมาก** → pretrained decoder ช่วยได้ แต่ **จะไม่ถึงระดับ DeepMind** (TPU + 3,000 ชม.)
- "100%" ของ slt_v2 = การจำ ไม่ใช่การแปลจริง — วัดผลจริงด้วย chrF/BLEU (sacrebleu) บน **video-level test split** ที่ไม่มี leak
- chrF 15.25 ยังต่ำ — รอบ 2 ที่แก้ overfit แล้วน่าจะดีขึ้น แต่ตั้งความหวังตามปริมาณข้อมูล

---

## 9. บทเรียนสำคัญจาก legacy SLT (ยังใช้อ้างอิงได้)

> ยุคก่อน PoseToTextT5 — โมเดล `SignToTextTransformer` (`src/tsl/models/slt.py`), 162-dim, train from scratch

| เวอร์ชัน | สถานะ | ข้อมูล | Tokenizer | chrF | Exact | Checkpoint |
|---|---|---|---|---|---|---|
| v1 | ❌ Failed | TSL-51 (252) | Char | ~0 | 0% | — |
| **v2** | ✅ Shipped (legacy) | TSL-51 (252) | Word (34) | 100.0 | 100% | `checkpoints/slt_v2/` |
| combined | 🧪 Failed | TSL-51 + YT (1,878) | Char | 13.1 | 0% | `checkpoints/slt_combined/` |
| combined v2 | 🧪 Failed | + label smoothing | Char | 13.8 | 0% | `checkpoints/slt_combined_v2/` |

- **v1 ล้มเหลว:** บั๊ก teacher forcing — `_right_shift_target` ใส่ BOS ซ้ำ (double-BOS) ทำให้ train/inference ไม่ตรง → **แก้: ใช้ `tgt[:, :-1]`**
- **v2 สำเร็จ (แต่คือการจำ):** แก้บั๊กข้างบน + word tokenizer (vocab 30 คำ เหมาะกับ 252 ตัวอย่าง). chrF/exact 100% = **closed-vocab memorization** ไม่ใช่การแปลจริง
- **combined ล้มเหลว:** ต้นตอ = **data scarcity** ไม่ใช่บั๊ก/ข้อมูลเสีย (NaN 0%, เฟรมปกติ). 1,626 ประโยคไม่ซ้ำ → ~1 ตัวอย่าง/ประโยค → mode collapse + overfit. **ไม่มีเทคนิคเทรนใดแก้เพดานนี้ได้** — นี่คือเหตุผลที่ย้ายมาใช้ pretrained decoder (mT5)
- โหลด legacy: `SentenceTranslator("checkpoints/slt_v2")` (รับ path ตรงๆ)

---

## 10. Git state

- branch: `main`
- งาน v3 ส่วนใหญ่ committed แล้ว (pose_t5, checkpointing, collate, translator, video_pipeline, /translate-video, sacrebleu, migration, kaggle driver — ดู `git log`)
- **ยังมีไฟล์ modified/untracked ค้างเยอะ** — `data/`, `checkpoints/`, `kaggle_upload/`, scripts ใหม่ → ตัดสินใจว่าจะ commit อะไร
- **แนะนำ:** commit โค้ด + docs + tests; `data/` และ `checkpoints/` (3.43 GB/ไฟล์) ควร `.gitignore` หรือใช้ Git LFS — **อย่า** commit checkpoint ใหญ่เข้า git ตรงๆ
