/** Thai UI string constants. */
export const th = {
  appTitle: "วาทยากร",
  appSubtitle: "บันทึกภาษามือแล้วรับคำแปลภาษาไทย",
  assistantLabel: "ผลลัพธ์",
  cameraPanelLabel: "กล้อง",
  chatPanelLabel: "บทสนทนา",

  // Camera states
  cameraInit: "กำลังเตรียมกล้อง...",
  cameraReady: "กล้องพร้อม",
  cameraError: "ไม่สามารถเข้าถึงกล้องได้",
  cameraErrorHint: "กรุณาอนุญาตการใช้กล้องในเบราว์เซอร์",
  cameraRetry: "ลองใหม่",

  // Recording
  recordStart: "เริ่มบันทึก",
  recordStop: "หยุดบันทึก",
  recording: "กำลังบันทึก...",
  frames: (n: number) => `${n} เฟรม`,

  // Translation
  translating: "กำลังแปล...",
  resultPlaceholder: "—",
  confidence: (pct: number) => `ความมั่นใจ ${pct}%`,
  noFrames: "ไม่พบการเคลื่อนไหว ลองเริ่มบันทึกใหม่",

  // Model picker
  modelLabel: "โมเดล",
  modelUnavailable: "ไม่พร้อมใช้งาน",
  modelLoading: "กำลังโหลดโมเดล...",
  modelLoadError: "โหลดรายชื่อโมเดลล้มเหลว",

  // Errors
  errorModelUnavailable: "โมเดลนี้ยังไม่พร้อมใช้งาน",
  errorGeneric: "เกิดข้อผิดพลาด กรุณาลองใหม่",

  // Supported phrases panel
  supportedPhrasesTitle: "วลีที่รองรับ",
  supportedPhrasesScope: "โมเดลปัจจุบันรู้จักวลีจากชุดข้อมูล TSL-51 เท่านั้น",
  supportedPhrasesUnavailable: "ไม่สามารถโหลดรายการวลีได้",
  supportedPhrasesEmpty: "ยังไม่มีข้อมูลวลีที่รองรับ",
  supportedPhrasesCount: (n: number) => `${n} วลี`,
  supportedPhrasesShow: "ดูวลีที่รองรับ",
  supportedPhrasesHide: "ซ่อน",

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
  actionCopy: "คัดลอก",
  actionDelete: "ลบ",
  actionShare: "แชร์",
  copied: "คัดลอกแล้ว",

  // Settings
  settingsTitle: "ตั้งค่า",
  settingsLanguage: "ภาษา",
  settingsLandmarks: "เส้นโครงร่าง",
  settingsDevMode: "โหมดนักพัฒนา",
  settingsClearHistory: "ล้างประวัติ",
  confirmClear: "ล้างประวัติทั้งหมด?",

  // Network / offline
  offline: "ออฟไลน์",
  offlineHint: "ไม่มีการเชื่อมต่ออินเทอร์เน็ต",
  retry: "ลองใหม่",

  // Misc actions
  close: "ปิด",

  // Relative time
  timeJustNow: "เมื่อสักครู่",
  timeMinutesAgo: (n: number) => `${n} นาทีที่แล้ว`,
  timeHoursAgo: (n: number) => `${n} ชั่วโมงที่แล้ว`,
  timeDaysAgo: (n: number) => `${n} วันที่แล้ว`,

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
};
