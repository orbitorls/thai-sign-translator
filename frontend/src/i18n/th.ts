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

  // Settings
  settingsTitle: "ตั้งค่า",
  settingsLanguage: "ภาษา",
  settingsLandmarks: "เส้นโครงร่าง",
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
};
