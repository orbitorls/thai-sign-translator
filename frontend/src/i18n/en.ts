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
  actionCopy: "Copy",
  actionDelete: "Delete",
  actionShare: "Share",
  copied: "Copied",

  // Settings
  settingsTitle: "Settings",
  settingsLanguage: "Language",
  settingsLandmarks: "Landmarks",
  settingsClearHistory: "Clear history",
  confirmClear: "Clear all history?",

  // Network / offline
  offline: "Offline",
  offlineHint: "No internet connection",
  retry: "Retry",

  // Misc actions
  close: "Close",

  // Relative time
  timeJustNow: "Just now",
  timeMinutesAgo: (n: number) => `${n} min ago`,
  timeHoursAgo: (n: number) => `${n} hr ago`,
  timeDaysAgo: (n: number) => `${n} days ago`,
};
