export type ConsentScope =
  | "service"
  | "model_improvement"
  | "video_research"
  | "academic_publication";

export const CONSENT_VERSION = "2026-06-28-v1";

export const ALL_CONSENT_SCOPES: ConsentScope[] = [
  "service",
  "model_improvement",
  "video_research",
  "academic_publication",
];

export interface ConsentState {
  version: string;
  scopes: Record<ConsentScope, boolean>;
  updatedAt: string;
  modalComplete: boolean;
}

const USER_ID_KEY = "signbridge:user-id";
const CONSENT_KEY = "signbridge:consent";
const SETTINGS_KEY = "signbridge:settings";

const DEFAULT_SCOPES: Record<ConsentScope, boolean> = {
  service: false,
  model_improvement: false,
  video_research: false,
  academic_publication: false,
};

export const DEFAULT_CONSENT_STATE: ConsentState = {
  version: CONSENT_VERSION,
  scopes: { ...DEFAULT_SCOPES },
  updatedAt: "",
  modalComplete: false,
};

function newUserId(): string {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `uid_${Date.now()}_${Math.random().toString(36).slice(2)}`;
}

export function getOrCreateUserId(): string {
  try {
    let id = localStorage.getItem(USER_ID_KEY);
    if (!id) {
      id = newUserId();
      localStorage.setItem(USER_ID_KEY, id);
    }
    return id;
  } catch {
    return "anonymous";
  }
}

function migrateLegacyFeedbackOptIn(state: ConsentState): ConsentState {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (!raw) return state;
    const parsed = JSON.parse(raw) as { feedbackOptIn?: boolean; _consentMigrated?: boolean };
    if (parsed._consentMigrated || !parsed.feedbackOptIn) return state;
    parsed._consentMigrated = true;
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(parsed));
    return {
      ...state,
      scopes: { ...state.scopes, model_improvement: true },
    };
  } catch {
    return state;
  }
}

export function loadConsentState(): ConsentState {
  try {
    const raw = localStorage.getItem(CONSENT_KEY);
    if (!raw) {
      return migrateLegacyFeedbackOptIn(DEFAULT_CONSENT_STATE);
    }
    const parsed = JSON.parse(raw) as Partial<ConsentState>;
    const merged: ConsentState = {
      ...DEFAULT_CONSENT_STATE,
      ...parsed,
      scopes: { ...DEFAULT_SCOPES, ...(parsed.scopes ?? {}) },
    };
    return migrateLegacyFeedbackOptIn(merged);
  } catch {
    return DEFAULT_CONSENT_STATE;
  }
}

export function saveConsentState(state: ConsentState): void {
  try {
    localStorage.setItem(CONSENT_KEY, JSON.stringify(state));
  } catch {
    // ignore
  }
}

export function hasConsentScope(scope: ConsentScope): boolean {
  return loadConsentState().scopes[scope] === true;
}

export function feedbackSessionId(): string {
  try {
    let id = localStorage.getItem("signbridge:feedback-session");
    if (!id) {
      id = `sess_${Date.now()}_${Math.random().toString(36).slice(2)}`;
      localStorage.setItem("signbridge:feedback-session", id);
    }
    return id;
  } catch {
    return "anonymous";
  }
}
