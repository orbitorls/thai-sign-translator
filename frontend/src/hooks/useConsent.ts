import { useCallback, useEffect, useState } from "react";
import {
  ALL_CONSENT_SCOPES,
  CONSENT_VERSION,
  ConsentScope,
  ConsentState,
  DEFAULT_CONSENT_STATE,
  loadConsentState,
  saveConsentState,
} from "../privacy/consentStorage";
import {
  deleteUserData,
  syncConsentScope,
  type ConsentSource,
} from "../api/client";
import { createMockConsentState, MOCKUP_MODE } from "../mockup";

export interface UseConsent {
  consent: ConsentState;
  setScope: (scope: ConsentScope, granted: boolean, source?: ConsentSource) => Promise<void>;
  completeModal: (scopes: Partial<Record<ConsentScope, boolean>>) => Promise<void>;
  withdrawAndDelete: () => Promise<number>;
  hasScope: (scope: ConsentScope) => boolean;
  needsConsentModal: boolean;
}

export function useConsent(): UseConsent {
  const [consent, setConsent] = useState<ConsentState>(() =>
    MOCKUP_MODE ? createMockConsentState() : loadConsentState(),
  );

  useEffect(() => {
    saveConsentState(consent);
  }, [consent]);

  const setScope = useCallback(
    async (scope: ConsentScope, granted: boolean, source: ConsentSource = "settings_toggle") => {
      setConsent((prev) => ({
        ...prev,
        version: CONSENT_VERSION,
        updatedAt: new Date().toISOString(),
        scopes: { ...prev.scopes, [scope]: granted },
      }));
      if (MOCKUP_MODE) return;
      try {
        await syncConsentScope({ scope, granted, source });
      } catch {
        // Local consent is authoritative for UI; sync when backend is available.
      }
    },
    []
  );

  const completeModal = useCallback(async (scopes: Partial<Record<ConsentScope, boolean>>) => {
    setConsent((prev) => ({
      ...prev,
      version: CONSENT_VERSION,
      updatedAt: new Date().toISOString(),
      modalComplete: true,
      scopes: {
        ...prev.scopes,
        ...scopes,
      },
    }));
    if (MOCKUP_MODE) return;
    const entries = Object.entries(scopes) as [ConsentScope, boolean][];
    await Promise.allSettled(
      entries
        .filter(([, granted]) => granted)
        .map(([scope]) =>
          syncConsentScope({ scope, granted: true, source: "consent_modal" })
        )
    );
  }, []);

  const withdrawAndDelete = useCallback(async () => {
    if (MOCKUP_MODE) {
      setConsent({
        ...createMockConsentState(),
        scopes: {
          service: false,
          model_improvement: false,
          video_research: false,
          academic_publication: false,
        },
      });
      return 0;
    }
    const deleted = await deleteUserData();
    const clearedScopes = ALL_CONSENT_SCOPES.reduce(
      (acc, scope) => {
        acc[scope] = false;
        return acc;
      },
      {} as Record<ConsentScope, boolean>
    );
    for (const scope of ["model_improvement", "video_research", "academic_publication"] as ConsentScope[]) {
      await syncConsentScope({ scope, granted: false, source: "withdrawal" });
    }
    setConsent({
      ...DEFAULT_CONSENT_STATE,
      modalComplete: true,
      scopes: clearedScopes,
      updatedAt: new Date().toISOString(),
    });
    return deleted;
  }, []);

  const hasScope = useCallback(
    (scope: ConsentScope) => consent.scopes[scope] === true,
    [consent.scopes]
  );

  const needsConsentModal = !consent.modalComplete || !consent.scopes.service;

  return {
    consent,
    setScope,
    completeModal,
    withdrawAndDelete,
    hasScope,
    needsConsentModal,
  };
}
