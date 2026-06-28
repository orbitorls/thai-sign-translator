import React, { useState } from "react";
import { useSettings } from "../hooks/SettingsProvider";
import { useHistory } from "../hooks/HistoryProvider";
import { useConsent } from "../hooks/useConsent";
import { useT } from "../i18n";
import { Toggle } from "./settings/Toggle";
import { ModelPicker } from "./ModelPicker";
import { PrivacyScreen } from "./PrivacyScreen";
import { ConfirmModal } from "./ui/ConfirmModal";
import type { ConsentScope } from "../privacy/consentStorage";

interface SettingsScreenProps {
  onOpenPrivacy?: () => void;
}

function SettingsDivider() {
  return <div className="settings-divider" role="separator" aria-hidden="true" />;
}

function SettingsControl({ children }: { children: React.ReactNode }) {
  return <div className="settings-control">{children}</div>;
}

export function SettingsScreen({ onOpenPrivacy }: SettingsScreenProps) {
  const { t } = useT();
  const { settings, update, reset, setLang, setShowLandmarks, lang, showLandmarks } = useSettings();
  const { clear } = useHistory();
  const { hasScope, setScope } = useConsent();
  const [view, setView] = useState<"main" | "privacy">("main");
  const [confirmOpen, setConfirmOpen] = useState(false);

  if (view === "privacy") {
    return <PrivacyScreen onBack={() => setView("main")} />;
  }

  const consentRows: { scope: ConsentScope; label: string }[] = [
    { scope: "model_improvement", label: t.consentModelTitle },
    { scope: "video_research", label: t.consentVideoTitle },
    { scope: "academic_publication", label: t.consentAcademicTitle },
  ];

  return (
    <div className="screen-sheet">
      <h2 className="sheet-title settings-page-title">{t.settingsTitle}</h2>

      <div className="settings-list">
        <div className="settings-row">
          <span className="settings-label">{t.modelLabel}</span>
          <SettingsControl>
            <ModelPicker className="glass-chip settings-select" />
          </SettingsControl>
        </div>

        <div className="settings-row">
          <span className="settings-label">{t.settingsLanguage}</span>
          <SettingsControl>
            <div className="seg seg--settings" role="group" aria-label={t.settingsLanguage}>
              <button type="button" className={lang === "th" ? "on" : ""} onClick={() => setLang("th")} aria-pressed={lang === "th"}>
                ไทย
              </button>
              <button type="button" className={lang === "en" ? "on" : ""} onClick={() => setLang("en")} aria-pressed={lang === "en"}>
                EN
              </button>
            </div>
          </SettingsControl>
        </div>

        <SettingsDivider />

        <div className="settings-row">
          <span className="settings-label">{t.settingsLandmarks}</span>
          <SettingsControl>
            <Toggle
              id="landmarks"
              checked={showLandmarks}
              onChange={setShowLandmarks}
              aria-label={t.settingsLandmarks}
            />
          </SettingsControl>
        </div>

        <div className="settings-row">
          <span className="settings-label">{t.cameraFacingLabel}</span>
          <SettingsControl>
            <div className="seg seg--settings" role="group" aria-label={t.cameraFacingLabel}>
              <button
                type="button"
                className={settings.cameraFacing === "user" ? "on" : ""}
                onClick={() => update("cameraFacing", "user")}
                aria-pressed={settings.cameraFacing === "user"}
              >
                {t.cameraFacingUser}
              </button>
              <button
                type="button"
                className={settings.cameraFacing === "environment" ? "on" : ""}
                onClick={() => update("cameraFacing", "environment")}
                aria-pressed={settings.cameraFacing === "environment"}
              >
                {t.cameraFacingEnvironment}
              </button>
            </div>
          </SettingsControl>
        </div>

        <SettingsDivider />

        <div className="settings-row">
          <span className="settings-label">{t.diagnosticsEnabledLabel}</span>
          <SettingsControl>
            <Toggle
              id="diagnostics"
              checked={settings.diagnosticsEnabled}
              onChange={(v) => update("diagnosticsEnabled", v)}
              aria-label={t.diagnosticsEnabledLabel}
            />
          </SettingsControl>
        </div>

        <div className="settings-row">
          <span className="settings-label">{t.speakAloud}</span>
          <SettingsControl>
            <Toggle
              id="speak-aloud"
              checked={settings.speakAloud}
              onChange={(v) => update("speakAloud", v)}
              aria-label={t.speakAloud}
            />
          </SettingsControl>
        </div>

        <div className="settings-row">
          <span className="settings-label">{t.transcriptFontSize}</span>
          <SettingsControl>
            <select
              className="glass-chip settings-select"
              value={settings.fontSize}
              onChange={(e) => update("fontSize", e.target.value as typeof settings.fontSize)}
              aria-label={t.transcriptFontSize}
            >
              <option value="small">{t.fontSizeSmall}</option>
              <option value="medium">{t.fontSizeMedium}</option>
              <option value="large">{t.fontSizeLarge}</option>
            </select>
          </SettingsControl>
        </div>

        <SettingsDivider />

        {consentRows.map((row) => (
          <div key={row.scope} className="settings-row">
            <span className="settings-label">{row.label}</span>
            <SettingsControl>
              <Toggle
                id={`consent-${row.scope}`}
                checked={hasScope(row.scope)}
                onChange={(v) => void setScope(row.scope, v)}
                aria-label={row.label}
              />
            </SettingsControl>
          </div>
        ))}

        <SettingsDivider />

        <button
          type="button"
          className="settings-row settings-action"
          onClick={() => {
            onOpenPrivacy?.();
            setView("privacy");
          }}
        >
          <span className="settings-label">{t.privacyPolicy}</span>
          <span className="settings-chevron" aria-hidden="true">›</span>
        </button>

        <button
          type="button"
          className="settings-row settings-action settings-action--muted"
          onClick={() => setConfirmOpen(true)}
        >
          <span className="settings-label">{t.settingsClearHistory}</span>
        </button>

        <button type="button" className="settings-row settings-action settings-action--muted" onClick={reset}>
          <span className="settings-label">{t.resetToDefaults}</span>
        </button>
      </div>

      <ConfirmModal
        open={confirmOpen}
        title={t.settingsClearHistory}
        message={t.confirmClear}
        onConfirm={() => { clear(); setConfirmOpen(false); }}
        onCancel={() => setConfirmOpen(false)}
        danger
      />
    </div>
  );
}
