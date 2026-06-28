import React from "react";
import { useSettings } from "../hooks/SettingsProvider";
import { useHistory } from "../hooks/HistoryProvider";
import { useI18n } from "../i18n";

const stroke = {
  fill: "none" as const,
  stroke: "currentColor",
  strokeWidth: 2,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
};

export function SettingsScreen() {
  const { lang, setLang, showLandmarks, setShowLandmarks } = useSettings();
  const { clear } = useHistory();
  const th = useI18n();

  return (
    <div className="screen-sheet">
      <h2 className="sheet-title" style={{ marginBottom: "var(--space-5)" }}>
        {th.settingsTitle}
      </h2>
      <div className="settings-list">
        {/* Language */}
        <div className="settings-row">
          <svg viewBox="0 0 24 24" aria-hidden="true" {...stroke}>
            <circle cx="12" cy="12" r="10" />
            <line x1="2" y1="12" x2="22" y2="12" />
            <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
          </svg>
          <span className="settings-label">{th.settingsLanguage}</span>
          <div className="seg">
            <button type="button" className={lang === "th" ? "on" : ""} onClick={() => setLang("th")} aria-pressed={lang === "th"}>
              ไทย
            </button>
            <button type="button" className={lang === "en" ? "on" : ""} onClick={() => setLang("en")} aria-pressed={lang === "en"}>
              EN
            </button>
          </div>
        </div>

        {/* Landmarks toggle */}
        <div className="settings-row">
          <svg viewBox="0 0 24 24" aria-hidden="true" {...stroke}>
            <polygon points="12 2 15 9 22 9 16 14 18 21 12 17 6 21 8 14 2 9 9 9" />
          </svg>
          <span className="settings-label">{th.settingsLandmarks}</span>
          <button
            type="button"
            role="switch"
            aria-checked={showLandmarks}
            aria-label={th.settingsLandmarks}
            className={`toggle${showLandmarks ? " on" : ""}`}
            onClick={() => setShowLandmarks(!showLandmarks)}
          />
        </div>

        {/* Clear history */}
        <button
          type="button"
          className="settings-row settings-action"
          onClick={() => {
            if (window.confirm(th.confirmClear)) clear();
          }}
        >
          <svg viewBox="0 0 24 24" aria-hidden="true" {...stroke}>
            <polyline points="3 6 5 6 21 6" />
            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
          </svg>
          <span className="settings-label">{th.settingsClearHistory}</span>
        </button>
      </div>
    </div>
  );
}
