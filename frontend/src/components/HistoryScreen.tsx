import React from "react";
import { useHistory } from "../hooks/HistoryProvider";
import { useSpeech } from "../hooks/useSpeech";
import { useSettings } from "../hooks/SettingsProvider";
import { useI18n } from "../i18n";

const stroke = {
  fill: "none" as const,
  stroke: "currentColor",
  strokeWidth: 2,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
};

function formatTime(ts: number, lang: "th" | "en"): string {
  const locale = lang === "th" ? "th-TH" : "en-US";
  return new Date(ts).toLocaleString(locale, {
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function HistoryScreen() {
  const { items, clear } = useHistory();
  const { speak } = useSpeech();
  const { lang } = useSettings();
  const th = useI18n();

  return (
    <div className="screen-sheet">
      <div className="sheet-head">
        <h2 className="sheet-title">{th.historyTitle}</h2>
        {items.length > 0 && (
          <button
            type="button"
            className="glass-chip"
            onClick={() => {
              if (window.confirm(th.confirmClear)) clear();
            }}
          >
            ✕ {th.settingsClearHistory}
          </button>
        )}
      </div>

      {items.length === 0 ? (
        <div className="empty-state">
          <svg viewBox="0 0 24 24" {...stroke}>
            <circle cx="12" cy="12" r="9" />
            <path d="M12 7v5l3 2" />
          </svg>
          <p>{th.historyEmpty}</p>
        </div>
      ) : (
        <ul className="history-list">
          {items.map((it) => (
            <li key={it.id}>
              <button type="button" className="history-row" onClick={() => speak(it.sentence)}>
                <div className="history-word">
                  <b>{it.sentence}</b>
                  <small>{formatTime(it.ts, lang)}</small>
                </div>
                <svg viewBox="0 0 24 24" {...stroke}>
                  <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
                  <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
                  <path d="M19.07 4.93a10 10 0 0 1 0 14.14" />
                </svg>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
