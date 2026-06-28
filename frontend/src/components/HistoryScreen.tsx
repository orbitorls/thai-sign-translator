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

function formatRelative(ts: number, t: ReturnType<typeof useI18n>, lang: "th" | "en"): string {
  const diff = Date.now() - ts;
  const min = Math.floor(diff / 60000);
  if (min < 1) return t.timeJustNow;
  if (min < 60) return t.timeMinutesAgo(min);
  const hr = Math.floor(min / 60);
  if (hr < 24) return t.timeHoursAgo(hr);
  const day = Math.floor(hr / 24);
  if (day < 7) return t.timeDaysAgo(day);
  return new Date(ts).toLocaleDateString(lang === "th" ? "th-TH" : "en-US", { day: "numeric", month: "short" });
}

export function HistoryScreen() {
  const { items, clear, remove } = useHistory();
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
          <svg viewBox="0 0 24 24" aria-hidden="true" {...stroke}>
            <circle cx="12" cy="12" r="9" />
            <path d="M12 7v5l3 2" />
          </svg>
          <p>{th.historyEmpty}</p>
        </div>
      ) : (
        <ul className="history-list">
          {items.map((it) => (
            <li key={it.id} className="history-row">
              <button type="button" className="history-word-btn" onClick={() => speak(it.sentence)}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{ width: 18, height: 18, flexShrink: 0, marginRight: 8, opacity: 0.6 }}>
                  <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/>
                </svg>
                <div className="history-word">
                  <b>{it.sentence}</b>
                  <small>{formatRelative(it.ts, th, lang)}</small>
                </div>
              </button>
              <div className="history-actions">
                <button
                  type="button"
                  aria-label={th.actionCopy}
                  title={th.actionCopy}
                  onClick={() => { if (navigator.clipboard) navigator.clipboard.writeText(it.sentence); }}
                >
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <rect x="9" y="9" width="13" height="13" rx="2"/>
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                  </svg>
                </button>
                {typeof navigator !== "undefined" && "share" in navigator && (
                  <button
                    type="button"
                    aria-label={th.actionShare}
                    title={th.actionShare}
                    onClick={() => { navigator.share({ text: it.sentence }).catch(() => {}); }}
                  >
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                      <circle cx="18" cy="5" r="3"/>
                      <circle cx="6" cy="12" r="3"/>
                      <circle cx="18" cy="19" r="3"/>
                      <line x1="8.6" y1="13.5" x2="15.4" y2="17.5"/>
                      <line x1="15.4" y1="6.5" x2="8.6" y2="10.5"/>
                    </svg>
                  </button>
                )}
                <button
                  type="button"
                  aria-label={th.actionDelete}
                  title={th.actionDelete}
                  onClick={() => remove(it.id)}
                >
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <polyline points="3 6 5 6 21 6"/>
                    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                  </svg>
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
