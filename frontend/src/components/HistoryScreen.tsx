import React, { useState } from "react";
import { useHistory } from "../hooks/HistoryProvider";
import { useSpeech } from "../hooks/useSpeech";
import { useSettings } from "../hooks/SettingsProvider";
import { useI18n } from "../i18n";
import { ConfirmModal } from "./ui/ConfirmModal";

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
  const { items, clear } = useHistory();
  const { speak } = useSpeech();
  const { lang } = useSettings();
  const th = useI18n();
  const [confirmOpen, setConfirmOpen] = useState(false);

  return (
    <div className="screen-sheet">
      <div className="sheet-head">
        <h2 className="sheet-title">{th.historyTitle}</h2>
        {items.length > 0 && (
          <button
            type="button"
            className="glass-chip"
            onClick={() => setConfirmOpen(true)}
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
            <li key={it.id}>
              <button type="button" className="history-row" onClick={() => speak(it.sentence)}>
                <div className="history-word">
                  <b>{it.sentence}</b>
                  <small>{formatRelative(it.ts, th, lang)}</small>
                </div>
                <svg viewBox="0 0 24 24" aria-hidden="true" {...stroke}>
                  <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
                  <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
                  <path d="M19.07 4.93a10 10 0 0 1 0 14.14" />
                </svg>
              </button>
            </li>
          ))}
        </ul>
      )}

      <ConfirmModal
        open={confirmOpen}
        title={th.settingsClearHistory}
        message={th.confirmClear}
        onConfirm={() => { clear(); setConfirmOpen(false); }}
        onCancel={() => setConfirmOpen(false)}
        danger
      />
    </div>
  );
}
