import React from "react";
import { useI18n } from "../i18n";

export type Screen = "camera" | "teach" | "history" | "settings";

const stroke = {
  fill: "none" as const,
  stroke: "currentColor",
  strokeWidth: 2,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
};

const CameraIcon = () => (
  <svg viewBox="0 0 24 24" {...stroke}>
    <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
    <circle cx="12" cy="13" r="4" />
  </svg>
);
const HistoryIcon = () => (
  <svg viewBox="0 0 24 24" {...stroke}>
    <circle cx="12" cy="12" r="9" />
    <path d="M12 7v5l3 2" />
  </svg>
);
const SettingsIcon = () => (
  <svg viewBox="0 0 24 24" {...stroke}>
    <circle cx="12" cy="12" r="3" />
    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
  </svg>
);
const TeachIcon = () => (
  <svg viewBox="0 0 24 24" {...stroke}>
    <path d="M12 3 2 8l10 5 10-5-10-5z" />
    <path d="M6 10.5V16c0 1 2.7 2.5 6 2.5s6-1.5 6-2.5v-5.5" />
  </svg>
);

interface BottomNavProps {
  active: Screen;
  onChange: (s: Screen) => void;
}

export function BottomNav({ active, onChange }: BottomNavProps) {
  const th = useI18n();
  const items: { key: Screen; label: string; icon: JSX.Element }[] = [
    { key: "camera", label: th.navCamera, icon: <CameraIcon /> },
    { key: "teach", label: th.navTeach, icon: <TeachIcon /> },
    { key: "history", label: th.navHistory, icon: <HistoryIcon /> },
    { key: "settings", label: th.navSettings, icon: <SettingsIcon /> },
  ];
  return (
    <nav className="bottom-nav-float">
      {items.map((it) => (
        <button
          key={it.key}
          type="button"
          className={`bottom-nav-item${active === it.key ? " on" : ""}`}
          onClick={() => onChange(it.key)}
          aria-current={active === it.key ? "page" : undefined}
          aria-label={it.label}
        >
          <span className="bottom-nav-ico" aria-hidden="true">{it.icon}</span>
          <span>{it.label}</span>
        </button>
      ))}
    </nav>
  );
}
